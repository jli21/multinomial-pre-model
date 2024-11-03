import os
import time
from processing import *

st.set_page_config(layout="wide")

if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)

num_cores = os.cpu_count() - 2

def shutdown_ray():
    if ray.is_initialized():
        ray.shutdown()

st.session_state["_ray_shutdown"] = shutdown_ray

st.sidebar.markdown(
    """
    <style>
    .center-btn { display: flex; justify-content: center; margin-bottom: 20px; }
    .center-btn > button { width: 80%; }
    </style>
    """, unsafe_allow_html=True
)

def main():
    st.sidebar.markdown('<div class="center-btn">', unsafe_allow_html=True)
    run_button = st.sidebar.button("Run", key="run_button")
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    # DATA SELECTION
    st.sidebar.header("📊 Data Selection")
    years = [2019, 2020, 2021, 2022, 2023]
    train_years = st.sidebar.multiselect("Select Training Years", years, default=[2021, 2022])
    test_years = st.sidebar.multiselect("Select Testing Years", years, default=[2023])

    if any(year in train_years for year in test_years):
        st.sidebar.warning("Training and testing years should not overlap.")

    # MODEL HYPERPARAMETERS
    st.sidebar.header("⚙️ Model Hyperparameters")
    model_parameters = {
        "XGBoost": {"learning_rate": 0.02, "max_depth": 7, "n_estimators": 150},
        "LightGBM": {"learning_rate": 0.02, "max_depth": 7, "n_estimators": 150},
        "Logistic Regression": {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 100},
        "Random Forest": {"n_estimators": 150, "max_depth": 7, "min_samples_split": 6, "min_samples_leaf": 3},
        "Gradient Boosting": {"learning_rate": 0.02, "max_depth": 7, "n_estimators": 150, "min_samples_split": 6}
    }

    params_xgb = {
        "learning_rate": st.sidebar.number_input("XGBoost Learning Rate", 0.01, 0.1, model_parameters["XGBoost"]["learning_rate"]),
        "max_depth": st.sidebar.slider("XGBoost Max Depth", 3, 10, model_parameters["XGBoost"]["max_depth"]),
        "n_estimators": st.sidebar.number_input("XGBoost Estimators", 50, 500, model_parameters["XGBoost"]["n_estimators"])
    }
    params_lgbm = {
        "learning_rate": st.sidebar.number_input("LightGBM Learning Rate", 0.01, 0.1, model_parameters["LightGBM"]["learning_rate"]),
        "max_depth": st.sidebar.slider("LightGBM Max Depth", 3, 10, model_parameters["LightGBM"]["max_depth"]),
        "n_estimators": st.sidebar.number_input("LightGBM Estimators", 50, 500, model_parameters["LightGBM"]["n_estimators"])
    }
    params_lr = {
        "C": st.sidebar.number_input("Logistic Regression C", 0.1, 10.0, model_parameters["Logistic Regression"]["C"]),
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": model_parameters["Logistic Regression"]["max_iter"]
    }
    params_rf = {
        "n_estimators": st.sidebar.number_input("Random Forest Estimators", 50, 500, model_parameters["Random Forest"]["n_estimators"]),
        "max_depth": st.sidebar.slider("Random Forest Max Depth", 3, 10, model_parameters["Random Forest"]["max_depth"]),
        "min_samples_split": model_parameters["Random Forest"]["min_samples_split"],
        "min_samples_leaf": model_parameters["Random Forest"]["min_samples_leaf"],
        "bootstrap": True
    }
    params_gb = {
        "learning_rate": st.sidebar.number_input("Gradient Boosting Learning Rate", 0.01, 0.1, model_parameters["Gradient Boosting"]["learning_rate"]),
        "max_depth": st.sidebar.slider("Gradient Boosting Max Depth", 3, 10, model_parameters["Gradient Boosting"]["max_depth"]),
        "n_estimators": st.sidebar.number_input("Gradient Boosting Estimators", 50, 500, model_parameters["Gradient Boosting"]["n_estimators"]),
        "min_samples_split": model_parameters["Gradient Boosting"]["min_samples_split"]
    }

    # MODEL SELECTION 
    st.sidebar.header("📈 Select Model Display")
    selected_models = {name: st.sidebar.checkbox(name, value=True) for name in model_parameters.keys()}

    @st.cache_data(show_spinner=False)
    def get_model_data(model_name, _clf):
        Y_prob = binary_classifier(_clf, X_train, Y_train, X_test, Y_test, cv_value=10)
        odds_df = pd.read_csv("bsktball/returns/odds_raw.csv", index_col=[0, 1])
        returns = preprocess_odds(odds_df, ['Unibet', 'bet365'])
        prep_df = prep_simulation(Y_prob, Y_test, time_id, returns, factor=0.2)
        res_dict = find_total(prep_df.to_dict('index'))
        res = pd.DataFrame(res_dict).T
        res["Model"] = model_name
        return Y_prob, res.reset_index()

    if run_button:
        st.cache_data.clear()
        start_time = time.time()

        # DATA
        df = combine_df(columns_selected=[
            'y.status', 'season', 'Marathonbet', '1xBet', 'Pinnacle', 'Unibet',
            'William Hill', 'team.elo.booker.lm', 'opp.elo.booker.lm', 'team.elo.booker.combined',
            'opp.elo.booker.combined', 'elo.prob', 'predict.prob.booker', 'predict.prob.combined',
            'elo.court30.prob', 'raptor.court30.prob', 'booker_odds.Pinnacle', 'elo_prob', 'raptor_prob'
        ])
        time_id = combine_df(columns_selected=['datetime', 'endtime'])
        df = prep_data(df)
        X_train, X_test, Y_train, Y_test = split_data(df, train_years, test_years, shuffle=False)

        # CLASSIFIER
        classifiers = {
            "XGBoost": get_classifier("xgboost", **params_xgb),
            "LightGBM": get_classifier("lightgbm", **params_lgbm),
            "Logistic Regression": get_classifier("logistic_regression", **params_lr),
            "Random Forest": get_classifier("random_forest", **params_rf),
            "Gradient Boosting": get_classifier("gradient_boosting", **params_gb)
        }

        model_data = {name: get_model_data(name, clf) for name, clf in classifiers.items()}
        st.session_state["model_data"] = model_data
        st.session_state["classifier_names"] = list(classifiers.keys())

        end_time = time.time()
        st.write(f"Time taken: {end_time - start_time:.2f} seconds")

    model_data = st.session_state.get("model_data", {})
    classifier_names = st.session_state.get("classifier_names", [])

    tabs = st.tabs(["Graph", "Model Output Probabilities and Bets"])

    # PLOT 
    with tabs[0]:  
        combined_data = [model_data[model_name][1] for model_name in selected_models if selected_models[model_name] and model_name in model_data]
        if combined_data:
            combined_df = pd.concat(combined_data, ignore_index=True)
            plot_bankroll(combined_df)  

    # PROBABILITY OUTPUTS
    with tabs[1]:  
        for i in range(0, len(classifier_names), 2):
            model_col1, model_col2 = st.columns(2)

            model_name1 = classifier_names[i]
            if model_name1 in model_data:
                with model_col1:
                    st.subheader(model_name1)
                    Y_prob1, res1 = model_data[model_name1]
                    st.expander("Probability Output", expanded=True).write(Y_prob1)
                    st.expander("Betting Timelapse", expanded=True).write(res1)

            if i + 1 < len(classifier_names):
                model_name2 = classifier_names[i + 1]
                if model_name2 in model_data:
                    with model_col2:
                        st.subheader(model_name2)
                        Y_prob2, res2 = model_data[model_name2]
                        st.expander("Probability Output", expanded=True).write(Y_prob2)
                        st.expander("Betting Timelapse", expanded=True).write(res2)

if __name__ == "__main__":
    main()
