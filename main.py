from test_bsktball import *

# DATA 
columns_selected = ['y.status', 'season', 'Marathonbet', '1xBet', 'Pinnacle', 'Unibet', 
                    'William Hill', 'team.elo.booker.lm', 'opp.elo.booker.lm', 'team.elo.booker.combined', 
                    'opp.elo.booker.combined', 'elo.prob', 'predict.prob.booker', 'predict.prob.combined', 
                    'elo.court30.prob', 'raptor.court30.prob', 'booker_odds.Pinnacle', 'elo_prob', 'raptor_prob']

df = combine_df(columns_selected=columns_selected)
time_id = combine_df(columns_selected=['datetime', 'endtime'])
df = prep_data(df)

# SPLIT DATA
train_years = [2021,2022]
test_year = [2023]
X_train, X_test, Y_train, Y_test = split_data(df, train_years, test_year, False)

train_years_nn = [2020,2021]
val_year_nn = [2022]
test_year_nn = [2023]
X_train_nn, X_val, X_test_nn, Y_train_nn, Y_val, Y_test_nn = split_data_nn(df, train_years_nn, val_year_nn, test_year_nn)

# CLF 
clf = get_classifier(
    model_name =       'xgboost',
    learning_rate =     0.02,
    max_depth =         7,
    min_child_weight =  6,
    n_estimators =      150
)

clf = get_classifier(
    model_name =        'lightgbm',
    learning_rate =      0.02,
    max_depth =          7,
    min_child_samples =  6,  
    n_estimators =       150
)

clf = get_classifier(
    model_name =        'logistic_regression',
    C =                  1.0,              
    penalty =           'l2',       
    solver =            'lbfgs',     
    max_iter =           100         
)

clf = get_classifier(
    model_name =        'random_forest',
    n_estimators =       150,
    max_depth =          7,
    min_samples_split =  6,
    min_samples_leaf =   3,
    bootstrap =         True
)

clf = get_classifier(
    model_name =        'gradient_boosting',
    learning_rate =      0.02,
    max_depth =          7,
    min_samples_split =  6,
    n_estimators =       150
)

# CLF NEURAL NETWORK 
clf_nn = get_classifier(
    model_name =        'neural_network',
    input_dim =         X_train_nn.shape[1], 
    hidden_units_1 =    64,
    hidden_units_2 =    32
)

Y_prob = binary_classifier(
    clf,
    X_train, 
    Y_train, 
    X_test, 
    Y_test, 
    10
)

Y_prob_nn = binary_classifier_nn(
    clf_nn, 
    X_train_nn, 
    Y_train_nn, 
    X_test_nn, 
    Y_test_nn, 
    epochs = 10, 
    batch_size = 32, 
    validation_data = (X_val, Y_val)
)

odds_df = pd.read_csv("bsktball/returns/odds_raw.csv", index_col = [0,1])
returns = preprocess_odds(odds_df, ['Unibet', 'bet365'])

# ORDINARY CLF
prep_df = prep_simulation(Y_prob, Y_test, time_id, returns, factor=0.2)
res_dict = find_total(prep_df.to_dict('index'))

res = pd.DataFrame(res_dict).T
plot_bankroll(res)

# NEURAL NETWORK
prep_df = prep_simulation(Y_prob_nn, Y_test_nn, time_id, returns, factor=0.2)
res_dict = find_total(prep_df.to_dict('index'))

res = pd.DataFrame(res_dict).T
plot_bankroll(res)

# SIMULATION 
res_sim = run_single_simulation(Y_prob, time_id, returns, 0.2)
plot_bankroll(res_sim)
