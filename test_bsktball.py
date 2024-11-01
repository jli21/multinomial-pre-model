import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

from model import * 

def combine_df(folder_name='bsktball', columns_selected=None, drop_na=True):
    """
    Combine columns from all CSV files in the specified folder into a single dataframe.

    Parameters:
    folder_name (str): The name of the folder containing the CSV files.

    Returns:
    pd.DataFrame: A dataframe containing the combined columns from all CSV files.
    """
    df_list = []

    folder_path = os.path.join(os.getcwd(), folder_name)

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path, index_col=[0, 1])
            df_list.append(df)

    combined_df = pd.concat(df_list, axis=1)[columns_selected]
    if drop_na:
        combined_df = combined_df.dropna(how='any')
    
    return combined_df

def prep_data(df):
    """
    Preprocess the dataframe by dropping duplicate columns, converting all columns to floats,
    and setting the second level index to a new column named 'home_status'.
    
    Parameters:
    df (pd.DataFrame): The input dataframe with a multiindex.
    
    Returns:
    pd.DataFrame: The preprocessed dataframe.
    """
    df_T = df.T.drop_duplicates().T

    df = df_T.astype(float)

    df['home_status'] = df.index.get_level_values(1)
    return df

def split_data(df, train_years, test_year, shuffle=False):
    """
    Split the dataframe into training and testing sets based on the specified seasons and drop the 'season' column.
    
    Parameters:
    df (pd.DataFrame): The input dataframe.
    train_years (list): List of seasons to be used for training.
    test_year (int): The season to be used for testing.
    shuffle (bool): Whether to shuffle the data before splitting. Default is False.
    """
    train_df = df[df['season'].isin(train_years)]
    test_df = df[df['season'].isin(test_year)]
    
    X_train = train_df.drop(columns=['y.status', 'season'])
    Y_train = train_df['y.status']
    
    X_test = test_df.drop(columns=['y.status', 'season'])
    Y_test = test_df['y.status']
    
    if shuffle:
        X_train, Y_train = train_test_split(
            train_df.drop(columns=['y.status', 'season']), train_df['y.status'],
            test_size=0.2, random_state=42, shuffle=shuffle, stratify=train_df['season']
        )
        X_test, Y_test = train_test_split(
            test_df.drop(columns=['y.status', 'season']), test_df['y.status'],
            test_size=0.2, random_state=42, shuffle=shuffle, stratify=test_df['season']
        )
    
    return X_train, X_test, Y_train, Y_test

def split_data_nn(df, train_years, val_year, test_year):
    """
    Split the dataframe into training, validation, and testing sets based on specified seasons.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    train_years (list): List of seasons to be used for training.
    val_year (int): The season to be used for validation.
    test_year (int): The season to be used for testing.

    Returns:
    Tuple of (X_train, X_val, X_test, Y_train, Y_val, Y_test)
    """
    train_df = df[df['season'].isin(train_years)]
    val_df = df[df['season'].isin(val_year)]
    test_df = df[df['season'].isin(test_year)]

    X_train = train_df.drop(columns=['y.status', 'season'])
    Y_train = train_df['y.status']

    X_val = val_df.drop(columns=['y.status', 'season'])
    Y_val = val_df['y.status']

    X_test = test_df.drop(columns=['y.status', 'season'])
    Y_test = test_df['y.status']

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def binary_classifier(clf, X_train, Y_train, X_test, Y_test, cv_value, normalize_probs=True):
    clf.fit(X_train, Y_train)
    
    calibrated_clf = CalibratedClassifierCV(clf, cv=cv_value)
    calibrated_clf.fit(X_train, Y_train)
    
    Y_train_pred_adj = calibrated_clf.predict_proba(X_train)[:, 1]
    Y_pred_prob_adj = calibrated_clf.predict_proba(X_test)[:, 1]
    
    if normalize_probs:
        Y_train_pred = Y_train_pred_adj / (np.repeat(Y_train_pred_adj[0::2] + Y_train_pred_adj[1::2], 2))
        Y_pred_prob = Y_pred_prob_adj / (np.repeat(Y_pred_prob_adj[0::2] + Y_pred_prob_adj[1::2], 2))
    else:
        Y_train_pred = Y_train_pred_adj
        Y_pred_prob = Y_pred_prob_adj
    
    Y_pred = [1 if p > 0.5 else 0 for p in Y_pred_prob]
    Y_train_pred_bin = [1 if p > 0.5 else 0 for p in Y_train_pred]
    
    if hasattr(clf, 'feature_importances_'):
        feature_importances = clf.feature_importances_
        print(pd.DataFrame(data = feature_importances, index = X_train.columns, columns=["score"]).sort_values(by="score", ascending=False).head(30))
    else:
        print("Model does not have feature_importances_ attribute.")
    
    Y_test = Y_test.dropna()
    if Y_test.size == 0:
        print("\nTest Accuracy is not available")
    elif len(Y_test) == len(Y_pred):
        acc = accuracy_score(Y_test, Y_pred)
        print("\nTest Accuracy is %s" % (acc))
    else:
        acc = accuracy_score(Y_test, Y_pred[:len(Y_test)])
        print("\nTest Accuracy is %s" % (acc))
    
    print("Train Accuracy is %.3f" % accuracy_score(Y_train, Y_train_pred_bin))
    return pd.DataFrame(index=X_test.index, data=Y_pred_prob, columns=['Y_prob'])

def binary_classifier_nn(clf, X_train, Y_train, X_test, Y_test, 
                         epochs = 10, batch_size = 32, validation_data = None):

    clf.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

    Y_train_pred_adj = clf.predict(X_train).flatten()
    Y_pred_prob_adj = clf.predict(X_test).flatten()

    Y_pred = [1 if p > 0.5 else 0 for p in Y_pred_prob_adj]
    Y_train_pred_bin = [1 if p > 0.5 else 0 for p in Y_train_pred_adj]

    if not Y_test.empty and len(Y_test) == len(Y_pred):
        print(f"\nTest Accuracy: {accuracy_score(Y_test, Y_pred):.3f}")
    print(f"Train Accuracy: {accuracy_score(Y_train, Y_train_pred_bin):.3f}")

    return pd.DataFrame(index = X_test.index, data=Y_pred_prob_adj, columns=['Y_prob'])

def kelly_bet_size(prob, returns, alpha):
    bet_size = alpha * ((prob * (returns)) - (1 - prob)) / (returns)
    return max(0, bet_size)

def preprocess_odds(odds_df, cols):
    def american_to_decimal(american_odds):
        if american_odds > 0:
            return (american_odds / 100) 
        else:
            return (100 / abs(american_odds)) 
    
    for col in odds_df.columns:
        odds_df[col] = odds_df[col].apply(american_to_decimal)

    return odds_df[cols]

def assign_kelly_bets(Y_prob, returns, factor):

    df_all = Y_prob.join(returns)

    df_all['kelly_bet'] = df_all.apply(lambda row: kelly_bet_size(row['Y_prob'], row[returns.columns].max(), factor), axis=1)
    
    return df_all

def prep_simulation(Y_prob, Y_test, time_data, returns, factor):
    df = assign_kelly_bets(Y_prob, returns, factor)
    df = df.join(Y_test).join(time_data)
    
    prep_dict = {}

    for idx, row in df.iterrows():
        game_id, home_status = idx
        start_time = row['datetime']
        end_time = row['endtime']

        odds = row[returns.columns].max()
        implied_prob = 1 / odds
        adjusted_implied_prob = implied_prob / (1 + implied_prob)
        
        if row['kelly_bet'] > 0:

            prep_dict[(game_id, 1)] = {
                'datetime': start_time,
                'home_status': home_status,
                'prob': row['Y_prob'],
                'kelly_bet': row['kelly_bet'],
                'status': row['y.status'],
                'odd': odds,
                'return': (odds) * row['y.status'] * row['kelly_bet'],
                'implied_prob': adjusted_implied_prob
            }
        
            prep_dict[(game_id, 0)] = {
                'datetime': end_time,
                'home_status': home_status,
                'prob': row['Y_prob'],
                'kelly_bet': row['kelly_bet'],
                'status': row['y.status'],
                'odd': odds,
                'return': (odds) * row['y.status'] * row['kelly_bet'],
                'implied_prob': adjusted_implied_prob
            }

    prep_df = pd.DataFrame.from_dict(prep_dict, orient='index').sort_values(by='datetime')
    return prep_df

def find_total(dict_returns):
    """
    Calculate the total pool of money available after each bet, considering bet sizes and returns from resolved bets.

    Parameters:
    dict_returns (dict): Dictionary with bet details and outcomes.

    Returns:
    dict: Updated dictionary with pre_total and total calculated.
    """
    dict_returns[list(dict_returns)[0]]['pre_total'] = 1
    dict_returns[list(dict_returns)[0]]['total'] = 1 - dict_returns[list(dict_returns)[0]]['kelly_bet']

    keys = list(dict_returns.keys())
    for k in keys[1:]:
        dict_returns[k]['pre_total'] = dict_returns[keys[keys.index(k) - 1]]['total']
        if k[1] == 1:
            dict_returns[k]['total'] = dict_returns[k]['pre_total'] * (1 - dict_returns[k]['kelly_bet'])
        if k[1] == 0:
            if dict_returns[k]['return'] == 0:
                dict_returns[k]['total'] = dict_returns[k]['pre_total']
            else:
                dict_returns[k]['total'] = dict_returns[k]['pre_total'] + dict_returns[(k[0], 1)]['pre_total'] * (dict_returns[k]['return'] + dict_returns[k]['kelly_bet'])
    return dict_returns

def plot_bankroll(df):
    if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
        df['datetime'] = pd.to_datetime(df['datetime'])

    df.set_index('datetime', inplace=True)
    daily_data = df['total'].resample('D').mean()
    plt.figure(figsize=(10, 6))

    plt.plot(daily_data.index, daily_data.values, label='Bankroll', color='blue')
    plt.gcf().autofmt_xdate() 
    plt.xlabel('Time')
    plt.ylabel('Bankroll')
    plt.title('Bankroll Over Time')
    plt.legend()
    plt.tight_layout() 
    plt.show()

# SIMULATION 

def generate_y_test(y_prob_df):
    y_test_dict = {}

    for game_id, group in y_prob_df.groupby(level=0):
        prob_0 = group.loc[(game_id, 0), 'Y_prob']
        prob_1 = group.loc[(game_id, 1), 'Y_prob']
        
        result_0 = np.random.choice([1, 0], p=[prob_0, 1 - prob_0])
        result_1 = 1 - result_0  
        y_test_dict[(game_id, 0)] = result_0
        y_test_dict[(game_id, 1)] = result_1

    y_test_series = pd.Series(y_test_dict)
    y_test_series.index = pd.MultiIndex.from_tuples(y_test_series.index, names=['game_id', 'homeStatus'])
    y_test_series.name = 'y.status'

    return y_test_series

def run_single_simulation(y_prob_df, time_data, returns_df, factor):
    Y_test = generate_y_test(y_prob_df) 
    prep_df = prep_simulation(y_prob_df, Y_test, time_data, returns_df, factor)  
    res_dict = find_total(prep_df.to_dict('index'))  
    return pd.DataFrame(res_dict).T

