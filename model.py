def get_classifier(model_name, **kwargs):
    """
    Returns a classifier instance based on the specified model name.

    Parameters:
    - model_name (str): The name of the model. Supported models include:
        'xgboost', 'lightgbm', 'logistic_regression', 'random_forest', 'svc', 'gradient_boosting', etc.
    - **kwargs: Additional keyword arguments to pass to the classifier's constructor.

    Returns:
    - clf: An instantiated classifier object.

    Raises:
    - ValueError: If an unsupported model name is provided.
    """
    model_name = model_name.lower()
    
    if model_name == 'xgboost':
        from xgboost import XGBClassifier
        clf = XGBClassifier(**kwargs)
        
    elif model_name == 'lightgbm':
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(**kwargs)
        
    elif model_name == 'logistic_regression':
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(**kwargs)
        
    elif model_name == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(**kwargs)
        
    elif model_name == 'svc':
        from sklearn.svm import SVC
        clf = SVC(**kwargs)
        
    elif model_name == 'gradient_boosting':
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(**kwargs)

    elif model_name == 'neural_network':
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense

        clf = Sequential()
        clf.add(Dense(kwargs.get('hidden_units_1', 64), input_dim=kwargs.get('input_dim', 10), activation='relu'))
        clf.add(Dense(kwargs.get('hidden_units_2', 32), activation='relu'))
        clf.add(Dense(1, activation='sigmoid'))  
        
        clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    else:
        raise ValueError(f"Model '{model_name}' is not supported. Choose from 'xgboost', 'lightgbm', 'logistic_regression', 'random_forest', 'svc', 'gradient_boosting', 'neural_network'.")

        
    return clf
