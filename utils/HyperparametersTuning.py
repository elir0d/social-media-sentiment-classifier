import pandas as pd
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, RandomizedSearchCV

def model_hypertuning(Xarg, yarg, classifiers, hyper_params):
    """
    Performs hyperparameter tuning for the specified model using GridSearchCV.

    Args:
        Xarg (DataFrame or ndarray): The input data.
        yarg (Series or ndarray): The target variable.
        classifiers (dict): A dictionary where each key is the name of a model and the value is the model object.
        hyper_params (dict): A dictionary where each key is the name of a model and the value is a dictionary of hyperparameters for that model.

    Returns:
        tuple: A tuple containing:
            - best_params (dict): The best parameters found by GridSearchCV.
            - report_df (pandas.DataFrame): A DataFrame containing the performance metrics for each model.
    """
    # Define the train data
    X, y = Xarg, yarg
    models = classifiers
    grid_params = hyper_params
    best_params = {}
    report_data_frame = []

    # hypertuning the models provided
    for model_name, model in models.items():
        # Define grid search
        grid = grid_params[model_name]
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=69)
        grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
        random_search = RandomizedSearchCV(model, grid, random_state=69)
    
        # Fit the grid search to the data
        print(100*'-'+f'\nTuning: {model_name} Classifier with the follow hyperparameters:\n{grid}\n'+100*'-')
        grid_result = random_search.fit(X, y)
        best_params[model_name] = grid_result.best_params_

        # Store the correspondent params
        report_data_frame.append({
            "Model": model_name,
            "Best Params": grid_result.best_params_,
            "Accuracy": grid_result.best_score_,
        })
        
    report_df = pd.DataFrame(report_data_frame)
    
    return best_params, report_df