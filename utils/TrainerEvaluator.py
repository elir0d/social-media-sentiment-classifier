from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, RocCurveDisplay
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def train_model(Xarg, yarg, models, threshold=0.7, params=None):
    """
    This Method trains a specified model on the given data.

    Parameters:
    Xarg (DataFrame or ndarray): The input data.
    yarg (Series or ndarray): The target variable.
    model (str): The name of the model to be trained.
    params (dict, optional): The parameters to be set for the model. If None, default parameters are used.

    Returns:
    model_fitted (estimator): The trained model.
    """
    cutoff = threshold
    # Define the dataset
    X, y = Xarg, yarg

    # Create a dictionary of model classifiers
    classifier = models
    
    # Instantiate the model
    classifier = classifier[model]
    if params is not None:
        classifier.set_params(**params)
    
    # Train the model
    print(100*'-' + f'\nTraining: {model} Classifier with the follow hyperparameters:\n{params}\n' + 100*'-')
    model_fitted = classifier.fit(X, y)
    print('Training completed successfully!')
    return model_fitted

def make_predictions(Xarg, model_fited):
    """
    This Method evaluates a trained model on the given data.

    Parameters:
    Xarg (DataFrame or ndarray): The input data.
    model_fited (trained estimator): The trained model.

    Returns:
    predict (ndarray): The predictions of the model on the input data.
    """
    
    # Make predictions
    predict = model_fited.predict(Xarg)
    
    # Generate and return the classification report
    return predict

def compare_models(base_model, tuned_model, test_labels):
    """
    Compares the performance of two classification models.

    Args:
        base_model: The base model (trained without hyperparameter tuning).
        tuned_model: The tuned model (trained with hyperparameter tuning/feature selection or both).
        X_test_features: Test features.
        y_test_labels: Test labels.

    Returns:
        float: Accuracy gain after tuning (percentage).
    """
    # Predictions from base model
    # Predictions from tuned model
    base_model_predictions = base_model
    tuned_model_predictions = tuned_model
    print(base_model_predictions)
    print(tuned_model_predictions)
    
    # Calculate accuracy for both models
    base_accuracy = (base_model_predictions == test_labels).mean() * 100
    tuned_accuracy = (tuned_model_predictions == test_labels).mean() * 100
    
    # Calculate accuracy gain
    accuracy_gain = (tuned_accuracy - base_accuracy)
    print(15*"=" + "Comparison Results" + 15*"=")
    print(f"Base Model Accuracy: {base_accuracy:>26.2f} %")
    print(f"Tuned Model Accuracy: {tuned_accuracy:>25.2f} %")
    print(f"Accuracy Gain after Tuning: {accuracy_gain:>19.2f} %")
    print(15*"=" + "__________________" + 15*"=")

    return accuracy_gain

def evaluate_multiple_models(train_data, test_data, classifiers, grid_params):
    """
    Trains and evaluates multiple machine learning models using scikit-learn.

    Args:
        classifiers (dict): A dictionary where each key is the name of a model and the value is the model object.
        grid_params (dict): A dictionary where each key is the name of a model and the value is a dictionary of hyperparameters for that model.
        train_data (tuple): A pair of X_train and y_train, representing the training data.
        test_data (tuple): A pair of X_test and y_test, representing the test data.

    Returns:
        pandas.DataFrame: A DataFrame containing the performance metrics for each model.
    """
    # define train and test data
    X_train, y_train = train_data
    X_test, y_test = test_data
    models = classifiers
    hyper_params = grid_params
    metrics = []

    # Loop to iterate between models and train them one by one
    print(100*'=')
    for model_name, model in models.items():
        # Train the model
        params = model.get_params()
        if bool(hyper_params):
            params = hyper_params[model_name]
        model.set_params(**params)
        print(100*'-'+f'\nTraining: {model_name} Classifier with the follow hyperparameters:\n{params}')
        model.fit(X_train, y_train)
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        fscore = f1_score(y_test, y_pred, average='weighted')
        
        # Store the results
        metrics.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": fscore,
        })
        
    print(100*'=')
    # Create and return a DataFrame with the results
    metrics_df = pd.DataFrame(metrics)
    return metrics_df

def evaluate_model(Xarg, yarg, model_fited):
    """
    This Method evaluates a trained model on the given data.

    Parameters:
    Xarg (DataFrame or ndarray): The input data.
    yarg (Series or ndarray): The target variable.
    model (estimator): The trained model.

    Returns:
    predict (ndarray): The predictions of the model on the input data.
    report (str): The classification report showing the main classification metrics.
    """
    
    # Make predictions
    predict = model_fited.predict(Xarg)
    
    # Generate and return the classification report
    return predict, classification_report(yarg, predict, digits=3)
    
def cross_validate_multiple_models(train_data, classifiers, grid_params, threshold=0.7):
    """
    Trains and evaluates multiple machine learning models using scikit-learn.

    Args:
        classifiers (dict): A dictionary where each key is the name of a model and the value is the model object.
        grid_params (dict): A dictionary where each key is the name of a model and the value is a dictionary of hyperparameters for that model.
        train_data (tuple): A pair of X_train and y_train, representing the training data.

    Returns:
        pandas.DataFrame: A DataFrame containing the performance metrics for each model.
        metrics(Dict): A dictionary containing the performance metrics for each model
    """
    # Define train data
    X_train, y_train = train_data
    models = classifiers
    hyper_params = grid_params
    scores = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    metrics = []
    kfold = 10
    cutoff = threshold
    
    # Loop to iterate between models and train them one by one
    print(100 * '=')
    for model_name, model in models.items():
        # Train the model
        params = model.get_params()
        if bool(hyper_params):
            params = hyper_params[model_name]
            model.set_params(**params)
        print(100 * '-' + f'\n Cross validating: {model_name} Classifier with the following hyperparameters:\n{model.get_params()}')
        cv_scores = cross_validate(model, X_train, y_train, cv=kfold, scoring=scores)

        accuracy = cv_scores['test_accuracy']
        precision = cv_scores['test_precision_macro']
        recall = cv_scores['test_recall_macro']
        fscore = cv_scores['test_f1_macro']
        
        metrics.append({
            "Model": model_name,
            "Accuracy": accuracy.mean(),
            "Precision": precision.mean(),
            "Recall": recall.mean(),
            "F1-score": fscore.mean(),
        })

    # Create a DataFrame with the metrics
    metrics_df = pd.DataFrame(metrics)
    return metrics_df

def plot_roc_curve(train_data, classifiers, hyper_params):
    # Defina seus modelos
    models = classifiers
    X, y = train_data
    y = y.replace({'positivo':1, 'negativo':0})
    params = hyper_params
    # Para cada modelo
    for model_name, model in models.items():
        # Calibre o modelo
        params = model.get_params()
        if bool(hyper_params):
            params = hyper_params[model_name]
            model.set_params(**params)
        calibrated_model = CalibratedClassifierCV(model, cv=10)
        
        # Use cross_val_predict para obter as probabilidades de previsão
        y_scores = cross_val_predict(calibrated_model, X, y, cv=10, method='predict_proba')

        name = {
                'LogisticRegression': 'Regressão Logística',
                'MultinomialNB': 'Naive Bayes',
                'LinearSVC': 'SVM',
                'DecisionTreeClassifier': 'Árvore de Decisão'
        }
        
        
        # Calcule a curva ROC
        fpr, tpr, thresholds = roc_curve(y, y_scores[:, 1])  # Supondo que a classe positiva é 1
        
        # Calcule a AUC (área sob a curva)
        roc_auc = auc(fpr, tpr)
        
        # Plote a curva ROC
        plt.plot(fpr, tpr, label='%s (AUC = %0.2f)' % (name[model_name], roc_auc))

    # Adicione detalhes ao gráfico
    plt.plot([0, 1], [0, 1], 'k--')  # Linha de chance
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curvas ROC')
    plt.legend(loc="lower right")
    plt.show()