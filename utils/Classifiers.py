from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np

"""
This Python script defines machine learning models and their hyperparameters.

Models:
- LogisticRegression
- MultinomialNB
- LinearSVC
- DecisionTreeClassifier

Hyperparameters:
- For LogisticRegression:
    - max_iter: [10000]
    - class_weight: ['balanced']
    - solver: ['newton-cg', 'lbfgs', 'saga', 'sag']
    - penalty: ['l2']
    - C: range(1, 101)

- For MultinomialNB:
    - alpha: range(1, 101)
    - force_alpha: [True]
    - fit_prior: [True]
    - class_prior: [None]

- For LinearSVC:
    - penalty: ['l2']
    - loss: ['hinge', 'squared_hinge']
    - dual: [False]
    - C: range(1, 101)
    - class_weight: ['balanced']
    - max_iter: [10000]

- For DecisionTreeClassifier:
    - criterion: ['gini', 'entropy']
    - splitter: ['best', 'random']
    - max_depth: range(2, 51)
    - min_samples_split: range(2, 21)
    - min_samples_leaf: range(2, 101)
    - max_features: ['None', 'sqrt', 'log2']
"""



models = {
        'LogisticRegression': LogisticRegression(),
        'MultinomialNB': MultinomialNB(),
        'LinearSVC': LinearSVC(),
        'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=2)

}

hyper_params = {
    'LogisticRegression':  {
        'max_iter': [10000],
        'class_weight': ['balanced'],
        'solver': ['newton-cg', 'lbfgs', 'saga', 'sag'],
        'penalty': ['l2'],
        'C': np.arange(1, 101, 1).tolist()
        
    },
    
    'MultinomialNB':  {
        'alpha': np.arange(1, 101, 1).tolist(),
        'force_alpha': [True],
        'fit_prior': [True],
        'class_prior': [None]
    },
    
    'LinearSVC': {
        'penalty': ['l2'], 
        'loss': ['hinge','squared_hinge'], 
        'dual':[False], 
        'C': np.arange(1, 101, 1).tolist(), 
        'class_weight': ['balanced'], 
        'max_iter': [10000]
    },

    'DecisionTreeClassifier': {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': np.arange(2, 51).tolist(),
        'min_samples_split': np.arange(2, 21).tolist(),
        'min_samples_leaf': np.arange(2, 101).tolist(),
        'max_features': ['None','sqrt', 'log2']
    }
}

