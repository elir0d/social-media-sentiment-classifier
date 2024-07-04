import pandas as pd    
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import RFE

def remove_features_with_low_variance(VectorizedCorpus, vectorizer=None):
    """
    This Method removes features with low variance from the dataset

    Parameters:
    VectorizedCorpus (SparseMatrix): The input data with features to be selected.
    vectorizer (Vectorizer, optional): The vectorizer used for feature extraction - CountVectorizer or TfidfVectorizer
    It can be either CountVectorizer or TfidfVectorizer. If None, all features are selected.

    Returns:
    X_selected (ndarray): The data with only the selected features.
    X_selected_df (DataFrame): The DataFrame version of X_selected.
    selected_features (Index): The names of the selected features.
    """
    
    # Define the dataframe
    dataframe = VectorizedCorpus
    
    # Create the variance selector
    # The threshold for variance depends on the vectorizer used on the sparse matrix
    vector_gate = {CountVectorizer: 0.8 * (1 - 0.8), TfidfVectorizer: 0.01}
    threshold = vector_gate.get(vectorizer, 0.01)
    
    # Initialize the VarianceThreshold selector
    # Fit the selector to the data and transform the data
    selector = VarianceThreshold(threshold=threshold)
    X_selected = selector.fit_transform(dataframe)
    
    # Get the names of the original features
    # Select only the features that were kept
    feature_names = dataframe.columns[selector.get_support()]
    X_selected_df = pd.DataFrame.sparse.from_spmatrix(X_selected, columns=feature_names)
    
    return X_selected, X_selected_df, feature_names

#-----------------------------------------------------------------------------------------------

# Applying Univariate feature selection to select the best features
def univariate_feature_selection(VectorizedCorpus, Yarg, k=50):
    """
    This Method applies Univariate Feature Selection for feature selection.

    Parameters:
    VectorizedCorpus (SparseMatrix): The input data with features to be selected.
    Yarg (Series): The target variable.
    k (int, optional): The number of top features to select. If None, all features are selected.

    Returns:
    X_selected (ndarray): The data with only the selected features.
    X_selected_df (DataFrame): The DataFrame version of X_selected.
    selected_features (Index): The names of the selected features.
    """

    # This lambda function calculates the number of features to retain in the dataset.
    # It multiplies the total number of features (len(VectorizedCorpus)) by the specified percentage (k)
    # and then divides by 100 to determine the final count of features to keep.
    # For example, if `len(VectorizedCorpus)` is 1000 and `k` is 75, the result will be 750 (75% of 1000).
    k_percent = lambda dataset_len, percentual: (dataset_len.shape[1] * percentual) // 100
    
    # Define the dataframe and target column
    # Choose the number of best features (k)
    dataframe = VectorizedCorpus
    target_column = Yarg
    features_to_keep = k_percent(VectorizedCorpus, k)
    
    # Initialize the selector
    # Fit and transform the data
    selector = SelectKBest(score_func=chi2, k=features_to_keep)
    X_selected = selector.fit_transform(dataframe, target_column)
    
    # Get the names of the original features
    # Select only the features that were kept
    feature_names = dataframe.columns[selector.get_support()]
    X_selected_df = pd.DataFrame.sparse.from_spmatrix(X_selected, columns=feature_names)

    return X_selected, X_selected_df, feature_names

#-----------------------------------------------------------------------------------------------
def recursive_feature_elimination(VectorizedCorpus, Yarg, model, k=10):
    """
    This Method applies Recursive Feature Elimination (RFE) to reduce the number of features in the dataset.
    It can be apply on the follow estimators choosed on this project:
        LogisticRegression,
        RandomForestClassifier,
        XGBClassifier
        LinearSVC,

    Parameters:
    VectorizedCorpus (SparseMatrix): The input data with features to be selected.
    Yarg (Series): The target variable.
    model (estimator): The machine learning estimator to be used for feature ranking. It must have a `fit` method.
    k (int, optional): The percentul of top features to select. If None, half of the features are selected.

    Returns:
    X_selected (ndarray): The data with only the selected features.
    X_selected_df (DataFrame): The DataFrame version of X_selected.
    selected_features (Index): The names of the selected features.
    """

    # This lambda function calculates the number of features to retain in the dataset.
    # It multiplies the total number of features (len(VectorizedCorpus)) by the specified percentage (k)
    # and then divides by 100 to determine the final count of features to keep.
    # For example, if `len(VectorizedCorpus)` is 1000 and `k` is 75, the result will be 750 (75% of 1000).
    k_percent = lambda dataset_len, percentual: round((len(dataset_len.columns) * percentual) // 100)
    
    # Deifne the X dataframe, define Y as a target value and define a K number
    dataframe = VectorizedCorpus
    target_column = Yarg
    features_to_keep = k_percent(VectorizedCorpus, k)
    
    # Define the estimator and Fit the RFE selector to the data
    estimator = model
    selector = RFE(estimator, n_features_to_select=features_to_keep)
    selector.fit(dataframe, target_column)
    
    # Get the names of the original features
    # Select only the features that were kept
    # Convert the selected data to a DataFrame
    feature_names = dataframe.columns[selector.support_]
    X_selected = selector.transform(dataframe)
    X_selected_df = pd.DataFrame.sparse.from_spmatrix(X_selected, columns=feature_names)
    
    return X_selected, X_selected_df, feature_names