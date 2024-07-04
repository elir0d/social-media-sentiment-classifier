from sklearn.model_selection import train_test_split

def split_dataset(X, y, test_size=0.3, random_state=69):
    """
    Splits the dataset into training and test sets.

    Args:
        X (pandas.DataFrame or numpy.ndarray): Features (input data).
        y (pandas.Series or numpy.ndarray): Target variable (labels).
        test_size (float, optional): Proportion of the test set. Default is 0.3 (30%).
        random_state (int, optional): Seed for shuffling the data. Default is 69.

    Returns:
        pandas.DataFrame or numpy.ndarray, pandas.DataFrame or numpy.ndarray:
            Training features, test features, training labels, test labels.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
