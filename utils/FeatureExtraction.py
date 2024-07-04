import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def bag_of_words(corpus):
    """
    This function converts the corpus into a Bag of Words representation.

    Parameters:
    corpus (list): A list of documents where each document is a string.

    Returns:
    X_matrix (sparse matrix): The Bag of Words representation of the corpus.
    feature_names (list): The names of the features in the Bag of Words representation.
    df: (Dataframe): a X and feature names pandas dataframe
    """
    # Initialize the CountVectorizer
    vectorizer = CountVectorizer(input='content',
                                 encoding='utf-8',
                                 strip_accents='unicode',
                                 lowercase=True,
                                 ngram_range=(1, 1),
                                 analyzer='word',
                                 max_df=1.0,
                                 min_df=1,
                                 max_features=None)

    # Fit the vectorizer to the corpus and transform the corpus into a sparse matrix
    # Get the names of the features in the Bag of Words representation
    # Convert X_matrix ro array and combine with names of the features ro create a datafrma
    X_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    df = pd.DataFrame.sparse.from_spmatrix(X_matrix, columns=feature_names)
    
    return X_matrix, feature_names, df
 

def tf_idf(corpus):
    """
    This function converts the corpus into a TF-IDF representation.

    Parameters:
    corpus (list): A list of documents where each document is a string.

    Returns:
    X_matrix (sparse matrix): The TF-IDF representation of the corpus.
    feature_names (list): The names of the features in the TF-IDF representation.
    df: (Dataframe): a X and feature names pandas dataframe
    """
    # Initialize the TfidfVectorizer
    vectorizer = TfidfVectorizer(input='content',
                                 encoding='utf-8',
                                 strip_accents='unicode',
                                 lowercase=True,
                                 ngram_range=(1, 1),
                                 analyzer='word',
                                 max_df=1.0,
                                 min_df=1,
                                 max_features=None,
                                 norm='l2',
                                 use_idf=True,
                                 smooth_idf=True,
                                 sublinear_tf=False)

    # Fit the vectorizer to the corpus and transform the corpus into a sparse matrix
    # Get the names of the features in the TF-IDF representation
    # Convert X_matrix ro array and combine with names of the features ro create a datafrma
    X_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    df = pd.DataFrame.sparse.from_spmatrix(X_matrix, columns=feature_names)

    return X_matrix, feature_names, df
    