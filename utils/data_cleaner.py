import re
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

nlp = spacy.load('pt_core_news_lg')

def remove_usernames(text):
    """
    Method to remove social media usernames from the text.
    :param text: str, input text from which usernames are to be removed.
    :return: str, text with usernames removed.
    """
    return re.sub('@[^\s]+', '', text)

def remove_urls(text):
    """
    Method to remove URLs from the text.
    :param text: str, input text from which URLs are to be removed.
    :return: str, text with URLs removed.
    """
    return re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

def remove_hashtags(text):
    """
    Method to remove hashtags from the text.
    :param text: str, input text from which hashtags are to be removed.
    :return: str, text with hashtags removed.
    """
    return re.sub('#[^\s]+', '', text)

def remove_non_alphabets(text):
    """
    Method to remove non-alphabetic characters.
    :param text: str, input text from which non-alphabetic characters are to be removed.
    :return: str, text with non-alphabetic characters removed.
    """
    return re.sub('[^a-zA-Z]', ' ', text)

def to_lower(text):
    """
    Method to convert the text to lowercase.
    :param text: str, input text to be converted to lowercase.
    :return: str, text converted to lowercase.
    """
    return text.lower()

def bag_of_words( corpus):
    """
    Method to convert the corpus into a Bag of Words representation.
    :param corpus: list, a list of documents where each document is a string.
    :return: tuple, a tuple containing the Bag of Words representation and the feature names.
    """
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus).todense()
    return vectorizer.get_feature_names_out()

def tf_idf( corpus):
    """
    Method to convert the corpus into a TF-IDF representation.
    :param corpus: list, a list of documents where each document is a string.
    :return: tuple, a tuple containing the TF-IDF representation and the feature names.
    """
    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(corpus).todense() 
    X = vectorizer.get_feature_names_out()
    return X

def tokenize(text):
    """
    Method to tokenize the text.
    :param text: str, input text to be tokenized.
    :return: list, a list of tokens.
    """
    doc = nlp(text)
    return [token.text for token in doc]

def remove_stopwords(text):
    """
    Method to remove irrelevant words (stop words).
    :param text: str, input text from which stop words are to be removed.
    :return: str, text with stop words removed.
    """
    doc = nlp(text)
    return ' '.join([token.text for token in doc if not token.is_stop])

def lemmatize(text):
    """
    Method to apply lemmatization, which is the process of converting a word to its base form.
    :param text: str, input text to be lemmatized.
    :return: str, text with words lemmatized.
    """
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

def pos_tagging(text):
    """
    Method to identify the part of speech of each word.
    :param text: str, input text to be part-of-speech tagged.
    :return: list, a list of tuples where each tuple contains a word and its part of speech.
    """
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]