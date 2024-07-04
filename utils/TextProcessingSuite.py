import spacy

# Load the Portuguese language model
nlp = spacy.load('pt_core_news_lg')

def tokenize(text):
    """
    This Method tokenizes the text.

    Parameters:
    text (str): The input text to be tokenized.

    Returns:
    str: A string of tokens separated by spaces.
    """
    # Use the Portuguese language model to tokenize the text
    doc = nlp(text)
    
    # Return a string of tokens separated by spaces
    return ' '.join([token.text for token in doc])

def remove_stopwords(text):
    """
    This Method removes irrelevant words (stop words) from the text.

    Parameters:
    text (str): The input text from which stop words are to be removed.

    Returns:
    str: The text with stop words removed.
    """
    # Use the Portuguese language model to tokenize the text
    doc = nlp(text)
    
    # Return a string of tokens (excluding stop words) separated by spaces
    return ' '.join([token.text for token in doc if not token.is_stop])

def lemmatize(text):
    """
    This Method applies lemmatization to the text, which is the process of converting a word to its base form.

    Parameters:
    text (str): The input text to be lemmatized.

    Returns:
    str: The text with words lemmatized.
    """
    # Use the Portuguese language model to tokenize the text
    doc = nlp(text)
    
    # Return a string of lemmas separated by spaces
    return ' '.join([token.lemma_ for token in doc])

def pos_tagging(text):
    """
    This Method identifies the part of speech of each word in the text.

    Parameters:
    text (str): The input text to be part-of-speech tagged.

    Returns:
    list: A list of tuples where each tuple contains a word and its part of speech.
    """
    # Use the Portuguese language model to tokenize the text
    doc = nlp(text)
    
    # Return a list of tuples where each tuple contains a word and its part of speech
    return [(token.text, token.pos_) for token in doc]
