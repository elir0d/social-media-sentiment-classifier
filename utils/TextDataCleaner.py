import re

def remove_usernames(text):
    """
    This Method removes social media usernames from the text.

    Parameters:
    text (str): The input text from which usernames are to be removed.

    Returns:
    str: The text with usernames removed.
    """
    # Use regex to find and remove all usernames (words that start with '@')
    return re.sub('@[^\\s]+', '', text)

def remove_urls(text):
    """
    This Method removes URLs from the text.

    Parameters:
    text (str): The input text from which URLs are to be removed.

    Returns:
    str: The text with URLs removed.
    """
    # Use regex to find and remove all URLs
    return re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\\\(\\\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

def remove_hashtags(text):
    """
    This Method removes hashtags from the text.

    Parameters:
    text (str): The input text from which hashtags are to be removed.

    Returns:
    str: The text with hashtags removed.
    """
    # Use regex to find and remove all hashtags (words that start with '#')
    return re.sub('#[^\\s]+', '', text)

def remove_new_lines_char(text):
    """
    This Method removes new line characters from the text.

    Parameters:
    text (str): The input text from which new line characters are to be removed.

    Returns:
    str: The text with new line characters removed.
    """
    # Use regex to find and remove all new line characters
    return re.sub('\\[^\\s]+', '', text)

def remove_non_alphabets(text):
    """
    This Method removes non-alphabetic characters from the text.

    Parameters:
    text (str): The input text from which non-alphabetic characters are to be removed.

    Returns:
    str: The text with non-alphabetic characters removed.
    """
    # Use regex to find and remove all non-alphabetic characters
    text = re.sub('[^A-Za-zÀ-ÖØ-öø-ÿ]', ' ', text, flags=re.UNICODE)

    # Use regex to find and remove all numbers
    text = re.sub('\\\\d+', ' ', text)

    # Use regex to find and remove multiple spaces
    text = re.sub(r'\\s+', ' ', text)
    return text

def to_lower(text):
    """
    This Method converts the text to lowercase.

    Parameters:
    text (str): The input text to be converted to lowercase.

    Returns:
    str: The text converted to lowercase.
    """
    # Use the lower() method to convert the text to lowercase
    return text.lower()

def remove_accents(text):
    """
    This Method removes accents from a Portuguese vocabulary.

    Parameters:
    text (str): The text to process.

    Returns:
    str: The text with accents removed.
    """
    # Define a dictionary mapping accented characters to their unaccented counterparts
    accent_map = {'á': 'a', 'à': 'a', 'ã': 'a','â': 'a', 
                  'é': 'e', 'è': 'e', 'ê': 'e', 
                  'í': 'i', 'ì': 'i', 'î': 'i',
                  'ó': 'o', 'ò': 'o', 'õ': 'o', 'ô': 'o',
                  'ú': 'u', 'ù': 'u', 'û': 'u',
                  'ç': 'c'
    }
    
    # Use a regular expression to replace accented characters with their unaccented counterparts
    pattern = re.compile('|'.join(accent_map.keys()))
    clean_text = pattern.sub(lambda m: accent_map[m.group(0)], text)
    return clean_text

def remove_words_with_repeating_letters(text):
  """
  This function removes words with 3 or more repeating letters from a given text.

  Args:
    text: The input text.

  Returns:
    The text with words containing repeating letters removed.
  """

  # Create a regular expression to match words with 3 or more repeating letters
  regex = r"\b\w*((\w)\2{2,})\w*\b"

  # Remove words with repeating letters from the text
  text = re.sub(regex, "", text)

  # Return the result
  return text