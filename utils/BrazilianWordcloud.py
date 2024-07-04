import numpy as np
from PIL import Image
from matplotlib import pyplot
from wordcloud import WordCloud

def wordcloud(data, title, mask=None):
    """
    This method generates a custom word cloud of any format but it will generate a 
    Brazilian geo map custom wordcloud

    Parameters:
    data (str): The text corpus to generate the word cloud from.
    title (str): The title of the word cloud.
    mask (ndarray, optional): A mask image to shape the word cloud. If None, the word cloud is rectangular.

    Returns:
    None. The function displays the word cloud.
    """
    # Initialize the WordCloud object
    cloud = WordCloud(scale=3,
                      max_words=150,
                      colormap='Blues',
                      mask=mask,
                      background_color='white',
                      collocations=True,
                      ).generate_from_text(data)
    
    # Create a figure
    pyplot.figure(figsize=(10,8))
    
    # Display the word cloud
    pyplot.imshow(cloud)
    pyplot.axis('off')
    pyplot.title(title)
    pyplot.show()
    
    # Uncomment the line below to save the word cloud as an image file #
    # cloud.to_file('./assets/wordcloud.png')
    return cloud
    
def image_to_mask(imagepath):
    """
    This method converts an image to a mask for the word cloud.

    Parameters:
    imagepath (str): The path to the image file.

    Returns:
    mask (ndarray): The mask to be used for the word cloud.
    """
    # Open the image file and convert it to a numpy array
    return np.array(Image.open(imagepath))
