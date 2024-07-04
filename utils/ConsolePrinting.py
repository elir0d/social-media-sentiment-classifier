def _printing(string, object):
    """
    Prints a centered title followed by a separator line and an object.

    Args:
        string (str): The title string to be centered.
        object: The object to be printed below the title.

    Returns:
        None
    """
    # Calculate the length of the title string
    string_length = len(string)
    
    # Calculate the empty space on the left and right
    empty_space = (70 - len(string)) // 2
    
    # Print the centered title
    print(70 * '=')
    print(f"{' ' * empty_space}{string}{' ' * empty_space}")
    print(70 * '=')
    print(object)
    print(70 * '=')