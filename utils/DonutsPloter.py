from matplotlib import pyplot

def donuts_plot(percent, classe_counts):
    """
    This Method generates a custom donut plot.

    Parameters:
    percent (Series): The percentage of each class.
    classe_counts (Series): The number of observations in each class.

    Returns:
    None. The function displays the donut plot.
    """
    
    # Get the class labels
    classes = classe_counts.index
    
    # Create a figure and an axis
    fig, ax = pyplot.subplots()
    
    # Create a pie chart with a hole in the middle (donut plot)
    wedges, texts, autotexts = ax.pie(percent,
                                      labels=classes,
                                      autopct="%1.1f%%",
                                      startangle=-70,
                                      wedgeprops=dict(width=0.3))
    
    # Add a white circle in the middle to create the donut effect
    centre_circle = pyplot.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)
    
    # Ensure the pie chart is circular
    ax.axis('equal')
    
    # Display the donut plot
    pyplot.tight_layout()
    pyplot.show()
