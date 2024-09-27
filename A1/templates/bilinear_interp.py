import numpy as np
from numpy.linalg import inv

def bilinear_interp(I, pt):
    """
    Performs bilinear interpolation for a given image point.

    Given the (x, y) location of a point in an input image, use the surrounding
    four pixels to conmpute the bilinearly-interpolated output pixel intensity.

    Note that images are (usually) integer-valued functions (in 2D), therefore
    the intensity value you return must be an integer (use round()).

    This function is for a *single* image band only - for RGB images, you will 
    need to call the function once for each colour channel.

    Parameters:
    -----------
    I   - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).
    pt  - 2x1 np.array of point in input image (x, y), with subpixel precision.

    Returns:
    --------
    b  - Interpolated brightness or intensity value (whole number >= 0).
    """
    #--- FILL ME IN ---

    if pt.shape != (2, 1):
        raise ValueError('Point size is incorrect.')
    
    x, y = pt[0, 0], pt[1, 0]
    
    # Get the four surrounding pixels based on point location (x1, y1), (x2, y2), (x1, y2), (x2, y1)
    x1, y1 = round(np.floor(x)), round(np.floor(y))
    x2, y2 = round(np.ceil(x)), round(np.ceil(y))

    # If points are outside of image, set to four corners of image
    if x1 < 0:
        x1 = 0
    if x2 >= I.shape[1]:
        x2 = I.shape[1] - 1
    if y1 < 0:
        y1 = 0
    if y2 >= I.shape[0]:
        y2 = I.shape[0] - 1

    # Handle case where x1 = x2 and y1 = y2
    if (x1 == x2 and y1 == y2):
        b = I[y1, x1]
    # Handle case where x1 = x2 (linear interpolation in y direction)
    elif (x1 == x2):
        b = round((y2 - y) * I[y1, x1] + (y - y1) * I[y2, x1])
    # Handle case where y1 = y2 (linear interpolation in x direction)
    elif (y1 == y2):
        b = round((x2 - x) * I[y1, x1] + (x - x1) * I[y1, x2])
    # General case
    else:
        # Interpolate pixel values in x direction
        f1 = (x2 - x) * I[y1, x1] + (x - x1) * I[y1, x2]
        f2 = (x2 - x) * I[y2, x1] + (x - x1) * I[y2, x2]
        # Interpolate pixel values in y direction to get final pixel value for that point
        b = round((y2 - y) * f1 + (y - y1) * f2)

    #------------------

    return b