import numpy as np

def histogram_eq(I):
    """
    Histogram equalization for greyscale image.

    Perform histogram equalization on the 8-bit greyscale intensity image I
    to produce a contrast-enhanced image J. Full details of the algorithm are
    provided in the Szeliski text.

    Parameters:
    -----------
    I  - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).

    Returns:
    --------
    J  - Contrast-enhanced greyscale intensity image, 8-bit np.array (i.e., uint8).
    """
    #--- FILL ME IN ---

    # Verify I is grayscale.
    if I.dtype != np.uint8:
        raise ValueError('Incorrect image format!')
    
    J = np.zeros_like(I)
    # Store count of each grey level value in the image
    histogram, _ = np.histogram(I.flatten(), bins=256, range=(0, 255))
    
    # Calculate cumulative distribution function for each intensity level. Each index of culm_dist is a cumulative sum of all previous intensity levels
    culm_dist = np.cumsum(histogram)
    culm_dist = culm_dist / culm_dist[-1]
    c = np.round(255 * culm_dist)

    # Map intensity levels to new values
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            # Scale pixel intensity to 0-255 based on cumulative distribution
            J[i, j] = c[I[i, j]]
    J = J.astype(np.uint8)

    #------------------

    return J