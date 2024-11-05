import numpy as np
from scipy.ndimage.filters import *

def stereo_disparity_fast(Il, Ir, bbox, maxd):
    """
    Fast stereo correspondence algorithm.

    This function computes a stereo disparity image from left stereo 
    image Il and right stereo image Ir. Only disparity values within
    the bounding box region (inclusive) are evaluated.

    Parameters:
    -----------
    Il    - Left stereo image, m x n pixel np.array, greyscale.
    Ir    - Right stereo image, m x n pixel np.array, greyscale.
    bbox  - 2x2 np.array, bounding box, relative to left image, from top left
            corner to bottom right corner (inclusive). (x, y) in columns.
    maxd  - Integer, maximum disparity value; disparities must be within zero
            to maxd inclusive (i.e., don't search beyond maxd).

    Returns:
    --------
    Id  - Disparity image (map) as np.array, same size as Il.
    """
    # Hints:
    #
    #  - Loop over each image row, computing the local similarity measure, then
    #    aggregate. At the border, you may replicate edge pixels, or just avoid
    #    using values outside of the image.
    #
    #  - You may hard-code any parameters you require in this function.
    #
    #  - Use whatever window size you think might be suitable.
    #
    #  - Optimize for runtime AND for clarity.

    #--- FILL ME IN ---

    # Define hyperparameters
    window_size = 15

    # Initialize disparity map Id
    Id = np.zeros_like(Il)

    # Define range to loop through based on bounding box
    xmin = bbox[0, 0]
    xmax = bbox[0, 1] + 1
    ymin = bbox[1, 0]
    ymax = bbox[1, 1] + 1

    # Pad left and right images based on half of the window width to avoid border issues
    Il_pad = np.pad(Il, window_size // 2, mode='edge')
    Ir_pad = np.pad(Ir, window_size // 2, mode='edge')

    # Loop over each image row and use SAD similarity measure (L1)
    for y in range(ymin, ymax):
        for x in range(xmin, xmax):
            # Due to padding of images, need to adjust x and y for correct window creation
            x_pad = x + window_size // 2
            y_pad = y + window_size // 2
            # Define window for left image
            window_l = Il_pad[(y_pad - (window_size // 2)): (y_pad + (window_size // 2) + 1), (x_pad - (window_size // 2)): (x_pad + (window_size // 2) + 1)]
            # Using epipolar line from left image, can look at range (0, maxd) from right image window and calculate L1 norm for similarity
            best_disparity = 0
            best_sim = np.inf
            for disp in range (maxd + 1):
                if x - disp - (window_size // 2) < 0:
                    # Skip if the window with disparity goes out of bounds in the right image
                    continue
                # Create window for right image based on disparity and window size, only checking disparity to the left since we are looking at the right image
                window_r = Ir_pad[(y_pad - (window_size // 2)): (y_pad + (window_size // 2) + 1), (x_pad - disp - (window_size // 2)): (x_pad - disp + (window_size // 2) + 1)]
                # Calculate L1 norm for similarity
                sad = np.sum(np.abs(window_r - window_l))
                # Update best similarity and disparity
                if sad < best_sim:
                    best_sim = sad
                    best_disparity = disp
            # Update disparity map Id
            Id[y, x] = best_disparity
    # Apply median filter to smooth disparity map
    Id = median_filter(Id, size=3)
    #------------------

    correct = isinstance(Id, np.ndarray) and Id.shape == Il.shape

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Id