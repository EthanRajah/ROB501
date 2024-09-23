import numpy as np
from numpy.linalg import inv, norm
from scipy.linalg import null_space

def dlt_homography(I1pts, I2pts):
    """
    Find perspective Homography between two images.

    Given 4 points from 2 separate images, compute the perspective homography
    (warp) between these points using the DLT algorithm.

    Parameters:
    ----------- 
    I1pts  - 2x4 np.array of points from Image 1 (each column is x, y).
    I2pts  - 2x4 np.array of points from Image 2 (in 1-to-1 correspondence).

    Returns:
    --------
    H  - 3x3 np.array of perspective homography (matrix map) between image coordinates.
    A  - 8x9 np.array of DLT matrix used to determine homography.
    """
    #--- FILL ME IN ---

    # Define matrix A (8x9) using the input correspondences
    A = np.zeros((8, 9))
    for i in range(4):
        x, y = I1pts[:, i]
        u, v = I2pts[:, i]
        A[2*i] = np.array([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A[2*i+1] = np.array([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    
    # The 1D null space of A is the solution to Ah = 0, allowing us to then get the solution space for h
    h = null_space(A)
    # If h is not a 1D array, we take the last column of h as the solution
    if (h.shape[1] > 1):
        h = h[:, -1]
    H = h.reshape(3, 3)

    # Since any multiple of all the values in the homography is the same homography, we normalize
    H = H / H[2, 2]
    
    #------------------

    return H, A