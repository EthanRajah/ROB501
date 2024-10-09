import numpy as np
from numpy.linalg import inv, lstsq
from scipy.linalg import null_space
from scipy.ndimage.filters import *
from matplotlib.path import Path

# You may add support functions here, if desired.
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

def saddle_point(I):
    """
    Locate saddle point in an image patch.

    The function identifies the subpixel centre of a cross-junction in the
    image patch I, by fitting a hyperbolic paraboloid to the patch, and then 
    finding the critical point of that paraboloid.

    Note that the location of 'pt' is relative to (-0.5, -0.5) at the upper
    left corner of the patch, i.e., the pixels are treated as covering an 
    area of one unit square.

    Parameters:
    -----------
    I  - Single-band (greyscale) image patch as np.array (e.g., uint8, float).

    Returns:
    --------
    pt  - 2x1 np.array (float64), subpixel location of saddle point in I (x, y).
    """
    #--- FILL ME IN ---
 
    # Define Jacobian matrix for least squares fitting of hyperbolic paraboloid with parameters alpha, beta, gamma, delta, epsilon and zeta to the image patch I
    # Also define b, the smoothed image patch pixel values at each pixel location
    J = np.zeros((I.size, 6))
    b = np.zeros((I.size, 1))
    row = 0
    for y in range(I.shape[0]):
        for x in range(I.shape[1]):
            J[row, :] = [x**2, x*y, y**2, x, y, 1]
            b[row] = I[y, x]
            row += 1
    
    # Using J and b, we can solve for the parameters of the hyperbolic paraboloid
    alpha, beta, gamma, delta, epsilon, zeta = lstsq(J, b, rcond=None)[0].T[0]

    # The saddle point is the intersection of the two lines 2*alpha*x + beta*y + delta = 0 and beta*x + 2*gamma*y + epsilon = 0
    A = np.array([[2*alpha, beta], [beta, 2*gamma]])
    v = np.array([[delta], [epsilon]])
    pt = -inv(A) @ v

    #------------------

    correct = isinstance(pt, np.ndarray) and \
        pt.dtype == np.float64 and pt.shape == (2, 1)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return pt

def cross_junctions(I, bpoly, Wpts):
    """
    Find cross-junctions in image with subpixel accuracy.

    The function locates a series of cross-junction points on a planar 
    calibration target, where the target is bounded in the image by the 
    specified quadrilateral. The number of cross-junctions identified 
    should be equal to the number of world points.

    Note also that the world and image points must be in *correspondence*,
    that is, the first world point should map to the first image point, etc.

    Parameters:
    -----------
    I      - Single-band (greyscale) image as np.array (e.g., uint8, float).
    bpoly  - 2x4 np.array, bounding polygon (clockwise from upper left).
    Wpts   - 3xn np.array of world points (in 3D, on calibration target).

    Returns:
    --------
    Ipts  - 2xn np.array of cross-junctions (x, y), relative to the upper
            left corner of the target. The array must contain float64 values.
    """
    #--- FILL ME IN ---
    
    # Define the bounding polygon as a Path object
    path = Path(bpoly.T)

    #------------------

    correct = isinstance(Ipts, np.ndarray) and \
        Ipts.dtype == np.float64 and Ipts.shape[0] == 2

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Ipts