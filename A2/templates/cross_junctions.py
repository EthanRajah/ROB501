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

    # Wpts contains 3D world points of junctions, with z = 0 (flat plane points)
    # print(Wpts)
    checker_square_size = Wpts[0, 1] - Wpts[0, 0]

    # Estimate target bounding box in world coordinates for homography using the fact that the checkered square length is 63.5mm
    # Bounding hyperparameters are estimated through trial and error checks to ensure the edges of the target are captured
    bounding_hyperparameters = [1.4, 1.25]
    W_minx = min(Wpts[0, :]) - bounding_hyperparameters[0]*checker_square_size
    W_miny = min(Wpts[1, :]) - bounding_hyperparameters[1]*checker_square_size
    W_maxx = max(Wpts[0, :]) + bounding_hyperparameters[0]*checker_square_size
    W_maxy = max(Wpts[1, :]) + bounding_hyperparameters[1]*checker_square_size
    Wbox = np.array([[W_minx, W_maxx, W_maxx, W_minx], [W_miny, W_miny, W_maxy, W_maxy]])

    # Perform homography transformation from Wpts (original/rectangular shape) to bpoly (warped shape)
    H, A = dlt_homography(Wbox, bpoly)

    # Use patch size (hyperparameter) to define the size of the image patch around each world point to look at for saddle points
    patch_size = 12
    Ipts = np.zeros((2, Wpts.shape[1]))

    # Compute vectorized homography
    Wpts[2, :] = 1
    Wpts_transform = H @ Wpts
    Wpts_transform = Wpts_transform / Wpts_transform[2]
    Wpts_transform = Wpts_transform[:-1]
    Wpts_transform = np.round(Wpts_transform).astype(int)

    # Using the transformed Wpts, the saddle points can be found in the image since the coordinates (indices) are now based on the perspective of the image
    for i in range(Wpts.shape[1]):
        x, y = Wpts_transform[:, i]
        img_patch_filter = gaussian_filter(I[y-patch_size:y+patch_size, x-patch_size:x+patch_size], sigma=1)
        junction_pt = saddle_point(img_patch_filter)
        # Get cross junction coordinates relative to the upper left corner of the target using computed saddle point
        Ipts[:, i] = (np.array([[x], [y]]) - patch_size + junction_pt).reshape(2)
    
    #------------------

    correct = isinstance(Ipts, np.ndarray) and \
        Ipts.dtype == np.float64 and Ipts.shape[0] == 2

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Ipts