import numpy as np
from numpy.linalg import inv, lstsq

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