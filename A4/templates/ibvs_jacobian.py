import numpy as np

def ibvs_jacobian(K, pt, z):
    """
    Determine the Jacobian for IBVS.

    The function computes the Jacobian for image-based visual servoing,
    given the camera matrix K, an image plane point, and the estimated
    depth of the point. The x and y focal lengths in K are guaranteed 
    to be identical.

    Parameters:
    -----------
    K  - 3x3 np.array, camera intrinsic calibration matrix.
    pt - 2x1 np.array, image plane point. 
    z  - Scalar depth value (estimated).

    Returns:
    --------
    J  - 2x6 np.array, Jacobian. The matrix must contain float64 values.
    """

    #--- FILL ME IN ---

    # Convert image plane point to normalized image plane coordinates
    pt = pt / z

    # Initialize Jacobian matrix and components needed for calculation
    J = np.zeros((2, 6))
    f = K[0, 0]
    x = pt[0]
    y = pt[1]

    # Calculate ubar, vbar to be the normalized image plane coordinates based on pixel coordinates
    ubar = f * x
    vbar = f * y

    # Calculate Jacobian matrix
    # First row
    J[0, 0] = -f / z
    J[0, 2] = ubar / z
    J[0, 3] = ubar * vbar / f
    J[0, 4] = -(f ** 2 + ubar ** 2) / f
    J[0, 5] = vbar
    # Second row
    J[1, 1] = -f / z
    J[1, 2] = vbar / z
    J[1, 3] = (f ** 2 + vbar ** 2) / f
    J[1, 4] = -ubar * vbar / f
    J[1, 5] = -ubar
    #------------------

    correct = isinstance(J, np.ndarray) and \
        J.dtype == np.float64 and J.shape == (2, 6)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return J