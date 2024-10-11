import numpy as np
from numpy.linalg import inv

def rpy_from_dcm(R):
    """
    Roll, pitch, yaw Euler angles from rotation matrix.

    The function computes roll, pitch and yaw angles from the
    rotation matrix R. The pitch angle p is constrained to the range
    (-pi/2, pi/2].  The returned angles are in radians.

    Inputs:
    -------
    R  - 3x3 orthonormal rotation matrix.

    Returns:
    --------
    rpy  - 3x1 np.array of roll, pitch, yaw Euler angles.
    """
    rpy = np.zeros((3, 1))

    # Roll.
    rpy[0] = np.arctan2(R[2, 1], R[2, 2])

    # Pitch.
    sp = -R[2, 0]
    cp = np.sqrt(R[0, 0]*R[0, 0] + R[1, 0]*R[1, 0])

    if np.abs(cp) > 1e-15:
      rpy[1] = np.arctan2(sp, cp)
    else:
      # Gimbal lock...
      rpy[1] = np.pi/2
  
      if sp < 0:
        rpy[1] = -rpy[1]

    # Yaw.
    rpy[2] = np.arctan2(R[1, 0], R[0, 0])

    return rpy

def dcm_from_rpy(rpy):
    """
    Rotation matrix from roll, pitch, yaw Euler angles.

    The function produces a 3x3 orthonormal rotation matrix R
    from the vector rpy containing roll angle r, pitch angle p, and yaw angle
    y.  All angles are specified in radians.  We use the aerospace convention
    here (see descriptions below).  Note that roll, pitch and yaw angles are
    also often denoted by phi, theta, and psi (respectively).

    The angles are applied in the following order:

     1.  Yaw   -> by angle 'y' in the local (body-attached) frame.
     2.  Pitch -> by angle 'p' in the local frame.
     3.  Roll  -> by angle 'r' in the local frame.  

    Note that this is exactly equivalent to the following fixed-axis
    sequence:

     1.  Roll  -> by angle 'r' in the fixed frame.
     2.  Pitch -> by angle 'p' in the fixed frame.
     3.  Yaw   -> by angle 'y' in the fixed frame.

    Parameters:
    -----------
    rpy  - 3x1 np.array of roll, pitch, yaw Euler angles.

    Returns:
    --------
    R  - 3x3 np.array, orthonormal rotation matrix.
    """
    cr = np.cos(rpy[0]).item()
    sr = np.sin(rpy[0]).item()
    cp = np.cos(rpy[1]).item()
    sp = np.sin(rpy[1]).item()
    cy = np.cos(rpy[2]).item()
    sy = np.sin(rpy[2]).item()

    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [  -sp,            cp*sr,            cp*cr]])

def find_jacobian(K, Twc, Wpt):
    """
    Determine the Jacobian for NLS camera pose optimization.

    The function computes the Jacobian of an image plane point with respect
    to the current camera pose estimate, given a landmark point. The 
    projection model is the simple pinhole model.

    Note that the homogeneous transformation matrix provided defines the
    transformation from the *camera frame* to the *world frame* (to 
    project into the image, you would need to invert this matrix).

    Parameters:
    -----------
    K    - 3x3 np.array, camera intrinsic calibration matrix.
    Twc  - 4x4 np.array, homogenous pose matrix, current guess for camera pose. 
    Wpt  - 3x1 world point on calibration target (one of n).

    Returns:
    --------
    J  - 2x6 np.array, Jacobian matrix (columns are tx, ty, tz, r, p, q).
         The Jacobian must contain float64 values.
    """
    #--- FILL ME IN ---
    
    J = np.zeros((2, 6))

    # Extract the rotation and translation components from the Twc matrix
    R = Twc[:3, :3]
    t = Twc[:3, 3].reshape(-1,1)

    # Compute the camera frame coordinates of the world point and seperate into x, y, z
    # Want rotation from world to camera frame so invert the rotation matrix
    P = (R.T).dot(Wpt - t)
    x_img, y_img, z_img = P[0], P[1], P[2]

    # Jacobian with respect to translation (differentiating camera frame coordinates u,v with respect to x,y,z)
    # du/dx = 1/z * K[0, 0] * (-r11 + x*r13/z)
    J[0, 0] = (K[0, 0]/z_img) * (-R[0, 0] + x_img*R[0, 2]/z_img)
    # du/dy = 1/z * K[0, 0] * (-r21 + x*r23/z)
    J[0, 1] = (K[0, 0]/z_img) * (-R[1, 0] + x_img*R[1, 2]/z_img)
    # du/dz = 1/z * K[0, 0] * (-r31 + x*r33/z)
    J[0, 2] = (K[0, 0]/z_img) * (-R[2, 0] + x_img*R[2, 2]/z_img)
    # dv/dx = 1/z * K[1, 1] * (-r12 + y*r13/z)
    J[1, 0] = (K[1, 1]/z_img) * (-R[0, 1] + y_img*R[0, 2]/z_img)
    # dv/dy = 1/z * K[1, 1] * (-r22 + y*r23/z)
    J[1, 1] = (K[1, 1]/z_img) * (-R[1, 1] + y_img*R[1, 2]/z_img)
    # dv/dz = 1/z * K[1, 1] * (-r32 + y*r33/z)
    J[1, 2] = (K[1, 1]/z_img) * (-R[2, 1] + y_img*R[2, 2]/z_img)

    # Jacobian with respect to rotation (differentiating camera frame coordinates u,v with respect to yaw, pitch, roll angles)

    # First seperate the rotation matrix into euler angle rotation matrices about an axis
    euler_angles = rpy_from_dcm(R)
    r, p, y = euler_angles[0][0], euler_angles[1][0], euler_angles[2][0]
    Croll = dcm_from_rpy(np.array([r, 0, 0]))
    Cpitch = dcm_from_rpy(np.array([0, p, 0]))
    Cyaw = dcm_from_rpy(np.array([0, 0, y]))

    # Define skew symmetric matrices for cross product with rotation matrix
    Sroll = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
    Spitch = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
    Syaw = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])

    # Use results to define dCu_roll, dCu_pitch, dCu_yaw
    dCu_roll = Cyaw @ Cpitch @ Sroll @ Croll
    dCu_pitch = Cyaw @ Spitch @ Cpitch @ Croll
    dCu_yaw = Syaw @ Cyaw @ Cpitch @ Croll

    # Transpose dCu matrices since we want the rotations from the world to camera frame and then multiply by world point translation
    dCu_roll = dCu_roll.T @ (Wpt - t)
    dCu_pitch = dCu_pitch.T @ (Wpt - t)
    dCu_yaw = dCu_yaw.T @ (Wpt - t)

    # Can now compute the rest of the Jacobian elements, du/droll, du/dpitch, du/dyaw, dv/droll, dv/dpitch, dv/dyaw
    # du/droll = 1/z * K[0, 0] * (dx/droll - (x/z) * dz/droll))
    J[0, 3] = (K[0, 0]/z_img) * (dCu_roll[0] - (x_img/z_img) * dCu_roll[2])
    # du/dpitch = 1/z * K[0, 0] * (dx/dpitch - (x/z) * dz/dpitch))
    J[0, 4] = (K[0, 0]/z_img) * (dCu_pitch[0] - (x_img/z_img) * dCu_pitch[2])
    # du/dyaw = 1/z * K[0, 0] * (dx/dyaw - (x/z) * dz/dyaw))
    J[0, 5] = (K[0, 0]/z_img) * (dCu_yaw[0] - (x_img/z_img) * dCu_yaw[2])
    # dv/droll = 1/z * K[1, 1] * (dy/droll - (y/z) * dz/droll))
    J[1, 3] = (K[1, 1]/z_img) * (dCu_roll[1] - (y_img/z_img) * dCu_roll[2])
    # dv/dpitch = 1/z * K[1, 1] * (dy/dpitch - (y/z) * dz/dpitch))
    J[1, 4] = (K[1, 1]/z_img) * (dCu_pitch[1] - (y_img/z_img) * dCu_pitch[2])
    # dv/dyaw = 1/z * K[1, 1] * (dy/dyaw - (y/z) * dz/dyaw))
    J[1, 5] = (K[1, 1]/z_img) * (dCu_yaw[1] - (y_img/z_img) * dCu_yaw[2])

    #------------------

    correct = isinstance(J, np.ndarray) and \
        J.dtype == np.float64 and J.shape == (2, 6)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return J