# Billboard hack script file.
import numpy as np
from matplotlib.path import Path
from imageio import imread, imwrite

from dlt_homography import dlt_homography
from bilinear_interp import bilinear_interp
from histogram_eq import histogram_eq

def billboard_hack():
    """
    Hack and replace the billboard!

    Parameters:
    ----------- 

    Returns:
    --------
    Ihack  - Hacked RGB intensity image, 8-bit np.array (i.e., uint8).
    """
    # Bounding box in Y & D Square image - use if you find useful.
    bbox = np.array([[404, 490, 404, 490], [38,  38, 354, 354]])

    # Point correspondences.
    Iyd_pts = np.array([[416, 485, 488, 410], [40,  61, 353, 349]])
    Ist_pts = np.array([[2, 218, 218, 2], [2, 2, 409, 409]])

    Iyd = imread('../images/yonge_dundas_square.jpg')
    Ist = imread('../images/uoft_soldiers_tower_light.png')

    Ihack = np.asarray(Iyd)
    Ist = np.asarray(Ist)

    #--- FILL ME IN ---

    # Let's do the histogram equalization first.
    Ist = histogram_eq(Ist)
    # Compute the perspective homography we need...
    H, A = dlt_homography(Iyd_pts, Ist_pts)
    # Main 'for' loop to do the warp and insertion - 
    # this could be vectorized to be faster if needed!
    # You may wish to make use of the contains_points() method
    # available in the matplotlib.path.Path class!
    edge_walk = Path(Iyd_pts.T)
    # Loop through all the pixels in the billboard section of the image we want to replace
    for x in range(min(bbox[0]), max(bbox[0])+1):
        for y in range(min(bbox[1]), max(bbox[1])+1):
            if (edge_walk.contains_point((x,y))):
                # Perform homography transformation from Iyd (warped shape) to Ist (original/rectangular shape)
                pt = np.array(([x], [y], [1]))
                pt_transform = H @ pt
                pt_transform = pt_transform / pt_transform[2]
                pt_transform = pt_transform[:-1]
                # Interpolate pixel values from Ist now that coordinates are transformed and account for case where Ist is an RGB image
                if (Ist.ndim == 3):
                    pixel = (bilinear_interp(Ist[:,:,0], pt_transform), bilinear_interp(Ist[:,:,1], pt_transform), bilinear_interp(Ist[:,:,2], pt_transform))
                    Ihack[y, x] = pixel
                else:
                    pixel = bilinear_interp(Ist, pt_transform)
                    Ihack[y, x] = np.repeat(pixel, 3)

    #------------------

    # Visualize the result, if desired...
    #import matplotlib.pyplot as plt
    #plt.imshow(Ihack)
    #plt.show()
    #imwrite(Ihack, 'billboard_hacked.png')

    return Ihack

if __name__ == "__main__":
    billboard_hack()