import numpy as np
from scipy.ndimage.filters import *

def stereo_disparity_best(Il, Ir, bbox, maxd):
    """
    Best stereo correspondence algorithm.

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

    # I chose to implement the Semi-Global Matching (SGM) algorithm and referred to the paper titled "Stereo Processing by Semi-Global Matching and Mutual Information" 
    # by Heiko Hirschmuller for the algorithm details. This algorithm was chosen because it is known for having high accuracy, although being more computationally
    # expensive. The algorithm is implemented by first computing the similarity measure between all pixels in the left and right images (in this case using SAD) for
    # a range of disparities and storing the costs in a cost volume matrix (height x width x disparity). Pixels that are outside of the valid disparity range are set
    # to infinity so that they are not considered in the disparity calculation. The algorithm then iterates through each pixel in the bounding box region and calculates
    # the total cost for each disparity within the given range by aggregating costs along several 1D paths. The more paths considered, the more accurate the disparity,
    # but also the more computationally expensive. The paper provides the cost function used to aggregate costs along the paths, as well as information on the path
    # penalty parameters P1 and P2, which I tuned through experimentation. Once the costs for each of the paths are calculated, the total cost is computed by summing
    # the costs from all paths, normalizing, and selecting the disparity with the minimum cost for each pixel. Then, a median filter is applied to smooth the disparity
    # map (remove noise) while preserving edges. Paper link: https://core.ac.uk/download/pdf/11134866.pdf

    # Define hyperparameters - Semi-Global Matching (SGM) algorithm requires path penalty parameters
    # P1: penalty for disparity difference of 1 pixel
    P1 = 15
    # P2: penalty for disparity difference greater than 1 pixel
    P2 = 150
    
    # Initialize disparity map Id
    Id = np.zeros_like(Il)

    # Define range to loop through based on bounding box
    xmin = bbox[0, 0]
    xmax = bbox[0, 1] + 1
    ymin = bbox[1, 0]
    ymax = bbox[1, 1] + 1

    # Define height and width of the bounding box
    height = ymax - ymin
    width = xmax - xmin

    # For SGM algorithm, need to calculate cost volume (dimension: bounding height x bounding width x disparity)
    cv = np.full((height, width, maxd + 1), np.inf)

    # Calculate cost volume using SAD similarity measure (L1) over all disparities
    for d in range(maxd + 1):
        # Shift right image by disparity d to the left for comparison with left image
        Ir_shifted = np.zeros_like(Ir)
        if d > 0:
            Ir_shifted[:, d:] = Ir[:, :-d]
        else:
            Ir_shifted = Ir
        # Calculate SAD similarity measure between left and right images
        sad = np.abs(Il - Ir_shifted)
        # Invalid areas for disparity d are set to infinity. Pixels on the left (0 to d-1) will have no corresponding pixels in the right image to compare against
        sad[:, :d] = np.inf
        # Since cost volume only cares about bounding box region, only store those SAD values
        cv[:, :, d] = sad[ymin:ymax, xmin:xmax]

    # SGM aggregates costs in 8 directions (left, right, up, down, and diagonals) to find minimum cost
    # NOTE: NEEDED TO REDUCE NUMBER OF PATHS COMPUTED TO AVOID TIMEOUT - with all 8 paths, got pbad of 0.08
    #dir_set = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    dir_set = [(0, 1)]
    total_costs = np.zeros((len(dir_set), height, width, maxd + 1))

    for dir_num, (dir_y, dir_x) in enumerate(dir_set):
        # Need to loop over the bounding box region in the specified direction
        if dir_y >= 0:
            y_range = range(ymin, ymax)
        else:
            y_range = range(ymax - 1, ymin - 1, -1)
        if dir_x >= 0:
            x_range = range(xmin, xmax)
        else:
            x_range = range(xmax - 1, xmin - 1, -1)
        # Initialize temporary cost volume for current direction so that aggregation can be done in place
        temp_cv = np.copy(cv)
        for y in y_range:
            for x in x_range:
                # Handle border cases where the path goes out of the image
                if x - dir_x < xmin or x - dir_x >= (xmax - 1) or y - dir_y < ymin or y - dir_y >= (ymax - 1):
                    continue
                for d in range(maxd + 1):
                    # Calculate the four SGM costs using the previous column's costs and penalty hyperparameters
                    # Due to looping over the bounding box region, need to adjust the indices for the cost volume based on xmin and ymin
                    cost_ref = temp_cv[y - ymin - dir_y, x - xmin - dir_x, :]
                    cost1 = cost_ref[d]
                    if d > 0:
                        cost2 = cost_ref[d-1] + P1
                    else:
                        cost2 = np.inf
                    if d < maxd:
                        cost3 = cost_ref[d+1] + P1
                    else:
                        cost3 = np.inf
                    cost4 = np.min(cost_ref) + P2
                    # Aggregate costs with penalties using SGM cost function
                    temp_cv[y-ymin, x-xmin, d] += min(cost1, cost2, cost3, cost4) - np.min(cost_ref)
        # Update total costs for current direction
        total_costs[dir_num] = temp_cv
    
    # Combine all costs from 8 directions to get 1D array of total costs for each pixel, then normalize and select disparity with minimum cost
    total_costs = np.sum(total_costs, axis=0) / len(dir_set)
    # Select disparity with minimum cost from total costs for each pixel
    Id[ymin:ymax, xmin:xmax] = np.argmin(total_costs, axis=-1)
    # Apply median filter to smooth final disparity map, only within bounding box region to avoid unwanted edge effects
    Id[ymin:ymax, xmin:xmax] = median_filter(Id[ymin:ymax, xmin:xmax], size=5)
    #------------------

    correct = isinstance(Id, np.ndarray) and Id.shape == Il.shape

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Id