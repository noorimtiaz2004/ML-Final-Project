# Author: Jayden Vinson

import numpy as np

# generate test and training da
def generate_data(data_points: int, sigma: float, dimension: int) -> tuple:

    # needed to make gaussian vectors, or multivariate normal distribution
    rng = np.random.default_rng()

    # covaraince matrix
    cov = (sigma ** 2) * np.eye(dimension)

    # creates labels -1 or 1 for each data point
    all_labels, nums = _make_all_labels(data_points)

    # mean vectors for -1 and 1 
    mu0 = np.array([-1/4, -1/4, -1/4, -1/4])
    mu1 = np.array([1/4, 1/4, 1/4, 1/4])

    # negative numbers
    u_neg =rng.multivariate_normal(
        mean= mu0,
        cov= cov,
        size= nums[0]
    )
    
    #positive numbers
    u_pos =rng.multivariate_normal(
        mean= mu1,
        cov= cov,
        size= nums[1]
    )

    # necessary to get return 
    empty = _combined_multivariate_normal(data_points, dimension, all_labels, u_neg, u_pos)

    # final necessary thing
    features = _euclidean_proj(data_points, dimension, empty)

    return features, all_labels

def _make_all_labels(data_points: int) -> tuple[np.ndarray, list]:
   # Create all labels, -1 or 1, at once 
    all_labels = np.empty(data_points, dtype=int)
    nums = [0,0]
    for i in range(data_points):
        num = np.random.choice([-1, 1])
        if(num < 0):
            nums[0] += 1    
        else:            
            nums[1] += 1
        all_labels[i] = num
    return all_labels, nums


def _euclidean_proj(data_points: int, dimension: int, empty: np.ndarray) -> np.ndarray:
    features = np.empty((data_points, dimension))
    for i in range(data_points):
        # Calculate the length (Euclidean norm) of the current vector
        length = np.linalg.norm(empty[i])
        
        # If it stepped outside the bounds of 1, pull it back to the edge
        if length > 1:
            features[i] = empty[i] / length
        # Otherwise, keep it exactly as it is
        else:
            features[i] = empty[i]
    return features

def _combined_multivariate_normal(data_points: int, dimension: int, all_labels: np.ndarray, u_neg: np.ndarray, u_pos: np.ndarray) -> np.ndarray:
    # Initialize the empty array with the correct dimensions
    empty = np.empty((data_points, dimension))
    
    neg_index = 0
    pos_index = 0
    
    for i in range(data_points):
        if all_labels[i] == -1:
            # Grab the next available negative vector and put it in row i
            empty[i] = u_neg[neg_index]
            neg_index += 1  # Move to the next negative vector
        else:
            # Grab the next available positive vector and put it in row i
            empty[i] = u_pos[pos_index]
            pos_index += 1  # Move to the next positive vector

    return empty