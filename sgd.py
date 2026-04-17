# Author: Noor Imtiaz

import math
import matplotlib.pyplot as plt
import numpy as np

def run_sgd(train_features, train_labels, n_iterations):
    # Initialize w as a 5d vector 
    w = np.array([0.0] * 5)
    
    # Store all weight vectors for Polyak-Ruppert averaging
    w_sum = np.zeros(5)
    
    for t in range(1, n_iterations + 1):
        # Get random example (stochastic sampling)
        idx = np.random.randint(0, len(train_features))
        x = train_features[idx]
        y = train_labels[idx]
        
        # Create augmented feature vector x_tilde = (x, 1)
        x_tilde = np.append(x, 1.0)
        
        # Learning rate (time-decaying)
        eta = 1.0 / math.sqrt(t)
        
        # Compute gradient of logistic loss
        # gradient = -y * x_tilde / (1 + exp(y * <w, x_tilde>))
        dot_product = np.dot(w, x_tilde)
        
        
        exponent = y * dot_product
        exponent = np.clip(exponent, -500, 500)  # Prevent overflow

        denominator = 1.0 + math.exp(exponent)
        gradient = (-y * x_tilde) / denominator
        
        # Update weight vector
        w = w - eta * gradient
        
        # Project w back onto unit ball (constrained optimization)
        w_norm = np.linalg.norm(w)
        if w_norm > 1:
            w = w / w_norm
        
        # Accumulate weights for averaging
        w_sum += w
    
    # Return Polyak-Ruppert average
    w_bar = w_sum / n_iterations
    return w_bar




