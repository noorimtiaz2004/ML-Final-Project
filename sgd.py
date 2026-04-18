# Author: Noor Imtiaz

import math
import matplotlib.pyplot as plt
import numpy as np

def run_sgd(train_features, train_labels, n_iterations):
    """
    Stochastic Gradient Descent (SGD) Algorithm
    
    Parameters:
        train_features: Training data features
        train_labels: Training data labels
        n_iterations: Integer T > 0 (number of iterations)
    
    Returns:
        w_bar: Averaged weight vector 
    """
    # Parameter T
    T = n_iterations
    
    # Initialize: w^(1) = 0
    w = np.array([0.0] * 5)
    
    # Accumulator for output: w̄ = (1/T) * Σ w^(t)
    w_sum = np.zeros(5)
    
    # Algorithm Loop: For t = 1, 2, ..., T
    for t in range(1, T + 1):
        # Step 1: Sample z ~ D (sample training example)
        idx = np.random.randint(0, len(train_features))
        z = (train_features[idx], train_labels[idx])
        x, y = z
        
        # Create augmented feature vector x_tilde = (x, 1)
        x_tilde = np.append(x, 1.0)
        
        # Step 2: Pick v_t ∈ ∂ℓ(w^(t), z) (compute subgradient)
        # For logistic loss: v_t = -y * x_tilde / (1 + exp(y * <w, x_tilde>))
        dot_product = np.dot(w, x_tilde)
        exponent = y * dot_product
        exponent = np.clip(exponent, -500, 500)  # Numerical stability
        denominator = 1.0 + math.exp(exponent)
        v_t = (-y * x_tilde) / denominator
        
        # Step 3: Set learning rate η_t = 1/√(2t)
        eta_t = 1.0 / math.sqrt(2 * t)
        
        # Step 4: Update w^(t+1) = w^(t) - η_t * v_t
        w = w - eta_t * v_t
        
        # Step 5: Project w^(t+1) onto unit ball
        # If ||w^(t+1)|| > 1, then w^(t+1) = w^(t+1) / ||w^(t+1)||
        w_norm = np.linalg.norm(w)
        if w_norm > 1:
            w = w / w_norm
        
        # Accumulate for averaging
        w_sum += w
    
    # Output: w̄ = (1/T) * Σ_{t=1}^T w^(t)
    w_bar = w_sum / T
    return w_bar




