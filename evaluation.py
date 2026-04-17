# Author: Noor Imtiaz

import numpy as np
import math

def evaluate_predictor(w, test_features, test_labels):
   
    N = len(test_labels)
    total_loss = 0.0
    total_errors = 0
    
    for i in range(N):
        # Create augmented feature vector
        x_tilde = np.append(test_features[i], 1.0)
        y = test_labels[i]
        
        dot_product = np.dot(w, x_tilde)
        
        # Compute logistic loss with numerical stability
        exponent = -y * dot_product
        exponent = np.clip(exponent, -500, 500)
        total_loss += math.log(1.0 + math.exp(exponent))
        
        # Compute classification error (0-1 loss)
        prediction = 1 if dot_product >= 0 else -1
        if prediction != y:
            total_errors += 1
    
    avg_logistic_loss = total_loss / N
    classification_error = total_errors / N
    
    return avg_logistic_loss, classification_error
