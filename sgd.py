import math
import matplotlib.pyplot as plt

def run_sgd(train_features, train_labels, n_iterations):
    # Initialize w as a 5d vector 
    w = [0.0] * 5 
    
    for t in range(1, n_iterations + 1):
        # Convert numpy feature to list and append 1 for x_tilde 
        x = list(train_features[t-1])
        x_tilde = x + [1.0]
        y = train_labels[t-1]
        
        # Set the learning rate
        eta = 1.0 / math.sqrt(t)
        
        # gradient = -y * x_tilde / (1 + exp(y * <w, xtilde>)
        dot_product = sum(w[i] * x_tilde[i] for i in range(5))
        denom = 1 + math.exp(y * dot_product)
        gradient = [(-y * xi) / denom for xi in x_tilde]
        
        #  Update weight vector
        w = [w[i] - eta * gradient[i] for i in range(5)]
    
        # Project w back to the unit ball if its norm exceeds 1 dajkd
        w_norm = math.sqrt(sum(val**2 for val in w))
        if w_norm > 1:
            w = [val / w_norm for val in w]
            
    return w



def evaluate_logistic_loss(w, test_features, test_labels):
    total_loss = 0
    total_errors = 0
    N = len(test_labels)
    
    for i in range(N):
        x_tilde = list(test_features[i]) + [1.0]
        y = test_labels[i]
        
        dot_product = sum(w[j] * x_tilde[j] for j in range(5))
        
        
        total_loss += math.log(1 + math.exp(-y * dot_product))
        
   
        prediction = 1 if dot_product >= 0 else -1
        if prediction != y:
            total_errors += 1
            
    return total_loss / N, total_errors / N




