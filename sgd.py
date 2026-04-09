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
    
        # Project w back to the unit ball if its norm exceeds 1
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




def create_plots(n_values, sigma_results):
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    styles = {0.2: ('-o', 'blue'), 0.4: ('--s', 'red')} # Different styles 
    
    for sigma in [0.2, 0.4]:
        res = sigma_results[sigma]
        style, color = styles[sigma]
        
        # Plot 1: Expected Excess Risk 
        ax1.errorbar(n_values, res['excess_risk'], yerr=res['risk_std'], 
                     fmt=style, color=color, label=f'σ = {sigma}', capsize=5)
        
        # Plot 2: Expected Classification Error 
        ax2.errorbar(n_values, res['mean_error'], yerr=res['error_std'], 
                     fmt=style, color=color, label=f'σ = {sigma}', capsize=5)

    # Formatting 
    ax1.set_title("Expected Excess Risk vs n")
    ax1.set_xlabel("Number of training examples (n)")
    ax1.set_ylabel("Excess Risk")
    ax1.legend()

    ax2.set_title("Expected Classification Error vs n")
    ax2.set_xlabel("Number of training examples (n)")
    ax2.set_ylabel("Classification Error")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()