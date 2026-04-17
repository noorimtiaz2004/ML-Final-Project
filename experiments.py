import numpy as np
from evaluation import evaluate_predictor
from sgd import run_sgd
def run_experiments():
    
    # Fixed parameters
    N_TEST = 400
    N_TRIALS = 30
    dimension = 4
    
    # Variable parameters
    sigma_values = [0.2, 0.4]
    n_values = [50, 100, 500, 1000]
    
    # Store all results
    all_results = {}
    
    for sigma in sigma_values:
        from dd import generate_data
        print(f"\n{'='*60}")
        print(f"Running experiments for σ = {sigma}")
        print(f"{'='*60}")
        
        # Generate ONE test set for this sigma (reused across all trials)
        test_features, test_labels = generate_data(N_TEST, sigma, dimension)
        
        sigma_results = {
            'n': [],
            'mean_loss': [],
            'std_loss': [],
            'min_loss': [],
            'excess_risk': [],
            'mean_error': [],
            'std_error': []
        }
        
        for n in n_values:
            print(f"\nTesting with n = {n} training examples...")
            
            # Store results from all trials
            trial_losses = []
            trial_errors = []
            
            # Run 30 trials
            for trial in range(N_TRIALS):
                # Generate fresh training data for this trial
                train_features, train_labels = generate_data(n, sigma, dimension)
                
                # Run SGD
                w = run_sgd(train_features, train_labels, n)
                
                # Evaluate on test set
                loss, error = evaluate_predictor(w, test_features, test_labels)
                
                trial_losses.append(loss)
                trial_errors.append(error)
            
            # Compute statistics
            mean_loss = np.mean(trial_losses)
            std_loss = np.std(trial_losses, ddof=1)
            min_loss = np.min(trial_losses)
            excess_risk = mean_loss - min_loss
            
            mean_error = np.mean(trial_errors)
            std_error = np.std(trial_errors, ddof=1)
            
            # Store results
            sigma_results['n'].append(n)
            sigma_results['mean_loss'].append(mean_loss)
            sigma_results['std_loss'].append(std_loss)
            sigma_results['min_loss'].append(min_loss)
            sigma_results['excess_risk'].append(excess_risk)
            sigma_results['mean_error'].append(mean_error)
            sigma_results['std_error'].append(std_error)
            
            print(f"  Mean Loss: {mean_loss:.6f}, Std: {std_loss:.6f}")
            print(f"  Excess Risk: {excess_risk:.6f}")
            print(f"  Classification Error: {mean_error:.6f}, Std: {std_error:.6f}")
        
        all_results[sigma] = sigma_results
    
    return all_results
