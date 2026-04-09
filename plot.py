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