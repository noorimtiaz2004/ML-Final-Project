import matplotlib.pyplot as plt
def create_plots(results):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    styles = {0.2: ('o-', 'blue', 'solid'), 
              0.4: ('s--', 'red', 'dashed')}
    
    for sigma in [0.2, 0.4]:
        res = results[sigma]
        marker_style, color, line_style = styles[sigma]

        # Plot 1: Excess Risk with error band
        ax1.plot(res['n'], res['excess_risk'], marker=marker_style[0], color=color, label=f'σ = {sigma}', linewidth=2, markersize=10)
        ax1.fill_between(res['n'],
                        [m-s for m, s in zip(res['excess_risk'], res['std_loss'])],
                        [m+s for m, s in zip(res['excess_risk'], res['std_loss'])],
                        color=color, alpha=0.15)
        # Annotate last point
        ax1.annotate(f"{res['excess_risk'][-1]:.3f}", (res['n'][-1], res['excess_risk'][-1]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12)

        # Plot 2: Classification Error with error band
        ax2.plot(res['n'], res['mean_error'], marker=marker_style[0], color=color, label=f'σ = {sigma}', linewidth=2, markersize=10)
        ax2.fill_between(res['n'],
                        [m-s for m, s in zip(res['mean_error'], res['std_error'])],
                        [m+s for m, s in zip(res['mean_error'], res['std_error'])],
                        color=color, alpha=0.15)
        ax2.annotate(f"{res['mean_error'][-1]:.3f}", (res['n'][-1], res['mean_error'][-1]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12)

    # Format Plot 1
    ax1.set_title("Expected Excess Risk vs Training Set Size", fontsize=16, fontweight='bold', pad=15)
    ax1.set_xlabel("Number of training examples (n)", fontsize=12)
    ax1.set_ylabel("Excess Risk (mean - min)", fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Format Plot 2
    ax2.set_title("Expected Classification Error vs Training Set Size", fontsize=16, fontweight='bold', pad=15)
    ax2.set_xlabel("Number of training examples (n)", fontsize=12)
    ax2.set_ylabel("Classification Error Rate", fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('sgd_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_results_table(results):
    """
    Print results in the table format required by the report.
    """
    print("\n" + "="*120)
    print("RESULTS TABLE")
    print("="*120)
    print(f"{'σ':<6} {'n':<6} {'N':<6} {'#trials':<9} {'Mean Loss':<12} {'Std Loss':<12} "
          f"{'Min Loss':<12} {'Excess Risk':<14} {'Mean Error':<12} {'Std Error':<12}")
    print("-"*120)
    
    for sigma in [0.2, 0.4]:
        res = results[sigma]
        for i, n in enumerate(res['n']):
            print(f"{sigma:<6.1f} {n:<6} {400:<6} {30:<9} "
                  f"{res['mean_loss'][i]:<12.6f} {res['std_loss'][i]:<12.6f} "
                  f"{res['min_loss'][i]:<12.6f} {res['excess_risk'][i]:<14.6f} "
                  f"{res['mean_error'][i]:<12.6f} {res['std_error'][i]:<12.6f}")
    print("="*120)

