from experiments import run_experiments
from plot import print_results_table, create_plots

if __name__ == "__main__":
    print("CSE 5523: SGD for Logistic Regression Project")
    print("Author: Jayden Vinson and Noor Imtiaz")
    print("\nStarting experiments...")
    # Run all experiments
    results = run_experiments()
    # Print results table
    print_results_table(results)
    # Create plots
    create_plots(results)
    print("\nExperiments complete! Plots saved as 'sgd_results.png'")
