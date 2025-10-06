# In main.py

# Import your custom tools
from preprocessing import load_and_preprocess
from modelling import evaluate_model, get_scores_over_time
from visualizations import plot_psd_and_tfr, plot_csp_patterns, plot_scores_over_time

# --- 1. SET PARAMETERS ---
file_path = 'S004R04.edf'
RUN_SLOW_ANALYSIS = True  # Set to True to run the time-consuming analysis

# --- 2. PREPROCESSING ---
print("--- Starting Preprocessing ---")
# Define the faulty channel you found by inspecting the data
noisy_channels = ['FT8']
epochs = load_and_preprocess(file_path,
    bad_channels=noisy_channels
)

if epochs:
    # Now you can proceed with your analysis on the clean data
    print("\nAnalysis can now proceed on the cleaned epochs.")
if epochs:
    # --- 3. VISUALIZE & EVALUATE ---
    plot_psd_and_tfr(epochs)
    
    print("\n--- Evaluating Overall Model Performance ---")
    mean_accuracy = evaluate_model(epochs)
    print(f"\nMean CV Accuracy: {mean_accuracy * 100:.2f}%")
    
    print("\n--- Visualizing CSP Patterns ---")
    plot_csp_patterns(epochs)
    
    # --- 4. (OPTIONAL) SLOW ANALYSIS ---
    if RUN_SLOW_ANALYSIS:
        print("\n--- Running 'Accuracy Over Time' Analysis (this may take several minutes) ---")
        scores_over_time = get_scores_over_time(epochs)
        plot_scores_over_time(epochs, scores_over_time)

print("\n--- Analysis Complete ---")