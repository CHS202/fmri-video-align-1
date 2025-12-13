import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---
# STEP 1: LOAD AND PREPARE DATA
# ---
def load_data(file_map):
    """
    Loads data from the four CSV files and combines them into
    a single 'tidy' DataFrame.
    
    Assumes each CSV has the columns:
    model, split1, split2, split3, split4, avg, std
    """
    all_dfs = []
    
    for filename, condition_name in file_map.items():
        if not os.path.exists(filename):
            print(f"Warning: File not found at '{filename}'. Skipping.")
            continue
            
        try:
            df = pd.read_csv(filename)
            # Add the 'condition' column to identify the data source
            df['condition'] = condition_name
            all_dfs.append(df)
            print(f"  - Successfully loaded '{filename}' as '{condition_name}'")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            
    if not all_dfs:
        print("Error: No data was loaded. Please check your file_map and CSV files.")
        return pd.DataFrame(), pd.DataFrame()

    # Concatenate all DataFrames into one
    df_all = pd.concat(all_dfs, ignore_index=True)
    
    # "Melt" the DataFrame to make 'split' a variable,
    # which is much easier for plotting.
    df_long = df_all.melt(
        id_vars=['model', 'condition', 'avg', 'std'],
        value_vars=['split-01', 'split-02', 'split-03', 'split-04'],
        var_name='split',
        value_name='performance'
    )
    
    # Re-order the 'split' column for correct plotting
    df_long['split'] = pd.Categorical(df_long['split'], 
                                      categories=['split-01', 'split-02', 'split-03', 'split-04'],
                                      ordered=True)
    
    return df_all, df_long

# ---
# STEP 2: REPLICATE PLOTS AND ANALYSIS
# ---

def plot_comparison_chart(df):
    """
    Replicates the 'comparison' bar chart.
    Shows average performance by method, grouped by model.
    """
    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=df,
        x='model',
        y='avg',
        hue='condition'
    )
    plt.title('Average Performance by Method and Model', fontsize=16, fontweight='bold')
    plt.xlabel('Model')
    plt.ylabel('Average Performance')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Condition')
    plt.ylim(0.6, 0.85) # Set Y-axis limits like in the React chart
    plt.tight_layout()
    plt.savefig('plot_1_comparison.png')
    print("  - Saved 'plot_1_comparison.png'")

def plot_stability_chart(df):
    """
    Replicates the 'stability' scatter plot.
    Shows Avg Performance vs. Standard Deviation.
    """
    plt.figure(figsize=(10, 7))
    # Using 'style' for model adds another dimension (shape)
    sns.scatterplot(
        data=df,
        x='avg',
        y='std',
        hue='condition',
        style='model',
        s=150 # Make markers larger
    )
    plt.title('Performance vs. Stability (Avg vs. Std Dev)', fontsize=16, fontweight='bold')
    plt.xlabel('Average Performance')
    plt.ylabel('Standard Deviation (Lower is more stable)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    # Add annotation
    plt.annotate(
        'Ideal: Bottom-Right\n(High Avg, Low Std)', 
        xy=(0.05, 0.95), 
        xycoords='axes fraction',
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1, alpha=0.8)
    )
    
    plt.tight_layout()
    plt.savefig('plot_2_stability.png')
    print("  - Saved 'plot_2_stability.png'")

def plot_splits_chart(df_long):
    """
    Replicates the 'splits' line chart.
    This uses a FacetGrid as a static alternative to an interactive dropdown.
    """
    g = sns.FacetGrid(
        df_long, 
        col='model',  # Creates one plot per model
        col_wrap=3,   # Wraps to the next line after 3 plots
        height=5,
        sharey=True   # All plots share the same Y-axis
    )
    
    # Map a line plot onto each facet
    g.map_dataframe(
        sns.lineplot, 
        x='split', 
        y='performance', 
        hue='condition', 
        marker='o'
    )
    
    g.set_axis_labels('Data Split', 'Performance')
    g.set_titles(col_template="{col_name}", fontweight='bold')
    g.add_legend(title='Condition', loc='upper right', bbox_to_anchor=(1, 0.9), borderaxespad=0.)
    g.fig.suptitle('Performance Across Data Splits (by Model)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('plot_3_splits.png')
    print("  - Saved 'plot_3_splits.png'")

def print_rankings_analysis(df):
    """
    Replicates the 'rankings' view and key findings.
    """
    print("\n--- 📊 Key Findings and Rankings ---")
    
    # Find the best score (max avg) for each model
    idx_best = df.groupby('model')['avg'].idxmax()
    df_best = df.loc[idx_best].sort_values('avg', ascending=False).reset_index(drop=True)
    
    print("\n## 🏆 Model Rankings (by Best Avg. Performance) ##\n")
    for idx, row in df_best.iterrows():
        print(f"  #{idx+1}: {row['model']}")
        print(f"     Best Avg: {row['avg']:.3f}")
        print(f"     Condition: {row['condition']}\n")
        
    # Detailed scores per model
    print("\n## 📈 Detailed Scores by Model ##\n")
    all_models = sorted(df['model'].unique())
    for model in all_models:
        print(f"  --- {model} ---")
        model_data = df[df['model'] == model][['condition', 'avg', 'std']]
        print(model_data.to_string(index=False, float_format="%.3f"))
        print("") # Newline

# ---
# STEP 3: MAIN EXECUTION
# ---
def main():
    
    # ---
    # ⚠️ **EDIT THIS SECTION** ⚠️
    # ---
    # Update this dictionary to match your CSV filenames.
    # The 'key' is the path to your file (e.g., 'data/baseline.csv')
    # The 'value' is the label you want in the plot legends.
    
    # file_map = {
    #     'sig_test/rt/baseline.csv': 'Baseline (No Co-train)',
    #     'sig_test_lstm_feature/rt/cotrain_ce_cka.csv': 'Co-train (CE + CKA)',
    #     'sig_test_add_project_layer_mse/rt/cotrain_ce_mse.csv': 'Co-train (CE + MSE)',
    #     'sig_test/rt/cotrain_no_lstm.csv': 'Co-train (CE + CKA, No LSTM)'
    # }

    file_map = {
        'sig_test/rt/baseline.csv': 'Baseline (No Co-train)',
        'sig_test_lstm_feature/rt/cotrain_ce_cka.csv': 'Co-train (CE + CKA)',
        'sig_test_add_project_layer_mse/rt/cotrain_ce_mse.csv': 'Co-train (CE + MSE)',
        'sig_test/rt/cotrain_no_lstm.csv': 'Co-train (CE + CKA, No LSTM)',
        # 'sig_test_lstm_feature_add_mse/rt/cotrain_ce_cka_mse.csv': 'Co-train (CE + CKA + MSE)',
        # 'sig_test_lstm_feature_rho4rdm/rt/cotrain_ce_cka_rho4rdm.csv': 'Co-train (CE + CKA, Rho for RDM)',
        'sig_test_neurostorm_feature/rt/cotrain_ce_cka_neurostorm.csv': 'Co-train (CE + CKA, Neurostorm)',
        'sig_test_neurostorm_feature_cls/rt/cotrain_ce_cka_neurostorm_cls.csv': 'Co-train (CE + CKA, Neurostorm cls)',
        'sig_test_add_project_layer_mse_neurostorm_cls/rt/cotrain_ce_mse_neurostorm_cls.csv':  'Co-train (CE + MSE, Neurostorm cls)',
    }
    
    # 1. Load and process data from your CSVs
    print("Loading and processing data...")
    df_all, df_long = load_data(file_map)
    
    if df_all.empty:
        print("No data loaded. Exiting.")
        return
        
    print("Data loaded successfully.")
    
    # 2. Set plot style
    sns.set_theme(style="whitegrid", palette="muted")
    
    # 3. Generate analysis and plots
    print_rankings_analysis(df_all)
    
    print("\n--- 🖼️ Generating Plots ---")
    plot_comparison_chart(df_all)
    plot_stability_chart(df_all)
    plot_splits_chart(df_long)
    
    print("\nAll plots saved as .png files.")
    

if __name__ == "__main__":
    main()