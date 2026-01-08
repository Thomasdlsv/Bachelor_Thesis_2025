"""
Generate IEEE conference-style visualizations for imputation method comparison results.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy import stats

# Set IEEE style formatting
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.figsize': (7.16, 4.8),  # IEEE single column width
    'figure.autolayout': True
})

# Mapping for method display names
METHOD_NAME_MAPPING = {
    'BGAIN': 'BGAN',
    'BN_AUG_Imputer': 'BN-BGAN'
}

def create_rmse_comparison_plot(df):
    """Create box plot comparing RMSE across methods, missing patterns, and datasets."""
    # Filter for complete_train scenario
    df_complete = df[df['scenario'] == 'complete_train']
    
    # Create subplot for each dataset
    datasets = ['hepatitis', 'heart']
    patterns = ['MAR', 'MCAR', 'MNAR']
    methods = ['BGAIN', 'BN_AUG_Imputer', 'KNN', 'MICE', 'MissForest']
    
    # Create figure with subplots for each dataset
    fig = plt.figure(figsize=(7.16, 8))
    fig.set_constrained_layout(True)  # Use constrained_layout instead of tight_layout
    
    # Create GridSpec with more control over spacing
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 0.2])
    
    # Create color palette
    colors = sns.color_palette("husl", len(methods))
    
    for row, dataset in enumerate(datasets):
        for col, pattern in enumerate(patterns):
            ax = fig.add_subplot(gs[row, col])
            data = df_complete[(df_complete['pattern'] == pattern) & 
                             (df_complete['dataset'] == dataset)]
            
            # Create boxplot
            bp = ax.boxplot([data[data['method'] == method]['continuous_rmse_mean'] 
                           for method in methods],
                           patch_artist=True,
                           medianprops=dict(color="black", linewidth=1.5),
                           flierprops=dict(marker='o', markerfacecolor='gray', markersize=4))
            
            # Color boxes
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Set titles and labels
            if row == 0:
                ax.set_title(f'{pattern}', pad=10)
            if col == 0:
                ax.set_ylabel(f'{dataset.capitalize()}\nRMSE')
            
            ax.set_xticks(range(1, len(methods) + 1))
            if row == 1:  # Only show method names on bottom row
                # Map method names for display
                display_names = [METHOD_NAME_MAPPING.get(m, m) for m in methods]
                ax.set_xticklabels(display_names, rotation=45, ha='right')
            else:
                ax.set_xticklabels([])
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Create legend in the bottom row of GridSpec
    legend_ax = fig.add_subplot(gs[2, :])
    legend_ax.axis('off')
    # Map method names for legend
    display_names = [METHOD_NAME_MAPPING.get(m, m) for m in methods]
    legend_elements = [Patch(facecolor=color, alpha=0.7, label=display_name)
                      for color, display_name in zip(colors, display_names)]
    legend_ax.legend(handles=legend_elements, loc='center', 
                    ncol=3, frameon=False, 
                    bbox_to_anchor=(0.5, 0.5))
    
    # Add title with proper spacing
    fig.suptitle('Imputation Performance Comparison by Dataset and Missing Pattern', 
                y=1.02, fontsize=12)
    
    # Save figure
    plt.savefig('rmse_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
def create_missing_rate_impact_plot(df):
    
    # Filter for complete_train scenario and specific methods
    df_complete = df[
        (df['scenario'] == 'complete_train') & 
        (df['method'].isin(['BGAIN', 'BN_AUG_Imputer']))
    ]
    
    # Create figure with subplots for each pattern and dataset
    fig = plt.figure(figsize=(7.16, 9))
    fig.set_constrained_layout(True)  # Use constrained_layout instead of tight_layout
    
    # Create GridSpec with space for legend at bottom
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1, 0.2])
    
    datasets = ['hepatitis', 'heart']
    patterns = ['MAR', 'MCAR', 'MNAR']
    
    # Color scheme for methods
    colors = {'BGAIN': '#1f77b4', 'BN_AUG_Imputer': '#ff7f0e'}
    markers = {'BGAIN': 'o', 'BN_AUG_Imputer': 's'}
    
    # Create subplots for each pattern and dataset
    for i, pattern in enumerate(patterns):
        for j, dataset in enumerate(datasets):
            ax = fig.add_subplot(gs[i, j])
            
            dataset_data = df_complete[
                (df_complete['dataset'] == dataset) & 
                (df_complete['pattern'] == pattern)
            ]
            
            # Plot each method
            for method in ['BGAIN', 'BN_AUG_Imputer']:
                method_data = dataset_data[dataset_data['method'] == method]
                
                # Sort by missing rate to ensure proper line connection
                method_data = method_data.sort_values('missing_rate')
                
                # Use display name for label
                display_name = METHOD_NAME_MAPPING.get(method, method)
                ax.errorbar(method_data['missing_rate'], 
                          method_data['continuous_rmse_mean'],
                          yerr=method_data['continuous_rmse_std'],
                          label=display_name,
                          color=colors[method],
                          marker=markers[method],
                          markersize=6,
                          capsize=3,
                          capthick=1,
                          linewidth=1.5,
                          elinewidth=1)
            
            # Customize each subplot
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Only add labels on outer edges for cleaner appearance
            if i == 2:  # Bottom row
                ax.set_xlabel('Missing Rate (%)', fontsize=10)
            if j == 0:  # Left column
                ax.set_ylabel('RMSE', fontsize=10)
            
            # Add pattern label on left side (y-axis area)
            if j == 0:
                # Create pattern description
                pattern_desc = {
                    'MAR': 'Missing at Random',
                    'MCAR': 'Missing Completely at Random',
                    'MNAR': 'Missing Not at Random'
                }
                ax.text(-0.45, 0.5, pattern_desc[pattern],
                       transform=ax.transAxes, fontsize=10, weight='bold',
                       rotation=90, verticalalignment='center',
                       horizontalalignment='center')
            
            # Add dataset label at the top in a subtle way
            if i == 0:
                dataset_labels = {'hepatitis': 'Hepatitis Dataset', 'heart': 'Heart Disease Dataset'}
                ax.text(0.5, 1.12, dataset_labels[dataset],
                       transform=ax.transAxes, fontsize=10, weight='bold',
                       horizontalalignment='center')
            
            # Set consistent y-axis limits for each dataset
            if dataset == 'hepatitis':
                ax.set_ylim(0, 50)
            else:
                ax.set_ylim(0, 45)
            
            # Format x-axis with percentage labels
            ax.set_xticks([0.1, 0.2, 0.3])
            ax.set_xticklabels(['10%', '20%', '30%'])
    
    # Add legend in the bottom row
    legend_ax = fig.add_subplot(gs[3, :])
    legend_ax.axis('off')
    
    # Create legend with both methods using display names
    legend_elements = []
    for method in ['BGAIN', 'BN_AUG_Imputer']:
        display_name = METHOD_NAME_MAPPING.get(method, method)
        legend_elements.append(
            plt.Line2D([0], [0], color=colors[method], marker=markers[method],
                      label=display_name, markersize=6, linewidth=1.5)
        )
    legend_ax.legend(handles=legend_elements, loc='center', ncol=2,
                    frameon=False, bbox_to_anchor=(0.5, 0.5))
    
    # Add overall title with proper spacing
    fig.suptitle('Performance Sensitivity to Missing Data Mechanisms and Rate',
                y=1.02, fontsize=12, weight='bold')
    
    # Save figure
    plt.savefig('missing_rate_impact.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_statistical_summary_table(df):
    """Create statistical summary tables for each dataset."""
    # Filter for complete_train scenario
    df_complete = df[df['scenario'] == 'complete_train']
    
    # Create LaTeX table
    with open('statistical_summary.tex', 'w') as f:
        f.write('\\begin{table*}[t]\n')
        f.write('\\centering\n')
        f.write('\\caption{Statistical Summary of Imputation Methods by Dataset}\n')
        f.write('\\begin{tabular}{llcccccc}\n')
        f.write('\\hline\n')
        f.write('Dataset & Method & \\multicolumn{2}{c}{MAR} & \\multicolumn{2}{c}{MCAR} & \\multicolumn{2}{c}{MNAR} \\\\\n')
        f.write('\\cline{3-8}\n')
        f.write('& & Mean & Std & Mean & Std & Mean & Std \\\\\n')
        f.write('\\hline\n')
        
        # Process each dataset
        for dataset in ['hepatitis', 'heart']:
            dataset_data = df_complete[df_complete['dataset'] == dataset]
            
            # Process each method
            for method in ['BGAIN', 'BN_AUG_Imputer', 'KNN', 'MICE', 'MissForest']:
                method_data = dataset_data[dataset_data['method'] == method]
                
                # Calculate statistics for each pattern
                stats = []
                for pattern in ['MAR', 'MCAR', 'MNAR']:
                    pattern_data = method_data[method_data['pattern'] == pattern]
                    mean = pattern_data['continuous_rmse_mean'].mean()
                    std = pattern_data['continuous_rmse_mean'].std()
                    stats.extend([mean, std])
                
                # Write row to table
                f.write(f"{dataset.capitalize() if method == 'BGAIN' else ''} & "
                       f"{method} & "
                       f"{stats[0]:.2f} & {stats[1]:.2f} & "
                       f"{stats[2]:.2f} & {stats[3]:.2f} & "
                       f"{stats[4]:.2f} & {stats[5]:.2f} \\\\\n")
            
            # Add horizontal line between datasets
            if dataset == 'hepatitis':
                f.write('\\hline\n')
        
        f.write('\\hline\n')
        f.write('\\end{tabular}\n')
        f.write('\\label{tab:method_statistics}\n')
        f.write('\\end{table*}\n')
        
        # Add table notes
        f.write('\n% Table Notes:\n')
        f.write('% - Mean: Average RMSE across all missing rates\n')
        f.write('% - Std: Standard deviation of RMSE across all missing rates\n')
        f.write('% - Results shown for complete_train scenario only\n')

def create_comprehensive_summary(df):
    """Create a single boxplot comparing overall RMSE performance of all methods."""
    # Filter for complete_train scenario
    df_complete = df[df['scenario'] == 'complete_train']
    
    # Create figure
    plt.figure(figsize=(7.16, 5))
    
    # Prepare data for plotting
    methods = ['BGAIN', 'BN_AUG_Imputer', 'KNN', 'MICE', 'MissForest']
    plot_data = []
    for method in methods:
        method_data = df_complete[df_complete['method'] == method]['continuous_rmse_mean']
        plot_data.append(method_data.values)
    
    # Create boxplot
    bp = plt.boxplot(plot_data, 
                    patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='gray', 
                                  markersize=4, alpha=0.5),
                    widths=0.7)
    
    # Color boxes with professional color scheme
    colors = sns.color_palette("husl", len(methods))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Customize plot
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.ylabel('RMSE')
    plt.title('Overall Imputation Performance Comparison', pad=20)
    
    # Set x-axis labels with display names
    display_names = [METHOD_NAME_MAPPING.get(m, m) for m in methods]
    plt.xticks(range(1, len(methods) + 1), display_names, rotation=45, ha='right')
    
    # Add statistical annotations
    # Perform Wilcoxon test between BGAIN and BN_AUG_Imputer
    bgain_data = df_complete[df_complete['method'] == 'BGAIN']['continuous_rmse_mean']
    bnaug_data = df_complete[df_complete['method'] == 'BN_AUG_Imputer']['continuous_rmse_mean']
    stat_results = stats.wilcoxon(bgain_data, bnaug_data)
    
    # Add p-value annotation
    plt.text(0.98, 0.98, f'Wilcoxon test p={stat_results.pvalue:.2e}',
             transform=plt.gca().transAxes,
             horizontalalignment='right',
             verticalalignment='top',
             bbox=dict(facecolor='white', edgecolor='black', alpha=0.8, pad=5))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig('comprehensive_summary.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all visualizations."""
    print("Loading data and generating IEEE-style visualizations...")
    
    # Load data
    df = pd.read_csv('results_quality_summary.csv')
    
    # Generate visualizations
    create_rmse_comparison_plot(df)
    create_missing_rate_impact_plot(df)
    create_statistical_summary_table(df)
    
    create_comprehensive_summary(df)
    
    print("\nVisualization completed! Files generated:")
    print("1. rmse_comparison.pdf - Box plots comparing methods across missing patterns")
    print("2. missing_rate_impact.pdf - Line plot showing impact of missing rates")
    print("3. method_statistics.csv - Detailed statistical summary")
    print("4. statistical_summary.tex - LaTeX table for publication")
    print("5. comprehensive_summary.pdf - Single-page summary of all key findings")

if __name__ == '__main__':
    main()