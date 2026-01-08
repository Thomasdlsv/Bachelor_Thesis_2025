"""
Generate IEEE conference-style plots for uncertainty analysis comparing BGAIN and BN_AUG_Imputer.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.gridspec import GridSpec

# Set style for IEEE conference-quality plots
plt.style.use('default')
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.2)
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300
})
plt.rcParams['figure.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Mapping for method display names
METHOD_NAME_MAPPING = {
    'BGAIN': 'BGAN',
    'BN_AUG_Imputer': 'BN-BGAN'
}
def load_uncertainty_data():
    """Load uncertainty analysis results."""
    test_stds = pd.read_csv('test_uncertainty_per_cell_stds.csv')
    train_stds = pd.read_csv('uncertainty_per_cell_stds.csv')
    return test_stds, train_stds

def create_ieee_boxplot(test_stds):
    """Create IEEE conference-style box plot for uncertainty comparison."""
    fig, ax = plt.subplots(figsize=(6.5, 4))  # IEEE conference column width
    
    plot_data = pd.DataFrame({
        'BGAIN': test_stds['BGAIN_std'],
        'BN-AUG': test_stds['BN_AUG_std']
    }).melt(var_name='Method', value_name='Uncertainty (σ)')
    
    # Map method names for display
    plot_data['Method'] = plot_data['Method'].map({
        'BGAIN': METHOD_NAME_MAPPING['BGAIN'],
        'BN-AUG': METHOD_NAME_MAPPING['BN_AUG_Imputer']
    })
    
    # Create box plot with IEEE style
    sns.boxplot(data=plot_data, x='Method', y='Uncertainty (σ)', 
                width=0.5, ax=ax, color='lightgray',
                flierprops={'marker': 'o', 'markerfacecolor': 'gray', 'markersize': 4})
    
    # Statistical test
    wilcoxon_stat = stats.wilcoxon(test_stds['BGAIN_std'],
                                  test_stds['BN_AUG_std'])
    
    # Customize plot
    ax.set_title('Comparison of Imputation Uncertainty', pad=20)
    ax.set_xlabel('Imputation Method')
    ax.set_ylabel('Standard Deviation (σ)')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add Wilcoxon test result in the upper right corner
    ax.text(0.98, 0.95, f'p = {wilcoxon_stat.pvalue:.2e}',
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', edgecolor='black', alpha=0.8, pad=5))
    
    plt.tight_layout()
    plt.savefig('uncertainty_boxplot.pdf', bbox_inches='tight', dpi=300)
    plt.close()

def create_ieee_table(test_stds):
    """Create detailed IEEE-style statistical table."""
    # Compute detailed statistics
    stats_dict = {
        'Metric': [
            'Mean (σ)',
            'Median (σ)',
            'Std. Dev. (σ)',
            'Min (σ)',
            'Max (σ)',
            'Q1 (σ)',
            'Q3 (σ)',
            'IQR',
            'Skewness',
            'Kurtosis'
        ],
        'BGAIN': [
            f"{test_stds['BGAIN_std'].mean():.4f}",
            f"{test_stds['BGAIN_std'].median():.4f}",
            f"{test_stds['BGAIN_std'].std():.4f}",
            f"{test_stds['BGAIN_std'].min():.4f}",
            f"{test_stds['BGAIN_std'].max():.4f}",
            f"{test_stds['BGAIN_std'].quantile(0.25):.4f}",
            f"{test_stds['BGAIN_std'].quantile(0.75):.4f}",
            f"{test_stds['BGAIN_std'].quantile(0.75) - test_stds['BGAIN_std'].quantile(0.25):.4f}",
            f"{test_stds['BGAIN_std'].skew():.4f}",
            f"{test_stds['BGAIN_std'].kurtosis():.4f}"
        ],
        'BN-AUG': [
            f"{test_stds['BN_AUG_std'].mean():.4f}",
            f"{test_stds['BN_AUG_std'].median():.4f}",
            f"{test_stds['BN_AUG_std'].std():.4f}",
            f"{test_stds['BN_AUG_std'].min():.4f}",
            f"{test_stds['BN_AUG_std'].max():.4f}",
            f"{test_stds['BN_AUG_std'].quantile(0.25):.4f}",
            f"{test_stds['BN_AUG_std'].quantile(0.75):.4f}",
            f"{test_stds['BN_AUG_std'].quantile(0.75) - test_stds['BN_AUG_std'].quantile(0.25):.4f}",
            f"{test_stds['BN_AUG_std'].skew():.4f}",
            f"{test_stds['BN_AUG_std'].kurtosis():.4f}"
        ]
    }
    
    # Create figure for table
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table with IEEE formatting
    table = ax.table(cellText=[[stats_dict['Metric'][i], stats_dict['BGAIN'][i], stats_dict['BN-AUG'][i]] 
                              for i in range(len(stats_dict['Metric']))],
                    colLabels=['Metric', METHOD_NAME_MAPPING['BGAIN'], METHOD_NAME_MAPPING['BN_AUG_Imputer']],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    # Add title
    plt.title('Statistical Analysis of Imputation Uncertainty', pad=20)
    
    # Style header
    for j, cell in enumerate(table._cells[(0, j)] for j in range(3)):
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#E6E6E6')
    
    plt.savefig('uncertainty_statistics.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    """Create summary statistics plot."""
    plt.figure(figsize=(12, 6))
    
    # Compute summary statistics
    stats_dict = {
        'Method': ['BGAIN', 'BN-AUG'],
        'Mean': [test_stds['BGAIN_std'].mean(), test_stds['BN_AUG_std'].mean()],
        'Median': [test_stds['BGAIN_std'].median(), test_stds['BN_AUG_std'].median()],
        'Std': [test_stds['BGAIN_std'].std(), test_stds['BN_AUG_std'].std()]
    }
    
    # Plot as grouped bar chart
    stats_df = pd.DataFrame(stats_dict)
    stats_melted = stats_df.melt(id_vars=['Method'], var_name='Metric', value_name='Value')
    
    g = sns.barplot(data=stats_melted, x='Metric', y='Value', hue='Method')
    plt.title('Summary Statistics of Uncertainty Estimates')
    plt.ylabel('Value')
    
    # Add value labels on bars
    for container in g.containers:
        g.bar_label(container, fmt='%.3f')
    
    plt.tight_layout()
    plt.savefig('uncertainty_summary.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate IEEE conference-style uncertainty analysis outputs."""
    test_stds, train_stds = load_uncertainty_data()
    
    print("Generating IEEE conference-style uncertainty analysis outputs...")
    
    # Generate separate box plot and detailed statistics table
    create_ieee_boxplot(test_stds)
    create_ieee_table(test_stds)
    
    print("\nAnalysis completed!")
    print("Outputs saved as:")
    print("- uncertainty_boxplot.pdf (Visualization)")
    print("- uncertainty_statistics.pdf (Detailed Statistics)")

if __name__ == '__main__':
    main()