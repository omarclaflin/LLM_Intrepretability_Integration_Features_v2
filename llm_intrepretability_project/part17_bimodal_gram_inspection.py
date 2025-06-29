"""
part17_bimodal_gram_inspection.py

Bimodal Gram Matrix Analysis: Feature Orthogonality vs Architecture Contributions

This script analyzes the relationship between feature orthogonality (squared norms from Gram matrix)
and feature contributions across different architecture components (SAE residual, NFM linear, NFM interaction).

Usage:
python part17_bimodal_gram_inspection.py --joint_model_path checkpoints_joint/best_joint_sae_nfm_model.pt --input_dim 3200 --num_features 50000 --sae_k 1024 --nfm_embedding_dim 300 --output_dir bimodal_analysis
"""

# Computing Spearman correlations...
# nfm_total_ratio: r = -0.9866, p = 0.000000
# linear_ratio: r = -0.8657, p = 0.000000
# interaction_ratio: r = -0.9217, p = 0.000000
# residual_ratio: r = 0.9866, p = 0.000000
# All visualizations saved to bimodal_analysis

# All results and visualizations saved to: bimodal_analysis

# ============================================================
# SUMMARY INSIGHTS
# ============================================================

# Key Findings:

# Correlations with Squared Norms:
#   nfm_total_ratio: negative strong correlation (r=-0.987, significant)
#   linear_ratio: negative strong correlation (r=-0.866, significant)
#   interaction_ratio: negative strong correlation (r=-0.922, significant)
#   residual_ratio: positive strong correlation (r=0.987, significant)

# Group Differences (< 0.2 vs > 0.2):
#   nfm_total_ratio: High squared norm features have significantly lower ratios
#     (High: 0.713 vs Low: 0.828, p=0.0e+00)
#   linear_ratio: High squared norm features have significantly lower ratios
#     (High: 0.290 vs Low: 0.331, p=0.0e+00)
#   interaction_ratio: High squared norm features have significantly lower ratios
#     (High: 0.423 vs Low: 0.497, p=0.0e+00)
#   residual_ratio: High squared norm features have significantly higher ratios
#     (High: 0.287 vs Low: 0.172, p=0.0e+00)



import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def make_json_serializable(obj):
    """Recursively convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif obj is None:
        return None
    else:
        return obj

class TopKSparseAutoencoder(nn.Module):
    """TopK SAE definition (matching your implementation)"""
    def __init__(self, input_dim, hidden_dim, k):
        super().__init__()
        self.k = k
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)

class NeuralFactorizationMachine(nn.Module):
    """NFM component (matching your implementation)"""
    def __init__(self, num_sae_features, embedding_dim, output_dim):
        super().__init__()
        self.num_sae_features = num_sae_features
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        
        self.feature_embeddings = nn.Embedding(num_sae_features, embedding_dim)
        self.linear = nn.Linear(num_sae_features, output_dim, bias=True)
        
        self.interaction_mlp = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, output_dim)
        )

class JointSAENFM(nn.Module):
    """Joint model definition (matching your implementation)"""
    def __init__(self, input_dim, sae_features, sae_k, nfm_embedding_dim):
        super().__init__()
        self.primary_sae = TopKSparseAutoencoder(input_dim, sae_features, sae_k)
        self.nfm = NeuralFactorizationMachine(sae_features, nfm_embedding_dim, input_dim)

def load_joint_model(model_path, input_dim, num_features, sae_k, nfm_embedding_dim):
    """Load the trained joint SAE+NFM model"""
    print(f"Loading joint model from {model_path}")
    
    joint_model = JointSAENFM(input_dim, num_features, sae_k, nfm_embedding_dim)
    state_dict = torch.load(model_path, map_location='cpu')
    joint_model.load_state_dict(state_dict)
    
    return joint_model

def compute_gram_matrix_diagonal(decoder_weights, threshold=1e-6):
    """
    Compute diagonal elements of Gram matrix (squared norms)
    
    Args:
        decoder_weights: [input_dim, num_features] tensor
        threshold: Filter features below this threshold
    
    Returns:
        squared_norms: Diagonal elements (squared norms) for valid features
        valid_indices: Indices of features that passed the threshold
    """
    print("Computing Gram matrix diagonal (squared norms)...")
    
    # Compute squared norms for all features
    squared_norms = torch.sum(decoder_weights ** 2, dim=0)
    
    # Filter valid features
    valid_mask = squared_norms > threshold
    valid_indices = torch.where(valid_mask)[0]
    valid_squared_norms = squared_norms[valid_mask]
    
    n_valid = len(valid_indices)
    n_total = len(squared_norms)
    
    print(f"Using {n_valid}/{n_total} features (filtered {n_total-n_valid} with squared norm < {threshold})")
    
    return valid_squared_norms.cpu().numpy(), valid_indices.cpu().numpy()

def compute_feature_contributions(joint_model, valid_indices):
    """
    Compute feature contributions across architecture components using embedding weights
    
    Args:
        joint_model: Trained JointSAENFM model
        valid_indices: Indices of valid features to analyze
    
    Returns:
        contributions: Dictionary with contribution metrics for each component
    """
    print("Computing feature contributions from embedding weights...")
    
    # Extract weights
    decoder_weights = joint_model.primary_sae.decoder.weight.data  # [input_dim, num_features]
    linear_weights = joint_model.nfm.linear.weight.data  # [output_dim, num_features]
    embedding_weights = joint_model.nfm.feature_embeddings.weight.data  # [num_features, embedding_dim]
    
    contributions = {
        'residual_sae': [],
        'nfm_linear': [],
        'nfm_interaction': [],
        'total': []
    }
    
    print(f"Analyzing {len(valid_indices)} valid features...")
    
    for i in tqdm(valid_indices, desc="Computing contributions"):
        # 1. Residual/SAE component: ||decoder_weight[:, i]||
        residual_contrib = torch.norm(decoder_weights[:, i]).item()
        
        # 2. NFM Linear component: ||NFM_linear_weight[:, i]||
        linear_contrib = torch.norm(linear_weights[:, i]).item()
        
        # 3. NFM Interaction component: ||NFM_embedding_weight[i, :]||
        interaction_contrib = torch.norm(embedding_weights[i, :]).item()
        
        # Total contribution
        total_contrib = residual_contrib + linear_contrib + interaction_contrib
        
        contributions['residual_sae'].append(residual_contrib)
        contributions['nfm_linear'].append(linear_contrib)
        contributions['nfm_interaction'].append(interaction_contrib)
        contributions['total'].append(total_contrib)
    
    # Convert to numpy arrays
    for key in contributions:
        contributions[key] = np.array(contributions[key])
    
    return contributions

def compute_contribution_ratios(contributions):
    """
    Compute contribution ratios for analysis
    
    Args:
        contributions: Dictionary with contribution arrays
    
    Returns:
        ratios: Dictionary with ratio arrays
    """
    print("Computing contribution ratios...")
    
    total = contributions['total']
    residual = contributions['residual_sae']
    linear = contributions['nfm_linear']
    interaction = contributions['nfm_interaction']
    
    # Avoid division by zero
    safe_total = np.where(total > 1e-8, total, 1e-8)
    
    ratios = {
        'nfm_total_ratio': (linear + interaction) / safe_total,
        'linear_ratio': linear / safe_total,
        'interaction_ratio': interaction / safe_total,
        'residual_ratio': residual / safe_total
    }
    
    return ratios

def perform_statistical_analysis(squared_norms, ratios):
    """
    Perform statistical analysis: t-tests and correlations
    
    Args:
        squared_norms: Array of squared norms
        ratios: Dictionary of ratio arrays
    
    Returns:
        stats_results: Dictionary with statistical results
    """
    print("Performing statistical analysis...")
    
    # Split data: <0.2 vs >0.2
    low_mask = squared_norms < 0.2
    high_mask = squared_norms > 0.2
    
    low_count = np.sum(low_mask)
    high_count = np.sum(high_mask)
    total_count = len(squared_norms)
    
    print(f"Data split: {low_count} features < 0.2, {high_count} features > 0.2 (total: {total_count})")
    
    stats_results = {
        'split_info': {
            'low_count': int(low_count),
            'high_count': int(high_count),
            'total_count': int(total_count),
            'low_percentage': float(low_count / total_count * 100),
            'high_percentage': float(high_count / total_count * 100)
        },
        't_tests': {},
        'correlations': {}
    }
    
    # T-tests for each ratio
    print("\nPerforming t-tests...")
    ratio_names = ['nfm_total_ratio', 'linear_ratio', 'interaction_ratio', 'residual_ratio']
    
    for ratio_name in ratio_names:
        ratio_values = ratios[ratio_name]
        
        low_values = ratio_values[low_mask]
        high_values = ratio_values[high_mask]
        
        if len(low_values) > 1 and len(high_values) > 1:
            # Perform two-sample t-test
            t_stat, p_value = stats.ttest_ind(low_values, high_values)
            
            # Compute descriptive statistics
            low_mean = np.mean(low_values)
            high_mean = np.mean(high_values)
            low_std = np.std(low_values)
            high_std = np.std(high_values)
            
            stats_results['t_tests'][ratio_name] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'low_mean': float(low_mean),
                'high_mean': float(high_mean),
                'low_std': float(low_std),
                'high_std': float(high_std),
                'mean_difference': float(high_mean - low_mean),
                'significant': bool(p_value < 0.05)
            }
            
            print(f"{ratio_name}:")
            print(f"  Low group (< 0.2): {low_mean:.4f} ± {low_std:.4f}")
            print(f"  High group (> 0.2): {high_mean:.4f} ± {high_std:.4f}")
            print(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.6f}")
            print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
        else:
            print(f"Warning: Insufficient data for t-test on {ratio_name}")
            stats_results['t_tests'][ratio_name] = None
    
    # Spearman correlations (full dataset)
    print("\nComputing Spearman correlations...")
    for ratio_name in ratio_names:
        ratio_values = ratios[ratio_name]
        
        # Remove any NaN or infinite values
        valid_mask = np.isfinite(ratio_values) & np.isfinite(squared_norms)
        clean_norms = squared_norms[valid_mask]
        clean_ratios = ratio_values[valid_mask]
        
        if len(clean_norms) > 2:
            corr_coef, corr_p = stats.spearmanr(clean_norms, clean_ratios)
            
            stats_results['correlations'][ratio_name] = {
                'correlation': float(corr_coef),
                'p_value': float(corr_p),
                'significant': bool(corr_p < 0.05),
                'n_samples': int(len(clean_norms))
            }
            
            print(f"{ratio_name}: r = {corr_coef:.4f}, p = {corr_p:.6f}")
        else:
            print(f"Warning: Insufficient data for correlation on {ratio_name}")
            stats_results['correlations'][ratio_name] = None
    
    return stats_results

def create_visualizations(squared_norms, ratios, stats_results, output_dir):
    """Create comprehensive visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    ratio_names = ['nfm_total_ratio', 'linear_ratio', 'interaction_ratio', 'residual_ratio']
    ratio_labels = ['NFM Total/Total', 'Linear/Total', 'Interaction/Total', 'Residual(SAE)/Total']
    
    # 1. Scatter plots: Squared norm vs ratios
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (ratio_name, ratio_label) in enumerate(zip(ratio_names, ratio_labels)):
        ax = axes[i]
        ratio_values = ratios[ratio_name]
        
        # Remove outliers for better visualization
        valid_mask = np.isfinite(ratio_values) & np.isfinite(squared_norms)
        clean_norms = squared_norms[valid_mask]
        clean_ratios = ratio_values[valid_mask]
        
        # Create scatter plot
        ax.scatter(clean_norms, clean_ratios, alpha=0.6, s=20)
        
        # Add vertical lines for split points
        ax.axvline(0.2, color='red', linestyle='--', alpha=0.7, label='Split < 0.2')
        ax.axvline(0.2, color='orange', linestyle='--', alpha=0.7, label='Split > 0.2')
        
        # Add correlation info
        if ratio_name in stats_results['correlations']:
            corr_info = stats_results['correlations'][ratio_name]
            ax.text(0.05, 0.95, f"r = {corr_info['correlation']:.3f}\np = {corr_info['p_value']:.1e}", 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Squared Norm (Gram Matrix Diagonal)')
        ax.set_ylabel(ratio_label)
        ax.set_title(f'Squared Norm vs {ratio_label}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_plots_squared_norm_vs_ratios.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Violin plots: Distribution of ratios by squared norm bins
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Create dataframe for violin plots
    plot_data = []
    
    low_mask = squared_norms < 0.2
    high_mask = squared_norms > 0.2
    
    for ratio_name, ratio_label in zip(ratio_names, ratio_labels):
        ratio_values = ratios[ratio_name]
        
        # Low group
        low_values = ratio_values[low_mask]
        for val in low_values:
            if np.isfinite(val):
                plot_data.append({
                    'ratio_name': ratio_label,
                    'group': 'Low (< 0.2)',
                    'value': val
                })
        
        # High group
        high_values = ratio_values[high_mask]
        for val in high_values:
            if np.isfinite(val):
                plot_data.append({
                    'ratio_name': ratio_label,
                    'group': 'High (> 0.2)',
                    'value': val
                })
    
    df = pd.DataFrame(plot_data)
    
    for i, (ratio_name, ratio_label) in enumerate(zip(ratio_names, ratio_labels)):
        ax = axes[i]
        
        # Filter data for this ratio
        ratio_df = df[df['ratio_name'] == ratio_label]
        
        if len(ratio_df) > 0:
            sns.violinplot(data=ratio_df, x='group', y='value', ax=ax)
            
            # Add t-test results
            if ratio_name in stats_results['t_tests']:
                t_info = stats_results['t_tests'][ratio_name]
                significance = '***' if t_info['p_value'] < 0.001 else ('**' if t_info['p_value'] < 0.01 else ('*' if t_info['p_value'] < 0.05 else 'ns'))
                ax.text(0.5, 0.95, f"p = {t_info['p_value']:.1e} {significance}", 
                       transform=ax.transAxes, ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(f'{ratio_label} by Squared Norm Groups')
        ax.set_ylabel(ratio_label)
        ax.set_xlabel('Squared Norm Group')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'violin_plots_ratios_by_groups.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Distribution of squared norms (show bimodal nature)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.hist(squared_norms, bins=100, alpha=0.7, density=True, color='skyblue', edgecolor='black')
    ax.axvline(0.2, color='red', linestyle='--', linewidth=2, label='Split < 0.2')
    ax.axvline(0.2, color='orange', linestyle='--', linewidth=2, label='Split > 0.2')
    ax.axvline(np.mean(squared_norms), color='green', linestyle='-', linewidth=2, label=f'Mean = {np.mean(squared_norms):.3f}')
    
    ax.set_xlabel('Squared Norm (Gram Matrix Diagonal)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Feature Squared Norms (Bimodal)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'squared_norms_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"All visualizations saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Bimodal Gram Matrix Analysis')
    parser.add_argument('--joint_model_path', type=str, required=True, help='Path to trained joint SAE+NFM model')
    parser.add_argument('--input_dim', type=int, default=3200, help='Input dimension')
    parser.add_argument('--num_features', type=int, default=50000, help='Number of SAE features')
    parser.add_argument('--sae_k', type=int, default=1024, help='TopK parameter for SAE')
    parser.add_argument('--nfm_embedding_dim', type=int, default=300, help='NFM embedding dimension')
    parser.add_argument('--threshold', type=float, default=1e-6, help='Threshold for filtering features')
    parser.add_argument('--output_dir', type=str, default='bimodal_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    print("="*60)
    print("BIMODAL GRAM MATRIX ANALYSIS")
    print("="*60)
    
    # Load the joint model
    joint_model = load_joint_model(
        args.joint_model_path, args.input_dim, args.num_features, 
        args.sae_k, args.nfm_embedding_dim
    )
    
    # Extract decoder weights for Gram matrix analysis
    decoder_weights = joint_model.primary_sae.decoder.weight.data  # [input_dim, num_features]
    print(f"Decoder weights shape: {decoder_weights.shape}")
    
    # 1. Compute squared norms (Gram matrix diagonal)
    squared_norms, valid_indices = compute_gram_matrix_diagonal(decoder_weights, args.threshold)
    
    print(f"\nSquared norms statistics:")
    print(f"  Mean: {np.mean(squared_norms):.6f}")
    print(f"  Std: {np.std(squared_norms):.6f}")
    print(f"  Min: {np.min(squared_norms):.6f}")
    print(f"  Max: {np.max(squared_norms):.6f}")
    print(f"  Median: {np.median(squared_norms):.6f}")
    
    # Check bimodal distribution
    below_02 = np.sum(squared_norms < 0.2)
    above_025 = np.sum(squared_norms > 0.2)
    between = len(squared_norms) - below_02 - above_025
    
    print(f"\nBimodal distribution check:")
    print(f"  < 0.2: {below_02} ({below_02/len(squared_norms)*100:.1f}%)")
    print(f"  0.2-0.2: {between} ({between/len(squared_norms)*100:.1f}%)")
    print(f"  > 0.2: {above_025} ({above_025/len(squared_norms)*100:.1f}%)")
    
    # 2. Compute feature contributions
    contributions = compute_feature_contributions(joint_model, valid_indices)
    
    print(f"\nContribution statistics:")
    for comp_name, comp_values in contributions.items():
        print(f"  {comp_name}: mean = {np.mean(comp_values):.6f}, std = {np.std(comp_values):.6f}")
    
    # 3. Compute contribution ratios
    ratios = compute_contribution_ratios(contributions)
    
    print(f"\nRatio statistics:")
    for ratio_name, ratio_values in ratios.items():
        print(f"  {ratio_name}: mean = {np.mean(ratio_values):.4f}, std = {np.std(ratio_values):.4f}")
    
    # 4. Perform statistical analysis
    stats_results = perform_statistical_analysis(squared_norms, ratios)
    
    # 5. Create visualizations
    create_visualizations(squared_norms, ratios, stats_results, args.output_dir)
    
    print(f"\nAll results and visualizations saved to: {args.output_dir}")
    
    # 6. Summary insights
    print(f"\n" + "="*60)
    print("SUMMARY INSIGHTS")
    print("="*60)
    
    print(f"\nKey Findings:")
    
    # Correlation insights
    print(f"\nCorrelations with Squared Norms:")
    for ratio_name in ['nfm_total_ratio', 'linear_ratio', 'interaction_ratio', 'residual_ratio']:
        if ratio_name in stats_results['correlations']:
            corr_info = stats_results['correlations'][ratio_name]
            direction = "positive" if corr_info['correlation'] > 0 else "negative"
            strength = "strong" if abs(corr_info['correlation']) > 0.5 else ("moderate" if abs(corr_info['correlation']) > 0.3 else "weak")
            significance = "significant" if corr_info['significant'] else "not significant"
            print(f"  {ratio_name}: {direction} {strength} correlation (r={corr_info['correlation']:.3f}, {significance})")
    
    # T-test insights
    print(f"\nGroup Differences (< 0.2 vs > 0.2):")
    for ratio_name in ['nfm_total_ratio', 'linear_ratio', 'interaction_ratio', 'residual_ratio']:
        if ratio_name in stats_results['t_tests']:
            t_info = stats_results['t_tests'][ratio_name]
            direction = "higher" if t_info['mean_difference'] > 0 else "lower"
            significance = "significantly" if t_info['significant'] else "not significantly"
            print(f"  {ratio_name}: High squared norm features have {significance} {direction} ratios")
            print(f"    (High: {t_info['high_mean']:.3f} vs Low: {t_info['low_mean']:.3f}, p={t_info['p_value']:.1e})")
    
    print(f"\nInterpretation:")
    print(f"• Features with higher squared norms (> 0.2) are more 'concentrated' or specialized")
    print(f"• Features with lower squared norms (< 0.2) are more 'distributed' across dimensions")
    print(f"• Check correlations to see if specialized features prefer certain architecture components")
    print(f"• Check t-tests to see if there are systematic differences between feature types")
    
    print(f"\nNext Steps:")
    print(f"1. Examine the generated plots in {args.output_dir}/")
    print(f"2. Look for patterns: Do specialized features contribute more to NFM components?")
    print(f"3. Consider feature interpretability analysis on high vs low squared norm features")
    print(f"4. Investigate whether this bimodal split correlates with monosemanticity")

if __name__ == "__main__":
    main()