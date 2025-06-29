"""
SAE Orthogonality Analysis Tool

Compares feature orthogonality between two SAE models using:
1. Cosine similarity analysis (mean + histograms)
2. PCA elbow curves and variance explained
3. Gram matrix analysis (diagonal/off-diagonal stats + histograms)

Usage:
python sae_orthogonality_analysis.py --sae1_path sae_topk_model.pt --sae2_path joint_sae_nfm_model.pt --k1 1024 --k2 1024
"""
#python part16_orthogonality_analysis.py --sae1_path checkpoints_topk\best_model.pt --sae2_path .\checkpoints_joint\best_joint_sae_nfm_model.pt --sae1_name "Standalone_SAE" --sae2_name "Joint_SAE" --k1 1024 --k2 1024 --output_dir orthogonality_results --input_dim 3200

# COMPARATIVE ANALYSIS
# ============================================================

# Cosine Similarity Comparison:
# Standalone_SAE mean abs similarity: 0.014675
# Joint_SAE mean abs similarity: 0.014750
# 🏆 Standalone_SAE has better orthogonality (lower similarity)

# PCA Comparison (components needed for 90% variance):
# Standalone_SAE: 866 components
# Joint_SAE: 863 components
# 🏆 Standalone_SAE has better feature diversity (needs more components)

# Gram Matrix Comparison (off-diagonal mean absolute values):
# Standalone_SAE: 0.005753
# 🏆 Joint_SAE has better orthogonality (lower interactions)


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.utils.extmath import randomized_svd
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

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
    
    def apply_topk(self, features):
        """Apply TopK sparsity - keep only top K activations per sample"""
        batch_size, num_features = features.shape
        topk_values, topk_indices = torch.topk(features, self.k, dim=1)
        sparse_features = torch.zeros_like(features)
        sparse_features.scatter_(1, topk_indices, topk_values)
        return sparse_features

class NeuralFactorizationMachine(nn.Module):
    """NFM component (minimal version for loading)"""
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

def load_sae_model(model_path, model_type, input_dim=4096, num_features=50000, k=1024):
    """Load SAE model from checkpoint"""
    print(f"Loading {model_type} model from {model_path}")
    
    if "joint" in model_path.lower():
        # Joint model - extract just the SAE part
        joint_model = JointSAENFM(input_dim, num_features, k, 300)  # NFM dim doesn't matter
        state_dict = torch.load(model_path, map_location='cpu')
        joint_model.load_state_dict(state_dict)
        sae_model = joint_model.primary_sae
    else:
        # Standalone SAE model
        sae_model = TopKSparseAutoencoder(input_dim, num_features, k)
        state_dict = torch.load(model_path, map_location='cpu')
        sae_model.load_state_dict(state_dict)
    
    return sae_model

def filter_low_values(tensor, threshold=1e-6, name="tensor"):
    """Filter out low values and return mask + filtered tensor"""
    mask = torch.abs(tensor) > threshold
    n_filtered = (~mask).sum().item()
    n_total = mask.numel()
    print(f"{name}: Filtered {n_filtered}/{n_total} ({n_filtered/n_total*100:.2f}%) values below {threshold}")
    return mask, tensor[mask]

def compute_cosine_similarity_efficient(decoder_weights, batch_size=1000, threshold=1e-6):
    """
    Efficiently compute cosine similarity matrix for decoder weights
    
    Args:
        decoder_weights: [input_dim, num_features] tensor
        batch_size: Process in batches to manage memory
        threshold: Filter values below this threshold
    
    Returns:
        similarity_matrix: [num_features, num_features]
        off_diagonal_similarities: flattened off-diagonal values
    """
    num_features = decoder_weights.shape[1]
    device = decoder_weights.device
    
    print(f"Computing cosine similarity for {num_features} features...")
    
    # Normalize decoder weights (each column is a feature vector)
    decoder_norm = torch.norm(decoder_weights, dim=0, keepdim=True)
    # Filter out near-zero features
    valid_features = decoder_norm.squeeze() > threshold
    n_valid = valid_features.sum().item()
    print(f"Using {n_valid}/{num_features} features (filtered {num_features-n_valid} with norm < {threshold})")
    
    if n_valid < 2:
        print("ERROR: Not enough valid features for similarity analysis")
        return None, None
    
    # Work with valid features only
    valid_decoder = decoder_weights[:, valid_features]
    valid_norm = decoder_norm[:, valid_features]
    normalized_decoder = valid_decoder / (valid_norm + 1e-8)
    
    # Compute similarity matrix in batches
    similarity_matrix = torch.zeros(n_valid, n_valid, device=device)
    
    for i in tqdm(range(0, n_valid, batch_size), desc="Computing cosine similarities"):
        end_i = min(i + batch_size, n_valid)
        batch_i = normalized_decoder[:, i:end_i]
        
        for j in range(0, n_valid, batch_size):
            end_j = min(j + batch_size, n_valid)
            batch_j = normalized_decoder[:, j:end_j]
            
            # Compute cosine similarity: (A^T @ B) where A, B are normalized
            sim_block = torch.mm(batch_i.T, batch_j)
            similarity_matrix[i:end_i, j:end_j] = sim_block
    
    # Extract off-diagonal elements
    mask = ~torch.eye(n_valid, dtype=torch.bool, device=device)
    off_diagonal_similarities = similarity_matrix[mask]
    
    return similarity_matrix.cpu(), off_diagonal_similarities.cpu()

def compute_pca_analysis(decoder_weights, n_components=1000, threshold=1e-6):
    """
    Compute PCA analysis with elbow curve
    
    Args:
        decoder_weights: [input_dim, num_features] tensor
        n_components: Number of components to analyze
        threshold: Filter features below this threshold
    
    Returns:
        explained_variance_ratio: Array of explained variance ratios
        cumulative_variance: Cumulative explained variance
        n_components_90: Number of components for 90% variance
        n_components_95: Number of components for 95% variance
    """
    print(f"Computing PCA analysis...")
    
    # Transpose to [num_features, input_dim] for PCA
    features_matrix = decoder_weights.T.cpu().numpy()
    
    # Filter out low-norm features
    feature_norms = np.linalg.norm(features_matrix, axis=1)
    valid_mask = feature_norms > threshold
    n_valid = valid_mask.sum()
    print(f"Using {n_valid}/{len(features_matrix)} features for PCA")
    
    if n_valid < n_components:
        n_components = min(n_valid - 1, 500)
        print(f"Reducing n_components to {n_components}")
    
    valid_features = features_matrix[valid_mask]
    
    # Use randomized PCA for efficiency with large matrices
    if n_valid > 5000:
        print("Using randomized SVD for efficiency...")
        U, s, Vt = randomized_svd(valid_features, n_components=n_components, random_state=42)
        # Compute explained variance ratio
        explained_variance = (s ** 2) / (n_valid - 1)
        total_variance = np.sum(explained_variance)
        explained_variance_ratio = explained_variance / total_variance
    else:
        print("Using standard PCA...")
        pca = PCA(n_components=n_components)
        pca.fit(valid_features)
        explained_variance_ratio = pca.explained_variance_ratio_
    
    # Compute cumulative variance
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Find components needed for 90% and 95% variance
    n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    
    return explained_variance_ratio, cumulative_variance, n_components_90, n_components_95

def compute_gram_matrix_analysis(decoder_weights, batch_size=1000, threshold=1e-6):
    """
    Compute Gram matrix analysis (W^T @ W)
    
    Args:
        decoder_weights: [input_dim, num_features] tensor
        batch_size: Process in batches to manage memory
        threshold: Filter values below this threshold
    
    Returns:
        diagonal_stats: Statistics for diagonal elements (squared norms)
        off_diagonal_stats: Statistics for off-diagonal elements
        gram_matrix: Full Gram matrix (if small enough)
    """
    print(f"Computing Gram matrix analysis...")
    
    num_features = decoder_weights.shape[1]
    device = decoder_weights.device
    
    # Filter valid features
    feature_norms = torch.norm(decoder_weights, dim=0)
    valid_mask = feature_norms > threshold
    n_valid = valid_mask.sum().item()
    print(f"Using {n_valid}/{num_features} features for Gram matrix")
    
    if n_valid < 2:
        print("ERROR: Not enough valid features for Gram matrix analysis")
        return None, None, None
    
    valid_decoder = decoder_weights[:, valid_mask]
    
    # Compute Gram matrix: G = W^T @ W
    # This gives us G[i,j] = dot product of feature i and feature j
    if n_valid <= 5000:  # Compute full matrix if manageable
        gram_matrix = torch.mm(valid_decoder.T, valid_decoder)
        
        # Extract diagonal (squared norms)
        diagonal_elements = torch.diag(gram_matrix)
        
        # Extract off-diagonal elements
        mask = ~torch.eye(n_valid, dtype=torch.bool, device=device)
        off_diagonal_elements = gram_matrix[mask]
        
        gram_matrix_cpu = gram_matrix.cpu()
    else:
        print("Large matrix - computing statistics without storing full Gram matrix...")
        # Compute diagonal elements (squared norms)
        diagonal_elements = torch.sum(valid_decoder ** 2, dim=0)
        
        # Compute off-diagonal elements in batches
        off_diagonal_elements = []
        
        for i in tqdm(range(0, n_valid, batch_size), desc="Computing Gram matrix"):
            end_i = min(i + batch_size, n_valid)
            batch_i = valid_decoder[:, i:end_i]
            
            # Compute dot products with all other features
            dots = torch.mm(batch_i.T, valid_decoder)  # [batch_size, n_valid]
            
            # Extract off-diagonal elements for this batch
            for local_idx, global_idx in enumerate(range(i, end_i)):
                # Get row, exclude diagonal element
                row = dots[local_idx]
                off_diag_row = torch.cat([row[:global_idx], row[global_idx+1:]])
                off_diagonal_elements.append(off_diag_row)
        
        off_diagonal_elements = torch.cat(off_diagonal_elements)
        gram_matrix_cpu = None  # Too large to return
    
    # Compute statistics
    diagonal_stats = {
        'mean': diagonal_elements.mean().item(),
        'std': diagonal_elements.std().item(),
        'min': diagonal_elements.min().item(),
        'max': diagonal_elements.max().item(),
        'median': diagonal_elements.median().item()
    }
    
    off_diagonal_stats = {
        'mean': off_diagonal_elements.mean().item(),
        'std': off_diagonal_elements.std().item(),
        'min': off_diagonal_elements.min().item(),
        'max': off_diagonal_elements.max().item(),
        'median': off_diagonal_elements.median().item(),
        'mean_abs': torch.abs(off_diagonal_elements).mean().item()
    }
    
    return diagonal_stats, off_diagonal_stats, (gram_matrix_cpu, diagonal_elements.cpu(), off_diagonal_elements.cpu())

def plot_analysis_results(results_dict, output_dir):
    """Create comprehensive plots for orthogonality analysis"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Cosine Similarity Histograms
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    for i, (model_name, data) in enumerate(results_dict.items()):
        if data['cosine_similarity']['off_diagonal_similarities'] is not None:
            similarities = data['cosine_similarity']['off_diagonal_similarities']
            axes[i].hist(similarities, bins=100, alpha=0.7, density=True, label=model_name)
            axes[i].axvline(similarities.mean(), color='red', linestyle='--', 
                          label=f'Mean: {similarities.mean():.4f}')
            axes[i].set_xlabel('Cosine Similarity')
            axes[i].set_ylabel('Density')
            axes[i].set_title(f'{model_name} - Off-Diagonal Cosine Similarities')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cosine_similarity_histograms.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. PCA Elbow Curves
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Individual variance explained
    for model_name, data in results_dict.items():
        pca_data = data['pca_analysis']
        if pca_data['explained_variance_ratio'] is not None:
            n_components = len(pca_data['explained_variance_ratio'])
            axes[0].plot(range(1, n_components+1), pca_data['explained_variance_ratio'], 
                        label=f"{model_name}", marker='o', markersize=2)
    
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('Individual Component Variance Explained')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 50)  # Focus on first 50 components
    
    # Cumulative variance explained
    for model_name, data in results_dict.items():
        pca_data = data['pca_analysis']
        if pca_data['cumulative_variance'] is not None:
            n_components = len(pca_data['cumulative_variance'])
            axes[1].plot(range(1, n_components+1), pca_data['cumulative_variance'], 
                        label=f"{model_name}")
            
            # Mark 90% and 95% variance lines
            axes[1].axhline(0.90, color='red', linestyle='--', alpha=0.5)
            axes[1].axhline(0.95, color='orange', linestyle='--', alpha=0.5)
            axes[1].axvline(pca_data['n_components_90'], color='red', linestyle=':', alpha=0.5)
            axes[1].axvline(pca_data['n_components_95'], color='orange', linestyle=':', alpha=0.5)
    
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Variance Explained')
    axes[1].set_title('Cumulative Variance Explained')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, min(500, max([len(d['pca_analysis']['cumulative_variance']) 
                                   for d in results_dict.values() 
                                   if d['pca_analysis']['cumulative_variance'] is not None])))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_elbow_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Gram Matrix Histograms
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    plot_idx = 0
    for model_name, data in results_dict.items():
        gram_data = data['gram_matrix']
        if gram_data['gram_results'] is not None:
            _, diagonal_elements, off_diagonal_elements = gram_data['gram_results']
            
            # Diagonal elements (squared norms)
            axes[plot_idx].hist(diagonal_elements, bins=50, alpha=0.7, density=True)
            axes[plot_idx].axvline(diagonal_elements.mean(), color='red', linestyle='--')
            axes[plot_idx].set_xlabel('Squared Feature Norms')
            axes[plot_idx].set_ylabel('Density')
            axes[plot_idx].set_title(f'{model_name} - Diagonal Elements (Squared Norms)')
            axes[plot_idx].grid(True, alpha=0.3)
            
            # Off-diagonal elements
            axes[plot_idx + 1].hist(off_diagonal_elements, bins=100, alpha=0.7, density=True)
            axes[plot_idx + 1].axvline(off_diagonal_elements.mean(), color='red', linestyle='--')
            axes[plot_idx + 1].axvline(0, color='black', linestyle='-', alpha=0.5)
            axes[plot_idx + 1].set_xlabel('Feature Dot Products')
            axes[plot_idx + 1].set_ylabel('Density')
            axes[plot_idx + 1].set_title(f'{model_name} - Off-Diagonal Elements (Dot Products)')
            axes[plot_idx + 1].grid(True, alpha=0.3)
            
            plot_idx += 2
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gram_matrix_histograms.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"All plots saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='SAE Orthogonality Analysis')
    parser.add_argument('--sae1_path', type=str, required=True, help='Path to first SAE model')
    parser.add_argument('--sae2_path', type=str, required=True, help='Path to second SAE model')
    parser.add_argument('--sae1_name', type=str, default='SAE1', help='Name for first SAE')
    parser.add_argument('--sae2_name', type=str, default='SAE2', help='Name for second SAE')
    parser.add_argument('--k1', type=int, default=1024, help='TopK parameter for first SAE')
    parser.add_argument('--k2', type=int, default=1024, help='TopK parameter for second SAE')
    parser.add_argument('--input_dim', type=int, default=4096, help='Input dimension')
    parser.add_argument('--num_features', type=int, default=50000, help='Number of SAE features')
    parser.add_argument('--threshold', type=float, default=1e-6, help='Threshold for filtering low values')
    parser.add_argument('--output_dir', type=str, default='orthogonality_analysis', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for large matrix operations')
    parser.add_argument('--pca_components', type=int, default=1000, help='Number of PCA components to analyze')
    
    args = parser.parse_args()
    
    # Load models
    print("="*60)
    print("LOADING SAE MODELS")
    print("="*60)
    
    sae1 = load_sae_model(args.sae1_path, args.sae1_name, args.input_dim, args.num_features, args.k1)
    sae2 = load_sae_model(args.sae2_path, args.sae2_name, args.input_dim, args.num_features, args.k2)
    
    # Extract decoder weights
    decoder1 = sae1.decoder.weight.data  # [input_dim, num_features]
    decoder2 = sae2.decoder.weight.data  # [input_dim, num_features]
    
    print(f"\nDecoder shapes: {decoder1.shape}, {decoder2.shape}")
    
    results = {}
    
    # Analyze both models
    for model_name, decoder_weights in [(args.sae1_name, decoder1), (args.sae2_name, decoder2)]:
        print(f"\n{'='*60}")
        print(f"ANALYZING {model_name}")
        print(f"{'='*60}")
        
        results[model_name] = {}
        
        # 1. Cosine Similarity Analysis
        print(f"\n1. COSINE SIMILARITY ANALYSIS")
        print("-" * 40)
        similarity_matrix, off_diag_sims = compute_cosine_similarity_efficient(
            decoder_weights, args.batch_size, args.threshold
        )
        
        if off_diag_sims is not None:
            cosine_stats = {
                'mean_similarity': off_diag_sims.mean().item(),
                'std_similarity': off_diag_sims.std().item(),
                'mean_abs_similarity': torch.abs(off_diag_sims).mean().item(),
                'median_similarity': off_diag_sims.median().item(),
                'min_similarity': off_diag_sims.min().item(),
                'max_similarity': off_diag_sims.max().item()
            }
            
            print(f"Mean cosine similarity (off-diagonal): {cosine_stats['mean_similarity']:.6f}")
            print(f"Mean absolute cosine similarity: {cosine_stats['mean_abs_similarity']:.6f}")
            print(f"Std cosine similarity: {cosine_stats['std_similarity']:.6f}")
        else:
            cosine_stats = None
        
        results[model_name]['cosine_similarity'] = {
            'stats': cosine_stats,
            'off_diagonal_similarities': off_diag_sims
        }
        
        # 2. PCA Analysis
        print(f"\n2. PCA ANALYSIS")
        print("-" * 40)
        explained_var_ratio, cumulative_var, n_comp_90, n_comp_95 = compute_pca_analysis(
            decoder_weights, args.pca_components, args.threshold
        )
        
        if explained_var_ratio is not None:
            print(f"Components for 90% variance: {n_comp_90}")
            print(f"Components for 95% variance: {n_comp_95}")
            print(f"First component explains: {explained_var_ratio[0]:.4f} of variance")
            print(f"Top 10 components explain: {cumulative_var[9]:.4f} of variance")
        
        results[model_name]['pca_analysis'] = {
            'explained_variance_ratio': explained_var_ratio,
            'cumulative_variance': cumulative_var,
            'n_components_90': n_comp_90,
            'n_components_95': n_comp_95
        }
        
        # 3. Gram Matrix Analysis
        print(f"\n3. GRAM MATRIX ANALYSIS")
        print("-" * 40)
        diag_stats, off_diag_stats, gram_results = compute_gram_matrix_analysis(
            decoder_weights, args.batch_size, args.threshold
        )
        
        if diag_stats is not None and off_diag_stats is not None:
            print(f"Diagonal (squared norms) - Mean: {diag_stats['mean']:.6f}, Std: {diag_stats['std']:.6f}")
            print(f"Off-diagonal (dot products) - Mean: {off_diag_stats['mean']:.6f}, Mean Abs: {off_diag_stats['mean_abs']:.6f}")
            print(f"Off-diagonal std: {off_diag_stats['std']:.6f}")
        
        results[model_name]['gram_matrix'] = {
            'diagonal_stats': diag_stats,
            'off_diagonal_stats': off_diag_stats,
            'gram_results': gram_results
        }
    
    # Comparative Analysis
    print(f"\n{'='*60}")
    print("COMPARATIVE ANALYSIS")
    print("="*60)
    
    # Compare cosine similarities
    if (results[args.sae1_name]['cosine_similarity']['stats'] is not None and 
        results[args.sae2_name]['cosine_similarity']['stats'] is not None):
        
        cos1 = results[args.sae1_name]['cosine_similarity']['stats']
        cos2 = results[args.sae2_name]['cosine_similarity']['stats']
        
        print(f"\nCosine Similarity Comparison:")
        print(f"{args.sae1_name} mean abs similarity: {cos1['mean_abs_similarity']:.6f}")
        print(f"{args.sae2_name} mean abs similarity: {cos2['mean_abs_similarity']:.6f}")
        
        if cos1['mean_abs_similarity'] < cos2['mean_abs_similarity']:
            print(f"🏆 {args.sae1_name} has better orthogonality (lower similarity)")
        else:
            print(f"🏆 {args.sae2_name} has better orthogonality (lower similarity)")
    
    # Compare PCA results
    pca1 = results[args.sae1_name]['pca_analysis']
    pca2 = results[args.sae2_name]['pca_analysis']
    
    if pca1['n_components_90'] is not None and pca2['n_components_90'] is not None:
        print(f"\nPCA Comparison (components needed for 90% variance):")
        print(f"{args.sae1_name}: {pca1['n_components_90']} components")
        print(f"{args.sae2_name}: {pca2['n_components_90']} components")
        
        if pca1['n_components_90'] > pca2['n_components_90']:
            print(f"🏆 {args.sae1_name} has better feature diversity (needs more components)")
        else:
            print(f"🏆 {args.sae2_name} has better feature diversity (needs more components)")
    
    # Compare Gram matrix results
    if (results[args.sae1_name]['gram_matrix']['off_diagonal_stats'] is not None and
        results[args.sae2_name]['gram_matrix']['off_diagonal_stats'] is not None):
        
        gram1 = results[args.sae1_name]['gram_matrix']['off_diagonal_stats']
        gram2 = results[args.sae2_name]['gram_matrix']['off_diagonal_stats']
        
        print(f"\nGram Matrix Comparison (off-diagonal mean absolute values):")
        print(f"{args.sae1_name}: {gram1['mean_abs']:.6f}")
        print(f"{args.sae2_name}: {gram2['mean_abs']:.6f}")
        
        if gram1['mean_abs'] < gram2['mean_abs']:
            print(f"🏆 {args.sae1_name} has better orthogonality (lower interactions)")
        else:
            print(f"🏆 {args.sae2_name} has better orthogonality (lower interactions)")
    
    # Create plots
    plot_analysis_results(results, args.output_dir)
    
    # Save detailed results
    output_path = Path(args.output_dir) / 'orthogonality_analysis_results.json'
    
    # Convert tensors to lists for JSON serialization
    json_results = {}
    for model_name, model_results in results.items():
        json_results[model_name] = {}
        
        # Cosine similarity
        cos_data = model_results['cosine_similarity']
        json_results[model_name]['cosine_similarity'] = {
            'stats': cos_data['stats']
            # Skip storing the full similarity arrays
        }
        
        # PCA
        pca_data = model_results['pca_analysis']
        json_results[model_name]['pca_analysis'] = {
            'n_components_90': int(pca_data['n_components_90']) if pca_data['n_components_90'] is not None else None,
            'n_components_95': int(pca_data['n_components_95']) if pca_data['n_components_95'] is not None else None,
            'first_component_variance': float(pca_data['explained_variance_ratio'][0]) if pca_data['explained_variance_ratio'] is not None else None,
            'top10_cumulative_variance': float(pca_data['cumulative_variance'][9]) if (pca_data['cumulative_variance'] is not None and len(pca_data['cumulative_variance']) > 9) else None
        }
        
        # Gram matrix
        gram_data = model_results['gram_matrix']
        json_results[model_name]['gram_matrix'] = {
            'diagonal_stats': gram_data['diagonal_stats'],
            'off_diagonal_stats': gram_data['off_diagonal_stats']
        }
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_path}")
    
    # Summary recommendations
    print(f"\n{'='*60}")
    print("SUMMARY & RECOMMENDATIONS")
    print("="*60)
    
    print(f"\nKey Metrics Summary:")
    print(f"{'Metric':<25} {'SAE1':<15} {'SAE2':<15} {'Winner'}")
    print("-" * 65)
    
    # Cosine similarity (lower is better)
    if (results[args.sae1_name]['cosine_similarity']['stats'] is not None and 
        results[args.sae2_name]['cosine_similarity']['stats'] is not None):
        cos1_val = results[args.sae1_name]['cosine_similarity']['stats']['mean_abs_similarity']
        cos2_val = results[args.sae2_name]['cosine_similarity']['stats']['mean_abs_similarity']
        winner = args.sae1_name if cos1_val < cos2_val else args.sae2_name
        print(f"{'Mean |cos similarity|':<25} {cos1_val:<15.6f} {cos2_val:<15.6f} {winner}")
    
    # PCA components (higher is better for diversity)
    if pca1['n_components_90'] is not None and pca2['n_components_90'] is not None:
        pca1_val = pca1['n_components_90']
        pca2_val = pca2['n_components_90']
        winner = args.sae1_name if pca1_val > pca2_val else args.sae2_name
        print(f"{'PCA 90% components':<25} {pca1_val:<15d} {pca2_val:<15d} {winner}")
    
    # Gram matrix off-diagonal (lower is better)
    if (results[args.sae1_name]['gram_matrix']['off_diagonal_stats'] is not None and
        results[args.sae2_name]['gram_matrix']['off_diagonal_stats'] is not None):
        gram1_val = results[args.sae1_name]['gram_matrix']['off_diagonal_stats']['mean_abs']
        gram2_val = results[args.sae2_name]['gram_matrix']['off_diagonal_stats']['mean_abs']
        winner = args.sae1_name if gram1_val < gram2_val else args.sae2_name
        print(f"{'Mean |dot products|':<25} {gram1_val:<15.6f} {gram2_val:<15.6f} {winner}")
    
    print(f"\nInterpretation Guide:")
    print(f"• Lower cosine similarity = more orthogonal features")
    print(f"• More PCA components for 90% variance = better feature diversity")
    print(f"• Lower Gram matrix off-diagonal values = less feature interference")
    print(f"• Check histograms for distribution shapes and outliers")
    
    print(f"\nNext Steps:")
    print(f"1. Examine the generated plots in {args.output_dir}/")
    print(f"2. Look for multimodal distributions in histograms (may indicate feature clusters)")
    print(f"3. Consider feature activation analysis on actual data")
    print(f"4. Run monosemanticity tests on top-performing model")

if __name__ == "__main__":
    main()