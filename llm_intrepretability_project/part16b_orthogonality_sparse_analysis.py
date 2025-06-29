# python part16b_orthogonality_sparse_analysis.py --sae1_path checkpoints_topk\best_model.pt --sae2_path .\checkpoints_joint\best_joint_sae_nfm_model.pt --sae1_name "Standalone_SAE" --sae2_name "Joint_SAE" --k1 1024 --k2 1024 --input_dim 3200 --topk_dims 250 --output_dir sparse_orthogonality_results --max_pairs 1000000
"""
SAE Sparse Orthogonality Analysis Tool (part16b_orthogonality_analysis.py)

Compares feature orthogonality between two SAE models using sparsity-aware methods:
1. Top-K Union Cosine Similarity (focuses on most important dimensions)
2. Top-K Intersection Cosine Similarity (conservative interference detection)
3. Top-K Jaccard Similarity (overlap of important dimensions)

Usage:
python part16b_orthogonality_analysis.py --sae1_path sae_topk_model.pt --sae2_path joint_sae_nfm_model.pt --k1 1024 --k2 1024 --topk_dims 250
"""

# ============================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================

# Key Sparse Metrics Summary:
# Metric                         SAE1            SAE2            Winner          P-value
# -------------------------------------------------------------------------------------
# Mean |union cosine sim|        0.030875        0.032879        Standalone_SAE  9.8533e-01
# Mean |intersection cos sim|    0.183704        0.183814        Standalone_SAE  1.7674e-05
# Mean Jaccard similarity        0.041891        0.041807        Joint_SAE       5.0916e-11

# Interpretation Guide:
# • Union Cosine: Lower values = more orthogonal on important dimensions
# • Intersection Cosine: Lower values = more orthogonal on shared important dims
# • Jaccard: Lower values = less overlap in important dimension sets
# • Top-250 analysis focuses on most important dimensions per feature

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import warnings
import random
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

def compute_topk_indices(decoder_weights, k=250, threshold=1e-6):
    """
    Compute top-k most important dimensions for each feature
    
    Args:
        decoder_weights: [input_dim, num_features] tensor
        k: Number of top dimensions to consider
        threshold: Filter features below this threshold
    
    Returns:
        topk_indices: Dict mapping feature_idx -> tensor of top-k dimension indices
        valid_features: Boolean mask of features with sufficient norm
    """
    print(f"Computing top-{k} dimensions for each feature...")
    
    num_features = decoder_weights.shape[1]
    input_dim = decoder_weights.shape[0]
    
    # Filter out near-zero features
    feature_norms = torch.norm(decoder_weights, dim=0)
    valid_features = feature_norms > threshold
    n_valid = valid_features.sum().item()
    print(f"Using {n_valid}/{num_features} features (filtered {num_features-n_valid} with norm < {threshold})")
    
    if n_valid < 2:
        print("ERROR: Not enough valid features for analysis")
        return None, None
    
    # Adjust k if necessary
    actual_k = min(k, input_dim)
    if actual_k < k:
        print(f"Reducing top-k from {k} to {actual_k} (limited by input dimension)")
    
    topk_indices = {}
    valid_feature_indices = torch.where(valid_features)[0]
    
    for i, feature_idx in enumerate(tqdm(valid_feature_indices, desc="Computing top-k dimensions")):
        feature_vec = decoder_weights[:, feature_idx]  # [input_dim]
        _, top_indices = torch.topk(torch.abs(feature_vec), k=actual_k)
        topk_indices[feature_idx.item()] = top_indices
    
    return topk_indices, valid_features

def compute_topk_union_cosine(decoder_weights, topk_indices, max_pairs=1_000_000):
    """
    Compute cosine similarity using union of top-k dimensions for each pair
    
    Args:
        decoder_weights: [input_dim, num_features] tensor
        topk_indices: Dict mapping feature_idx -> top-k dimension indices
        max_pairs: Maximum number of pairs to process for efficiency
    
    Returns:
        similarities: List of similarity values
        pair_info: List of (feature_i, feature_j, union_size) tuples
    """
    print("Computing Top-K Union Cosine Similarities...")
    
    feature_list = list(topk_indices.keys())
    n_features = len(feature_list)
    similarities = []
    pair_info = []
    
    # Calculate total possible pairs
    total_possible_pairs = n_features * (n_features - 1) // 2
    actual_pairs = min(max_pairs, total_possible_pairs)
    
    print(f"Processing {actual_pairs:,} pairs out of {total_possible_pairs:,} possible pairs")
    
    # Generate random pairs efficiently - NO FULL LIST
    if actual_pairs < total_possible_pairs:
        print("Using random sampling of feature pairs...")
        
        processed_pairs = set()
        sampled_pairs = []
        
        while len(sampled_pairs) < actual_pairs:
            i = random.randint(0, n_features - 2)
            j = random.randint(i + 1, n_features - 1)
            pair = (i, j)
            
            if pair not in processed_pairs:
                processed_pairs.add(pair)
                sampled_pairs.append(pair)
    else:
        # Use all pairs if max_pairs >= total pairs
        sampled_pairs = [(i, j) for i in range(n_features) for j in range(i + 1, n_features)]
    
    with tqdm(total=actual_pairs, desc="Computing union similarities") as pbar:
        for i, j in sampled_pairs:
            feature_i = feature_list[i]
            feature_j = feature_list[j]
            
            # Create union of top-k dimensions
            union_indices = torch.cat([topk_indices[feature_i], topk_indices[feature_j]]).unique()
            union_size = len(union_indices)
            
            if union_size > 0:
                # Extract vectors for union dimensions
                vec1_subset = decoder_weights[union_indices, feature_i]
                vec2_subset = decoder_weights[union_indices, feature_j]
                
                # Compute cosine similarity
                cos_sim = F.cosine_similarity(vec1_subset.unsqueeze(0), vec2_subset.unsqueeze(0), dim=1).item()
                similarities.append(cos_sim)
                pair_info.append((feature_i, feature_j, union_size))
            
            pbar.update(1)
    
    return similarities, pair_info

def compute_topk_intersection_cosine(decoder_weights, topk_indices, max_pairs=1_000_000):
    """
    Compute cosine similarity using intersection of top-k dimensions for each pair
    
    Args:
        decoder_weights: [input_dim, num_features] tensor
        topk_indices: Dict mapping feature_idx -> top-k dimension indices
        max_pairs: Maximum number of pairs to process for efficiency
    
    Returns:
        similarities: List of similarity values
        pair_info: List of (feature_i, feature_j, intersection_size) tuples
    """
    print("Computing Top-K Intersection Cosine Similarities...")
    
    feature_list = list(topk_indices.keys())
    n_features = len(feature_list)
    similarities = []
    pair_info = []
    
    total_possible_pairs = n_features * (n_features - 1) // 2
    actual_pairs = min(max_pairs, total_possible_pairs)
    
    print(f"Processing {actual_pairs:,} pairs out of {total_possible_pairs:,} possible pairs")
    
    # Generate random pairs efficiently - NO FULL LIST
    if actual_pairs < total_possible_pairs:
        print("Using random sampling of feature pairs...")
        
        processed_pairs = set()
        sampled_pairs = []
        
        while len(sampled_pairs) < actual_pairs:
            i = random.randint(0, n_features - 2)
            j = random.randint(i + 1, n_features - 1)
            pair = (i, j)
            
            if pair not in processed_pairs:
                processed_pairs.add(pair)
                sampled_pairs.append(pair)
    else:
        # Use all pairs if max_pairs >= total pairs
        sampled_pairs = [(i, j) for i in range(n_features) for j in range(i + 1, n_features)]
    
    with tqdm(total=actual_pairs, desc="Computing intersection similarities") as pbar:
        for i, j in sampled_pairs:
            feature_i = feature_list[i]
            feature_j = feature_list[j]
            
            # Create intersection of top-k dimensions
            set_i = set(topk_indices[feature_i].tolist())
            set_j = set(topk_indices[feature_j].tolist())
            intersection_indices = torch.tensor(list(set_i & set_j))
            intersection_size = len(intersection_indices)
            
            if intersection_size > 0:
                # Extract vectors for intersection dimensions
                vec1_subset = decoder_weights[intersection_indices, feature_i]
                vec2_subset = decoder_weights[intersection_indices, feature_j]
                
                # Compute cosine similarity
                cos_sim = F.cosine_similarity(vec1_subset.unsqueeze(0), vec2_subset.unsqueeze(0), dim=1).item()
                similarities.append(cos_sim)
                pair_info.append((feature_i, feature_j, intersection_size))
            # If no intersection, skip the pair entirely
            
            pbar.update(1)
    
    return similarities, pair_info

def compute_topk_jaccard_similarity(topk_indices, max_pairs=1_000_000):
    """
    Compute Jaccard similarity on top-k dimensions for each pair
    
    Args:
        topk_indices: Dict mapping feature_idx -> top-k dimension indices
        max_pairs: Maximum number of pairs to process for efficiency
    
    Returns:
        similarities: List of Jaccard similarity values
        pair_info: List of (feature_i, feature_j, intersection_size, union_size) tuples
    """
    print("Computing Top-K Jaccard Similarities...")
    
    feature_list = list(topk_indices.keys())
    n_features = len(feature_list)
    similarities = []
    pair_info = []
    
    total_possible_pairs = n_features * (n_features - 1) // 2
    actual_pairs = min(max_pairs, total_possible_pairs)
    
    print(f"Processing {actual_pairs:,} pairs out of {total_possible_pairs:,} possible pairs")
    
    # Generate random pairs efficiently - NO FULL LIST
    if actual_pairs < total_possible_pairs:
        print("Using random sampling of feature pairs...")
        
        processed_pairs = set()
        sampled_pairs = []
        
        while len(sampled_pairs) < actual_pairs:
            i = random.randint(0, n_features - 2)
            j = random.randint(i + 1, n_features - 1)
            pair = (i, j)
            
            if pair not in processed_pairs:
                processed_pairs.add(pair)
                sampled_pairs.append(pair)
    else:
        # Use all pairs if max_pairs >= total pairs
        sampled_pairs = [(i, j) for i in range(n_features) for j in range(i + 1, n_features)]
    
    with tqdm(total=actual_pairs, desc="Computing Jaccard similarities") as pbar:
        for i, j in sampled_pairs:
            feature_i = feature_list[i]
            feature_j = feature_list[j]
            
            # Convert to sets for intersection/union operations
            set_i = set(topk_indices[feature_i].tolist())
            set_j = set(topk_indices[feature_j].tolist())
            
            intersection = set_i & set_j
            union = set_i | set_j
            
            intersection_size = len(intersection)
            union_size = len(union)
            
            # Jaccard similarity = |intersection| / |union|
            jaccard_sim = intersection_size / union_size if union_size > 0 else 0.0
            
            similarities.append(jaccard_sim)
            pair_info.append((feature_i, feature_j, intersection_size, union_size))
            
            pbar.update(1)
    
    return similarities, pair_info

def plot_sparse_analysis_results(results_dict, output_dir):
    """Create comprehensive plots for sparse orthogonality analysis"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Union Cosine Similarity Histograms
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    for i, (model_name, data) in enumerate(results_dict.items()):
        similarities = data['union_cosine']['similarities']
        if similarities:
            axes[i].hist(similarities, bins=50, alpha=0.7, density=True, label=model_name)
            mean_sim = np.mean(similarities)
            axes[i].axvline(mean_sim, color='red', linestyle='--', 
                          label=f'Mean: {mean_sim:.4f}')
            axes[i].set_xlabel('Cosine Similarity (Union)')
            axes[i].set_ylabel('Density')
            axes[i].set_title(f'{model_name} - Top-K Union Cosine Similarities')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'topk_union_cosine_histograms.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Intersection Cosine Similarity Histograms
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    for i, (model_name, data) in enumerate(results_dict.items()):
        similarities = data['intersection_cosine']['similarities']
        
        if similarities:
            axes[i].hist(similarities, bins=50, alpha=0.7, density=True, label=model_name)
            mean_sim = np.mean(similarities)
            axes[i].axvline(mean_sim, color='red', linestyle='--', 
                          label=f'Mean: {mean_sim:.4f}')
            axes[i].set_xlabel('Cosine Similarity (Intersection)')
            axes[i].set_ylabel('Density')
            axes[i].set_title(f'{model_name} - Top-K Intersection Cosine')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'topk_intersection_cosine_histograms.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Jaccard Similarity Histograms
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    for i, (model_name, data) in enumerate(results_dict.items()):
        similarities = data['jaccard']['similarities']
        if similarities:
            axes[i].hist(similarities, bins=50, alpha=0.7, density=True, label=model_name)
            mean_sim = np.mean(similarities)
            axes[i].axvline(mean_sim, color='red', linestyle='--', 
                          label=f'Mean: {mean_sim:.4f}')
            axes[i].set_xlabel('Jaccard Similarity')
            axes[i].set_ylabel('Density')
            axes[i].set_title(f'{model_name} - Top-K Jaccard Similarities')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'topk_jaccard_histograms.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Comparison Summary Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    model_names = list(results_dict.keys())
    metrics = ['Union Cosine', 'Intersection Cosine', 'Jaccard']
    metric_keys = ['union_cosine', 'intersection_cosine', 'jaccard']
    
    for i, (metric, key) in enumerate(zip(metrics, metric_keys)):
        means = []
        stds = []
        
        for model_name in model_names:
            similarities = results_dict[model_name][key]['similarities']
            if similarities:
                means.append(np.mean(similarities))
                stds.append(np.std(similarities))
            else:
                means.append(0)
                stds.append(0)
        
        x_pos = np.arange(len(model_names))
        bars = axes[i].bar(x_pos, means, yerr=stds, alpha=0.7, capsize=5)
        axes[i].set_xlabel('Model')
        axes[i].set_ylabel('Mean Similarity')
        axes[i].set_title(f'{metric} - Mean ± Std')
        axes[i].set_xticks(x_pos)
        axes[i].set_xticklabels(model_names, rotation=45)
        axes[i].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for j, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                        f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sparse_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"All sparse analysis plots saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='SAE Sparse Orthogonality Analysis')
    parser.add_argument('--sae1_path', type=str, required=True, help='Path to first SAE model')
    parser.add_argument('--sae2_path', type=str, required=True, help='Path to second SAE model')
    parser.add_argument('--sae1_name', type=str, default='SAE1', help='Name for first SAE')
    parser.add_argument('--sae2_name', type=str, default='SAE2', help='Name for second SAE')
    parser.add_argument('--k1', type=int, default=1024, help='TopK parameter for first SAE')
    parser.add_argument('--k2', type=int, default=1024, help='TopK parameter for second SAE')
    parser.add_argument('--input_dim', type=int, default=4096, help='Input dimension')
    parser.add_argument('--num_features', type=int, default=50000, help='Number of SAE features')
    parser.add_argument('--threshold', type=float, default=1e-6, help='Threshold for filtering low values')
    parser.add_argument('--topk_dims', type=int, default=250, help='Number of top dimensions to consider for each feature')
    parser.add_argument('--output_dir', type=str, default='sparse_orthogonality_analysis', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for large operations')
    parser.add_argument('--max_pairs', type=int, default=1_000_000, help='Maximum number of feature pairs to analyze for efficiency')
    
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
    print(f"Using top-{args.topk_dims} dimensions for sparsity-aware analysis")
    
    results = {}
    
    # Analyze both models
    for model_name, decoder_weights in [(args.sae1_name, decoder1), (args.sae2_name, decoder2)]:
        print(f"\n{'='*60}")
        print(f"ANALYZING {model_name}")
        print(f"{'='*60}")
        
        results[model_name] = {}
        
        # Compute top-k indices for this model
        topk_indices, valid_features = compute_topk_indices(
            decoder_weights, args.topk_dims, args.threshold
        )
        
        if topk_indices is None:
            print(f"Skipping {model_name} due to insufficient valid features")
            continue
        
        n_valid_features = len(topk_indices)
        total_pairs = n_valid_features * (n_valid_features - 1) // 2
        print(f"Will analyze {total_pairs} feature pairs")
        
        # 1. Top-K Union Cosine Similarity
        print(f"\n1. TOP-K UNION COSINE SIMILARITY")
        print("-" * 40)
        union_similarities, union_info = compute_topk_union_cosine(
            decoder_weights, topk_indices, args.max_pairs
        )
        
        union_stats = {
            'mean': np.mean(union_similarities) if union_similarities else 0,
            'std': np.std(union_similarities) if union_similarities else 0,
            'mean_abs': np.mean(np.abs(union_similarities)) if union_similarities else 0,
            'median': np.median(union_similarities) if union_similarities else 0,
            'min': np.min(union_similarities) if union_similarities else 0,
            'max': np.max(union_similarities) if union_similarities else 0
        }
        
        print(f"Mean union cosine similarity: {union_stats['mean']:.6f}")
        print(f"Mean absolute union cosine similarity: {union_stats['mean_abs']:.6f}")
        print(f"Std: {union_stats['std']:.6f}")
        
        results[model_name]['union_cosine'] = {
            'similarities': union_similarities,
            'stats': union_stats,
            'pair_info': union_info
        }
        
        # 2. Top-K Intersection Cosine Similarity
        print(f"\n2. TOP-K INTERSECTION COSINE SIMILARITY")
        print("-" * 40)
        intersection_similarities, intersection_info = compute_topk_intersection_cosine(
            decoder_weights, topk_indices, args.max_pairs
        )
        
        intersection_stats = {
            'mean': np.mean(intersection_similarities) if intersection_similarities else 0,
            'std': np.std(intersection_similarities) if intersection_similarities else 0,
            'mean_abs': np.mean(np.abs(intersection_similarities)) if intersection_similarities else 0,
            'median': np.median(intersection_similarities) if intersection_similarities else 0,
            'min': np.min(intersection_similarities) if intersection_similarities else 0,
            'max': np.max(intersection_similarities) if intersection_similarities else 0
        }
        
        print(f"Mean intersection cosine similarity: {intersection_stats['mean']:.6f}")
        print(f"Mean absolute intersection cosine similarity: {intersection_stats['mean_abs']:.6f}")
        print(f"Std: {intersection_stats['std']:.6f}")
        print(f"Number of pairs with intersection: {len(intersection_similarities)}")
        
        results[model_name]['intersection_cosine'] = {
            'similarities': intersection_similarities,
            'stats': intersection_stats,
            'pair_info': intersection_info
        }
        
        # 3. Top-K Jaccard Similarity
        print(f"\n3. TOP-K JACCARD SIMILARITY")
        print("-" * 40)
        jaccard_similarities, jaccard_info = compute_topk_jaccard_similarity(topk_indices, args.max_pairs)
        
        jaccard_stats = {
            'mean': np.mean(jaccard_similarities) if jaccard_similarities else 0,
            'std': np.std(jaccard_similarities) if jaccard_similarities else 0,
            'median': np.median(jaccard_similarities) if jaccard_similarities else 0,
            'min': np.min(jaccard_similarities) if jaccard_similarities else 0,
            'max': np.max(jaccard_similarities) if jaccard_similarities else 0
        }
        
        print(f"Mean Jaccard similarity: {jaccard_stats['mean']:.6f}")
        print(f"Median Jaccard similarity: {jaccard_stats['median']:.6f}")
        print(f"Std: {jaccard_stats['std']:.6f}")
        
        results[model_name]['jaccard'] = {
            'similarities': jaccard_similarities,
            'stats': jaccard_stats,
            'pair_info': jaccard_info
        }
    
    # Comparative Analysis
    print(f"\n{'='*60}")
    print("COMPARATIVE ANALYSIS")
    print("="*60)
    
    if len(results) == 2:
        model_names = list(results.keys())
        model1, model2 = model_names
        
        # Compare Union Cosine
        if (results[model1]['union_cosine']['similarities'] and 
            results[model2]['union_cosine']['similarities']):
            
            union1_data = results[model1]['union_cosine']['similarities']
            union2_data = results[model2]['union_cosine']['similarities']
            union1 = results[model1]['union_cosine']['stats']['mean_abs']
            union2 = results[model2]['union_cosine']['stats']['mean_abs']
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(union1_data, union2_data)
            
            print(f"\nUnion Cosine Similarity Comparison:")
            print(f"{model1} mean abs similarity: {union1:.6f}")
            print(f"{model2} mean abs similarity: {union2:.6f}")
            print(f"T-test: t={t_stat:.4f}, p={p_value:.4e}")
            
            if union1 < union2:
                print(f"🏆 {model1} has better orthogonality (lower union similarity)")
            else:
                print(f"🏆 {model2} has better orthogonality (lower union similarity)")
        
        # Compare Intersection Results
        if (results[model1]['intersection_cosine']['similarities'] and 
            results[model2]['intersection_cosine']['similarities']):
            
            inter1_data = results[model1]['intersection_cosine']['similarities']
            inter2_data = results[model2]['intersection_cosine']['similarities']
            inter1 = results[model1]['intersection_cosine']['stats']['mean_abs']
            inter2 = results[model2]['intersection_cosine']['stats']['mean_abs']
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(inter1_data, inter2_data)
            
            print(f"\nIntersection Cosine Similarity Comparison:")
            print(f"{model1} mean abs similarity: {inter1:.6f}")
            print(f"{model2} mean abs similarity: {inter2:.6f}")
            print(f"T-test: t={t_stat:.4f}, p={p_value:.4e}")
            
            if inter1 < inter2:
                print(f"🏆 {model1} has better orthogonality (lower intersection similarity)")
            else:
                print(f"🏆 {model2} has better orthogonality (lower intersection similarity)")
        
        # Compare Jaccard
        if (results[model1]['jaccard']['similarities'] and 
            results[model2]['jaccard']['similarities']):
            
            jaccard1_data = results[model1]['jaccard']['similarities']
            jaccard2_data = results[model2]['jaccard']['similarities']
            jaccard1 = results[model1]['jaccard']['stats']['mean']
            jaccard2 = results[model2]['jaccard']['stats']['mean']
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(jaccard1_data, jaccard2_data)
            
            print(f"\nJaccard Similarity Comparison:")
            print(f"{model1} mean Jaccard similarity: {jaccard1:.6f}")
            print(f"{model2} mean Jaccard similarity: {jaccard2:.6f}")
            print(f"T-test: t={t_stat:.4f}, p={p_value:.4e}")
            
            if jaccard1 < jaccard2:
                print(f"🏆 {model1} has better orthogonality (lower Jaccard overlap)")
            else:
                print(f"🏆 {model2} has better orthogonality (lower Jaccard overlap)")
    
    # Create plots
    plot_sparse_analysis_results(results, args.output_dir)
    
    # Save detailed results
    output_path = Path(args.output_dir) / 'sparse_orthogonality_results.json'
    
    # Convert to JSON-serializable format
    json_results = {}
    for model_name, model_results in results.items():
        json_results[model_name] = {}
        
        for analysis_type in ['union_cosine', 'intersection_cosine', 'jaccard']:
            if analysis_type in model_results:
                json_results[model_name][analysis_type] = {
                    'stats': model_results[analysis_type]['stats']
                    # Skip storing full similarity arrays and pair info for space
                }
                
                # Add specific metrics - none needed for intersection now
                pass
    
    # Add t-test results if we have two models
    if len(results) == 2:
        model_names = list(results.keys())
        model1, model2 = model_names
        
        json_results['statistical_tests'] = {}
        
        # Union t-test
        if (results[model1]['union_cosine']['similarities'] and 
            results[model2]['union_cosine']['similarities']):
            t_stat, p_value = stats.ttest_ind(
                results[model1]['union_cosine']['similarities'],
                results[model2]['union_cosine']['similarities']
            )
            json_results['statistical_tests']['union_cosine'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value)
            }
        
        # Intersection t-test
        if (results[model1]['intersection_cosine']['similarities'] and 
            results[model2]['intersection_cosine']['similarities']):
            t_stat, p_value = stats.ttest_ind(
                results[model1]['intersection_cosine']['similarities'],
                results[model2]['intersection_cosine']['similarities']
            )
            json_results['statistical_tests']['intersection_cosine'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value)
            }
        
        # Jaccard t-test
        if (results[model1]['jaccard']['similarities'] and 
            results[model2]['jaccard']['similarities']):
            t_stat, p_value = stats.ttest_ind(
                results[model1]['jaccard']['similarities'],
                results[model2]['jaccard']['similarities']
            )
            json_results['statistical_tests']['jaccard'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value)
            }
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_path}")
    
    # Summary recommendations
    print(f"\n{'='*60}")
    print("SUMMARY & RECOMMENDATIONS")
    print("="*60)
    
    if len(results) == 2:
        print(f"\nKey Sparse Metrics Summary:")
        print(f"{'Metric':<30} {'SAE1':<15} {'SAE2':<15} {'Winner':<15} {'P-value'}")
        print("-" * 85)
        
        # Union cosine (lower is better)
        if (results[model1]['union_cosine']['similarities'] and 
            results[model2]['union_cosine']['similarities']):
            union1_val = results[model1]['union_cosine']['stats']['mean_abs']
            union2_val = results[model2]['union_cosine']['stats']['mean_abs']
            winner = model1 if union1_val < union2_val else model2
            
            t_stat, p_value = stats.ttest_ind(
                results[model1]['union_cosine']['similarities'],
                results[model2]['union_cosine']['similarities']
            )
            
            print(f"{'Mean |union cosine sim|':<30} {union1_val:<15.6f} {union2_val:<15.6f} {winner:<15} {p_value:.4e}")
        
        # Intersection cosine (lower is better)
        if (results[model1]['intersection_cosine']['similarities'] and 
            results[model2]['intersection_cosine']['similarities']):
            inter1_val = results[model1]['intersection_cosine']['stats']['mean_abs']
            inter2_val = results[model2]['intersection_cosine']['stats']['mean_abs']
            winner = model1 if inter1_val < inter2_val else model2
            
            t_stat, p_value = stats.ttest_ind(
                results[model1]['intersection_cosine']['similarities'],
                results[model2]['intersection_cosine']['similarities']
            )
            
            print(f"{'Mean |intersection cos sim|':<30} {inter1_val:<15.6f} {inter2_val:<15.6f} {winner:<15} {p_value:.4e}")
        
        # Jaccard similarity (lower is better)
        if (results[model1]['jaccard']['similarities'] and 
            results[model2]['jaccard']['similarities']):
            jaccard1_val = results[model1]['jaccard']['stats']['mean']
            jaccard2_val = results[model2]['jaccard']['stats']['mean']
            winner = model1 if jaccard1_val < jaccard2_val else model2
            
            t_stat, p_value = stats.ttest_ind(
                results[model1]['jaccard']['similarities'],
                results[model2]['jaccard']['similarities']
            )
            
            print(f"{'Mean Jaccard similarity':<30} {jaccard1_val:<15.6f} {jaccard2_val:<15.6f} {winner:<15} {p_value:.4e}")
    
    print(f"\nInterpretation Guide:")
    print(f"• Union Cosine: Lower values = more orthogonal on important dimensions")
    print(f"• Intersection Cosine: Lower values = more orthogonal on shared important dims")
    print(f"• Jaccard: Lower values = less overlap in important dimension sets")
    print(f"• Top-{args.topk_dims} analysis focuses on most important dimensions per feature")
    
    print(f"\nNext Steps:")
    print(f"1. Examine the generated plots in {args.output_dir}/")
    print(f"2. Consider different topk_dims values (try 100, 500) for sensitivity analysis")
    print(f"3. Intersection analysis shows truly orthogonal feature pairs")
    print(f"4. Union analysis shows overall similarity when features do interact")

if __name__ == "__main__":
    main()