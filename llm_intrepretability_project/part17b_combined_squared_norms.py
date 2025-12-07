"""
part17b_combined_squared_norms.py

Combined Squared Norms Distribution Plot

Combines the squared norms distributions from:
- Standalone SAE (from part16_orthogonality_analysis.py)
- Joint SAE+NFM (from part17_bimodal_gram_inspection.py)

Usage:
python part17b_combined_squared_norms.py --standalone_sae_path checkpoints_topk/best_model.pt --joint_model_path checkpoints_joint/best_joint_sae_nfm_model.pt --input_dim 3200 --num_features 50000 --k1 1024 --k2 1024 --nfm_embedding_dim 300 --output_dir combined_squared_norms
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from tqdm import tqdm
import warnings
import pickle
import tempfile
import os
warnings.filterwarnings('ignore')

# ============================================================================
# EXACT CODE FROM part16_orthogonality_analysis.py
# ============================================================================

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

# ============================================================================
# EXACT CODE FROM part17_bimodal_gram_inspection.py
# ============================================================================

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

# ============================================================================
# COMBINED PLOTTING FUNCTION
# ============================================================================

def create_combined_plot(standalone_diagonal_elements, joint_squared_norms, output_dir, standalone_name="Standalone_SAE"):
    """Create combined 1x1 figure with overlapping histograms"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    plt.style.use('default')
    
    # Set font sizes for publication
    # plt.rcParams.update({'font.size': 14})
    # plt.rcParams.update({'axes.titlesize': 16})
    # plt.rcParams.update({'axes.labelsize': 14})
    # plt.rcParams.update({'xtick.labelsize': 12})
    # plt.rcParams.update({'ytick.labelsize': 12})
    # plt.rcParams.update({'legend.fontsize': 12})
    plt.rcParams.update({'font.size': 29})
    plt.rcParams.update({'axes.titlesize': 32})
    plt.rcParams.update({'axes.labelsize': 29})
    plt.rcParams.update({'xtick.labelsize': 27})
    plt.rcParams.update({'ytick.labelsize': 27})
    plt.rcParams.update({'legend.fontsize': 32})    
    # Create single 1x1 figure
    # fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    
    # Plot Standalone SAE histogram (matching bin size to bimodal distribution)
    # Diagonal elements (squared norms)
    ax.hist(standalone_diagonal_elements, bins=100, alpha=0.7, density=True, label=f'{standalone_name} - Squared Norms', color='blue', edgecolor='black')
    ax.axvline(standalone_diagonal_elements.mean(), color='red', linestyle='--', label=f'{standalone_name} Mean = {np.mean(standalone_diagonal_elements):.3f}')
    
    # Plot Joint SAE histogram (from part17 exact code)
    # Distribution of squared norms (show bimodal nature)
    ax.hist(joint_squared_norms, bins=100, alpha=0.7, density=True, color='skyblue', edgecolor='black', label='Joint Architecture - Squared Norms')
    ax.axvline(0.2, color='red', linestyle='--', linewidth=2, label='Split < 0.2')
    ax.axvline(0.2, color='orange', linestyle='--', linewidth=2, label='Split > 0.2')
    ax.axvline(np.mean(joint_squared_norms), color='green', linestyle='-', linewidth=2, label=f'Joint Mean = {np.mean(joint_squared_norms):.3f}')
    
    ax.set_xlabel('Squared Norm (Gram Matrix Diagonal)', fontsize=29)
    ax.set_ylabel('Density', fontsize=29)
    ax.set_title('Distribution of Feature Squared Norms: Standalone vs Joint SAE', fontsize=32, fontweight='bold')
    ax.legend(fontsize=32)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_squared_norms_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined plot saved to {output_dir / 'combined_squared_norms_distribution.png'}")

def main():
    parser = argparse.ArgumentParser(description='Combined Squared Norms Distribution')
    parser.add_argument('--standalone_sae_path', type=str, required=True, help='Path to standalone SAE model')
    parser.add_argument('--joint_model_path', type=str, required=True, help='Path to joint SAE+NFM model')
    parser.add_argument('--standalone_name', type=str, default='Standalone_SAE', help='Name for standalone SAE')
    parser.add_argument('--k1', type=int, default=1024, help='TopK parameter for standalone SAE')
    parser.add_argument('--k2', type=int, default=1024, help='TopK parameter for joint SAE')
    parser.add_argument('--input_dim', type=int, default=4096, help='Input dimension')
    parser.add_argument('--num_features', type=int, default=50000, help='Number of SAE features')
    parser.add_argument('--nfm_embedding_dim', type=int, default=300, help='NFM embedding dimension')
    parser.add_argument('--threshold', type=float, default=1e-6, help='Threshold for filtering low values')
    parser.add_argument('--output_dir', type=str, default='combined_squared_norms', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for large matrix operations')
    
    args = parser.parse_args()
    
    # Create temp file for saving standalone squared norms
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
    temp_path = temp_file.name
    temp_file.close()
    
    try:
        # STEP 1: Load Standalone SAE and compute squared norms (from part16)
        print("="*60)
        print("STEP 1: PROCESSING STANDALONE SAE")
        print("="*60)
        
        sae1 = load_sae_model(args.standalone_sae_path, args.standalone_name, args.input_dim, args.num_features, args.k1)
        decoder1 = sae1.decoder.weight.data
        
        print(f"Decoder shape: {decoder1.shape}")
        
        diag_stats, off_diag_stats, gram_results = compute_gram_matrix_analysis(
            decoder1, args.batch_size, args.threshold
        )
        
        if gram_results is None:
            print("ERROR: Failed to compute Gram matrix for standalone SAE")
            return
        
        _, diagonal_elements, _ = gram_results
        standalone_diagonal_elements = diagonal_elements.numpy()
        
        # Save to temp file
        with open(temp_path, 'wb') as f:
            pickle.dump(standalone_diagonal_elements, f)
        
        print(f"Saved standalone squared norms to temp file")
        
        # Clear memory
        del sae1, decoder1, diag_stats, off_diag_stats, gram_results, diagonal_elements
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # STEP 2: Load Joint model and compute squared norms (from part17)
        print("\n" + "="*60)
        print("STEP 2: PROCESSING JOINT SAE+NFM")
        print("="*60)
        
        joint_model = load_joint_model(
            args.joint_model_path, args.input_dim, args.num_features, 
            args.k2, args.nfm_embedding_dim
        )
        
        decoder2 = joint_model.primary_sae.decoder.weight.data
        print(f"Decoder weights shape: {decoder2.shape}")
        
        joint_squared_norms, valid_indices = compute_gram_matrix_diagonal(decoder2, args.threshold)
        
        # Load saved standalone squared norms
        with open(temp_path, 'rb') as f:
            standalone_diagonal_elements = pickle.load(f)
        
        print(f"Loaded standalone squared norms from temp file")
        
        # STEP 3: Create combined plot
        print("\n" + "="*60)
        print("STEP 3: CREATING COMBINED PLOT")
        print("="*60)
        
        create_combined_plot(standalone_diagonal_elements, joint_squared_norms, args.output_dir, args.standalone_name)
        
        print(f"\nAll done! Combined plot saved to {args.output_dir}")
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    main()

