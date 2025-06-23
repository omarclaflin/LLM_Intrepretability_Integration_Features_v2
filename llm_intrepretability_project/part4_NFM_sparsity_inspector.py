import torch
import numpy as np
import argparse
from pathlib import Path

def analyze_weight_sparsity(weight_matrix, matrix_name="weight_matrix", analyze_dimension="rows"):
    """
    Analyze sparsity patterns in a weight matrix.
    
    Args:
        weight_matrix: torch.Tensor of shape [output_dim, input_dim]
        matrix_name: str, name of the matrix for reporting
        analyze_dimension: str, "rows" (output dims) or "columns" (input dims)
        
    Returns:
        dict with analysis results
    """
    print(f"\n{'='*60}")
    print(f"SPARSITY ANALYSIS: {matrix_name}")
    print(f"Analyzing {analyze_dimension}")
    print(f"{'='*60}")
    
    # Ensure we're working with the right orientation
    if weight_matrix.dim() != 2:
        raise ValueError(f"Expected 2D matrix, got {weight_matrix.dim()}D")
    
    num_rows, num_cols = weight_matrix.shape
    print(f"Matrix shape: [{num_rows}, {num_cols}]")
    
    # Convert to numpy for easier analysis
    W = weight_matrix.detach().cpu().numpy()
    
    if analyze_dimension == "columns":
        # Transpose to analyze columns as rows
        W = W.T
        num_analyzed = num_cols
        other_dim = num_rows
        print(f"Analyzing {num_cols} columns (input dimensions)")
        print(f"Each column has {num_rows} elements (output connections)")
    else:
        num_analyzed = num_rows
        other_dim = num_cols
        print(f"Analyzing {num_rows} rows (output dimensions)")
        print(f"Each row has {num_cols} elements (input connections)")
    
    # Compute sparsity metrics for each dimension
    dim_stats = []
    
    for i in range(num_analyzed):
        vec = W[i, :]
        
        # L0 "norm" (count of non-zero elements)
        l0_count = np.count_nonzero(np.abs(vec) > 1e-8)
        l0_sparsity = 1.0 - (l0_count / other_dim)  # Fraction of zeros
        
        # L1 norm
        l1_norm = np.sum(np.abs(vec))
        
        # L2 norm
        l2_norm = np.sqrt(np.sum(vec ** 2))
        
        # Max absolute value
        max_abs = np.max(np.abs(vec))
        
        # Standard deviation (measure of concentration)
        std_dev = np.std(vec)
        
        # Gini coefficient (inequality measure - closer to 1 means more sparse)
        # Fixed calculation to handle edge cases
        sorted_abs = np.sort(np.abs(vec))
        n = len(sorted_abs)
        total_sum = np.sum(sorted_abs)
        if total_sum > 1e-12:
            cumsum = np.cumsum(sorted_abs)
            gini = (2 * np.sum((np.arange(1, n+1) * sorted_abs))) / (n * total_sum) - (n+1)/n
            gini = np.clip(gini, 0, 1)  # Ensure valid range
        else:
            gini = 0.0
        
        # Effective rank (how many dimensions significantly contribute)
        vec_abs = np.abs(vec)
        vec_sum = np.sum(vec_abs)
        if vec_sum > 1e-12:
            vec_prob = vec_abs / vec_sum
            # Remove zeros to avoid log(0)
            vec_prob = vec_prob[vec_prob > 1e-12]
            if len(vec_prob) > 0:
                # Shannon entropy-based effective rank with numerical stability
                entropy = -np.sum(vec_prob * np.log(vec_prob + 1e-15))
                effective_rank = np.exp(entropy)
                # Cap effective rank to reasonable bounds
                effective_rank = min(effective_rank, other_dim)
            else:
                effective_rank = 1.0
        else:
            effective_rank = 1.0
        
        dim_stats.append({
            'dim_idx': i,
            'l0_sparsity': l0_sparsity,
            'l0_count': l0_count,
            'l1_norm': l1_norm,
            'l2_norm': l2_norm,
            'max_abs': max_abs,
            'std_dev': std_dev,
            'gini': gini,
            'effective_rank': effective_rank
        })
    
    # Convert to arrays for analysis with numerical stability
    l0_sparsities = np.array([s['l0_sparsity'] for s in dim_stats])
    l1_norms = np.array([s['l1_norm'] for s in dim_stats])
    l2_norms = np.array([s['l2_norm'] for s in dim_stats])
    gini_coeffs = np.array([s['gini'] for s in dim_stats])
    effective_ranks = np.array([s['effective_rank'] for s in dim_stats])
    
    # Find sparsest dimensions by different metrics
    sparsest_l0_idx = np.argmax(l0_sparsities)
    sparsest_gini_idx = np.argmax(gini_coeffs)
    
    # Handle effective ranks safely
    valid_rank_mask = np.isfinite(effective_ranks) & (effective_ranks > 0)
    if np.any(valid_rank_mask):
        valid_effective_ranks = effective_ranks[valid_rank_mask]
        valid_indices = np.where(valid_rank_mask)[0]
        lowest_effective_rank_idx = valid_indices[np.argmin(valid_effective_ranks)]
    else:
        lowest_effective_rank_idx = 0
    
    dim_type = "Column" if analyze_dimension == "columns" else "Row"
    print(f"\n--- SPARSEST {analyze_dimension.upper()} ---")
    print(f"Most L0-sparse {dim_type.lower()} (most zeros): {dim_type} {sparsest_l0_idx}")
    print(f"  L0 sparsity: {l0_sparsities[sparsest_l0_idx]:.4f} ({(1-l0_sparsities[sparsest_l0_idx])*100:.1f}% non-zero)")
    print(f"  Non-zero elements: {dim_stats[sparsest_l0_idx]['l0_count']}/{other_dim}")
    print(f"  L1 norm: {dim_stats[sparsest_l0_idx]['l1_norm']:.6f}")
    print(f"  L2 norm: {dim_stats[sparsest_l0_idx]['l2_norm']:.6f}")
    print(f"  Gini coefficient: {dim_stats[sparsest_l0_idx]['gini']:.4f}")
    print(f"  Effective rank: {dim_stats[sparsest_l0_idx]['effective_rank']:.2f}")
    
    print(f"\nHighest Gini coefficient {dim_type.lower()} (most unequal): {dim_type} {sparsest_gini_idx}")
    print(f"  Gini coefficient: {gini_coeffs[sparsest_gini_idx]:.4f}")
    print(f"  L0 sparsity: {l0_sparsities[sparsest_gini_idx]:.4f}")
    print(f"  Effective rank: {effective_ranks[sparsest_gini_idx]:.2f}")
    
    print(f"\nLowest effective rank {dim_type.lower()} (most concentrated): {dim_type} {lowest_effective_rank_idx}")
    print(f"  Effective rank: {effective_ranks[lowest_effective_rank_idx]:.2f}")
    print(f"  L0 sparsity: {l0_sparsities[lowest_effective_rank_idx]:.4f}")
    print(f"  Gini coefficient: {gini_coeffs[lowest_effective_rank_idx]:.4f}")
    
    # Overall statistics with numerical stability
    print(f"\n--- OVERALL STATISTICS ---")
    print(f"L0 Sparsity (fraction of zeros):")
    print(f"  Mean: {np.mean(l0_sparsities):.4f}")
    print(f"  Std:  {np.std(l0_sparsities):.4f}")
    print(f"  Min:  {np.min(l0_sparsities):.4f} (densest {dim_type.lower()})")
    print(f"  Max:  {np.max(l0_sparsities):.4f} (sparsest {dim_type.lower()})")
    print(f"  Median: {np.median(l0_sparsities):.4f}")
    
    # Safe statistics calculation
    def safe_stats(arr, name):
        if len(arr) == 0:
            print(f"\n{name}: No valid values")
            return
        
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        min_val = np.min(arr)
        max_val = np.max(arr)
        
        # Check for overflow/underflow
        if not np.isfinite(std_val):
            std_val = "overflow/underflow"
        else:
            std_val = f"{std_val:.6f}"
            
        print(f"\n{name}:")
        print(f"  Mean: {mean_val:.6f}")
        print(f"  Std:  {std_val}")
        print(f"  Min:  {min_val:.6f}")
        print(f"  Max:  {max_val:.6f}")
    
    safe_stats(l1_norms, "L1 Norms")
    safe_stats(l2_norms, "L2 Norms")
    
    print(f"\nGini Coefficients (0=equal, 1=maximally unequal):")
    print(f"  Mean: {np.mean(gini_coeffs):.4f}")
    print(f"  Std:  {np.std(gini_coeffs):.4f}")
    print(f"  Min:  {np.min(gini_coeffs):.4f}")
    print(f"  Max:  {np.max(gini_coeffs):.4f}")
    
    print(f"\nEffective Ranks (lower = more concentrated):")
    valid_ranks = effective_ranks[valid_rank_mask]
    if len(valid_ranks) > 0:
        print(f"  Mean: {np.mean(valid_ranks):.2f}")
        std_ranks = np.std(valid_ranks)
        if np.isfinite(std_ranks):
            print(f"  Std:  {std_ranks:.2f}")
        else:
            print(f"  Std:  overflow/underflow")
        print(f"  Min:  {np.min(valid_ranks):.2f}")
        print(f"  Max:  {np.max(valid_ranks):.2f}")
        print(f"  Valid values: {len(valid_ranks)}/{len(effective_ranks)}")
    else:
        print(f"  All values are invalid")
    
    # Show the actual sparsest dimension values
    print(f"\n--- SPARSEST {dim_type.upper()} DETAILS ---")
    sparsest_vec = W[sparsest_l0_idx, :]
    non_zero_indices = np.where(np.abs(sparsest_vec) > 1e-8)[0]
    non_zero_values = sparsest_vec[non_zero_indices]
    
    print(f"{dim_type} {sparsest_l0_idx} (sparsest by L0):")
    print(f"Non-zero elements: {len(non_zero_indices)}/{other_dim}")
    if len(non_zero_indices) <= 20:  # Show all if not too many
        for idx, val in zip(non_zero_indices, non_zero_values):
            print(f"  [{idx}]: {val:.6f}")
    else:  # Show top 10 by magnitude
        top_indices = np.argsort(np.abs(non_zero_values))[-10:]
        print(f"Top 10 by magnitude:")
        for i in top_indices:
            idx, val = non_zero_indices[i], non_zero_values[i]
            print(f"  [{idx}]: {val:.6f}")
    
    return {
        'dim_stats': dim_stats,
        'sparsest_l0_idx': sparsest_l0_idx,
        'sparsest_gini_idx': sparsest_gini_idx,
        'lowest_effective_rank_idx': lowest_effective_rank_idx,
        'overall_stats': {
            'l0_sparsity': {'mean': np.mean(l0_sparsities), 'std': np.std(l0_sparsities), 
                           'min': np.min(l0_sparsities), 'max': np.max(l0_sparsities)},
            'l1_norms': {'mean': np.mean(l1_norms), 'std': np.std(l1_norms) if np.isfinite(np.std(l1_norms)) else float('nan'),
                        'min': np.min(l1_norms), 'max': np.max(l1_norms)},
            'gini_coeffs': {'mean': np.mean(gini_coeffs), 'std': np.std(gini_coeffs),
                           'min': np.min(gini_coeffs), 'max': np.max(gini_coeffs)}
        }
    }

def load_and_analyze_model(model_path, component='embeddings'):
    """
    Load a model and analyze specified weight matrix.
    
    Args:
        model_path: Path to the saved model (.pt file)
        component: Which component to analyze ('embeddings', 'linear', 'interaction_mlp')
    """
    print(f"Loading model from: {model_path}")
    
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        print("Model loaded successfully!")
        
        print(f"\nAvailable parameters:")
        for name, param in state_dict.items():
            print(f"  {name}: {param.shape}")
        
        # Analyze requested component
        if component == 'embeddings':
            if 'feature_embeddings.weight' in state_dict:
                weight_matrix = state_dict['feature_embeddings.weight']
                # For embeddings: analyze rows (which SAE features have sparse embeddings across K dimensions)
                analyze_weight_sparsity(weight_matrix, "Feature Embeddings", "rows")
            else:
                print("No feature_embeddings.weight found in model!")
                
        elif component == 'linear':
            if 'linear.weight' in state_dict:
                weight_matrix = state_dict['linear.weight']
                # For linear: analyze columns (which input SAE features are sparsely represented)
                analyze_weight_sparsity(weight_matrix, "Linear Layer", "columns")
            else:
                print("No linear.weight found in model!")
                
        elif component == 'interaction_mlp':
            mlp_layers = [name for name in state_dict.keys() if 'interaction_mlp' in name and 'weight' in name]
            if mlp_layers:
                for layer_name in mlp_layers:
                    weight_matrix = state_dict[layer_name]
                    # For MLP: analyze both rows and columns
                    analyze_weight_sparsity(weight_matrix, f"Interaction MLP - {layer_name}", "rows")
                    analyze_weight_sparsity(weight_matrix, f"Interaction MLP - {layer_name}", "columns")
            else:
                print("No interaction_mlp weight layers found in model!")
                
        elif component == 'all':
            # Analyze all components
            components_analyzed = []
            
            if 'feature_embeddings.weight' in state_dict:
                analyze_weight_sparsity(state_dict['feature_embeddings.weight'], "Feature Embeddings", "rows")
                components_analyzed.append("embeddings")
            
            if 'linear.weight' in state_dict:
                analyze_weight_sparsity(state_dict['linear.weight'], "Linear Layer", "columns")
                components_analyzed.append("linear")
            
            mlp_layers = [name for name in state_dict.keys() if 'interaction_mlp' in name and 'weight' in name]
            for layer_name in mlp_layers:
                analyze_weight_sparsity(state_dict[layer_name], f"Interaction MLP - {layer_name}", "rows")
                components_analyzed.append(layer_name)
            
            print(f"\nAnalyzed components: {components_analyzed}")
        else:
            print(f"Unknown component: {component}. Choose from: embeddings, linear, interaction_mlp, all")
            
    except Exception as e:
        print(f"Error loading or analyzing model: {e}")

def main():
    parser = argparse.ArgumentParser(description='Analyze weight matrix sparsity patterns')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved model file (.pt)')
    parser.add_argument('--component', type=str, default='all',
                        choices=['embeddings', 'linear', 'interaction_mlp', 'all'],
                        help='Which component to analyze (default: all)')
    
    args = parser.parse_args()
    
    if not Path(args.model_path).exists():
        print(f"Model file not found: {args.model_path}")
        return
    
    load_and_analyze_model(args.model_path, args.component)

if __name__ == "__main__":
    main()