import torch
import numpy as np
import argparse
from pathlib import Path
import json


# === STEP 1: Top K Dimension for Feature 32806 ===
# Top K dimension: 140
# Weight strength: 0.177124
# Absolute weight: 0.177124

# === STEP 2: Strongest Other Feature for K Dimension 140 ===
# Strongest other feature: 40595
# Weight strength: 0.232422
# Absolute weight: 0.232422

# === TOP 5 CONTRIBUTORS TO K DIMENSION 140 ===
# Rank 1: Feature 40595, Weight: 0.232422
# Rank 2: Feature 23360, Weight: -0.197998
# Rank 3: Feature 28166, Weight: -0.197510
# Rank 4: Feature 17210, Weight: -0.196777
# Rank 5: Feature 13158, Weight: 0.193237

# Target feature 32806 rank in this K dimension: 23

# ================================================================================
# SUMMARY
# ================================================================================
# Target Feature: 32806
# Strongest Interacting Feature: 40595
# They interact through K dimension: 140
# Target feature weight: 0.177124
# Other feature weight: 0.232422


def find_strongest_interacting_feature(model_path, target_feature_idx):
    """
    Find the feature that interacts most strongly with the target feature in the NFM.
    
    Args:
        model_path: Path to the NFM model
        target_feature_idx: Feature index to analyze (e.g., 32806)
    
    Returns:
        Dictionary with results
    """
    print(f"Loading NFM model from: {model_path}")
    
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        print("Model loaded successfully!")
        
        print(f"\nAvailable parameters:")
        for name, param in state_dict.items():
            print(f"  {name}: {param.shape}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Check if we have the feature embeddings
    if 'feature_embeddings.weight' not in state_dict:
        print("Error: No feature_embeddings.weight found in model!")
        return None
    
    embeddings = state_dict['feature_embeddings.weight']  # [50000, 150]
    print(f"\nFeature embeddings shape: {embeddings.shape}")
    
    # Validate target feature index
    if target_feature_idx >= embeddings.shape[0]:
        print(f"Error: Target feature {target_feature_idx} is out of range (max: {embeddings.shape[0]-1})")
        return None
    
    # Step 1: Find the top K dimension that target feature contributes to most strongly
    target_embedding = embeddings[target_feature_idx, :]  # [150]
    
    # Find the dimension with maximum absolute weight
    abs_weights = torch.abs(target_embedding)
    top_k_dim = torch.argmax(abs_weights).item()
    top_k_weight = target_embedding[top_k_dim].item()
    
    print(f"\n=== STEP 1: Top K Dimension for Feature {target_feature_idx} ===")
    print(f"Top K dimension: {top_k_dim}")
    print(f"Weight strength: {top_k_weight:.6f}")
    print(f"Absolute weight: {abs(top_k_weight):.6f}")
    
    # Step 2: Find which other feature contributes most strongly to this K dimension
    k_dim_weights = embeddings[:, top_k_dim]  # [50000] - all features' weights for this K dimension
    
    # Set target feature weight to 0 to exclude it from search
    k_dim_weights_copy = k_dim_weights.clone()
    k_dim_weights_copy[target_feature_idx] = 0
    
    # Find feature with maximum absolute contribution to this K dimension
    abs_k_weights = torch.abs(k_dim_weights_copy)
    strongest_other_feature = torch.argmax(abs_k_weights).item()
    strongest_other_weight = k_dim_weights[strongest_other_feature].item()
    
    print(f"\n=== STEP 2: Strongest Other Feature for K Dimension {top_k_dim} ===")
    print(f"Strongest other feature: {strongest_other_feature}")
    print(f"Weight strength: {strongest_other_weight:.6f}")
    print(f"Absolute weight: {abs(strongest_other_weight):.6f}")
    
    # Additional analysis: show top 5 contributors to this K dimension for context
    print(f"\n=== TOP 5 CONTRIBUTORS TO K DIMENSION {top_k_dim} ===")
    top5_indices = torch.topk(abs_k_weights, k=5).indices
    for i, idx in enumerate(top5_indices):
        weight = k_dim_weights[idx.item()].item()
        print(f"Rank {i+1}: Feature {idx.item()}, Weight: {weight:.6f}")
    
    # Check if target feature would be in top 5 (for reference)
    target_rank = (abs_k_weights > abs(top_k_weight)).sum().item() + 1
    print(f"\nTarget feature {target_feature_idx} rank in this K dimension: {target_rank}")
    
    # Prepare results
    results = {
        'target_feature': target_feature_idx,
        'top_k_dimension': top_k_dim,
        'target_feature_weight_in_k_dim': top_k_weight,
        'strongest_other_feature': strongest_other_feature,
        'strongest_other_weight_in_k_dim': strongest_other_weight,
        'target_feature_rank_in_k_dim': target_rank,
        'k_dimension_analysis': {
            'dimension_index': top_k_dim,
            'total_features_contributing': embeddings.shape[0],
            'top_5_contributors': [
                {
                    'feature_idx': int(idx.item()),
                    'weight': float(k_dim_weights[idx.item()].item()),
                    'abs_weight': float(abs_k_weights[idx.item()].item())
                }
                for idx in top5_indices
            ]
        }
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Find strongest interacting feature in NFM")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the NFM model (.pt file)")
    parser.add_argument("--target_feature", type=int, required=True,
                        help="Target feature index to analyze (e.g., 32806)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Optional output JSON file to save results")
    
    args = parser.parse_args()
    
    # Validate file exists
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    # Run analysis
    results = find_strongest_interacting_feature(args.model_path, args.target_feature)
    
    if results is None:
        print("Analysis failed!")
        return
    
    # Print summary
    print(f"\n" + "="*80)
    print(f"SUMMARY")
    print(f"="*80)
    print(f"Target Feature: {results['target_feature']}")
    print(f"Strongest Interacting Feature: {results['strongest_other_feature']}")
    print(f"They interact through K dimension: {results['top_k_dimension']}")
    print(f"Target feature weight: {results['target_feature_weight_in_k_dim']:.6f}")
    print(f"Other feature weight: {results['strongest_other_weight_in_k_dim']:.6f}")
    
    print(f"\nNext step: Run feature analysis on feature {results['strongest_other_feature']}")
    print(f"Command:")
    print(f"python your_feature_analysis_script.py --features \"{results['strongest_other_feature']}\" --model_path YOUR_MODEL --sae_path YOUR_SAE --output_dir ./feature_{results['strongest_other_feature']}_analysis")
    
    # Save results if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")

if __name__ == "__main__":
    main()