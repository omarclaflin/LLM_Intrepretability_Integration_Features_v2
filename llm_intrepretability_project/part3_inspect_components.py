import torch
import numpy as np
import argparse

def load_and_inspect_nfm_weights(model_path):
    """
    Load and inspect NFM weights to compare linear vs interaction components.
    """
    print(f"Loading NFM model from: {model_path}")
    
    # Load the state dict
    state_dict = torch.load(model_path, map_location='cpu')
    
    print("\n" + "="*60)
    print("NFM MODEL WEIGHT ANALYSIS")
    print("="*60)
    
    # Print all parameter names and shapes
    print("\nAll Parameters:")
    for name, param in state_dict.items():
        print(f"  {name}: {param.shape}")
    
    print("\n" + "="*60)
    print("LINEAR COMPONENT ANALYSIS")
    print("="*60)
    
    # Analyze linear component
    if 'linear.weight' in state_dict:
        linear_weight = state_dict['linear.weight']
        linear_bias = state_dict['linear.bias'] if 'linear.bias' in state_dict else None
        
        print(f"Linear Weight Shape: {linear_weight.shape}")
        print(f"Linear Weight Stats:")
        print(f"  Min: {linear_weight.min().item():.6f}")
        print(f"  Max: {linear_weight.max().item():.6f}")
        print(f"  Mean: {linear_weight.mean().item():.6f}")
        print(f"  Std: {linear_weight.std().item():.6f}")
        print(f"  Non-zero elements: {(torch.abs(linear_weight) > 1e-8).sum().item()}/{linear_weight.numel()}")
        
        if linear_bias is not None:
            print(f"\nLinear Bias Shape: {linear_bias.shape}")
            print(f"Linear Bias Stats:")
            print(f"  Min: {linear_bias.min().item():.6f}")
            print(f"  Max: {linear_bias.max().item():.6f}")
            print(f"  Mean: {linear_bias.mean().item():.6f}")
            print(f"  Std: {linear_bias.std().item():.6f}")
        
        # Analyze which SAE features have largest linear weights
        feature_norms = torch.norm(linear_weight, dim=1)  # Norm across output dimensions
        top_features = torch.topk(feature_norms, k=10)
        
        print(f"\nTop 10 SAE features by linear weight magnitude:")
        for i, (norm, idx) in enumerate(zip(top_features.values, top_features.indices)):
            print(f"  SAE Feature {idx.item()}: L2 norm = {norm.item():.6f}")
    else:
        print("No linear component found in model!")
    
    print("\n" + "="*60)
    print("INTERACTION COMPONENT ANALYSIS")
    print("="*60)
    
    # Analyze embeddings
    if 'feature_embeddings.weight' in state_dict:
        embeddings = state_dict['feature_embeddings.weight']
        
        print(f"Embedding Shape: {embeddings.shape}")
        print(f"Embedding Stats:")
        print(f"  Min: {embeddings.min().item():.6f}")
        print(f"  Max: {embeddings.max().item():.6f}")
        print(f"  Mean: {embeddings.mean().item():.6f}")
        print(f"  Std: {embeddings.std().item():.6f}")
        
        # Analyze which SAE features have largest embedding norms
        embedding_norms = torch.norm(embeddings, dim=1)  # Norm across K dimensions
        top_embedding_features = torch.topk(embedding_norms, k=10)
        
        print(f"\nTop 10 SAE features by embedding norm (interaction strength):")
        for i, (norm, idx) in enumerate(zip(top_embedding_features.values, top_embedding_features.indices)):
            print(f"  SAE Feature {idx.item()}: L2 norm = {norm.item():.6f}")
    else:
        print("No feature embeddings found in model!")
    
    # Analyze interaction MLP
    mlp_params = [name for name in state_dict.keys() if 'interaction_mlp' in name]
    if mlp_params:
        print(f"\nInteraction MLP Parameters:")
        for name in mlp_params:
            param = state_dict[name]
            print(f"  {name}: {param.shape}")
            print(f"    Min: {param.min().item():.6f}, Max: {param.max().item():.6f}")
            print(f"    Mean: {param.mean().item():.6f}, Std: {param.std().item():.6f}")
    else:
        print("No interaction MLP parameters found!")
    
    print("\n" + "="*60)
    print("COMPONENT COMPARISON")
    print("="*60)
    
    # Compare linear vs interaction component magnitudes
    if 'linear.weight' in state_dict and 'feature_embeddings.weight' in state_dict:
        linear_weight = state_dict['linear.weight']
        embeddings = state_dict['feature_embeddings.weight']
        
        # Compare feature importance between linear and interaction
        linear_feature_norms = torch.norm(linear_weight, dim=1)
        embedding_feature_norms = torch.norm(embeddings, dim=1)
        
        print(f"Linear component feature importance:")
        print(f"  Mean feature weight norm: {linear_feature_norms.mean().item():.6f}")
        print(f"  Std feature weight norm: {linear_feature_norms.std().item():.6f}")
        print(f"  Max feature weight norm: {linear_feature_norms.max().item():.6f}")
        
        print(f"\nInteraction component feature importance:")
        print(f"  Mean embedding norm: {embedding_feature_norms.mean().item():.6f}")
        print(f"  Std embedding norm: {embedding_feature_norms.std().item():.6f}")
        print(f"  Max embedding norm: {embedding_feature_norms.max().item():.6f}")
        
        # Check overlap between top features
        top_linear_indices = set(torch.topk(linear_feature_norms, k=100).indices.tolist())
        top_embedding_indices = set(torch.topk(embedding_feature_norms, k=100).indices.tolist())
        overlap = len(top_linear_indices.intersection(top_embedding_indices))
        
        print(f"\nTop 100 feature overlap between linear and interaction: {overlap}/100")
        print(f"This suggests {'high' if overlap > 50 else 'low'} overlap between linear and interaction importance")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Inspect NFM model weights')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved NFM model file (.pt)')
    
    args = parser.parse_args()
    
    try:
        load_and_inspect_nfm_weights(args.model_path)
    except FileNotFoundError:
        print(f"Model file not found: {args.model_path}")
        print("Please check the file path and try again.")
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    main()