#This presumes the secondary SAE is already trained (part 10) but does the part6 script using hte SAE instead of hte NFM K directly

"""
Find Strongest Feature Interactions via Secondary SAE

This script finds which Primary SAE features interact most strongly by analyzing
their effects through the Secondary SAE in the complete NFM pipeline:

Pipeline: Primary SAE Feature → NFM Interaction → Secondary SAE → Find Other Primary Features

Logic:
1. For target Primary SAE feature, find which Secondary SAE feature it activates most
2. For that Secondary SAE feature, find which other Primary SAE features contribute most
3. This reveals the strongest interaction partner via the Secondary SAE pathway
"""

# Top 10 Primary features contributing to Secondary Feature 2020:
#   Rank 1: Primary Feature 41307, Contribution: 0.327409
#   Rank 2: Primary Feature 39767, Contribution: 0.327056
#   Rank 3: Primary Feature 38091, Contribution: 0.324165
#   Rank 4: Primary Feature 16056, Contribution: 0.323231
#   Rank 5: Primary Feature 25114, Contribution: 0.323095
#   Rank 6: Primary Feature 5640, Contribution: 0.322695
#   Rank 7: Primary Feature 42540, Contribution: 0.322233
#   Rank 8: Primary Feature 46910, Contribution: 0.321479
#   Rank 9: Primary Feature 12744, Contribution: 0.321355
#   Rank 10: Primary Feature 12191, Contribution: 0.321176

# === INTERACTION ANALYSIS RESULTS ===
# Target Primary Feature: 32806
# Most activated Secondary Feature: 2020 (activation: 0.308940)
# Strongest interacting Primary Feature: 41307
# Interaction strength via Secondary SAE: 0.327409

# ================================================================================
# SECONDARY SAE INTERACTION ANALYSIS SUMMARY
# ================================================================================
# Target Primary Feature: 32806
# Strongest Interacting Primary Feature: 41307
# Mediating Secondary Feature: 2020
# Target → Secondary activation: 0.308940
# Interactor → Secondary activation: 0.327409
# Overall interaction strength: 0.327409

import torch
import numpy as np
import argparse
from pathlib import Path
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

# TopK Sparse Autoencoder (from part11c)
class TopKSparseAutoencoder(torch.nn.Module):
    """TopK Sparse Autoencoder module."""
    def __init__(self, input_dim, hidden_dim, k):
        super().__init__()
        self.k = k
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Linear(hidden_dim, input_dim, bias=False)

    def forward(self, x):
        features = self.encoder(x)
        sparse_features = self.apply_topk(features)
        reconstruction = self.decoder(sparse_features)
        return sparse_features, reconstruction
    
    def apply_topk(self, features):
        """Apply TopK sparsity - keep only top K activations per sample"""
        batch_size, num_features = features.shape
        
        # Get TopK values and indices for each sample
        topk_values, topk_indices = torch.topk(features, self.k, dim=1)
        
        # Create sparse feature tensor
        sparse_features = torch.zeros_like(features)
        
        # Scatter the TopK values back to their original positions
        sparse_features.scatter_(1, topk_indices, topk_values)
        
        return sparse_features

class NeuralFactorizationModel(torch.nn.Module):
    """Neural Factorization Model for analyzing feature interactions."""
    def __init__(self, num_features, k_dim, output_dim):
        super().__init__()
        self.feature_embeddings = torch.nn.Embedding(num_features, k_dim)
        self.linear = torch.nn.Linear(num_features, output_dim)
        self.interaction_mlp = torch.nn.Sequential(
            torch.nn.Identity(),  # Layer 0 - placeholder
            torch.nn.Linear(k_dim, k_dim),  # Layer 1
            torch.nn.ReLU(),  # Layer 2  
            torch.nn.Linear(k_dim, output_dim)  # Layer 3
        )
    
    def forward(self, x):
        # Linear component
        linear_out = self.linear(x)
        
        # Interaction component - this gives us the embedding layer
        embeddings = self.feature_embeddings.weight.T  # [k_dim, num_features]
        weighted_embeddings = torch.matmul(x, embeddings.T)  # [batch, k_dim]
        interaction_out = self.interaction_mlp(weighted_embeddings)
        
        return linear_out + interaction_out, linear_out, interaction_out, weighted_embeddings

def load_sae_model(checkpoint_path, device="cuda"):
    """Load SAE model, detecting whether it's TopK or regular SAE."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract the actual model state dict
    if 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint
    
    # Determine dimensions
    if 'decoder.weight' in state_dict:
        input_dim = state_dict['decoder.weight'].shape[0]
        hidden_dim = state_dict['decoder.weight'].shape[1]
    elif 'encoder.0.weight' in state_dict:
        encoder_weight = state_dict['encoder.0.weight']
        hidden_dim, input_dim = encoder_weight.shape
    else:
        raise ValueError("Cannot determine SAE dimensions from state dict")
    
    # Check if this is a TopK SAE
    is_topk = False
    k_value = None
    
    # Check filename for TopK indicators
    filename = str(checkpoint_path).lower()
    if 'topk' in filename or 'top_k' in filename:
        is_topk = True
        import re
        k_matches = re.findall(r'topk(\d+)', filename)
        if k_matches:
            k_value = int(k_matches[0])
        else:
            k_matches = re.findall(r'_k(\d+)_', filename)
            if k_matches:
                k_value = int(k_matches[0])
    
    # Check checkpoint metadata for TopK info
    if 'metrics_history' in checkpoint:
        metrics = checkpoint['metrics_history']
        if 'percent_active' in metrics and len(metrics['percent_active']) > 0:
            recent_active = metrics['percent_active'][-10:]
            if len(recent_active) > 1:
                std_active = np.std(recent_active)
                mean_active = np.mean(recent_active)
                if std_active < 0.1 and mean_active < 10:
                    is_topk = True
                    if k_value is None:
                        k_value = int((mean_active / 100.0) * hidden_dim)
    
    # Default K value if not found
    if is_topk and k_value is None:
        k_value = max(1, int(0.02 * hidden_dim))
        print(f"Warning: TopK SAE detected but K value not found. Using default K={k_value}")
    
    # Create appropriate model
    if is_topk:
        print(f"Loading as TopK SAE with K={k_value}")
        model = TopKSparseAutoencoder(input_dim, hidden_dim, k_value)
    else:
        print(f"Loading as regular SAE")
        # Regular SAE class
        class SparseAutoencoder(torch.nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.encoder = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, hidden_dim),
                    torch.nn.ReLU()
                )
                self.decoder = torch.nn.Linear(hidden_dim, input_dim, bias=False)

            def forward(self, x):
                features = self.encoder(x)
                reconstruction = self.decoder(features)
                return features, reconstruction
        
        model = SparseAutoencoder(input_dim, hidden_dim)
    
    model.load_state_dict(state_dict)
    model.to(device)
    
    return model

class SecondaryInteractionFinder:
    """Find strongest feature interactions via Secondary SAE analysis."""
    
    def __init__(self, primary_sae, nfm_model, secondary_sae, tokenizer, base_model, device="cuda", target_layer=16):
        self.primary_sae = primary_sae
        self.nfm_model = nfm_model
        self.secondary_sae = secondary_sae
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.device = device
        self.target_layer = target_layer
        
        # Set models to eval mode
        self.primary_sae.eval()
        self.nfm_model.eval()
        self.secondary_sae.eval()
        self.base_model.eval()

    def test_single_primary_feature(self, primary_feature_idx, num_test_samples=100):
        """
        Test how a single Primary SAE feature affects Secondary SAE features.
        
        Returns:
            secondary_activations: [num_secondary_features] - mean activation for each secondary feature
        """
        print(f"Testing Primary SAE feature {primary_feature_idx} through complete pipeline...")
        
        # Create synthetic input with only this primary feature active
        # We'll use different activation strengths to get robust measurements
        activation_strengths = [0.5, 1.0, 1.5, 2.0, 2.5]
        all_secondary_activations = []
        
        for strength in activation_strengths:
            for sample in range(num_test_samples // len(activation_strengths)):
                # Create sparse primary vector with only target feature activated
                num_primary_features = self.primary_sae.encoder[0].out_features
                sparse_primary = torch.zeros(1, num_primary_features, device=self.device, dtype=torch.float32)
                sparse_primary[0, primary_feature_idx] = strength
                
                with torch.no_grad():
                    # Process through NFM Linear Pathway
                    nfm_linear_output = self.nfm_model.linear(sparse_primary)
                    
                    # Process through NFM Interaction Pathway
                    embeddings = self.nfm_model.feature_embeddings.weight.T
                    weighted_embeddings = torch.matmul(sparse_primary, embeddings.T)
                    
                    # MLP Layer 1
                    mlp_layer1_output = self.nfm_model.interaction_mlp[1](weighted_embeddings)
                    
                    # Secondary TopK SAE
                    secondary_features, _ = self.secondary_sae(
                        mlp_layer1_output.to(self.secondary_sae.encoder[0].weight.dtype)
                    )
                    
                    all_secondary_activations.append(secondary_features.squeeze().cpu().numpy())
        
        # Average across all test samples
        mean_secondary_activations = np.mean(all_secondary_activations, axis=0)
        
        return mean_secondary_activations

    def find_strongest_interacting_feature(self, target_primary_feature_idx, top_n_analyze=5):
        """
        Find the Primary SAE feature that interacts most strongly with the target feature
        via Secondary SAE analysis.
        
        Pipeline:
        1. Find which Secondary SAE feature is most activated by target Primary feature
        2. For that Secondary feature, find which other Primary features contribute most
        3. Return the strongest interaction partner
        
        Args:
            target_primary_feature_idx: Primary SAE feature to analyze
            top_n_analyze: Number of top Secondary features to analyze for interactions
            
        Returns:
            Dictionary with interaction analysis results
        """
        print(f"\n=== FINDING STRONGEST INTERACTION FOR PRIMARY FEATURE {target_primary_feature_idx} ===")
        
        # Step 1: Find which Secondary SAE features are most activated by target Primary feature
        print(f"\nStep 1: Testing target Primary feature {target_primary_feature_idx} through Secondary SAE...")
        secondary_activations = self.test_single_primary_feature(target_primary_feature_idx)
        
        # Find top N secondary features activated by this primary feature
        top_secondary_indices = np.argsort(secondary_activations)[-top_n_analyze:][::-1]
        top_secondary_values = secondary_activations[top_secondary_indices]
        
        print(f"\nTop {top_n_analyze} Secondary SAE features activated by Primary {target_primary_feature_idx}:")
        for i, (sec_idx, activation) in enumerate(zip(top_secondary_indices, top_secondary_values)):
            print(f"  Rank {i+1}: Secondary Feature {sec_idx}, Activation: {activation:.6f}")
        
        # Step 2: For the top Secondary feature, find which Primary features contribute most
        top_secondary_idx = top_secondary_indices[0]
        top_secondary_activation = top_secondary_values[0]
        
        print(f"\nStep 2: Analyzing Secondary Feature {top_secondary_idx} to find contributing Primary features...")
        
        # Test all Primary features to see which ones activate this Secondary feature
        num_primary_features = self.primary_sae.encoder[0].out_features
        primary_contributions = np.zeros(num_primary_features)
        
        # Sample a subset of primary features for efficiency (or test all if small enough)
        sample_size = min(1000, num_primary_features)  # Test top 1000 or all if fewer
        primary_indices_to_test = np.random.choice(num_primary_features, size=sample_size, replace=False)
        
        print(f"Testing {len(primary_indices_to_test)} Primary features for contribution to Secondary {top_secondary_idx}...")
        
        for i, primary_idx in enumerate(tqdm(primary_indices_to_test, desc="Testing Primary features")):
            if primary_idx == target_primary_feature_idx:
                continue  # Skip the target feature itself
                
            # Test this primary feature
            secondary_acts = self.test_single_primary_feature(primary_idx, num_test_samples=20)  # Fewer samples for efficiency
            primary_contributions[primary_idx] = secondary_acts[top_secondary_idx]
        
        # Find top contributing Primary features (excluding target)
        primary_contributions[target_primary_feature_idx] = 0  # Exclude target from results
        top_primary_contributors = np.argsort(primary_contributions)[-10:][::-1]
        top_primary_contributions = primary_contributions[top_primary_contributors]
        
        print(f"\nTop 10 Primary features contributing to Secondary Feature {top_secondary_idx}:")
        for i, (prim_idx, contribution) in enumerate(zip(top_primary_contributors, top_primary_contributions)):
            if contribution > 0:  # Only show non-zero contributions
                print(f"  Rank {i+1}: Primary Feature {prim_idx}, Contribution: {contribution:.6f}")
        
        # The strongest interacting feature is the top contributor
        strongest_interacting_feature = top_primary_contributors[0]
        strongest_contribution = top_primary_contributions[0]
        
        print(f"\n=== INTERACTION ANALYSIS RESULTS ===")
        print(f"Target Primary Feature: {target_primary_feature_idx}")
        print(f"Most activated Secondary Feature: {top_secondary_idx} (activation: {top_secondary_activation:.6f})")
        print(f"Strongest interacting Primary Feature: {strongest_interacting_feature}")
        print(f"Interaction strength via Secondary SAE: {strongest_contribution:.6f}")
        
        # Prepare detailed results
        results = {
            'target_primary_feature': int(target_primary_feature_idx),
            'strongest_interacting_primary_feature': int(strongest_interacting_feature),
            'interaction_strength': float(strongest_contribution),
            'mediating_secondary_feature': {
                'index': int(top_secondary_idx),
                'activation_by_target': float(top_secondary_activation),
                'activation_by_interactor': float(strongest_contribution)
            },
            'top_secondary_features_activated': [
                {
                    'secondary_feature_idx': int(idx),
                    'activation': float(val)
                }
                for idx, val in zip(top_secondary_indices, top_secondary_values)
            ],
            'top_primary_contributors_to_secondary': [
                {
                    'primary_feature_idx': int(idx),
                    'contribution_to_secondary': float(contrib)
                }
                for idx, contrib in zip(top_primary_contributors, top_primary_contributions)
                if contrib > 0
            ],
            'analysis_metadata': {
                'num_primary_features_tested': len(primary_indices_to_test),
                'total_primary_features': num_primary_features,
                'num_secondary_features': len(secondary_activations),
                'secondary_features_analyzed': top_n_analyze
            }
        }
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Find strongest feature interactions via Secondary SAE")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base LLM model")
    parser.add_argument("--primary_sae_path", type=str, required=True, help="Path to Primary SAE model")
    parser.add_argument("--nfm_path", type=str, required=True, help="Path to NFM model")
    parser.add_argument("--secondary_sae_path", type=str, required=True, help="Path to Secondary SAE model")
    parser.add_argument("--target_feature", type=int, required=True, help="Target Primary SAE feature to analyze")
    parser.add_argument("--output_file", type=str, default=None, help="Output JSON file to save results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--top_n_secondary", type=int, default=5, help="Number of top Secondary features to analyze")
    
    args = parser.parse_args()
    
    print("Loading models...")
    
    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")
    
    # Load Primary SAE
    print(f"Loading Primary SAE from {args.primary_sae_path}...")
    primary_sae = load_sae_model(args.primary_sae_path, args.device)
    
    # Load NFM model
    print(f"Loading NFM from {args.nfm_path}...")
    nfm_state_dict = torch.load(args.nfm_path, map_location=args.device)
    num_features = nfm_state_dict['feature_embeddings.weight'].shape[0]
    k_dim = nfm_state_dict['feature_embeddings.weight'].shape[1]
    output_dim = nfm_state_dict['linear.weight'].shape[0]
    
    nfm_model = NeuralFactorizationModel(num_features, k_dim, output_dim)
    nfm_model.load_state_dict(nfm_state_dict)
    nfm_model.to(args.device)
    
    # Load Secondary SAE
    print(f"Loading Secondary SAE from {args.secondary_sae_path}...")
    secondary_sae = load_sae_model(args.secondary_sae_path, args.device)
    
    print("All models loaded successfully!")
    
    # Print model information
    if hasattr(primary_sae, 'k'):
        primary_total = primary_sae.encoder[0].out_features
        primary_sparsity = (primary_sae.k / primary_total) * 100
        print(f"Primary TopK SAE: {primary_sae.encoder[0].in_features} → {primary_total} (K={primary_sae.k}, {primary_sparsity:.2f}% active)")
    else:
        print(f"Primary Regular SAE: {primary_sae.encoder[0].in_features} → {primary_sae.encoder[0].out_features}")
    
    print(f"NFM: {num_features} features → {k_dim} embedding dim")
    
    if hasattr(secondary_sae, 'k'):
        secondary_total = secondary_sae.encoder[0].out_features
        secondary_sparsity = (secondary_sae.k / secondary_total) * 100
        print(f"Secondary TopK SAE: {secondary_sae.encoder[0].in_features} → {secondary_total} (K={secondary_sae.k}, {secondary_sparsity:.2f}% active)")
    else:
        print(f"Secondary Regular SAE: {secondary_sae.encoder[0].in_features} → {secondary_sae.encoder[0].out_features}")
    
    # Validate target feature
    max_primary_features = primary_sae.encoder[0].out_features
    if args.target_feature >= max_primary_features:
        print(f"Error: Target feature {args.target_feature} is out of range (max: {max_primary_features-1})")
        return
    
    # Initialize interaction finder
    finder = SecondaryInteractionFinder(primary_sae, nfm_model, secondary_sae, tokenizer, base_model, args.device)
    
    # Run interaction analysis
    results = finder.find_strongest_interacting_feature(args.target_feature, top_n_analyze=args.top_n_secondary)
    
    if results is None:
        print("Analysis failed!")
        return
    
    # Print summary
    print(f"\n" + "="*80)
    print(f"SECONDARY SAE INTERACTION ANALYSIS SUMMARY")
    print(f"="*80)
    print(f"Target Primary Feature: {results['target_primary_feature']}")
    print(f"Strongest Interacting Primary Feature: {results['strongest_interacting_primary_feature']}")
    print(f"Mediating Secondary Feature: {results['mediating_secondary_feature']['index']}")
    print(f"Target → Secondary activation: {results['mediating_secondary_feature']['activation_by_target']:.6f}")
    print(f"Interactor → Secondary activation: {results['mediating_secondary_feature']['activation_by_interactor']:.6f}")
    print(f"Overall interaction strength: {results['interaction_strength']:.6f}")
    
    print(f"\nNext step: Analyze both features {results['target_primary_feature']} and {results['strongest_interacting_primary_feature']} together")
    print(f"Command suggestion:")
    print(f"python part11c_nfmSAE_analysis_w_distributions.py \\")
    print(f"    --model_path {args.model_path} \\")
    print(f"    --primary_sae_path {args.primary_sae_path} \\")
    print(f"    --nfm_path {args.nfm_path} \\")
    print(f"    --secondary_sae_path {args.secondary_sae_path} \\")
    print(f"    --primary_feature1 {results['target_primary_feature']} \\")
    print(f"    --primary_feature2 {results['strongest_interacting_primary_feature']} \\")
    print(f"    --output_dir ./interaction_analysis_{results['target_primary_feature']}_{results['strongest_interacting_primary_feature']}")
    
    # Save results if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")

if __name__ == "__main__":
    main()