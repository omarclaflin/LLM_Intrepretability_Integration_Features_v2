"""
NFM Secondary SAE Analysis Script - TOPK VERSION WITH COUNTEREXAMPLES

This script analyzes how Primary SAE features flow through the COMPLETE NFM pipeline 
(including Primary SAE reconstruction and NFM linear pathway) into Secondary SAE features.

CORRECTED Pipeline: Layer 16 → Primary TopK SAE → [3 Pathways: Primary Reconstruction + NFM Linear + NFM Interaction] → Secondary TopK SAE

NEW: COUNTEREXAMPLES - Also outputs features on the opposite side of each criteria to find 
non-interaction patterns as controls.
"""

import torch
import numpy as np
import argparse
from pathlib import Path
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
import logging
from scipy import stats
from typing import List, Tuple, Dict, Any

# Stimulus sets (keeping only first 10 items each for brevity)
STIMULUS_SETS = {
    "F1_high_F2_high": [
        "We profoundly regret the devastating error that has tarnished our reputation.",
        "It is with immense sorrow that I convey this tragic news.",
        "My heart is utterly broken by the egregious injustice perpetrated here.",
        "I hereby express my profound disgust at these deplorable findings today.",
        "The egregious suffering endured by citizens demands an immediate passionate response.",
        "With gravest apprehension, we anticipate the calamitous repercussions of this decision.",
        "An overwhelming sense of despair permeates the entire community currently.",
        "Our collective indignation at this flagrant disregard for truth is unyielding.",
        "I am compelled to voice my deepest anguish regarding this loss.",
        "The sheer terror of the unfolding catastrophe is beyond comprehension."
    ],
    "F1_high_F2_low": [
        "The aforementioned data substantiates the initial hypothesis with reasonable statistical confidence.",
        "It is imperative to review the procedural guidelines prior to implementation.",
        "The revised protocol stipulates adherence to stringent safety regulations throughout process.",
        "A comprehensive analysis of statistical variances was subsequently conducted by team.",
        "The committee's deliberations concluded without reaching a unanimous consensus on matter.",
        "This document delineates the parameters for forthcoming operational adjustments within department.",
        "The findings corroborate the established theoretical framework with acceptable precision levels.",
        "Please transmit the requisite documentation to the appropriate department in course.",
        "The stipulated deadlines must be strictly observed to ensure project continuity.",
        "Subsequent to evaluation, modifications to the existing infrastructure are recommended here."
    ],
    "F1_low_F2_high": [
        "OMG, I'm so hyped about that! This is seriously the best news!",
        "I literally can't even right now, this is just way too much!",
        "Dude, that was absolutely insane! I'm totally freaking out about it right!",
        "Seriously, I'm so mad about this I could just scream loudly!",
        "Holy cow, that's absolutely devastating! I feel so awful for them!",
        "I'm so thrilled about this, I might just cry the happiest tears!",
        "No way! This is like, completely mind-blowingly amazing and I can't believe!",
        "Ugh, I'm just so done with all this, it's making me furious!",
        "My heart just aches for everyone involved, it's such a tragic situation.",
        "I'm bouncing off the walls with excitement, you seriously have no idea!"
    ],
    "F1_low_F2_low": [
        "Yeah, the light switch is on the wall over there by door.",
        "It's raining outside today. What's up with you these days?",
        "He's over there somewhere, I think near the back of room.",
        "I gotta go now, catch ya later. Bye for now everyone.",
        "The cat's asleep on the couch in the living room right now.",
        "Cool, got it. Thanks for letting me know about that thing.",
        "Just chilling at home, nothing much going on today around here.",
        "The food's in the fridge if you want to grab something good.",
        "It's kinda quiet around here today, not much happening at all.",
        "He said okay, I guess that works for everyone involved here."
    ]
}

class TopKSparseAutoencoder(torch.nn.Module):
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
        batch_size, num_features = features.shape
        topk_values, topk_indices = torch.topk(features, self.k, dim=1)
        sparse_features = torch.zeros_like(features)
        sparse_features.scatter_(1, topk_indices, topk_values)
        return sparse_features

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

class NeuralFactorizationModel(torch.nn.Module):
    def __init__(self, num_features, k_dim, output_dim):
        super().__init__()
        self.feature_embeddings = torch.nn.Embedding(num_features, k_dim)
        self.linear = torch.nn.Linear(num_features, output_dim)
        self.interaction_mlp = torch.nn.Sequential(
            torch.nn.Identity(),
            torch.nn.Linear(k_dim, k_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(k_dim, output_dim)
        )
    
    def forward(self, x):
        linear_out = self.linear(x)
        embeddings = self.feature_embeddings.weight.T
        weighted_embeddings = torch.matmul(x, embeddings.T)
        interaction_out = self.interaction_mlp(weighted_embeddings)
        return linear_out + interaction_out, linear_out, interaction_out, weighted_embeddings

class NFMSAEAnalyzer:
    def __init__(self, primary_sae, nfm_model, secondary_sae, tokenizer, base_model, device="cuda", target_layer=16):
        self.primary_sae = primary_sae
        self.nfm_model = nfm_model
        self.secondary_sae = secondary_sae
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.device = device
        self.target_layer = target_layer
        
        self.primary_sae.eval()
        self.nfm_model.eval()
        self.secondary_sae.eval()
        self.base_model.eval()

    def get_complete_pipeline_activations(self, texts, batch_size=16):
        primary_activations = []
        primary_reconstructions = []
        nfm_linear_outputs = []
        nfm_interaction_outputs = []
        secondary_activations = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing complete TopK pipeline"):
            batch_texts = texts[i:i+batch_size]
            if not batch_texts:
                continue
                
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True,
                                truncation=True, max_length=100).to(self.device)
            
            with torch.no_grad():
                outputs = self.base_model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[self.target_layer]
                
                batch_size_inner, seq_len, hidden_dim = hidden_states.shape
                hidden_states_reshaped = hidden_states.reshape(-1, hidden_dim)
                
                primary_features, primary_reconstruction = self.primary_sae(
                    hidden_states_reshaped.to(self.primary_sae.encoder[0].weight.dtype)
                )
                
                primary_features = primary_features.reshape(batch_size_inner, seq_len, -1)
                primary_reconstruction = primary_reconstruction.reshape(batch_size_inner, seq_len, -1)
                
                nfm_linear_output = self.nfm_model.linear(primary_features)
                
                embeddings = self.nfm_model.feature_embeddings.weight.T
                weighted_embeddings = torch.matmul(primary_features, embeddings.T)
                
                mlp_layer1_output = self.nfm_model.interaction_mlp[1](weighted_embeddings)
                
                original_shape = mlp_layer1_output.shape
                mlp_layer1_flat = mlp_layer1_output.view(-1, original_shape[-1])
                
                secondary_features, _ = self.secondary_sae(
                    mlp_layer1_flat.to(self.secondary_sae.encoder[0].weight.dtype)
                )
                secondary_reconstruction = self.secondary_sae.decoder(secondary_features)
                secondary_reconstruction_reshaped = secondary_reconstruction.view(original_shape)
                
                relu_output = self.nfm_model.interaction_mlp[2](secondary_reconstruction_reshaped)
                nfm_interaction_output = self.nfm_model.interaction_mlp[3](relu_output)
                
                for b in range(primary_features.shape[0]):
                    seq_len_actual = torch.sum(inputs["attention_mask"][b]).item()
                    if seq_len_actual > 0:
                        mean_primary_features = torch.mean(primary_features[b, :seq_len_actual, :], dim=0)
                        mean_primary_reconstruction = torch.mean(primary_reconstruction[b, :seq_len_actual, :], dim=0)
                        mean_nfm_linear = torch.mean(nfm_linear_output[b, :seq_len_actual, :], dim=0)
                        mean_nfm_interaction = torch.mean(nfm_interaction_output[b, :seq_len_actual, :], dim=0)
                        
                        start_idx = b * seq_len_actual
                        end_idx = start_idx + seq_len_actual
                        if end_idx <= secondary_features.shape[0]:
                            mean_secondary_features = torch.mean(secondary_features[start_idx:end_idx, :], dim=0)
                        else:
                            secondary_features_reshaped = secondary_features.view(batch_size_inner, seq_len, -1)
                            mean_secondary_features = torch.mean(secondary_features_reshaped[b, :seq_len_actual, :], dim=0)
                        
                        primary_activations.append(mean_primary_features.cpu().numpy())
                        primary_reconstructions.append(mean_primary_reconstruction.cpu().numpy())
                        nfm_linear_outputs.append(mean_nfm_linear.cpu().numpy())
                        nfm_interaction_outputs.append(mean_nfm_interaction.cpu().numpy())
                        secondary_activations.append(mean_secondary_features.cpu().numpy())
        
        return (np.array(primary_activations), np.array(primary_reconstructions), 
                np.array(nfm_linear_outputs), np.array(nfm_interaction_outputs), 
                np.array(secondary_activations))

    def find_highest_activated_and_differencing_features(self, top_n_activation=10):
        print(f"\n=== ANALYSIS 3: TOP {top_n_activation} HIGHEST ACTIVATED AND DIFFERENCING SECONDARY TopK SAE FEATURES ===")
        
        condition_11_texts = STIMULUS_SETS['F1_high_F2_high']
        condition_00_texts = STIMULUS_SETS['F1_low_F2_low']
        
        (_, _, _, _, activations_11) = self.get_complete_pipeline_activations(condition_11_texts)
        (_, _, _, _, activations_00) = self.get_complete_pipeline_activations(condition_00_texts)
        
        mean_activations_11 = np.mean(activations_11, axis=0)
        mean_activations_00 = np.mean(activations_00, axis=0)
        
        top_11_indices = np.argsort(mean_activations_11)[-top_n_activation:][::-1]
        bottom_11_indices = np.argsort(mean_activations_11)[:top_n_activation]
        
        print(f"Top {top_n_activation} highest activated Secondary TopK SAE features in [1,1] condition:")
        for rank, idx in enumerate(top_11_indices):
            activation_11 = mean_activations_11[idx]
            activation_00 = mean_activations_00[idx]
            print(f"  Rank {rank+1}: Feature {idx}")
            print(f"    [1,1] activation: {activation_11:.6f}")
            print(f"    [0,0] activation: {activation_00:.6f}")
        
        print(f"\n🔍 COUNTEREXAMPLE: Top {top_n_activation} LOWEST activated Secondary TopK SAE features in [1,1] condition:")
        for rank, idx in enumerate(bottom_11_indices):
            activation_11 = mean_activations_11[idx]
            activation_00 = mean_activations_00[idx]
            print(f"  Rank {rank+1}: Feature {idx} (COUNTEREXAMPLE)")
            print(f"    [1,1] activation: {activation_11:.6f} (LOW)")
            print(f"    [0,0] activation: {activation_00:.6f}")
        
        differences = mean_activations_11 - mean_activations_00
        abs_differences = np.abs(differences)
        smallest_diff_indices = np.argsort(abs_differences)[:top_n_activation]
        
        print(f"\n🔍 COUNTEREXAMPLE: Top {top_n_activation} SMALLEST differencing Secondary TopK SAE features (no pattern):")
        for rank, idx in enumerate(smallest_diff_indices):
            difference = differences[idx]
            abs_diff = abs_differences[idx]
            activation_11 = mean_activations_11[idx]
            activation_00 = mean_activations_00[idx]
            print(f"  Rank {rank+1}: Feature {idx} (COUNTEREXAMPLE)")
            print(f"    Absolute difference: {abs_diff:.6f} (SMALL = no pattern)")
            print(f"    [1,1] activation: {activation_11:.6f}")
            print(f"    [0,0] activation: {activation_00:.6f}")
        
        # return {
        #     'top_11_features': [{'feature_idx': int(idx), 'activation_11': float(mean_activations_11[idx])} for idx in top_11_indices],
        #     'bottom_11_features': [{'feature_idx': int(idx), 'activation_11': float(mean_activations_11[idx])} for idx in bottom_11_indices],
        #     'smallest_diff_features': [{'feature_idx': int(idx), 'absolute_difference': float(abs_differences[idx])} for idx in smallest_diff_indices]
        # }
        # return {
        #     'top_11_features': [{'feature_idx': int(idx), 'activation_11': float(mean_activations_11[idx])} for idx in top_11_indices],
        #     'bottom_11_features': [{'feature_idx': int(idx), 'activation_11': float(mean_activations_11[idx])} for idx in bottom_11_indices],
        #     'smallest_diff_features': [{'feature_idx': int(idx), 'absolute_difference': float(abs_differences[idx])} for idx in smallest_diff_indices]
        # }
        return {
            'top_11_features': [{'feature_idx': int(idx), 'activation_11': float(mean_activations_11[idx])} for idx in top_11_indices],
            'bottom_11_features': [{'feature_idx': int(idx), 'activation_11': float(mean_activations_11[idx])} for idx in bottom_11_indices],
            'smallest_diff_features': [{'feature_idx': int(idx), 'absolute_difference': float(abs_differences[idx])} for idx in smallest_diff_indices]
        }

    def find_anova_sensitive_feature(self, top_n_anova=10):
        print(f"\n=== ANALYSIS 4: TOP {top_n_anova} SECONDARY TopK SAE FEATURES MOST SENSITIVE TO MODULATIONS (ANOVA) ===")
        
        condition_data = {}
        conditions = ['F1_high_F2_high', 'F1_high_F2_low', 'F1_low_F2_high', 'F1_low_F2_low']
        
        for condition in conditions:
            texts = STIMULUS_SETS[condition]
            (_, _, _, _, secondary_activations) = self.get_complete_pipeline_activations(texts)
            condition_data[condition] = secondary_activations
        
        num_secondary_features = condition_data[conditions[0]].shape[1]
        f_statistics = []
        p_values = []
        valid_features = []
        
        for sec_idx in range(num_secondary_features):
            groups = []
            for condition in conditions:
                groups.append(condition_data[condition][:, sec_idx])
            
            try:
                f_stat, p_val = stats.f_oneway(*groups)
                if np.isfinite(f_stat) and np.isfinite(p_val):
                    f_statistics.append(f_stat)
                    p_values.append(p_val)
                    valid_features.append(sec_idx)
            except:
                continue
        
        if not valid_features:
            return {'error': 'No valid ANOVA results'}
        
        top_f_indices = np.argsort(f_statistics)[-top_n_anova:][::-1]
        bottom_f_indices = np.argsort(f_statistics)[:top_n_anova]
        
        print(f"Top {top_n_anova} most ANOVA-sensitive Secondary TopK SAE features:")
        for rank, f_idx in enumerate(top_f_indices):
            actual_feature_idx = valid_features[f_idx]
            f_stat = f_statistics[f_idx]
            p_val = p_values[f_idx]
            print(f"  Rank {rank+1}: Feature {actual_feature_idx}")
            print(f"    F-statistic: {f_stat:.6f}, p-value: {p_val:.2e}")
        
        print(f"\n🔍 COUNTEREXAMPLE: Top {top_n_anova} LEAST ANOVA-sensitive Secondary TopK SAE features (no interaction):")
        for rank, f_idx in enumerate(bottom_f_indices):
            actual_feature_idx = valid_features[f_idx]
            f_stat = f_statistics[f_idx]
            p_val = p_values[f_idx]
            print(f"  Rank {rank+1}: Feature {actual_feature_idx} (COUNTEREXAMPLE)")
            print(f"    F-statistic: {f_stat:.6f} (LOW), p-value: {p_val:.2e}")
        
        return {
            'top_anova_features': [{'feature_idx': valid_features[f_idx], 'f_statistic': f_statistics[f_idx]} for f_idx in top_f_indices],
            'bottom_anova_features': [{'feature_idx': valid_features[f_idx], 'f_statistic': f_statistics[f_idx]} for f_idx in bottom_f_indices]
        }

def load_sae_model(checkpoint_path, device="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint
    
    if 'decoder.weight' in state_dict:
        input_dim = state_dict['decoder.weight'].shape[0]
        hidden_dim = state_dict['decoder.weight'].shape[1]
    elif 'encoder.0.weight' in state_dict:
        encoder_weight = state_dict['encoder.0.weight']
        hidden_dim, input_dim = encoder_weight.shape
    else:
        raise ValueError("Cannot determine SAE dimensions")
    
    filename = str(checkpoint_path).lower()
    is_topk = 'topk' in filename
    
    if is_topk:
        k_value = max(1, int(0.02 * hidden_dim))
        model = TopKSparseAutoencoder(input_dim, hidden_dim, k_value)
    else:
        model = SparseAutoencoder(input_dim, hidden_dim)
    
    model.load_state_dict(state_dict)
    model.to(device)
    return model

def convert_to_json_serializable(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
    
def main():
    parser = argparse.ArgumentParser(description="NFM Secondary SAE Analysis with Counterexamples")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--primary_sae_path", type=str, required=True)
    parser.add_argument("--nfm_path", type=str, required=True)
    parser.add_argument("--secondary_sae_path", type=str, required=True)
    parser.add_argument("--primary_feature1", type=int, required=True)
    parser.add_argument("--primary_feature2", type=int, required=True)
    parser.add_argument("--top_n_activation", type=int, default=1)
    parser.add_argument("--top_n_anova", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./nfm_sae_analysis")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading models...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")
    
    primary_sae = load_sae_model(args.primary_sae_path, args.device)
    
    nfm_state_dict = torch.load(args.nfm_path, map_location=args.device)
    num_features = nfm_state_dict['feature_embeddings.weight'].shape[0]
    k_dim = nfm_state_dict['feature_embeddings.weight'].shape[1]
    output_dim = nfm_state_dict['linear.weight'].shape[0]
    
    nfm_model = NeuralFactorizationModel(num_features, k_dim, output_dim)
    nfm_model.load_state_dict(nfm_state_dict)
    nfm_model.to(args.device)
    
    secondary_sae = load_sae_model(args.secondary_sae_path, args.device)
    
    print("Models loaded successfully!")
    
    analyzer = NFMSAEAnalyzer(primary_sae, nfm_model, secondary_sae, tokenizer, base_model, args.device)
    
    all_results = {}
    
    analysis3_results = analyzer.find_highest_activated_and_differencing_features(
        top_n_activation=args.top_n_activation
    )
    all_results['analysis3_highest_activated_and_differencing'] = analysis3_results
    
    analysis4_results = analyzer.find_anova_sensitive_feature(
        top_n_anova=args.top_n_anova
    )
    all_results['analysis4_anova_sensitivity'] = analysis4_results
    
    results_path = output_dir / "nfm_sae_analysis_results_with_counterexamples.json"
    with open(results_path, 'w') as f:
        #json.dump(all_results, f, indent=2)
        json.dump(convert_to_json_serializable(all_results), f, indent=2)
    
    print(f"\nAnalysis complete! Results saved to: {results_path}")
    
    print(f"\n🔍 COUNTEREXAMPLE SUMMARY:")
    if 'smallest_diff_features' in analysis3_results:
        print(f"Features with NO interaction pattern:")
        for feature in analysis3_results['smallest_diff_features'][:3]:
            print(f"  Feature {feature['feature_idx']}: abs_diff={feature['absolute_difference']:.6f}")
    
    if 'bottom_anova_features' in analysis4_results:
        print(f"Features with NO ANOVA sensitivity:")
        for feature in analysis4_results['bottom_anova_features'][:3]:
            print(f"  Feature {feature['feature_idx']}: F-stat={feature['f_statistic']:.6f}")

if __name__ == "__main__":
    main()