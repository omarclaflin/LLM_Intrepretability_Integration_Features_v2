#modified to include secondary SAE

"""
Complete Pipeline Stimulus-Response Analysis with Secondary SAE

This script performs stimulus-response analysis through the COMPLETE pipeline:
Primary SAE → NFM (Linear + Interaction) → Secondary SAE

Discovers top N Primary and Secondary SAE features that best distinguish between 
high/low conditions using t-tests, then runs full analysis on discovered features.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from pathlib import Path
import json
from tqdm.auto import tqdm
from typing import List, Tuple, Dict, Any
import pandas as pd
from scipy import stats

# Define feature names for clarity
feature1name = 'High Formality'
feature1bname = 'Low Formality'
feature2name = 'High Emotion'
feature2bname = 'Low Emotion'

STIMULUS_SETS = {
    "F1_high_F2_high": [
        "We profoundly regret the grievous error that has irrevocably tarnished our reputation.",
        "It is with immense sorrow that I must convey this devastating news to you all.",
        "My heart is utterly broken by the egregious injustice perpetrated against them.",
        "I hereby express my profound and utter disgust at the deplorable findings presented.",
        "The egregious suffering endured by the populace demands an immediate, impassioned response.",
        "With gravest apprehension, we anticipate the calamitous repercussions of this decision.",
        "An overwhelming sense of despair permeates the entire community in light of these events.",
        "Our collective indignation at this flagrant disregard for truth is unyielding.",
        "I am compelled to voice my deepest anguish regarding the tragic loss of life.",
        "The sheer terror of the unfolding catastrophe is almost beyond comprehension."
    ],
    "F1_high_F2_low": [
        "The aforementioned data substantiates the initial hypothesis effectively.",
        "It is imperative to review the procedural guidelines prior to implementation.",
        "The revised protocol stipulates adherence to stringent safety regulations.",
        "A comprehensive analysis of statistical variances was subsequently conducted.",
        "The committee's deliberations concluded without reaching a unanimous consensus.",
        "This document delineates the parameters for forthcoming operational adjustments.",
        "The findings corroborate the established theoretical framework accurately.",
        "Please transmit the requisite documentation to the appropriate department promptly.",
        "The stipulated deadlines must be strictly observed to ensure project continuity.",
        "Subsequent to evaluation, modifications to the existing infrastructure are advised."
    ],
    "F1_low_F2_high": [
        "OMG, I'm so hyped about that! Best news ever!",
        "I literally can't even, this is just too much!",
        "Dude, that was insane! I'm freaking out right now!",
        "Seriously, I'm so mad I could just scream!",
        "Holy cow, that's absolutely devastating! I feel awful for them.",
        "I'm so thrilled, I might just cry happy tears!",
        "No way! This is like, mind-blowingly amazing!",
        "Ugh, I'm just so done with this, it's infuriating!",
        "My heart just aches for everyone involved, it's truly tragic.",
        "I'm bouncing off the walls with excitement, you have no idea!"
    ],
    "F1_low_F2_low": [
        "Yeah, the light switch is on the wall.",
        "It's raining. What's up?",
        "He's over there, I think.",
        "I gotta go now, bye.",
        "The cat's asleep on the couch.",
        "Cool, got it. Thanks.",
        "Just chilling, nothing much.",
        "The food's in the fridge.",
        "It's kinda quiet today.",
        "He said okay, I guess."
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
        raise ValueError("Cannot determine SAE dimensions from state dict")
    
    is_topk = False
    k_value = None
    
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
    
    if is_topk and k_value is None:
        k_value = max(1, int(0.02 * hidden_dim))
        print(f"Warning: TopK SAE detected but K value not found. Using default K={k_value}")
    
    if is_topk:
        print(f"Loading as TopK SAE with K={k_value}")
        model = TopKSparseAutoencoder(input_dim, hidden_dim, k_value)
    else:
        print(f"Loading as regular SAE")
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

class CompletePipelineAnalyzer:
    def __init__(self, primary_sae, nfm_model, secondary_sae, tokenizer, base_model, device="cuda", target_layer=16, activation_method="mean"):
        self.primary_sae = primary_sae
        self.nfm_model = nfm_model
        self.secondary_sae = secondary_sae
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.device = device
        self.target_layer = target_layer
        self.activation_method = activation_method

        self.primary_sae.eval()
        self.nfm_model.eval()
        self.secondary_sae.eval()
        self.base_model.eval()
    
    def get_complete_pipeline_activations(self, texts, batch_size=16):
        primary_activations = []
        primary_reconstructions = []
        nfm_linear_outputs = []
        nfm_interaction_outputs = []
        nfm_weighted_embeddings = []
        secondary_activations = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing complete pipeline"):
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
                        if self.activation_method == "max":
                            mean_primary_features = torch.max(primary_features[b, :seq_len_actual, :], dim=0)[0]
                            mean_primary_reconstruction = torch.max(primary_reconstruction[b, :seq_len_actual, :], dim=0)[0]
                            mean_nfm_linear = torch.max(nfm_linear_output[b, :seq_len_actual, :], dim=0)[0]
                            mean_nfm_interaction = torch.max(nfm_interaction_output[b, :seq_len_actual, :], dim=0)[0]
                            mean_weighted_emb = torch.max(weighted_embeddings[b, :seq_len_actual, :], dim=0)[0]
                        else:
                            mean_primary_features = torch.mean(primary_features[b, :seq_len_actual, :], dim=0)
                            mean_primary_reconstruction = torch.mean(primary_reconstruction[b, :seq_len_actual, :], dim=0)
                            mean_nfm_linear = torch.mean(nfm_linear_output[b, :seq_len_actual, :], dim=0)
                            mean_nfm_interaction = torch.mean(nfm_interaction_output[b, :seq_len_actual, :], dim=0)
                            mean_weighted_emb = torch.mean(weighted_embeddings[b, :seq_len_actual, :], dim=0)
                        
                        start_idx = b * seq_len_actual
                        end_idx = start_idx + seq_len_actual
                        if end_idx <= secondary_features.shape[0]:
                            if self.activation_method == "max":
                                mean_secondary_features = torch.max(secondary_features[start_idx:end_idx, :], dim=0)[0]
                            else:
                                mean_secondary_features = torch.mean(secondary_features[start_idx:end_idx, :], dim=0)
                        else:
                            secondary_features_reshaped = secondary_features.view(batch_size_inner, seq_len, -1)
                            if self.activation_method == "max":
                                mean_secondary_features = torch.max(secondary_features_reshaped[b, :seq_len_actual, :], dim=0)[0]
                            else:
                                mean_secondary_features = torch.mean(secondary_features_reshaped[b, :seq_len_actual, :], dim=0)
                        
                        primary_activations.append(mean_primary_features.cpu().numpy())
                        primary_reconstructions.append(mean_primary_reconstruction.cpu().numpy())
                        nfm_linear_outputs.append(mean_nfm_linear.cpu().numpy())
                        nfm_interaction_outputs.append(mean_nfm_interaction.cpu().numpy())
                        nfm_weighted_embeddings.append(mean_weighted_emb.cpu().numpy())
                        secondary_activations.append(mean_secondary_features.cpu().numpy())
        
        return (np.array(primary_activations), np.array(primary_reconstructions), 
                np.array(nfm_linear_outputs), np.array(nfm_interaction_outputs), 
                np.array(nfm_weighted_embeddings), np.array(secondary_activations))
    
    def discover_top_primary_features(self, stimulus_sets, n_features=5, batch_size=16):
        print("\n=== PRIMARY FEATURE DISCOVERY PHASE ===")
        print(f"Discovering top {n_features} Primary SAE features for each category...")
        
        all_texts = []
        text_labels = []
        
        for condition, texts in stimulus_sets.items():
            all_texts.extend(texts)
            text_labels.extend([condition] * len(texts))
        
        (primary_activations, _, _, _, _, _) = self.get_complete_pipeline_activations(all_texts, batch_size)
        
        num_primary_features = primary_activations.shape[1]
        
        f1_high_mask = np.array([label in ['F1_high_F2_high', 'F1_high_F2_low'] for label in text_labels])
        f1_low_mask = np.array([label in ['F1_low_F2_high', 'F1_low_F2_low'] for label in text_labels])
        
        f1_high_activations = primary_activations[f1_high_mask]
        f1_low_activations = primary_activations[f1_low_mask]
        
        f1_t_stats = []
        f1_p_values = []
        
        for feat_idx in range(num_primary_features):
            t_stat, p_value = stats.ttest_ind(f1_high_activations[:, feat_idx], 
                                             f1_low_activations[:, feat_idx])
            f1_t_stats.append(abs(t_stat))
            f1_p_values.append(p_value)
        
        f1_top_indices = np.argsort(f1_t_stats)[-n_features:][::-1]
        
        f2_high_mask = np.array([label in ['F1_high_F2_high', 'F1_low_F2_high'] for label in text_labels])
        f2_low_mask = np.array([label in ['F1_high_F2_low', 'F1_low_F2_low'] for label in text_labels])
        
        f2_high_activations = primary_activations[f2_high_mask]
        f2_low_activations = primary_activations[f2_low_mask]
        
        f2_t_stats = []
        f2_p_values = []
        
        for feat_idx in range(num_primary_features):
            t_stat, p_value = stats.ttest_ind(f2_high_activations[:, feat_idx], 
                                             f2_low_activations[:, feat_idx])
            f2_t_stats.append(abs(t_stat))
            f2_p_values.append(p_value)
        
        f2_top_indices = np.argsort(f2_t_stats)[-n_features:][::-1]
        
        discovery_stats = {'feature1_stats': {}, 'feature2_stats': {}}
        
        print(f"\n=== TOP {n_features} PRIMARY FEATURES FOR FEATURE 1 ({feature1name} vs {feature1bname}) ===")
        for rank, idx in enumerate(f1_top_indices):
            high_vals = f1_high_activations[:, idx]
            low_vals = f1_low_activations[:, idx]
            
            stats_dict = {
                'feature_id': int(idx),
                't_statistic': float(f1_t_stats[idx]),
                'p_value': float(f1_p_values[idx]),
                'high_mean': float(np.mean(high_vals)),
                'high_std': float(np.std(high_vals)),
                'low_mean': float(np.mean(low_vals)),
                'low_std': float(np.std(low_vals))
            }
            
            discovery_stats['feature1_stats'][f'rank_{rank+1}'] = stats_dict
            
            print(f"\nRank {rank+1}: Primary Feature {idx}")
            print(f"  t-statistic: {stats_dict['t_statistic']:.4f}, p-value: {stats_dict['p_value']:.2e}")
            print(f"  High ({feature1name}): {stats_dict['high_mean']:.4f} ± {stats_dict['high_std']:.4f}")
            print(f"  Low ({feature1bname}): {stats_dict['low_mean']:.4f} ± {stats_dict['low_std']:.4f}")
        
        print(f"\n=== TOP {n_features} PRIMARY FEATURES FOR FEATURE 2 ({feature2name} vs {feature2bname}) ===")
        for rank, idx in enumerate(f2_top_indices):
            high_vals = f2_high_activations[:, idx]
            low_vals = f2_low_activations[:, idx]
            
            stats_dict = {
                'feature_id': int(idx),
                't_statistic': float(f2_t_stats[idx]),
                'p_value': float(f2_p_values[idx]),
                'high_mean': float(np.mean(high_vals)),
                'high_std': float(np.std(high_vals)),
                'low_mean': float(np.mean(low_vals)),
                'low_std': float(np.std(low_vals))
            }
            
            discovery_stats['feature2_stats'][f'rank_{rank+1}'] = stats_dict
            
            print(f"\nRank {rank+1}: Primary Feature {idx}")
            print(f"  t-statistic: {stats_dict['t_statistic']:.4f}, p-value: {stats_dict['p_value']:.2e}")
            print(f"  High ({feature2name}): {stats_dict['high_mean']:.4f} ± {stats_dict['high_std']:.4f}")
            print(f"  Low ({feature2bname}): {stats_dict['low_mean']:.4f} ± {stats_dict['low_std']:.4f}")
        
        return list(f1_top_indices), list(f2_top_indices), discovery_stats
    
    def discover_top_secondary_features(self, stimulus_sets, n_features=5, batch_size=16):
        print("\n=== SECONDARY FEATURE DISCOVERY PHASE ===")
        print(f"Discovering top {n_features} Secondary SAE features for each category...")
        
        all_texts = []
        text_labels = []
        
        for condition, texts in stimulus_sets.items():
            all_texts.extend(texts)
            text_labels.extend([condition] * len(texts))
        
        (_, _, _, _, _, secondary_activations) = self.get_complete_pipeline_activations(all_texts, batch_size)
        
        num_secondary_features = secondary_activations.shape[1]
        
        f1_high_mask = np.array([label in ['F1_high_F2_high', 'F1_high_F2_low'] for label in text_labels])
        f1_low_mask = np.array([label in ['F1_low_F2_high', 'F1_low_F2_low'] for label in text_labels])
        
        f1_high_secondary = secondary_activations[f1_high_mask]
        f1_low_secondary = secondary_activations[f1_low_mask]
        
        f1_secondary_t_stats = []
        f1_secondary_p_values = []
        
        for feat_idx in range(num_secondary_features):
            t_stat, p_value = stats.ttest_ind(f1_high_secondary[:, feat_idx], 
                                             f1_low_secondary[:, feat_idx])
            f1_secondary_t_stats.append(abs(t_stat))
            f1_secondary_p_values.append(p_value)
        
        f1_secondary_top_indices = np.argsort(f1_secondary_t_stats)[-n_features:][::-1]
        
        f2_high_mask = np.array([label in ['F1_high_F2_high', 'F1_low_F2_high'] for label in text_labels])
        f2_low_mask = np.array([label in ['F1_high_F2_low', 'F1_low_F2_low'] for label in text_labels])
        
        f2_high_secondary = secondary_activations[f2_high_mask]
        f2_low_secondary = secondary_activations[f2_low_mask]
        
        f2_secondary_t_stats = []
        f2_secondary_p_values = []
        
        for feat_idx in range(num_secondary_features):
            t_stat, p_value = stats.ttest_ind(f2_high_secondary[:, feat_idx], 
                                             f2_low_secondary[:, feat_idx])
            f2_secondary_t_stats.append(abs(t_stat))
            f2_secondary_p_values.append(p_value)
        
        f2_secondary_top_indices = np.argsort(f2_secondary_t_stats)[-n_features:][::-1]
        
        secondary_discovery_stats = {'feature1_stats': {}, 'feature2_stats': {}}
        
        print(f"\n=== TOP {n_features} SECONDARY FEATURES FOR FEATURE 1 ({feature1name} vs {feature1bname}) ===")
        for rank, idx in enumerate(f1_secondary_top_indices):
            high_vals = f1_high_secondary[:, idx]
            low_vals = f1_low_secondary[:, idx]
            
            stats_dict = {
                'feature_id': int(idx),
                't_statistic': float(f1_secondary_t_stats[idx]),
                'p_value': float(f1_secondary_p_values[idx]),
                'high_mean': float(np.mean(high_vals)),
                'high_std': float(np.std(high_vals)),
                'low_mean': float(np.mean(low_vals)),
                'low_std': float(np.std(low_vals))
            }
            
            secondary_discovery_stats['feature1_stats'][f'rank_{rank+1}'] = stats_dict
            
            print(f"\nRank {rank+1}: Secondary Feature {idx}")
            print(f"  t-statistic: {stats_dict['t_statistic']:.4f}, p-value: {stats_dict['p_value']:.2e}")
            print(f"  High ({feature1name}): {stats_dict['high_mean']:.4f} ± {stats_dict['high_std']:.4f}")
            print(f"  Low ({feature1bname}): {stats_dict['low_mean']:.4f} ± {stats_dict['low_std']:.4f}")
        
        print(f"\n=== TOP {n_features} SECONDARY FEATURES FOR FEATURE 2 ({feature2name} vs {feature2bname}) ===")
        for rank, idx in enumerate(f2_secondary_top_indices):
            high_vals = f2_high_secondary[:, idx]
            low_vals = f2_low_secondary[:, idx]
            
            stats_dict = {
                'feature_id': int(idx),
                't_statistic': float(f2_secondary_t_stats[idx]),
                'p_value': float(f2_secondary_p_values[idx]),
                'high_mean': float(np.mean(high_vals)),
                'high_std': float(np.std(high_vals)),
                'low_mean': float(np.mean(low_vals)),
                'low_std': float(np.std(low_vals))
            }
            
            secondary_discovery_stats['feature2_stats'][f'rank_{rank+1}'] = stats_dict
            
            print(f"\nRank {rank+1}: Secondary Feature {idx}")
            print(f"  t-statistic: {stats_dict['t_statistic']:.4f}, p-value: {stats_dict['p_value']:.2e}")
            print(f"  High ({feature2name}): {stats_dict['high_mean']:.4f} ± {stats_dict['high_std']:.4f}")
            print(f"  Low ({feature2bname}): {stats_dict['low_mean']:.4f} ± {stats_dict['low_std']:.4f}")
        
        return list(f1_secondary_top_indices), list(f2_secondary_top_indices), secondary_discovery_stats

def create_comprehensive_plots(results, primary_feature1_indices, primary_feature2_indices, 
                              secondary_feature1_indices, secondary_feature2_indices, output_dir):
    plt.style.use('default')
    sns.set_palette("husl")
    
    conditions = ['F1_high_F2_high', 'F1_high_F2_low', 'F1_low_F2_high', 'F1_low_F2_low']
    condition_labels = ['[1,1] '+feature1name+'+'+feature2name, '[1,0] '+feature1name+'+'+feature2bname, 
                       '[0,1] '+feature1bname+'+'+feature2name, '[0,0] '+feature1bname+'+'+feature2bname]
    
    fig1, axes1 = plt.subplots(2, 2, figsize=(15, 12))
    fig1.suptitle('Primary SAE Feature Analysis', fontsize=16, fontweight='bold')
    
    primary_data = []
    all_primary_indices = list(set(primary_feature1_indices + primary_feature2_indices))
    
    for cond in conditions:
        for feat_idx in all_primary_indices:
            for val in results[cond]['primary_activations'][feat_idx]:
                primary_data.append({
                    'Condition': condition_labels[conditions.index(cond)],
                    'Feature': f'P{feat_idx}',
                    'Value': val
                })
    
    df_primary = pd.DataFrame(primary_data)
    sns.boxplot(data=df_primary, x='Condition', y='Value', hue='Feature', ax=axes1[0, 0])
    axes1[0, 0].set_title('Primary SAE Activations - All Conditions')
    axes1[0, 0].tick_params(axis='x', rotation=45)
    
    linear_data = []
    for cond in conditions:
        for val in results[cond]['nfm_linear_outputs']:
            linear_data.append({
                'Condition': condition_labels[conditions.index(cond)],
                'Value': val
            })
    
    df_linear = pd.DataFrame(linear_data)
    sns.boxplot(data=df_linear, x='Condition', y='Value', ax=axes1[0, 1])
    axes1[0, 1].set_title('NFM Linear Output')
    axes1[0, 1].tick_params(axis='x', rotation=45)
    
    interaction_data = []
    for cond in conditions:
        for val in results[cond]['nfm_interaction_outputs']:
            interaction_data.append({
                'Condition': condition_labels[conditions.index(cond)],
                'Value': val
            })
    
    df_interaction = pd.DataFrame(interaction_data)
    sns.boxplot(data=df_interaction, x='Condition', y='Value', ax=axes1[1, 0])
    axes1[1, 0].set_title('NFM Interaction Output')
    axes1[1, 0].tick_params(axis='x', rotation=45)
    
# NFM Weighted Embeddings
    embedding_data = []
    for cond in conditions:
        for val in results[cond]['nfm_weighted_embeddings']:
            embedding_data.append({
                'Condition': condition_labels[conditions.index(cond)],
                'Value': val
            })
    
    df_embedding = pd.DataFrame(embedding_data)
    sns.boxplot(data=df_embedding, x='Condition', y='Value', ax=axes1[1, 1])
    axes1[1, 1].set_title('NFM Weighted Embeddings')
    axes1[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plot1_path = output_path / "primary_sae_nfm_analysis.png"
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create figure for Secondary SAE Analysis
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 12))
    fig2.suptitle('Secondary SAE Feature Analysis', fontsize=16, fontweight='bold')
    
    # Secondary SAE - Interaction conditions
    secondary_data = []
    all_secondary_indices = list(set(secondary_feature1_indices + secondary_feature2_indices))
    
    for cond in conditions:
        for feat_idx in all_secondary_indices:
            for val in results[cond]['secondary_activations'][feat_idx]:
                secondary_data.append({
                    'Condition': condition_labels[conditions.index(cond)],
                    'Feature': f'S{feat_idx}',
                    'Value': val
                })
    
    df_secondary = pd.DataFrame(secondary_data)
    sns.boxplot(data=df_secondary, x='Condition', y='Value', hue='Feature', ax=axes2[0, 0])
    axes2[0, 0].set_title('Secondary SAE Activations - All Conditions')
    axes2[0, 0].tick_params(axis='x', rotation=45)
    
    # Secondary SAE - Feature categories
    secondary_cat_data = []
    
    # Category 1 Secondary features
    for feat_idx in secondary_feature1_indices:
        high_values = []
        high_values.extend(results['F1_high_F2_high']['secondary_activations'][feat_idx])
        high_values.extend(results['F1_high_F2_low']['secondary_activations'][feat_idx])
        
        low_values = []
        low_values.extend(results['F1_low_F2_high']['secondary_activations'][feat_idx])
        low_values.extend(results['F1_low_F2_low']['secondary_activations'][feat_idx])
        
        for val in high_values:
            secondary_cat_data.append({'Category': f'Sec-Cat1 ({feature1name})', 'Level': 'High', 'Value': val})
        for val in low_values:
            secondary_cat_data.append({'Category': f'Sec-Cat1 ({feature1name})', 'Level': 'Low', 'Value': val})
    
    # Category 2 Secondary features
    for feat_idx in secondary_feature2_indices:
        high_values = []
        high_values.extend(results['F1_high_F2_high']['secondary_activations'][feat_idx])
        high_values.extend(results['F1_low_F2_high']['secondary_activations'][feat_idx])
        
        low_values = []
        low_values.extend(results['F1_high_F2_low']['secondary_activations'][feat_idx])
        low_values.extend(results['F1_low_F2_low']['secondary_activations'][feat_idx])
        
        for val in high_values:
            secondary_cat_data.append({'Category': f'Sec-Cat2 ({feature2name})', 'Level': 'High', 'Value': val})
        for val in low_values:
            secondary_cat_data.append({'Category': f'Sec-Cat2 ({feature2name})', 'Level': 'Low', 'Value': val})
    
    df_secondary_cat = pd.DataFrame(secondary_cat_data)
    sns.boxplot(data=df_secondary_cat, x='Category', y='Value', hue='Level', ax=axes2[0, 1])
    axes2[0, 1].set_title('Secondary SAE - Feature Categories')
    axes2[0, 1].tick_params(axis='x', rotation=45)
    
    # Summary: Primary vs Secondary Feature Counts
    summary_data = []
    for cond in conditions:
        primary_active = sum(1 for feat_idx in all_primary_indices 
                           for val in results[cond]['primary_activations'][feat_idx] if val > 0)
        
        secondary_active = sum(1 for feat_idx in all_secondary_indices
                             for val in results[cond]['secondary_activations'][feat_idx] if val > 0)
        
        summary_data.append({
            'Condition': condition_labels[conditions.index(cond)],
            'Type': 'Primary SAE',
            'Active Features': primary_active
        })
        summary_data.append({
            'Condition': condition_labels[conditions.index(cond)],
            'Type': 'Secondary SAE', 
            'Active Features': secondary_active
        })
    
    df_summary = pd.DataFrame(summary_data)
    sns.barplot(data=df_summary, x='Condition', y='Active Features', hue='Type', ax=axes2[1, 0])
    axes2[1, 0].set_title('Active Feature Counts: Primary vs Secondary SAE')
    axes2[1, 0].tick_params(axis='x', rotation=45)
    
    # Pipeline overview
    pipeline_data = []
    for cond in conditions:
        pipeline_data.append({
            'Condition': condition_labels[conditions.index(cond)],
            'Stage': 'Primary SAE',
            'Value': np.mean([np.mean(results[cond]['primary_activations'][feat_idx]) 
                             for feat_idx in all_primary_indices])
        })
        pipeline_data.append({
            'Condition': condition_labels[conditions.index(cond)],
            'Stage': 'Secondary SAE',
            'Value': np.mean([np.mean(results[cond]['secondary_activations'][feat_idx]) 
                             for feat_idx in all_secondary_indices])
        })
    
    df_pipeline = pd.DataFrame(pipeline_data)
    sns.barplot(data=df_pipeline, x='Condition', y='Value', hue='Stage', ax=axes2[1, 1])
    axes2[1, 1].set_title('Pipeline Stage Activations')
    axes2[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    plot2_path = output_path / "secondary_sae_analysis.png"
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Primary SAE + NFM analysis plot saved to: {plot1_path}")
    print(f"Secondary SAE analysis plot saved to: {plot2_path}")

def main():
    parser = argparse.ArgumentParser(description="Complete Pipeline Stimulus-Response Analysis with Secondary SAE")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base LLM model")
    parser.add_argument("--primary_sae_path", type=str, required=True, help="Path to trained Primary SAE model")
    parser.add_argument("--nfm_path", type=str, required=True, help="Path to trained NFM model")
    parser.add_argument("--secondary_sae_path", type=str, required=True, help="Path to trained Secondary SAE model")
    parser.add_argument("--output_dir", type=str, default="./complete_pipeline_results", 
                        help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--n_features", type=int, default=5, help="Number of top features to discover per category")
    parser.add_argument("--activation_method", type=str, default="mean", choices=["mean", "max"],
                    help="Method to aggregate activations across sequence: 'mean' or 'max' (default: mean)")
    
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
    
    print(f"NFM: {num_features} features → {k_dim} embedding dim")
    
    if hasattr(secondary_sae, 'k'):
        secondary_total = secondary_sae.encoder[0].out_features
        secondary_sparsity = (secondary_sae.k / secondary_total) * 100
        print(f"Secondary TopK SAE: {secondary_sae.encoder[0].in_features} → {secondary_total} (K={secondary_sae.k}, {secondary_sparsity:.2f}% active)")
    
    # Initialize analyzer
    analyzer = CompletePipelineAnalyzer(primary_sae, nfm_model, secondary_sae, tokenizer, base_model, args.device, activation_method=args.activation_method)
    
    # PRIMARY FEATURE DISCOVERY PHASE
    primary_feature1_indices, primary_feature2_indices, primary_discovery_stats = analyzer.discover_top_primary_features(
        STIMULUS_SETS, n_features=args.n_features, batch_size=args.batch_size
    )
    
    # SECONDARY FEATURE DISCOVERY PHASE
    secondary_feature1_indices, secondary_feature2_indices, secondary_discovery_stats = analyzer.discover_top_secondary_features(
        STIMULUS_SETS, n_features=args.n_features, batch_size=args.batch_size
    )
    
    # Save discovery statistics
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    primary_discovery_path = output_path / "primary_feature_discovery_stats.json"
    with open(primary_discovery_path, 'w') as f:
        json.dump(primary_discovery_stats, f, indent=2)
    
    secondary_discovery_path = output_path / "secondary_feature_discovery_stats.json"
    with open(secondary_discovery_path, 'w') as f:
        json.dump(secondary_discovery_stats, f, indent=2)
    
    print(f"\nPrimary feature discovery stats saved to: {primary_discovery_path}")
    print(f"Secondary feature discovery stats saved to: {secondary_discovery_path}")
    
    # Combine all discovered features for complete analysis
    all_primary_indices = list(set(primary_feature1_indices + primary_feature2_indices))
    all_secondary_indices = list(set(secondary_feature1_indices + secondary_feature2_indices))
    
    print(f"\n=== RUNNING COMPLETE PIPELINE ANALYSIS ===")
    print(f"Primary Feature 1 indices: {primary_feature1_indices}")
    print(f"Primary Feature 2 indices: {primary_feature2_indices}")
    print(f"Secondary Feature 1 indices: {secondary_feature1_indices}")
    print(f"Secondary Feature 2 indices: {secondary_feature2_indices}")
    
    # Store results
    results = {}
    
    # Process each stimulus condition
    for condition_name, texts in STIMULUS_SETS.items():
        print(f"\nProcessing condition: {condition_name}")
        
        # Get complete pipeline activations
        (primary_activations, primary_reconstructions, nfm_linear_outputs, 
         nfm_interaction_outputs, nfm_weighted_embeddings, secondary_activations) = analyzer.get_complete_pipeline_activations(texts, args.batch_size)
        
        # Store activations by feature for discovered features only
        primary_activations_by_feature = {}
        for feat_idx in all_primary_indices:
            primary_activations_by_feature[feat_idx] = primary_activations[:, feat_idx].tolist()
        
        secondary_activations_by_feature = {}
        for feat_idx in all_secondary_indices:
            secondary_activations_by_feature[feat_idx] = secondary_activations[:, feat_idx].tolist()
        
        # Store in results
        results[condition_name] = {
            'primary_activations': primary_activations_by_feature,
            'secondary_activations': secondary_activations_by_feature,
            'nfm_linear_outputs': np.mean(np.abs(nfm_linear_outputs), axis=1).tolist(),
            'nfm_interaction_outputs': np.mean(np.abs(nfm_interaction_outputs), axis=1).tolist(),
            'nfm_weighted_embeddings': np.mean(np.abs(nfm_weighted_embeddings), axis=1).tolist()
        }
    
    # Print summary statistics
    print("\n" + "="*80)
    print("COMPLETE PIPELINE SUMMARY STATISTICS")
    print("="*80)
    
    for condition_name in STIMULUS_SETS.keys():
        print(f"\n{condition_name}:")
        print(f"  NFM Linear Output: {np.mean(results[condition_name]['nfm_linear_outputs']):.4f} ± {np.std(results[condition_name]['nfm_linear_outputs']):.4f}")
        print(f"  NFM Interaction Output: {np.mean(results[condition_name]['nfm_interaction_outputs']):.4f} ± {np.std(results[condition_name]['nfm_interaction_outputs']):.4f}")
        print(f"  NFM Weighted Embeddings: {np.mean(results[condition_name]['nfm_weighted_embeddings']):.4f} ± {np.std(results[condition_name]['nfm_weighted_embeddings']):.4f}")
        
        # Primary SAE statistics
        print(f"  Primary SAE features:")
        for feat_idx in all_primary_indices:
            values = results[condition_name]['primary_activations'][feat_idx]
            category = "Cat1" if feat_idx in primary_feature1_indices else "Cat2"
            print(f"    P{feat_idx} [{category}]: {np.mean(values):.4f} ± {np.std(values):.4f}")
        
        # Secondary SAE statistics
        print(f"  Secondary SAE features:")
        for feat_idx in all_secondary_indices:
            values = results[condition_name]['secondary_activations'][feat_idx]
            category = "Cat1" if feat_idx in secondary_feature1_indices else "Cat2"
            print(f"    S{feat_idx} [{category}]: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    # Create comprehensive visualizations
    print("\nCreating comprehensive analysis plots...")
    create_comprehensive_plots(results, primary_feature1_indices, primary_feature2_indices,
                              secondary_feature1_indices, secondary_feature2_indices, args.output_dir)
    
    # Save detailed results
    results_path = output_path / "complete_pipeline_detailed_results.json"
    
    # Convert results to JSON-serializable format
    json_results = {}
    for condition_name, condition_data in results.items():
        json_results[condition_name] = {}
        for metric_name, metric_values in condition_data.items():
            if isinstance(metric_values, dict):
                json_results[condition_name][metric_name] = {str(k): [float(v) for v in val_list]
                                                               for k, val_list in metric_values.items()}
            elif isinstance(metric_values, list):
                json_results[condition_name][metric_name] = [float(v) for v in metric_values]
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Save feature mapping for reference
    feature_mapping = {
        'primary_features': {
            'feature1_indices': primary_feature1_indices,
            'feature2_indices': primary_feature2_indices,
            'feature1_name': feature1name,
            'feature2_name': feature2name
        },
        'secondary_features': {
            'feature1_indices': secondary_feature1_indices,
            'feature2_indices': secondary_feature2_indices,
            'feature1_name': feature1name,
            'feature2_name': feature2name
        },
        'analysis_settings': {
            'n_features': args.n_features,
            'activation_method': args.activation_method,
            'batch_size': args.batch_size
        }
    }
    
    feature_mapping_path = output_path / "feature_mapping.json"
    with open(feature_mapping_path, 'w') as f:
        json.dump(feature_mapping, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_path}")
    print(f"Feature mapping saved to: {feature_mapping_path}")
    
    # Summary of analysis
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Discovered Primary Features:")
    print(f"  Category 1 ({feature1name}): {primary_feature1_indices}")
    print(f"  Category 2 ({feature2name}): {primary_feature2_indices}")
    print(f"Discovered Secondary Features:")
    print(f"  Category 1 ({feature1name}): {secondary_feature1_indices}")
    print(f"  Category 2 ({feature2name}): {secondary_feature2_indices}")
    print(f"\nResults and plots saved to: {args.output_dir}")
    
    # Calculate pipeline efficiency metrics
    total_primary_features = primary_sae.encoder[0].out_features if hasattr(primary_sae, 'encoder') else 0
    total_secondary_features = secondary_sae.encoder[0].out_features if hasattr(secondary_sae, 'encoder') else 0
    
    if total_primary_features > 0:
        primary_selectivity = len(all_primary_indices) / total_primary_features * 100
        print(f"\nPipeline Efficiency:")
        print(f"  Primary SAE selectivity: {len(all_primary_indices)}/{total_primary_features} = {primary_selectivity:.2f}%")
    
    if total_secondary_features > 0:
        secondary_selectivity = len(all_secondary_indices) / total_secondary_features * 100
        print(f"  Secondary SAE selectivity: {len(all_secondary_indices)}/{total_secondary_features} = {secondary_selectivity:.2f}%")
    
    print(f"\nNext steps:")
    print(f"1. Examine the discovered feature patterns in the plots")
    print(f"2. Run part11c analysis on specific Primary feature pairs")
    print(f"3. Use part6_secondary_sae_interaction_finder.py to find interacting features")

if __name__ == "__main__":
    main()