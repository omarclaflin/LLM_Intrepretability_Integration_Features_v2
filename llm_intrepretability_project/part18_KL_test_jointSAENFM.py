"""
part18_KL_test_jointSAENFM.py

KL Divergence Test for Joint SAE+NFM Architecture vs. Feature Integration Hypothesis
Using WikiText-103 Dataset with Large Sample Sizes

This script tests the core hypothesis from Gurnee and Claflin papers using the new joint
end-to-end trained SAE+NFM architecture from part15_joint_SAENFM_training.py by comparing:
1. KL divergence (original, ε-random substitution)
2. KL divergence (original, Joint SAE+NFM)  
3. KL divergence (original, SAE only pathway)

Uses WikiText-103 dataset with proper chunking to match Gurnee's scale.
Pipeline interventions:
- ε-random: Layer 16 → Random vector at same L2 distance as Joint error
- Joint SAE+NFM: Layer 16 → Joint Model → Primary + Linear + Interaction reconstruction
- SAE only: Layer 16 → Joint Model SAE component only → Primary reconstruction only

Tests whether joint end-to-end training reduces pathological KL gap even further.
"""

import torch
import numpy as np
import argparse
import pandas as pd
from pathlib import Path
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
import logging
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

import torch.nn.functional as F
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import random

class TopKSparseAutoencoder(torch.nn.Module):
    """TopK Sparse Autoencoder module from joint training."""
    def __init__(self, input_dim, hidden_dim, k=None):
        super().__init__()
        self.k = k
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Linear(hidden_dim, input_dim, bias=False)

    def forward(self, x):
        features = self.encoder(x)
        if self.k is not None:
            sparse_features = self.apply_topk(features)
        else:
            sparse_features = features  # No top-k filtering
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

class NeuralFactorizationMachine(torch.nn.Module):
    """NFM from joint training."""
    def __init__(self, num_sae_features, embedding_dim, output_dim, use_linear_component=True, nfm_dropout=0.15):
        super().__init__()
        self.num_sae_features = num_sae_features
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        
        self.feature_embeddings = torch.nn.Embedding(num_sae_features, embedding_dim)
        self.linear = torch.nn.Linear(num_sae_features, output_dim, bias=True) if use_linear_component else None
        
        self.interaction_mlp = torch.nn.Sequential(
            torch.nn.Dropout(nfm_dropout),
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dim, output_dim)
        )
    
    def forward(self, sae_features):
        """Forward pass matching joint training architecture."""
        # Compute interaction vector using all embeddings (dense computation)
        all_embeddings = self.feature_embeddings.weight
        weighted_embeddings = sae_features.unsqueeze(-1) * all_embeddings.unsqueeze(0)
        
        sum_embeddings = torch.sum(weighted_embeddings, dim=1)
        square_embeddings = weighted_embeddings ** 2
        sum_squares = torch.sum(square_embeddings, dim=1)
        
        interaction_vector = 0.5 * (sum_embeddings ** 2 - sum_squares)
        interaction_out = self.interaction_mlp(interaction_vector)
        
        linear_out = None
        if self.linear is not None:
            linear_out = self.linear(sae_features)
            output = linear_out + interaction_out
        else:
            output = interaction_out
        
        return output, linear_out, interaction_out

class JointSAENFM(torch.nn.Module):
    """Joint SAE+NFM model from part15."""
    def __init__(self, input_dim, sae_features, sae_k, nfm_embedding_dim, use_linear_component=True):
        super().__init__()
        self.primary_sae = TopKSparseAutoencoder(input_dim, sae_features, sae_k)
        self.nfm = NeuralFactorizationMachine(sae_features, nfm_embedding_dim, input_dim, use_linear_component)
        
    def forward(self, layer_16_activations):
        """
        Joint forward pass: Layer 16 → SAE → NFM → Final Reconstruction
        
        Returns:
            final_reconstruction: Combined output from all three pathways
            primary_features: SAE features (for analysis)
            primary_recon: SAE reconstruction only
            linear_out: NFM linear component
            interaction_out: NFM interaction component
        """
        # Primary SAE path
        primary_features, primary_recon = self.primary_sae(layer_16_activations)
        
        # NFM path (operating on SAE features)
        nfm_output, linear_out, interaction_out = self.nfm(primary_features)
        
        # THREE-WAY RESIDUAL COMBINATION
        final_reconstruction = primary_recon + nfm_output
        
        return final_reconstruction, primary_features, primary_recon, linear_out, interaction_out

class JointKLDivergenceAnalyzer:
    """Analyzer for KL divergence comparison using Joint SAE+NFM architecture."""
    
    def __init__(self, joint_model, tokenizer, base_model, 
                 device="cuda", target_layer=16, max_token_length=128):
        self.joint_model = joint_model
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.device = device
        self.target_layer = target_layer
        self.max_token_length = max_token_length
        
        # Set models to eval mode
        self.joint_model.eval()
        self.base_model.eval()
        
        # Get vocabulary size for KL calculation
        self.vocab_size = len(tokenizer.vocab)
        print(f"Vocabulary size: {self.vocab_size}")

    def load_wiki_dataset(self, num_samples=50000):
        """
        Load WikiText-103 dataset with proper chunking like part14b.
        
        Args:
            num_samples: Number of text chunks to extract
        
        Returns:
            List of text chunks
        """
        print(f"Loading WikiText-103 dataset with chunking (target: {num_samples} samples)...")
        
        try:
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
            
            max_indices_to_check = min(len(dataset), num_samples * 3)
            
            filtered_texts = []
            for idx in tqdm(range(max_indices_to_check), desc="Processing WikiText", leave=False):
                text = dataset[idx]["text"]
                if text.strip():
                    full_inputs = self.tokenizer(text, return_tensors="pt", truncation=False)
                    full_length = torch.sum(full_inputs["attention_mask"][0]).item()
                    
                    if full_length > self.max_token_length:
                        # Chunk long text
                        clean_text = self.tokenizer.decode(full_inputs["input_ids"][0], skip_special_tokens=True)
                        tokens = self.tokenizer.encode(clean_text, add_special_tokens=False)
                        
                        for chunk_start in range(0, len(tokens), self.max_token_length):
                            chunk_tokens = tokens[chunk_start:chunk_start + self.max_token_length]
                            if len(chunk_tokens) >= self.max_token_length:
                                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                                filtered_texts.append(chunk_text)
                                
                                if len(filtered_texts) >= num_samples:
                                    break
                    else:
                        # Short text, truncate normally
                        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_token_length)
                        if torch.sum(inputs["attention_mask"][0]).item() >= self.max_token_length:
                            truncated_text = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
                            filtered_texts.append(truncated_text)
                            
                            if len(filtered_texts) >= num_samples:
                                break
                
                if len(filtered_texts) >= num_samples:
                    break

            print(f"Loaded {len(filtered_texts)} text chunks from WikiText-103 (max_token_length={self.max_token_length})")
            return filtered_texts

        except Exception as e:
            print(f"Error loading WikiText-103 dataset: {e}")
            # Fallback to simple prompts
            return [
                "I think that",
                "The situation is", 
                "My opinion on this is",
                "It seems to me that",
                "The best approach would be",
                "When considering the problem",
                "The analysis shows that",
                "Based on the evidence"
            ]

    def generate_epsilon_random(self, original_activation, joint_reconstruction):
        """
        Generate random vector at same L2 distance as Joint reconstruction error.
        
        Args:
            original_activation: Original activation vector [batch*seq, hidden]
            joint_reconstruction: Joint model reconstruction [batch*seq, hidden]
        
        Returns:
            Random vector with same L2 error distance as Joint model
        """
        # Calculate Joint reconstruction error
        joint_error = joint_reconstruction - original_activation  # [batch*seq, hidden]
        epsilon = torch.norm(joint_error, dim=-1, keepdim=True)  # [batch*seq, 1]
        
        # Generate random unit vectors
        random_direction = torch.randn_like(original_activation)  # [batch*seq, hidden]
        random_unit = F.normalize(random_direction, dim=-1)  # [batch*seq, hidden]
        
        # Scale to epsilon distance
        epsilon_random = original_activation + epsilon * random_unit  # [batch*seq, hidden]
        
        return epsilon_random

    def forward_with_original(self, input_ids, attention_mask):
        """
        Forward pass with no intervention (baseline).
        
        Returns:
            logits: Original model logits
        """
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                use_cache=False
            )
            logits = outputs.logits
        
        return logits

    def forward_with_epsilon_random(self, input_ids, attention_mask):
        """
        Forward pass with ε-random intervention at target layer.
        
        Returns:
            logits: Logits after ε-random substitution
            epsilon_distance: L2 distance used for random vector
        """
        intervention_data = {}
        
        def epsilon_random_hook(module, input, output):
            """Hook to apply ε-random intervention at layer 16."""
            layer_16_activations = output[0]  # [batch, seq, hidden]
            original_shape = layer_16_activations.shape
            layer_16_flat = layer_16_activations.view(-1, original_shape[-1])  # [batch*seq, hidden]
            
            # Get Joint model reconstruction to calculate epsilon
            joint_reconstruction, _, _, _, _ = self.joint_model(
                layer_16_flat.to(self.joint_model.primary_sae.encoder[0].weight.dtype)
            )
            
            # Generate ε-random vector at same distance as Joint error
            epsilon_random_flat = self.generate_epsilon_random(layer_16_flat, joint_reconstruction)
            
            # Store epsilon distance for reporting
            joint_error = joint_reconstruction - layer_16_flat
            epsilon_distance = torch.norm(joint_error, dim=-1).mean().item()
            intervention_data['epsilon_distance'] = epsilon_distance
            
            # Reshape back to original shape
            epsilon_random_reshaped = epsilon_random_flat.view(original_shape)
            
            # Return modified activations
            return (epsilon_random_reshaped,) + output[1:]
        
        # Register hook at target layer
        target_layer_module = self.base_model.model.layers[self.target_layer]
        hook_handle = target_layer_module.register_forward_hook(epsilon_random_hook)
        
        try:
            with torch.no_grad():
                outputs = self.base_model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    use_cache=False
                )
                logits = outputs.logits
                
        finally:
            hook_handle.remove()
        
        return logits, intervention_data.get('epsilon_distance', 0.0)

    def forward_with_sae_only(self, input_ids, attention_mask):
        """
        Forward pass with SAE component only (no NFM).
        
        Returns:
            logits: Logits after SAE-only substitution
        """
        def sae_only_hook(module, input, output):
            """Hook to apply SAE-only intervention at layer 16."""
            layer_16_activations = output[0]  # [batch, seq, hidden]
            original_shape = layer_16_activations.shape
            layer_16_flat = layer_16_activations.view(-1, original_shape[-1])  # [batch*seq, hidden]
            
            # SAE component only (no NFM)
            _, primary_features, primary_reconstruction, _, _ = self.joint_model(
                layer_16_flat.to(self.joint_model.primary_sae.encoder[0].weight.dtype)
            )
            
            # Reshape SAE reconstruction back to original shape
            sae_reconstruction_reshaped = primary_reconstruction.view(original_shape)
            
            # Return SAE reconstruction only
            return (sae_reconstruction_reshaped,) + output[1:]
        
        # Register hook at target layer
        target_layer_module = self.base_model.model.layers[self.target_layer]
        hook_handle = target_layer_module.register_forward_hook(sae_only_hook)
        
        try:
            with torch.no_grad():
                outputs = self.base_model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    use_cache=False
                )
                logits = outputs.logits
                
        finally:
            hook_handle.remove()
        
        return logits

    def forward_with_joint_sae_nfm(self, input_ids, attention_mask):
        """
        Forward pass with full Joint SAE+NFM pipeline.
        
        Returns:
            logits: Logits after Joint SAE+NFM substitution
        """
        def joint_sae_nfm_hook(module, input, output):
            """Hook to apply Joint SAE+NFM intervention at layer 16."""
            layer_16_activations = output[0]  # [batch, seq, hidden]
            original_shape = layer_16_activations.shape
            layer_16_flat = layer_16_activations.view(-1, original_shape[-1])  # [batch*seq, hidden]
            
            # Full Joint model processing
            joint_reconstruction, primary_features, primary_recon, linear_out, interaction_out = self.joint_model(
                layer_16_flat.to(self.joint_model.primary_sae.encoder[0].weight.dtype)
            )
            
            # Reshape joint reconstruction back to original shape
            joint_reconstruction_reshaped = joint_reconstruction.view(original_shape)
            
            # Return modified activations
            return (joint_reconstruction_reshaped,) + output[1:]
        
        # Register hook at target layer
        target_layer_module = self.base_model.model.layers[self.target_layer]
        hook_handle = target_layer_module.register_forward_hook(joint_sae_nfm_hook)
        
        try:
            with torch.no_grad():
                outputs = self.base_model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    use_cache=False
                )
                logits = outputs.logits
                
        finally:
            hook_handle.remove()
        
        return logits

    def calculate_kl_divergence(self, original_logits, modified_logits):
        """
        Calculate KL divergence between original and modified next-token probability distributions.
        
        Args:
            original_logits: [batch, seq, vocab] logits from original model
            modified_logits: [batch, seq, vocab] logits from modified model
        
        Returns:
            kl_divergences: KL divergence for each position [batch, seq]
        """
        # Convert logits to probabilities
        original_probs = F.softmax(original_logits, dim=-1)  # [batch, seq, vocab]
        modified_probs = F.softmax(modified_logits, dim=-1)  # [batch, seq, vocab]
        
        # Calculate KL divergence: KL(P || Q) = sum(P * log(P / Q))
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        log_ratio = torch.log((original_probs + eps) / (modified_probs + eps))
        kl_div = torch.sum(original_probs * log_ratio, dim=-1)  # [batch, seq]
        
        return kl_div

    def run_joint_kl_comparison_experiment(self, wiki_texts, num_epsilon_samples_per_text=3, batch_size=4):
        """
        Run KL divergence comparison experiment on WikiText-103 chunks using Joint architecture.
        
        Args:
            wiki_texts: List of WikiText-103 chunks
            num_epsilon_samples_per_text: Number of ε-random samples per text
            batch_size: Batch size for processing
        
        Returns:
            DataFrame with KL divergence results
        """
        print(f"\n=== JOINT SAE+NFM KL DIVERGENCE COMPARISON EXPERIMENT ===")
        print(f"Wiki texts: {len(wiki_texts)}")
        print(f"ε-random samples per text: {num_epsilon_samples_per_text}")
        print(f"Batch size: {batch_size}")
        print(f"Max token length: {self.max_token_length}")
        
        results = []
        
        for batch_start in tqdm(range(0, len(wiki_texts), batch_size), desc="Processing wiki batches"):
            batch_texts = wiki_texts[batch_start:batch_start + batch_size]
            batch_texts = [t for t in batch_texts if t.strip()]
            
            if not batch_texts:
                continue
            
            # Tokenize batch
            try:
                inputs = self.tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=self.max_token_length
                ).to(self.device)
                
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                
                # Skip if batch is too large for memory
                if input_ids.shape[0] * input_ids.shape[1] > 8192:  # Rough memory limit
                    continue
                
                # Get original logits (baseline)
                original_logits = self.forward_with_original(input_ids, attention_mask)
                
                # Get SAE-only logits
                sae_logits = self.forward_with_sae_only(input_ids, attention_mask)
                
                # Get Joint SAE+NFM logits
                joint_logits = self.forward_with_joint_sae_nfm(input_ids, attention_mask)
                
                # Calculate KL divergences for SAE and Joint
                kl_sae = self.calculate_kl_divergence(original_logits, sae_logits)
                kl_joint = self.calculate_kl_divergence(original_logits, joint_logits)
                
                # For ε-random, run multiple samples
                kl_epsilon_samples = []
                epsilon_distances = []
                
                for sample_idx in range(num_epsilon_samples_per_text):
                    epsilon_logits, epsilon_distance = self.forward_with_epsilon_random(input_ids, attention_mask)
                    kl_epsilon = self.calculate_kl_divergence(original_logits, epsilon_logits)
                    kl_epsilon_samples.append(kl_epsilon)
                    epsilon_distances.append(epsilon_distance)
                
                # Process results for each sequence position
                batch_size_actual, seq_len = kl_sae.shape
                
                for batch_idx in range(batch_size_actual):
                    # Get actual sequence length (excluding padding)
                    actual_seq_len = torch.sum(attention_mask[batch_idx]).item()
                    
                    # Get text for this batch item
                    text_content = batch_texts[batch_idx] if batch_idx < len(batch_texts) else ""
                    
                    for pos_idx in range(min(actual_seq_len, seq_len)):
                        # Get token at this position
                        token_id = input_ids[batch_idx, pos_idx].item()
                        token_text = self.tokenizer.decode([token_id])
                        
                        # SAE results
                        results.append({
                            'text_idx': batch_start + batch_idx,
                            'text_content': text_content[:100] + "..." if len(text_content) > 100 else text_content,
                            'position': pos_idx,
                            'token_id': token_id,
                            'token_text': token_text,
                            'intervention_type': 'SAE_only',
                            'kl_divergence': float(kl_sae[batch_idx, pos_idx].cpu()),
                            'sample_idx': 0,
                            'epsilon_distance': None
                        })
                        
                        # Joint SAE+NFM results
                        results.append({
                            'text_idx': batch_start + batch_idx,
                            'text_content': text_content[:100] + "..." if len(text_content) > 100 else text_content,
                            'position': pos_idx,
                            'token_id': token_id,
                            'token_text': token_text,
                            'intervention_type': 'Joint_SAE_NFM',
                            'kl_divergence': float(kl_joint[batch_idx, pos_idx].cpu()),
                            'sample_idx': 0,
                            'epsilon_distance': None
                        })
                        
                        # ε-random results (multiple samples)
                        for sample_idx, (kl_epsilon, epsilon_distance) in enumerate(zip(kl_epsilon_samples, epsilon_distances)):
                            results.append({
                                'text_idx': batch_start + batch_idx,
                                'text_content': text_content[:100] + "..." if len(text_content) > 100 else text_content,
                                'position': pos_idx,
                                'token_id': token_id,
                                'token_text': token_text,
                                'intervention_type': 'epsilon_random',
                                'kl_divergence': float(kl_epsilon[batch_idx, pos_idx].cpu()),
                                'sample_idx': sample_idx,
                                'epsilon_distance': epsilon_distance
                            })
                
            except Exception as e:
                print(f"Error processing batch starting at {batch_start}: {e}")
                continue
        
        print(f"Collected {len(results)} KL divergence measurements from WikiText-103")
        return pd.DataFrame(results)

    def analyze_joint_kl_results(self, results_df):
        """
        Analyze Joint model KL divergence results to test the pathological error hypothesis.
        
        Args:
            results_df: DataFrame from run_joint_kl_comparison_experiment
        
        Returns:
            Dictionary with analysis results
        """
        print(f"\n=== JOINT SAE+NFM KL DIVERGENCE ANALYSIS ===")
        
        # Calculate summary statistics by intervention type
        summary_stats = results_df.groupby('intervention_type')['kl_divergence'].agg([
            'count', 'mean', 'std', 'median', 'min', 'max'
        ]).round(6)
        
        print(f"\nSummary Statistics:")
        print(summary_stats)
        
        # Calculate ratios (Gurnee's key finding + Joint model performance)
        epsilon_mean = summary_stats.loc['epsilon_random', 'mean']
        sae_mean = summary_stats.loc['SAE_only', 'mean'] 
        joint_mean = summary_stats.loc['Joint_SAE_NFM', 'mean']
        
        sae_vs_epsilon_ratio = sae_mean / epsilon_mean
        joint_vs_epsilon_ratio = joint_mean / epsilon_mean
        joint_vs_sae_ratio = joint_mean / sae_mean
        
        print(f"\nKey Ratios:")
        print(f"  SAE vs ε-random: {sae_vs_epsilon_ratio:.3f}x")
        print(f"  Joint vs ε-random: {joint_vs_epsilon_ratio:.3f}x") 
        print(f"  Joint vs SAE: {joint_vs_sae_ratio:.3f}x")
        
        # Statistical tests
        epsilon_kls = results_df[results_df['intervention_type'] == 'epsilon_random']['kl_divergence']
        sae_kls = results_df[results_df['intervention_type'] == 'SAE_only']['kl_divergence']
        joint_kls = results_df[results_df['intervention_type'] == 'Joint_SAE_NFM']['kl_divergence']
        
        # T-tests
        sae_vs_epsilon_stat, sae_vs_epsilon_p = stats.ttest_ind(sae_kls, epsilon_kls)
        joint_vs_epsilon_stat, joint_vs_epsilon_p = stats.ttest_ind(joint_kls, epsilon_kls)
        joint_vs_sae_stat, joint_vs_sae_p = stats.ttest_ind(joint_kls, sae_kls)
        
        print(f"\nStatistical Tests (t-tests):")
        print(f"  SAE vs ε-random: t={sae_vs_epsilon_stat:.3f}, p={sae_vs_epsilon_p:.6f}")
        print(f"  Joint vs ε-random: t={joint_vs_epsilon_stat:.3f}, p={joint_vs_epsilon_p:.6f}")
        print(f"  Joint vs SAE: t={joint_vs_sae_stat:.3f}, p={joint_vs_sae_p:.6f}")
        
        # Interpretation
        print(f"\n=== HYPOTHESIS TESTING (JOINT ARCHITECTURE SCALE) ===")
        
        # Gurnee's pathological error hypothesis
        if sae_vs_epsilon_ratio > 1.5 and sae_vs_epsilon_p < 0.05:
            print(f"✓ GURNEE'S PATHOLOGICAL ERROR CONFIRMED: SAE errors are {sae_vs_epsilon_ratio:.1f}x worse than random")
        else:
            print(f"✗ Gurnee's pathological error NOT confirmed")
        
        # Joint end-to-end training hypothesis  
        if joint_vs_sae_ratio < 0.8 and joint_vs_sae_p < 0.05:
            print(f"✓ JOINT END-TO-END TRAINING HYPOTHESIS STRONGLY SUPPORTED: Joint reduces KL by {(1-joint_vs_sae_ratio)*100:.1f}%")
        elif joint_vs_sae_ratio < 1.0 and joint_vs_sae_p < 0.05:
            print(f"✓ JOINT END-TO-END TRAINING HYPOTHESIS SUPPORTED: Joint reduces KL by {(1-joint_vs_sae_ratio)*100:.1f}%")
        elif joint_vs_sae_ratio < 1.0:
            print(f"~ JOINT END-TO-END TRAINING HYPOTHESIS PARTIALLY SUPPORTED: Joint reduces KL by {(1-joint_vs_sae_ratio)*100:.1f}% (not significant)")
        else:
            print(f"✗ Joint end-to-end training hypothesis NOT supported")
        
        # Overall conclusion
        if joint_vs_epsilon_ratio < sae_vs_epsilon_ratio:
            improvement = ((sae_vs_epsilon_ratio - joint_vs_epsilon_ratio) / sae_vs_epsilon_ratio) * 100
            print(f"🎯 JOINT TRAINING REDUCES PATHOLOGICAL ERRORS by {improvement:.1f}%")
            print(f"   With {len(results_df):,} measurements from WikiText-103")
        else:
            print(f"❌ Joint training does not reduce pathological errors")
        
        # Compare joint vs sequential training improvement
        if joint_vs_sae_ratio < 1.0:
            joint_improvement = (1 - joint_vs_sae_ratio) * 100
            print(f"🔬 JOINT vs SAE-ONLY IMPROVEMENT: {joint_improvement:.1f}%")
            print(f"   Joint end-to-end training shows measurable benefit over SAE-only")
        
        # Sample size comparison with Gurnee
        total_measurements = len(results_df)
        gurnee_approx_size = 2_000_000 * 128  # 2M tokens × 128 seq length
        scale_ratio = total_measurements / gurnee_approx_size
        
        print(f"\nSample Size Comparison:")
        print(f"  This experiment: {total_measurements:,} measurements")
        print(f"  Gurnee's scale: ~{gurnee_approx_size:,} measurements")
        print(f"  Scale ratio: {scale_ratio:.4f}x of Gurnee's experiment")
        
        return {
            'summary_stats': summary_stats.to_dict(),
            'ratios': {
                'sae_vs_epsilon': sae_vs_epsilon_ratio,
                'joint_vs_epsilon': joint_vs_epsilon_ratio,
                'joint_vs_sae': joint_vs_sae_ratio
            },
            'statistical_tests': {
                'sae_vs_epsilon': {'statistic': sae_vs_epsilon_stat, 'p_value': sae_vs_epsilon_p},
                'joint_vs_epsilon': {'statistic': joint_vs_epsilon_stat, 'p_value': joint_vs_epsilon_p},
                'joint_vs_sae': {'statistic': joint_vs_sae_stat, 'p_value': joint_vs_sae_p}
            },
            'conclusions': {
                'gurnee_pathological_confirmed': sae_vs_epsilon_ratio > 1.5 and sae_vs_epsilon_p < 0.05,
                'joint_training_supported': joint_vs_sae_ratio < 0.8 and joint_vs_sae_p < 0.05,
                'joint_reduces_pathological': joint_vs_epsilon_ratio < sae_vs_epsilon_ratio,
                'total_measurements': total_measurements,
                'scale_vs_gurnee': scale_ratio
            }
        }

def create_joint_kl_comparison_plots(results_df, analysis_results, output_dir):
    """
    Create visualizations of Joint SAE+NFM KL divergence comparison results.
    """
    output_dir = Path(output_dir)
    plots_dir = output_dir / "joint_kl_comparison_plots"
    plots_dir.mkdir(exist_ok=True)
    
    plt.style.use('default')
    sns.set_palette("Set1")
    
    # 1. Box plot comparing distributions
    fig, ax = plt.subplots(figsize=(12, 8))
    
    intervention_order = ['epsilon_random', 'SAE_only', 'Joint_SAE_NFM']
    intervention_labels = ['ε-random', 'SAE only', 'Joint SAE+NFM']
    
    # Create box plot
    box_data = [results_df[results_df['intervention_type'] == itype]['kl_divergence'] 
                for itype in intervention_order]
    
    bp = ax.boxplot(box_data, labels=intervention_labels, patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('KL Divergence', fontsize=14)
    ax.set_title('Joint SAE+NFM KL Divergence Comparison: Testing End-to-End Training vs. Pathological Errors\n'
                 f'Scale: {len(results_df):,} measurements from real text', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add ratio annotations
    ratios = analysis_results['ratios']
    conclusions = analysis_results['conclusions']
    ax.text(0.02, 0.98, f"SAE vs ε-random: {ratios['sae_vs_epsilon']:.2f}x\n"
                         f"Joint vs ε-random: {ratios['joint_vs_epsilon']:.2f}x\n"
                         f"Joint vs SAE: {ratios['joint_vs_sae']:.2f}x\n\n"
                         f"Measurements: {conclusions['total_measurements']:,}\n"
                         f"Gurnee scale: {conclusions['scale_vs_gurnee']:.3f}x", 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    boxplot_path = plots_dir / "joint_kl_divergence_comparison_boxplot.png"
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Histogram comparison with sample sizes
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Joint SAE+NFM KL Divergence Distributions by Intervention Type', 
                 fontsize=16, fontweight='bold')
    
    for i, (itype, label, color) in enumerate(zip(intervention_order, intervention_labels, colors)):
        data = results_df[results_df['intervention_type'] == itype]['kl_divergence']
        axes[i].hist(data, bins=50, alpha=0.7, color=color, edgecolor='black')
        axes[i].set_title(f'{label}\nMean: {data.mean():.4f}\nN: {len(data):,}', fontsize=12)
        axes[i].set_xlabel('KL Divergence', fontsize=11)
        axes[i].set_ylabel('Frequency', fontsize=11)
        axes[i].grid(True, alpha=0.3)
        
        # Add statistics
        axes[i].axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.4f}')
        axes[i].axvline(data.median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {data.median():.4f}')
        axes[i].legend()
    
    plt.tight_layout()
    histogram_path = plots_dir / "joint_kl_divergence_histograms.png"
    plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Position-wise analysis
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Calculate mean KL by position for each intervention type (first 64 positions)
    max_pos = min(64, results_df['position'].max())
    position_stats = results_df[results_df['position'] <= max_pos].groupby(['intervention_type', 'position'])['kl_divergence'].mean().unstack(0)
    
    for itype, label in zip(intervention_order, intervention_labels):
        if itype in position_stats.columns:
            ax.plot(position_stats.index, position_stats[itype], 
                   marker='o', label=label, linewidth=2, markersize=3)
    
    ax.set_xlabel('Token Position in Sequence', fontsize=12)
    ax.set_ylabel('Mean KL Divergence', fontsize=12)
    ax.set_title('Joint SAE+NFM KL Divergence by Token Position\n'
                 'Shows how end-to-end training affects pathological errors across positions', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    position_path = plots_dir / "joint_kl_divergence_by_position.png"
    plt.savefig(position_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Improvement analysis plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Show improvement ratios
    improvement_data = [
        ('SAE vs Random', ratios['sae_vs_epsilon'], 'lightcoral'),
        ('Joint vs Random', ratios['joint_vs_epsilon'], 'lightgreen'),
        ('Joint vs SAE', ratios['joint_vs_sae'], 'lightyellow')
    ]
    
    labels, values, colors_imp = zip(*improvement_data)
    bars = ax.bar(labels, values, color=colors_imp, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}x', ha='center', va='bottom', fontweight='bold')
    
    # Add horizontal line at 1.0 for reference
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Baseline (no improvement)')
    
    ax.set_ylabel('KL Divergence Ratio', fontsize=12)
    ax.set_title('Joint SAE+NFM: KL Divergence Ratio Analysis\n'
                 'Lower is better (closer to random baseline)', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add improvement text
    joint_improvement = ((ratios['sae_vs_epsilon'] - ratios['joint_vs_epsilon']) / ratios['sae_vs_epsilon']) * 100
    sae_improvement = ((ratios['sae_vs_epsilon'] - ratios['joint_vs_sae']) / ratios['sae_vs_epsilon']) * 100
    
    ax.text(0.98, 0.98, f"Joint vs Random improvement: {joint_improvement:.1f}%\n"
                        f"Joint vs SAE improvement: {(1-ratios['joint_vs_sae'])*100:.1f}%", 
            transform=ax.transAxes, horizontalalignment='right', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    improvement_path = plots_dir / "joint_kl_improvement_analysis.png"
    plt.savefig(improvement_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Joint SAE+NFM KL comparison plots saved to: {plots_dir}")
    return {
        'plots_directory': str(plots_dir),
        'boxplot': str(boxplot_path),
        'histograms': str(histogram_path),
        'position_plot': str(position_path),
        'improvement_plot': str(improvement_path)
    }

def load_joint_model(args):
    """Load all required models including the joint SAE+NFM model."""
    print("Loading models...")
    
    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")
    
    # Load Joint SAE+NFM model
    print("Loading Joint SAE+NFM model...")
    joint_checkpoint = torch.load(args.joint_model_path, map_location=args.device)
    
    # Handle different checkpoint formats
    if 'model_state' in joint_checkpoint:
        joint_state_dict = joint_checkpoint['model_state']
    else:
        joint_state_dict = joint_checkpoint
    
    # Determine Joint model dimensions from state dict
    # SAE dimensions
    if 'primary_sae.decoder.weight' in joint_state_dict:
        sae_input_dim = joint_state_dict['primary_sae.decoder.weight'].shape[0]
        sae_hidden_dim = joint_state_dict['primary_sae.decoder.weight'].shape[1]
    elif 'primary_sae.encoder.0.weight' in joint_state_dict:
        encoder_weight = joint_state_dict['primary_sae.encoder.0.weight']
        sae_hidden_dim, sae_input_dim = encoder_weight.shape
    else:
        raise ValueError("Cannot determine SAE dimensions from joint model")
    
    # NFM dimensions
    if 'nfm.feature_embeddings.weight' in joint_state_dict:
        num_features = joint_state_dict['nfm.feature_embeddings.weight'].shape[0]
        nfm_k_dim = joint_state_dict['nfm.feature_embeddings.weight'].shape[1]
    else:
        raise ValueError("Cannot determine NFM dimensions from joint model")
    
    # Check if linear component exists
    use_linear_component = 'nfm.linear.weight' in joint_state_dict
    
    joint_model = JointSAENFM(
        input_dim=sae_input_dim,
        sae_features=num_features,
        sae_k=args.sae_k,
        nfm_embedding_dim=nfm_k_dim,
        use_linear_component=use_linear_component
    )
    joint_model.load_state_dict(joint_state_dict)
    joint_model.to(args.device)
    
    print(f"Models loaded successfully!")
    print(f"Joint SAE: {sae_input_dim} → {sae_hidden_dim} (k={args.sae_k})")
    print(f"Joint NFM: {num_features} features → {nfm_k_dim} embedding dim")
    print(f"Linear component: {'Yes' if use_linear_component else 'No'}")
    
    return tokenizer, base_model, joint_model

def save_joint_kl_results(results_df, analysis_results, output_dir):
    """Save Joint SAE+NFM KL divergence analysis results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw results
    results_csv_path = output_dir / "joint_kl_divergence_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"Joint KL divergence results saved to: {results_csv_path}")
    
    # Save analysis summary
    analysis_json_path = output_dir / "joint_kl_analysis_summary.json"
    
    # Convert numpy types to JSON-serializable types
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    analysis_to_save = convert_for_json(analysis_results)
    
    with open(analysis_json_path, 'w') as f:
        json.dump(analysis_to_save, f, indent=2)
    print(f"Joint KL analysis summary saved to: {analysis_json_path}")
    
    # Save detailed summary statistics
    summary_csv_path = output_dir / "joint_kl_summary_statistics.csv"
    summary_df = pd.DataFrame(analysis_results['summary_stats'])
    summary_df.to_csv(summary_csv_path)
    print(f"Joint summary statistics saved to: {summary_csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Joint SAE+NFM KL Divergence Test for End-to-End Training vs. Pathological Errors")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base LLM model")
    parser.add_argument("--joint_model_path", type=str, required=True, help="Path to Joint SAE+NFM model")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    
    parser.add_argument("--num_wiki_samples", type=int, default=10000,
                       help="Number of WikiText-103 chunks to analyze (default: 10000)")
    parser.add_argument("--num_epsilon_samples", type=int, default=3,
                       help="Number of ε-random samples per text for variance estimation")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for processing (default: 4)")
    parser.add_argument("--max_token_length", type=int, default=128,
                       help="Maximum token length for sequences (default: 128, matching Gurnee)")
    
    # Model parameters
    parser.add_argument("--sae_k", type=int, default=1024, 
                       help="TopK parameter for Joint SAE (default: 1024)")
    
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--target_layer", type=int, default=16, help="Target layer for intervention")
    
    args = parser.parse_args()
    
    # Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / "joint_kl_analysis.log"),
            logging.StreamHandler()
        ]
    )
    
    print(f"Starting Joint SAE+NFM KL Divergence Analysis")
    print(f"Testing end-to-end training vs. Gurnee's pathological error hypothesis")
    print(f"Joint SAE TopK: {args.sae_k}")
    print(f"Target layer: {args.target_layer}")
    print(f"WikiText-103 samples: {args.num_wiki_samples:,}")
    print(f"Max token length: {args.max_token_length} (matching Gurnee's 128)")
    print(f"ε-random samples per text: {args.num_epsilon_samples}")
    print(f"Batch size: {args.batch_size}")
    
    # Load models
    tokenizer, base_model, joint_model = load_joint_model(args)
    
    # Initialize analyzer
    analyzer = JointKLDivergenceAnalyzer(
        joint_model, tokenizer, base_model, 
        args.device, args.target_layer, args.max_token_length
    )
    
    # Load WikiText-103 dataset
    print("\n" + "="*60)
    print("LOADING WIKITEXT-103 DATASET")
    print("="*60)
    
    wiki_texts = analyzer.load_wiki_dataset(args.num_wiki_samples)
    
    if len(wiki_texts) < 100:
        print(f"Warning: Only loaded {len(wiki_texts)} texts. Consider reducing batch size or max_token_length.")
    
    # Run KL divergence comparison experiment
    print("\n" + "="*60)
    print("RUNNING JOINT SAE+NFM KL DIVERGENCE COMPARISON EXPERIMENT")
    print("="*60)
    
    results_df = analyzer.run_joint_kl_comparison_experiment(
        wiki_texts, args.num_epsilon_samples, args.batch_size
    )
    
    if len(results_df) == 0:
        print("Error: No results collected. Check your model paths and data.")
        return
    
    print(f"Collected {len(results_df):,} KL divergence measurements from WikiText-103")
    
    # Analyze results
    print("\n" + "="*60)
    print("ANALYZING JOINT SAE+NFM KL DIVERGENCE RESULTS")
    print("="*60)
    
    analysis_results = analyzer.analyze_joint_kl_results(results_df)
    
    # Create plots
    print("\n" + "="*60)
    print("CREATING JOINT VISUALIZATION PLOTS")
    print("="*60)
    
    plot_results = create_joint_kl_comparison_plots(results_df, analysis_results, args.output_dir)
    
    # Save results
    print("\n" + "="*60)
    print("SAVING JOINT RESULTS")
    print("="*60)
    
    save_joint_kl_results(results_df, analysis_results, args.output_dir)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL JOINT SAE+NFM SUMMARY")
    print("="*60)
    
    ratios = analysis_results['ratios']
    conclusions = analysis_results['conclusions']
    
    print(f"\nKEY FINDINGS (Joint SAE+NFM Architecture):")
    print(f"  • SAE vs ε-random ratio: {ratios['sae_vs_epsilon']:.3f}x")
    print(f"  • Joint vs ε-random ratio: {ratios['joint_vs_epsilon']:.3f}x")
    print(f"  • Joint vs SAE ratio: {ratios['joint_vs_sae']:.3f}x")
    print(f"  • Total measurements: {conclusions['total_measurements']:,}")
    print(f"  • Scale vs Gurnee: {conclusions['scale_vs_gurnee']:.4f}x")
    
    print(f"\nHYPOTHESIS TESTING:")
    if conclusions['gurnee_pathological_confirmed']:
        print(f"  ✓ GURNEE'S PATHOLOGICAL ERROR HYPOTHESIS: CONFIRMED")
    else:
        print(f"  ✗ Gurnee's pathological error hypothesis: NOT confirmed")
    
    if conclusions['joint_training_supported']:
        print(f"  ✓ JOINT END-TO-END TRAINING HYPOTHESIS: STRONGLY SUPPORTED")
    elif conclusions['joint_reduces_pathological']:
        print(f"  ~ JOINT END-TO-END TRAINING HYPOTHESIS: PARTIALLY SUPPORTED")
    else:
        print(f"  ✗ Joint end-to-end training hypothesis: NOT supported")
    
    if conclusions['joint_reduces_pathological']:
        improvement = ((ratios['sae_vs_epsilon'] - ratios['joint_vs_epsilon']) / ratios['sae_vs_epsilon']) * 100
        joint_improvement = (1 - ratios['joint_vs_sae']) * 100
        print(f"\n🎯 MAIN CONCLUSION: Joint end-to-end training reduces pathological errors by {improvement:.1f}%")
        print(f"   Joint model improves over SAE-only by {joint_improvement:.1f}%")
        print(f"   This supports the hypothesis that end-to-end optimization helps feature integration!")
        print(f"   Tested on {conclusions['total_measurements']:,} real WikiText-103 measurements")
    else:
        print(f"\n❌ MAIN CONCLUSION: Joint end-to-end training does not reduce pathological errors")
        print(f"   This suggests sequential training may be sufficient for feature integration")
    
    # Comparison with part14b results (if available)
    print(f"\nCOMPARISON ANALYSIS:")
    print(f"  🔬 Joint training effectiveness: {(1-ratios['joint_vs_sae'])*100:.1f}% improvement over SAE-only")
    print(f"  📊 Pathological error reduction: {((ratios['sae_vs_epsilon']-ratios['joint_vs_epsilon'])/ratios['sae_vs_epsilon']*100):.1f}%")
    print(f"  🎯 End-to-end vs Sequential: Compare these results with part14b NFM results")
    
    print(f"\nNEXT STEPS:")
    print(f"  1. Compare with part14b sequential training results")
    print(f"  2. Analyze component contributions (SAE vs Linear vs Interaction)")
    print(f"  3. Test feature monosemanticity improvements in joint model")
    print(f"  4. Run ablation studies on joint training hyperparameters")
    
    print(f"\nAll Joint SAE+NFM results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()