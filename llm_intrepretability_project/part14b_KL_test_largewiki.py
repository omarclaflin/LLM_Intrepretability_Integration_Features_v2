"""
part14b_KL_test_largewiki.py

KL Divergence Test for SAE Pathological Errors vs. Feature Integration Hypothesis
Using WikiText-103 Dataset with Large Sample Sizes

This script tests the core hypothesis from Gurnee and Claflin papers by comparing:
1. KL divergence (original, ε-random substitution)
2. KL divergence (original, Primary SAE only)  
3. KL divergence (original, Primary SAE + NFM)

Uses WikiText-103 dataset with proper chunking to match Gurnee's scale.
Pipeline interventions:
- ε-random: Layer 16 → Random vector at same L2 distance as SAE error
- SAE only: Layer 16 → Primary SAE → Primary reconstruction only
- SAE+NFM: Layer 16 → Primary SAE → NFM → Primary + Linear + Interaction

Tests whether NFM integration features reduce the pathological KL gap found by Gurnee.
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
    """TopK Sparse Autoencoder module that can handle k=None."""
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

class WikiKLDivergenceAnalyzer:
    """Analyzer for KL divergence comparison using WikiText-103 dataset."""
    
    def __init__(self, primary_sae, nfm_model, tokenizer, base_model, 
                 device="cuda", target_layer=16, max_token_length=128):
        self.primary_sae = primary_sae
        self.nfm_model = nfm_model
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.device = device
        self.target_layer = target_layer
        self.max_token_length = max_token_length
        
        # Set models to eval mode
        self.primary_sae.eval()
        self.nfm_model.eval()
        self.base_model.eval()
        
        # Get vocabulary size for KL calculation
        self.vocab_size = len(tokenizer.vocab)
        print(f"Vocabulary size: {self.vocab_size}")

    def load_wiki_dataset(self, num_samples=50000):
        """
        Load WikiText-103 dataset with proper chunking like part12 script.
        
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

    def generate_epsilon_random(self, original_activation, sae_reconstruction):
        """
        Generate random vector at same L2 distance as SAE reconstruction error.
        
        Args:
            original_activation: Original activation vector [batch*seq, hidden]
            sae_reconstruction: SAE reconstruction [batch*seq, hidden]
        
        Returns:
            Random vector with same L2 error distance as SAE
        """
        # Calculate SAE reconstruction error
        sae_error = sae_reconstruction - original_activation  # [batch*seq, hidden]
        epsilon = torch.norm(sae_error, dim=-1, keepdim=True)  # [batch*seq, 1]
        
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
            
            # Get SAE reconstruction to calculate epsilon
            primary_features, primary_reconstruction = self.primary_sae(
                layer_16_flat.to(self.primary_sae.encoder[0].weight.dtype)
            )
            
            # Generate ε-random vector at same distance as SAE error
            epsilon_random_flat = self.generate_epsilon_random(layer_16_flat, primary_reconstruction)
            
            # Store epsilon distance for reporting
            sae_error = primary_reconstruction - layer_16_flat
            epsilon_distance = torch.norm(sae_error, dim=-1).mean().item()
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
        Forward pass with Primary SAE reconstruction only (no NFM).
        
        Returns:
            logits: Logits after SAE-only substitution
        """
        def sae_only_hook(module, input, output):
            """Hook to apply SAE-only intervention at layer 16."""
            layer_16_activations = output[0]  # [batch, seq, hidden]
            original_shape = layer_16_activations.shape
            layer_16_flat = layer_16_activations.view(-1, original_shape[-1])  # [batch*seq, hidden]
            
            # Primary SAE processing only
            primary_features, primary_reconstruction = self.primary_sae(
                layer_16_flat.to(self.primary_sae.encoder[0].weight.dtype)
            )
            
            # Reshape primary reconstruction back to original shape
            primary_reconstruction_reshaped = primary_reconstruction.view(original_shape)
            
            # Return SAE reconstruction only
            return (primary_reconstruction_reshaped,) + output[1:]
        
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

    def forward_with_sae_nfm(self, input_ids, attention_mask):
        """
        Forward pass with full SAE + NFM pipeline.
        
        Returns:
            logits: Logits after SAE+NFM substitution
        """
        def sae_nfm_hook(module, input, output):
            """Hook to apply SAE+NFM intervention at layer 16."""
            layer_16_activations = output[0]  # [batch, seq, hidden]
            original_shape = layer_16_activations.shape
            layer_16_flat = layer_16_activations.view(-1, original_shape[-1])  # [batch*seq, hidden]
            
            # Primary SAE processing
            primary_features, primary_reconstruction = self.primary_sae(
                layer_16_flat.to(self.primary_sae.encoder[0].weight.dtype)
            )
            
            # NFM processing - Linear pathway
            linear_output = self.nfm_model.linear(primary_features)
            
            # NFM processing - Interaction pathway
            # Step 1: Feature embeddings
            embeddings = self.nfm_model.feature_embeddings.weight.T  # [k_dim, num_features]
            weighted_embeddings = torch.matmul(primary_features, embeddings.T)  # [batch*seq, k_dim]
            
            # Step 2: Interaction MLP Layer 1 [500 x 500]
            mlp_layer1_output = self.nfm_model.interaction_mlp[1](weighted_embeddings)  # [batch*seq, 500]
            
            # Step 3: ReLU activation (Layer 2)
            relu_output = self.nfm_model.interaction_mlp[2](mlp_layer1_output)  # [batch*seq, 500]
            
            # Step 4: Final linear layer (Layer 3) 
            interaction_output = self.nfm_model.interaction_mlp[3](relu_output)  # [batch*seq, 3200]
            
            # Step 5: THREE-WAY COMBINATION: Primary SAE reconstruction + Linear + Interaction
            # Reshape all components to match original shape
            primary_reconstruction_reshaped = primary_reconstruction.view(original_shape)
            linear_output_reshaped = linear_output.view(original_shape)
            interaction_output_reshaped = interaction_output.view(original_shape)
            
            # Combine all three pathways
            nfm_total_output = primary_reconstruction_reshaped + linear_output_reshaped + interaction_output_reshaped
            
            # Return modified activations
            return (nfm_total_output,) + output[1:]
        
        # Register hook at target layer
        target_layer_module = self.base_model.model.layers[self.target_layer]
        hook_handle = target_layer_module.register_forward_hook(sae_nfm_hook)
        
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

    def run_wiki_kl_comparison_experiment(self, wiki_texts, num_epsilon_samples_per_text=3, batch_size=4):
        """
        Run KL divergence comparison experiment on WikiText-103 chunks.
        
        Args:
            wiki_texts: List of WikiText-103 chunks
            num_epsilon_samples_per_text: Number of ε-random samples per text
            batch_size: Batch size for processing
        
        Returns:
            DataFrame with KL divergence results
        """
        print(f"\n=== WIKI KL DIVERGENCE COMPARISON EXPERIMENT ===")
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
                
                # Get SAE+NFM logits
                sae_nfm_logits = self.forward_with_sae_nfm(input_ids, attention_mask)
                
                # Calculate KL divergences for SAE and SAE+NFM
                kl_sae = self.calculate_kl_divergence(original_logits, sae_logits)
                kl_sae_nfm = self.calculate_kl_divergence(original_logits, sae_nfm_logits)
                
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
                        
                        # SAE+NFM results
                        results.append({
                            'text_idx': batch_start + batch_idx,
                            'text_content': text_content[:100] + "..." if len(text_content) > 100 else text_content,
                            'position': pos_idx,
                            'token_id': token_id,
                            'token_text': token_text,
                            'intervention_type': 'SAE_NFM',
                            'kl_divergence': float(kl_sae_nfm[batch_idx, pos_idx].cpu()),
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

    def analyze_kl_results(self, results_df):
        """
        Analyze KL divergence results to test the pathological error hypothesis.
        
        Args:
            results_df: DataFrame from run_wiki_kl_comparison_experiment
        
        Returns:
            Dictionary with analysis results
        """
        print(f"\n=== WIKI KL DIVERGENCE ANALYSIS ===")
        
        # Calculate summary statistics by intervention type
        summary_stats = results_df.groupby('intervention_type')['kl_divergence'].agg([
            'count', 'mean', 'std', 'median', 'min', 'max'
        ]).round(6)
        
        print(f"\nSummary Statistics:")
        print(summary_stats)
        
        # Calculate ratios (Gurnee's key finding)
        epsilon_mean = summary_stats.loc['epsilon_random', 'mean']
        sae_mean = summary_stats.loc['SAE_only', 'mean'] 
        sae_nfm_mean = summary_stats.loc['SAE_NFM', 'mean']
        
        sae_vs_epsilon_ratio = sae_mean / epsilon_mean
        sae_nfm_vs_epsilon_ratio = sae_nfm_mean / epsilon_mean
        sae_nfm_vs_sae_ratio = sae_nfm_mean / sae_mean
        
        print(f"\nKey Ratios:")
        print(f"  SAE vs ε-random: {sae_vs_epsilon_ratio:.3f}x")
        print(f"  SAE+NFM vs ε-random: {sae_nfm_vs_epsilon_ratio:.3f}x") 
        print(f"  SAE+NFM vs SAE: {sae_nfm_vs_sae_ratio:.3f}x")
        
        # Statistical tests
        epsilon_kls = results_df[results_df['intervention_type'] == 'epsilon_random']['kl_divergence']
        sae_kls = results_df[results_df['intervention_type'] == 'SAE_only']['kl_divergence']
        sae_nfm_kls = results_df[results_df['intervention_type'] == 'SAE_NFM']['kl_divergence']
        
        # T-tests
        sae_vs_epsilon_stat, sae_vs_epsilon_p = stats.ttest_ind(sae_kls, epsilon_kls)
        sae_nfm_vs_epsilon_stat, sae_nfm_vs_epsilon_p = stats.ttest_ind(sae_nfm_kls, epsilon_kls)
        sae_nfm_vs_sae_stat, sae_nfm_vs_sae_p = stats.ttest_ind(sae_nfm_kls, sae_kls)
        
        print(f"\nStatistical Tests (t-tests):")
        print(f"  SAE vs ε-random: t={sae_vs_epsilon_stat:.3f}, p={sae_vs_epsilon_p:.6f}")
        print(f"  SAE+NFM vs ε-random: t={sae_nfm_vs_epsilon_stat:.3f}, p={sae_nfm_vs_epsilon_p:.6f}")
        print(f"  SAE+NFM vs SAE: t={sae_nfm_vs_sae_stat:.3f}, p={sae_nfm_vs_sae_p:.6f}")
        
        # Interpretation
        print(f"\n=== HYPOTHESIS TESTING (LARGE WIKI SCALE) ===")
        
        # Gurnee's pathological error hypothesis
        if sae_vs_epsilon_ratio > 1.5 and sae_vs_epsilon_p < 0.05:
            print(f"✓ GURNEE'S PATHOLOGICAL ERROR CONFIRMED: SAE errors are {sae_vs_epsilon_ratio:.1f}x worse than random")
        else:
            print(f"✗ Gurnee's pathological error NOT confirmed")
        
        # Claflin's integration hypothesis  
        if sae_nfm_vs_sae_ratio < 0.8 and sae_nfm_vs_sae_p < 0.05:
            print(f"✓ CLAFLIN'S INTEGRATION HYPOTHESIS STRONGLY SUPPORTED: NFM reduces KL by {(1-sae_nfm_vs_sae_ratio)*100:.1f}%")
        elif sae_nfm_vs_sae_ratio < 1.0 and sae_nfm_vs_sae_p < 0.05:
            print(f"✓ CLAFLIN'S INTEGRATION HYPOTHESIS SUPPORTED: NFM reduces KL by {(1-sae_nfm_vs_sae_ratio)*100:.1f}%")
        elif sae_nfm_vs_sae_ratio < 1.0:
            print(f"~ CLAFLIN'S INTEGRATION HYPOTHESIS PARTIALLY SUPPORTED: NFM reduces KL by {(1-sae_nfm_vs_sae_ratio)*100:.1f}% (not significant)")
        else:
            print(f"✗ Claflin's integration hypothesis NOT supported")
        
        # Overall conclusion
        if sae_nfm_vs_epsilon_ratio < sae_vs_epsilon_ratio:
            improvement = ((sae_vs_epsilon_ratio - sae_nfm_vs_epsilon_ratio) / sae_vs_epsilon_ratio) * 100
            print(f"🎯 INTEGRATION REDUCES PATHOLOGICAL ERRORS by {improvement:.1f}%")
            print(f"   With {len(results_df):,} measurements from WikiText-103")
        else:
            print(f"❌ Integration does not reduce pathological errors")
        
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
                'sae_nfm_vs_epsilon': sae_nfm_vs_epsilon_ratio,
                'sae_nfm_vs_sae': sae_nfm_vs_sae_ratio
            },
            'statistical_tests': {
                'sae_vs_epsilon': {'statistic': sae_vs_epsilon_stat, 'p_value': sae_vs_epsilon_p},
                'sae_nfm_vs_epsilon': {'statistic': sae_nfm_vs_epsilon_stat, 'p_value': sae_nfm_vs_epsilon_p},
                'sae_nfm_vs_sae': {'statistic': sae_nfm_vs_sae_stat, 'p_value': sae_nfm_vs_sae_p}
            },
            'conclusions': {
                'gurnee_pathological_confirmed': sae_vs_epsilon_ratio > 1.5 and sae_vs_epsilon_p < 0.05,
                'claflin_integration_supported': sae_nfm_vs_sae_ratio < 0.8 and sae_nfm_vs_sae_p < 0.05,
                'integration_reduces_pathological': sae_nfm_vs_epsilon_ratio < sae_vs_epsilon_ratio,
                'total_measurements': total_measurements,
                'scale_vs_gurnee': scale_ratio
            }
        }

def create_wiki_kl_comparison_plots(results_df, analysis_results, output_dir):
    """
    Create visualizations of WikiText-103 KL divergence comparison results.
    """
    output_dir = Path(output_dir)
    plots_dir = output_dir / "wiki_kl_comparison_plots"
    plots_dir.mkdir(exist_ok=True)
    
    plt.style.use('default')
    sns.set_palette("Set1")
    
    # 1. Box plot comparing distributions
    fig, ax = plt.subplots(figsize=(12, 8))
    
    intervention_order = ['epsilon_random', 'SAE_only', 'SAE_NFM']
    intervention_labels = ['ε-random', 'SAE only', 'SAE + NFM']
    
    # Create box plot
    box_data = [results_df[results_df['intervention_type'] == itype]['kl_divergence'] 
                for itype in intervention_order]
    
    bp = ax.boxplot(box_data, labels=intervention_labels, patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('KL Divergence', fontsize=14)
    ax.set_title('WikiText-103 KL Divergence Comparison: Testing Pathological Errors vs. Integration\n'
                 f'Scale: {len(results_df):,} measurements from real text', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add ratio annotations
    ratios = analysis_results['ratios']
    conclusions = analysis_results['conclusions']
    ax.text(0.02, 0.98, f"SAE vs ε-random: {ratios['sae_vs_epsilon']:.2f}x\n"
                         f"SAE+NFM vs ε-random: {ratios['sae_nfm_vs_epsilon']:.2f}x\n"
                         f"SAE+NFM vs SAE: {ratios['sae_nfm_vs_sae']:.2f}x\n\n"
                         f"Measurements: {conclusions['total_measurements']:,}\n"
                         f"Gurnee scale: {conclusions['scale_vs_gurnee']:.3f}x", 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    boxplot_path = plots_dir / "wiki_kl_divergence_comparison_boxplot.png"
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Histogram comparison with sample sizes
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('WikiText-103 KL Divergence Distributions by Intervention Type', 
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
    histogram_path = plots_dir / "wiki_kl_divergence_histograms.png"
    plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Position-wise analysis for WikiText
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
    ax.set_title('WikiText-103 KL Divergence by Token Position\n'
                 'Shows how pathological errors vary across sequence positions', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    position_path = plots_dir / "wiki_kl_divergence_by_position.png"
    plt.savefig(position_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Sample size scaling plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Show cumulative means as function of sample size (for stability)
    sample_sizes = []
    cumulative_means = {itype: [] for itype in intervention_order}
    
    step_size = max(1, len(results_df) // 50)  # 50 points max
    
    for i in range(step_size, len(results_df), step_size):
        sample_subset = results_df.iloc[:i]
        sample_sizes.append(i)
        
        for itype in intervention_order:
            subset_data = sample_subset[sample_subset['intervention_type'] == itype]
            if len(subset_data) > 0:
                cumulative_means[itype].append(subset_data['kl_divergence'].mean())
            else:
                cumulative_means[itype].append(np.nan)
    
    for itype, label, color in zip(intervention_order, intervention_labels, colors):
        valid_points = [(s, m) for s, m in zip(sample_sizes, cumulative_means[itype]) if not np.isnan(m)]
        if valid_points:
            x_vals, y_vals = zip(*valid_points)
            ax.plot(x_vals, y_vals, label=label, linewidth=2, color=color)
    
    ax.set_xlabel('Number of Measurements', fontsize=12)
    ax.set_ylabel('Cumulative Mean KL Divergence', fontsize=12)
    ax.set_title('WikiText-103 KL Divergence Convergence with Sample Size\n'
                 'Shows statistical stability as more data is collected', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add final ratios
    final_ratios = analysis_results['ratios']
    ax.text(0.98, 0.02, f"Final Ratios:\nSAE/ε-random: {final_ratios['sae_vs_epsilon']:.3f}\n"
                        f"SAE+NFM/ε-random: {final_ratios['sae_nfm_vs_epsilon']:.3f}\n"
                        f"Improvement: {((final_ratios['sae_vs_epsilon']-final_ratios['sae_nfm_vs_epsilon'])/final_ratios['sae_vs_epsilon']*100):.1f}%", 
            transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    scaling_path = plots_dir / "wiki_kl_divergence_sample_scaling.png"
    plt.savefig(scaling_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"WikiText-103 KL comparison plots saved to: {plots_dir}")
    return {
        'plots_directory': str(plots_dir),
        'boxplot': str(boxplot_path),
        'histograms': str(histogram_path),
        'position_plot': str(position_path),
        'scaling_plot': str(scaling_path)
    }

def load_models(args):
    """Load all required models."""
    print("Loading models...")
    
    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")
    
    # Load Primary SAE
    print("Loading Primary SAE...")
    primary_sae_checkpoint = torch.load(args.primary_sae_path, map_location=args.device)
    
    # Handle different checkpoint formats
    if 'model_state' in primary_sae_checkpoint:
        primary_sae_state_dict = primary_sae_checkpoint['model_state']
    else:
        primary_sae_state_dict = primary_sae_checkpoint
    
    # Determine Primary SAE dimensions
    if 'decoder.weight' in primary_sae_state_dict:
        input_dim = primary_sae_state_dict['decoder.weight'].shape[0]
        hidden_dim = primary_sae_state_dict['decoder.weight'].shape[1]
    elif 'encoder.0.weight' in primary_sae_state_dict:
        encoder_weight = primary_sae_state_dict['encoder.0.weight']
        hidden_dim, input_dim = encoder_weight.shape
    else:
        raise ValueError("Cannot determine Primary SAE dimensions")
    
    primary_sae = TopKSparseAutoencoder(input_dim, hidden_dim, k=args.primary_k)
    primary_sae.load_state_dict(primary_sae_state_dict)
    primary_sae.to(args.device)
    
    # Load NFM model
    print("Loading NFM model...")
    nfm_state_dict = torch.load(args.nfm_path, map_location=args.device)
    num_features = nfm_state_dict['feature_embeddings.weight'].shape[0]
    k_dim = nfm_state_dict['feature_embeddings.weight'].shape[1]
    output_dim = nfm_state_dict['linear.weight'].shape[0]
    
    nfm_model = NeuralFactorizationModel(num_features, k_dim, output_dim)
    nfm_model.load_state_dict(nfm_state_dict)
    nfm_model.to(args.device)
    
    print(f"Models loaded successfully!")
    print(f"Primary SAE: {input_dim} → {hidden_dim} (k={args.primary_k})")
    print(f"NFM: {num_features} features → {k_dim} embedding dim")
    
    return tokenizer, base_model, primary_sae, nfm_model

def save_wiki_kl_results(results_df, analysis_results, output_dir):
    """Save WikiText-103 KL divergence analysis results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw results
    results_csv_path = output_dir / "wiki_kl_divergence_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"Wiki KL divergence results saved to: {results_csv_path}")
    
    # Save analysis summary
    analysis_json_path = output_dir / "wiki_kl_analysis_summary.json"
    
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
    print(f"Wiki KL analysis summary saved to: {analysis_json_path}")
    
    # Save detailed summary statistics
    summary_csv_path = output_dir / "wiki_kl_summary_statistics.csv"
    summary_df = pd.DataFrame(analysis_results['summary_stats'])
    summary_df.to_csv(summary_csv_path)
    print(f"Wiki summary statistics saved to: {summary_csv_path}")

def main():
    parser = argparse.ArgumentParser(description="WikiText-103 KL Divergence Test for SAE Pathological Errors vs. Integration Hypothesis")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base LLM model")
    parser.add_argument("--primary_sae_path", type=str, required=True, help="Path to Primary TopK SAE model")
    parser.add_argument("--nfm_path", type=str, required=True, help="Path to NFM model")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    
    parser.add_argument("--num_wiki_samples", type=int, default=10000,
                       help="Number of WikiText-103 chunks to analyze (default: 10000)")
    parser.add_argument("--num_epsilon_samples", type=int, default=3,
                       help="Number of ε-random samples per text for variance estimation")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for processing (default: 4)")
    parser.add_argument("--max_token_length", type=int, default=128,
                       help="Maximum token length for sequences (default: 128, matching Gurnee)")
    
    # TopK parameters
    parser.add_argument("--primary_k", type=int, default=1024, 
                       help="TopK parameter for Primary SAE (default: 1024)")
    
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
            logging.FileHandler(output_dir / "wiki_kl_analysis.log"),
            logging.StreamHandler()
        ]
    )
    
    print(f"Starting WikiText-103 KL Divergence Analysis")
    print(f"Testing Gurnee's pathological error hypothesis vs. Claflin's integration hypothesis")
    print(f"Primary SAE TopK: {args.primary_k}")
    print(f"Target layer: {args.target_layer}")
    print(f"WikiText-103 samples: {args.num_wiki_samples:,}")
    print(f"Max token length: {args.max_token_length} (matching Gurnee's 128)")
    print(f"ε-random samples per text: {args.num_epsilon_samples}")
    print(f"Batch size: {args.batch_size}")
    
    # Load models
    tokenizer, base_model, primary_sae, nfm_model = load_models(args)
    
    # Initialize analyzer
    analyzer = WikiKLDivergenceAnalyzer(
        primary_sae, nfm_model, tokenizer, base_model, 
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
    print("RUNNING WIKI KL DIVERGENCE COMPARISON EXPERIMENT")
    print("="*60)
    
    results_df = analyzer.run_wiki_kl_comparison_experiment(
        wiki_texts, args.num_epsilon_samples, args.batch_size
    )
    
    if len(results_df) == 0:
        print("Error: No results collected. Check your model paths and data.")
        return
    
    print(f"Collected {len(results_df):,} KL divergence measurements from WikiText-103")
    
    # Analyze results
    print("\n" + "="*60)
    print("ANALYZING WIKI KL DIVERGENCE RESULTS")
    print("="*60)
    
    analysis_results = analyzer.analyze_kl_results(results_df)
    
    # Create plots
    print("\n" + "="*60)
    print("CREATING WIKI VISUALIZATION PLOTS")
    print("="*60)
    
    plot_results = create_wiki_kl_comparison_plots(results_df, analysis_results, args.output_dir)
    
    # Save results
    print("\n" + "="*60)
    print("SAVING WIKI RESULTS")
    print("="*60)
    
    save_wiki_kl_results(results_df, analysis_results, args.output_dir)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL WIKI SUMMARY")
    print("="*60)
    
    ratios = analysis_results['ratios']
    conclusions = analysis_results['conclusions']
    
    print(f"\nKEY FINDINGS (WikiText-103 Scale):")
    print(f"  • SAE vs ε-random ratio: {ratios['sae_vs_epsilon']:.3f}x")
    print(f"  • SAE+NFM vs ε-random ratio: {ratios['sae_nfm_vs_epsilon']:.3f}x")
    print(f"  • SAE+NFM vs SAE ratio: {ratios['sae_nfm_vs_sae']:.3f}x")
    print(f"  • Total measurements: {conclusions['total_measurements']:,}")
    print(f"  • Scale vs Gurnee: {conclusions['scale_vs_gurnee']:.4f}x")
    
    print(f"\nHYPOTHESIS TESTING:")
    if conclusions['gurnee_pathological_confirmed']:
        print(f"  ✓ GURNEE'S PATHOLOGICAL ERROR HYPOTHESIS: CONFIRMED")
    else:
        print(f"  ✗ Gurnee's pathological error hypothesis: NOT confirmed")
    
    if conclusions['claflin_integration_supported']:
        print(f"  ✓ CLAFLIN'S INTEGRATION HYPOTHESIS: STRONGLY SUPPORTED")
    elif conclusions['integration_reduces_pathological']:
        print(f"  ~ CLAFLIN'S INTEGRATION HYPOTHESIS: PARTIALLY SUPPORTED")
    else:
        print(f"  ✗ Claflin's integration hypothesis: NOT supported")
    
    if conclusions['integration_reduces_pathological']:
        improvement = ((ratios['sae_vs_epsilon'] - ratios['sae_nfm_vs_epsilon']) / ratios['sae_vs_epsilon']) * 100
        print(f"\n🎯 MAIN CONCLUSION: Feature integration reduces pathological errors by {improvement:.1f}%")
        print(f"   This supports the hypothesis that SAE errors are due to missing feature integration!")
        print(f"   Tested on {conclusions['total_measurements']:,} real WikiText-103 measurements")
    else:
        print(f"\n❌ MAIN CONCLUSION: Feature integration does not reduce pathological errors")
        print(f"   This suggests other mechanisms may be responsible for SAE pathological errors")
    
    print(f"\nAll WikiText-103 results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()