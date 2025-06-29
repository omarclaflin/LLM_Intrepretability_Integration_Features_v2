"""
part14c_KL_test_largewiki_componentsseparated.py

KL Divergence Test with Linear vs. Interaction Component Ablation
Using WikiText-103 Dataset

This script extends the KL divergence analysis to test which NFM components
drive the reduction in pathological errors:

1. KL divergence (original, ε-random substitution)
2. KL divergence (original, Primary SAE only)  
3. KL divergence (original, Primary SAE + Linear only)     [NEW]
4. KL divergence (original, Primary SAE + Interaction only) [NEW] 
5. KL divergence (original, Primary SAE + NFM full)

Tests whether linear correction or interaction effects specifically
reduce SAE pathological errors found by Gurnee.
"""

# ============================================================
# LOADING WIKITEXT-103 DATASET
# ============================================================
# Loading WikiText-103 dataset with chunking (target: 20000 samples)...
# Loaded 19886 text chunks from WikiText-103 (max_token_length=128)

# ============================================================
# RUNNING COMPONENT ABLATION EXPERIMENT
# ============================================================

# === COMPONENT ABLATION KL DIVERGENCE EXPERIMENT ===
# Wiki texts: 19886
# ε-random samples per text: 3
# Batch size: 8
# Max token length: 128
# Testing: SAE only, SAE+Linear, SAE+Interaction, SAE+NFM full, ε-random
# Processing component ablation batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2486/2486 [2:39:31<00:00,  3.85s/it] 
# Collected 17,817,856 KL divergence measurements from WikiText-103 component ablation

# ============================================================
# ANALYZING COMPONENT ABLATION RESULTS
# ============================================================

# === COMPONENT ABLATION KL DIVERGENCE ANALYSIS ===

# Summary Statistics:
#                         count      mean       std    median       min        max
# intervention_type
# SAE_NFM_full          2545408  1.550403  1.492717  1.110366  0.000064  17.156267
# SAE_interaction_only  2545408  2.191868  1.728123  1.755464  0.000138  18.182688
# SAE_linear_only       2545408  1.552588  1.495939  1.110258  0.000064  17.382711
# SAE_only              2545408  2.211325  1.741549  1.769927  0.000154  18.190460
# epsilon_random        7636224  0.740148  1.355940  0.275957  0.000013  18.224291

# Key Ratios (vs. ε-random baseline):
#   SAE only vs ε-random:        2.988x
#   SAE+Linear vs ε-random:      2.098x
#   SAE+Interaction vs ε-random: 2.961x
#   SAE+NFM full vs ε-random:    2.095x

# Improvement Ratios (vs. SAE only baseline):
#   SAE+Linear vs SAE only:      0.702x (29.8% improvement)
#   SAE+Interaction vs SAE only: 0.991x (0.9% improvement)
#   SAE+NFM full vs SAE only:    0.701x (29.9% improvement)

# Statistical Tests vs. ε-random (t-tests):
#   SAE only: t=1390.445, p=0.000000
#   SAE+Linear: t=806.269, p=0.000000
#   SAE+Interaction: t=1375.810, p=0.000000
#   SAE+NFM full: t=804.601, p=0.000000

# Statistical Tests vs. SAE only (t-tests):
#   SAE+Linear vs SAE: t=-457.774, p=0.000000
#   SAE+Interaction vs SAE: t=-12.653, p=0.000000
#   SAE+NFM full vs SAE: t=-459.713, p=0.000000

# === COMPONENT ABLATION HYPOTHESIS TESTING ===

# Component Contribution Analysis:
#   Linear component alone:      29.8% improvement over SAE
#   Interaction component alone: 0.9% improvement over SAE
#   Both components together:    29.9% improvement over SAE
#   🎯 LINEAR COMPONENT is more important (29.8% vs 0.9%)
#   ➡️  ADDITIVE EFFECTS: Combined ≈ sum of parts (29.9% vs 30.7%)

# ✓ STATISTICALLY SIGNIFICANT improvements: Linear component, Interaction component, Full NFM

# Sample Size:
#   Total measurements: 17,817,856
#   Measurements per condition: 3,563,571

# ============================================================
# CREATING COMPONENT ABLATION PLOTS
# ============================================================
# Component ablation plots saved to: component_ablation_results\component_ablation_plots

# SAVING COMPONENT ABLATION RESULTS
# ============================================================
# Component ablation results saved to: component_ablation_results\component_ablation_kl_results.csv

# ============================================================
# FINAL COMPONENT ABLATION SUMMARY
# ============================================================

# COMPONENT EFFECTIVENESS:
#   • Linear component:      29.8% improvement
#   • Interaction component: 0.9% improvement
#   • Full NFM:              29.9% improvement

# KEY FINDINGS:
#   ✓ Results are statistically significant

# Total measurements: 17,817,856


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

class ComponentKLDivergenceAnalyzer:
    """Analyzer for KL divergence comparison with NFM component ablation."""
    
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
        """Load WikiText-103 dataset with proper chunking."""
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
        """Generate random vector at same L2 distance as SAE reconstruction error."""
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
        """Forward pass with no intervention (baseline)."""
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                use_cache=False
            )
            logits = outputs.logits
        
        return logits

    def forward_with_epsilon_random(self, input_ids, attention_mask):
        """Forward pass with ε-random intervention at target layer."""
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
        """Forward pass with Primary SAE reconstruction only (no NFM)."""
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

    def forward_with_sae_linear_only(self, input_ids, attention_mask):
        """
        Forward pass with Primary SAE + Linear component only (no interaction).
        
        Pipeline: Layer 16 → Primary SAE → NFM Linear → Primary + Linear + 0
        """
        def sae_linear_hook(module, input, output):
            """Hook to apply SAE + Linear only intervention at layer 16."""
            layer_16_activations = output[0]  # [batch, seq, hidden]
            original_shape = layer_16_activations.shape
            layer_16_flat = layer_16_activations.view(-1, original_shape[-1])  # [batch*seq, hidden]
            
            # Primary SAE processing
            primary_features, primary_reconstruction = self.primary_sae(
                layer_16_flat.to(self.primary_sae.encoder[0].weight.dtype)
            )
            
            # NFM Linear pathway ONLY
            linear_output = self.nfm_model.linear(primary_features)
            
            # Reshape components to match original shape
            primary_reconstruction_reshaped = primary_reconstruction.view(original_shape)
            linear_output_reshaped = linear_output.view(original_shape)
            
            # Two-way combination: Primary SAE reconstruction + Linear component + 0
            combined_output = primary_reconstruction_reshaped + linear_output_reshaped
            
            # Return modified activations
            return (combined_output,) + output[1:]
        
        # Register hook at target layer
        target_layer_module = self.base_model.model.layers[self.target_layer]
        hook_handle = target_layer_module.register_forward_hook(sae_linear_hook)
        
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

    def forward_with_sae_interaction_only(self, input_ids, attention_mask):
        """
        Forward pass with Primary SAE + Interaction component only (no linear).
        
        Pipeline: Layer 16 → Primary SAE → NFM Interaction → Primary + 0 + Interaction
        """
        def sae_interaction_hook(module, input, output):
            """Hook to apply SAE + Interaction only intervention at layer 16."""
            layer_16_activations = output[0]  # [batch, seq, hidden]
            original_shape = layer_16_activations.shape
            layer_16_flat = layer_16_activations.view(-1, original_shape[-1])  # [batch*seq, hidden]
            
            # Primary SAE processing
            primary_features, primary_reconstruction = self.primary_sae(
                layer_16_flat.to(self.primary_sae.encoder[0].weight.dtype)
            )
            
            # NFM Interaction pathway ONLY
            # Step 1: Feature embeddings
            embeddings = self.nfm_model.feature_embeddings.weight.T  # [k_dim, num_features]
            weighted_embeddings = torch.matmul(primary_features, embeddings.T)  # [batch*seq, k_dim]
            
            # Step 2: Interaction MLP Layer 1 [k_dim x k_dim]
            mlp_layer1_output = self.nfm_model.interaction_mlp[1](weighted_embeddings)  # [batch*seq, k_dim]
            
            # Step 3: ReLU activation (Layer 2)
            relu_output = self.nfm_model.interaction_mlp[2](mlp_layer1_output)  # [batch*seq, k_dim]
            
            # Step 4: Final linear layer (Layer 3) 
            interaction_output = self.nfm_model.interaction_mlp[3](relu_output)  # [batch*seq, output_dim]
            
            # Reshape components to match original shape
            primary_reconstruction_reshaped = primary_reconstruction.view(original_shape)
            interaction_output_reshaped = interaction_output.view(original_shape)
            
            # Two-way combination: Primary SAE reconstruction + 0 + Interaction component
            combined_output = primary_reconstruction_reshaped + interaction_output_reshaped
            
            # Return modified activations
            return (combined_output,) + output[1:]
        
        # Register hook at target layer
        target_layer_module = self.base_model.model.layers[self.target_layer]
        hook_handle = target_layer_module.register_forward_hook(sae_interaction_hook)
        
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

    def forward_with_sae_nfm_full(self, input_ids, attention_mask):
        """Forward pass with full SAE + NFM pipeline (both linear and interaction)."""
        def sae_nfm_hook(module, input, output):
            """Hook to apply full SAE+NFM intervention at layer 16."""
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
            
            # Step 2: Interaction MLP Layer 1 [k_dim x k_dim]
            mlp_layer1_output = self.nfm_model.interaction_mlp[1](weighted_embeddings)  # [batch*seq, k_dim]
            
            # Step 3: ReLU activation (Layer 2)
            relu_output = self.nfm_model.interaction_mlp[2](mlp_layer1_output)  # [batch*seq, k_dim]
            
            # Step 4: Final linear layer (Layer 3) 
            interaction_output = self.nfm_model.interaction_mlp[3](relu_output)  # [batch*seq, output_dim]
            
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
        """Calculate KL divergence between original and modified next-token probability distributions."""
        # Convert logits to probabilities
        original_probs = F.softmax(original_logits, dim=-1)  # [batch, seq, vocab]
        modified_probs = F.softmax(modified_logits, dim=-1)  # [batch, seq, vocab]
        
        # Calculate KL divergence: KL(P || Q) = sum(P * log(P / Q))
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        log_ratio = torch.log((original_probs + eps) / (modified_probs + eps))
        kl_div = torch.sum(original_probs * log_ratio, dim=-1)  # [batch, seq]
        
        return kl_div

    def run_component_ablation_kl_experiment(self, wiki_texts, num_epsilon_samples_per_text=3, batch_size=4):
        """
        Run KL divergence comparison experiment with NFM component ablation.
        
        Tests 5 conditions:
        1. ε-random substitution
        2. SAE only  
        3. SAE + Linear only (NEW)
        4. SAE + Interaction only (NEW)
        5. SAE + NFM full
        """
        print(f"\n=== COMPONENT ABLATION KL DIVERGENCE EXPERIMENT ===")
        print(f"Wiki texts: {len(wiki_texts)}")
        print(f"ε-random samples per text: {num_epsilon_samples_per_text}")
        print(f"Batch size: {batch_size}")
        print(f"Max token length: {self.max_token_length}")
        print(f"Testing: SAE only, SAE+Linear, SAE+Interaction, SAE+NFM full, ε-random")
        
        results = []
        
        for batch_start in tqdm(range(0, len(wiki_texts), batch_size), desc="Processing component ablation batches"):
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
                
                # Get logits for all SAE/NFM conditions
                sae_logits = self.forward_with_sae_only(input_ids, attention_mask)
                sae_linear_logits = self.forward_with_sae_linear_only(input_ids, attention_mask)
                sae_interaction_logits = self.forward_with_sae_interaction_only(input_ids, attention_mask)
                sae_nfm_logits = self.forward_with_sae_nfm_full(input_ids, attention_mask)
                
                # Calculate KL divergences for all SAE/NFM conditions
                kl_sae = self.calculate_kl_divergence(original_logits, sae_logits)
                kl_sae_linear = self.calculate_kl_divergence(original_logits, sae_linear_logits)
                kl_sae_interaction = self.calculate_kl_divergence(original_logits, sae_interaction_logits)
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
                        
                        # Store results for all intervention types
                        base_result = {
                            'text_idx': batch_start + batch_idx,
                            'text_content': text_content[:100] + "..." if len(text_content) > 100 else text_content,
                            'position': pos_idx,
                            'token_id': token_id,
                            'token_text': token_text,
                            'sample_idx': 0,
                            'epsilon_distance': None
                        }
                        
                        # SAE only results
                        sae_result = base_result.copy()
                        sae_result.update({
                            'intervention_type': 'SAE_only',
                            'kl_divergence': float(kl_sae[batch_idx, pos_idx].cpu())
                        })
                        results.append(sae_result)
                        
                        # SAE + Linear only results
                        sae_linear_result = base_result.copy()
                        sae_linear_result.update({
                            'intervention_type': 'SAE_linear_only',
                            'kl_divergence': float(kl_sae_linear[batch_idx, pos_idx].cpu())
                        })
                        results.append(sae_linear_result)
                        
                        # SAE + Interaction only results
                        sae_interaction_result = base_result.copy()
                        sae_interaction_result.update({
                            'intervention_type': 'SAE_interaction_only',
                            'kl_divergence': float(kl_sae_interaction[batch_idx, pos_idx].cpu())
                        })
                        results.append(sae_interaction_result)
                        
                        # SAE + NFM full results
                        sae_nfm_result = base_result.copy()
                        sae_nfm_result.update({
                            'intervention_type': 'SAE_NFM_full',
                            'kl_divergence': float(kl_sae_nfm[batch_idx, pos_idx].cpu())
                        })
                        results.append(sae_nfm_result)
                        
                        # ε-random results (multiple samples)
                        for sample_idx, (kl_epsilon, epsilon_distance) in enumerate(zip(kl_epsilon_samples, epsilon_distances)):
                            epsilon_result = base_result.copy()
                            epsilon_result.update({
                                'intervention_type': 'epsilon_random',
                                'kl_divergence': float(kl_epsilon[batch_idx, pos_idx].cpu()),
                                'sample_idx': sample_idx,
                                'epsilon_distance': epsilon_distance
                            })
                            results.append(epsilon_result)
                
            except Exception as e:
                print(f"Error processing batch starting at {batch_start}: {e}")
                continue
        
        print(f"Collected {len(results):,} KL divergence measurements from WikiText-103 component ablation")
        return pd.DataFrame(results)

    def analyze_component_kl_results(self, results_df):
        """Analyze component ablation KL divergence results."""
        print(f"\n=== COMPONENT ABLATION KL DIVERGENCE ANALYSIS ===")
        
        # Calculate summary statistics by intervention type
        summary_stats = results_df.groupby('intervention_type')['kl_divergence'].agg([
            'count', 'mean', 'std', 'median', 'min', 'max'
        ]).round(6)
        
        print(f"\nSummary Statistics:")
        print(summary_stats)
        
        # Calculate ratios relative to ε-random baseline
        epsilon_mean = summary_stats.loc['epsilon_random', 'mean']
        sae_mean = summary_stats.loc['SAE_only', 'mean'] 
        sae_linear_mean = summary_stats.loc['SAE_linear_only', 'mean']
        sae_interaction_mean = summary_stats.loc['SAE_interaction_only', 'mean']
        sae_nfm_mean = summary_stats.loc['SAE_NFM_full', 'mean']
        
        sae_vs_epsilon_ratio = sae_mean / epsilon_mean
        sae_linear_vs_epsilon_ratio = sae_linear_mean / epsilon_mean
        sae_interaction_vs_epsilon_ratio = sae_interaction_mean / epsilon_mean
        sae_nfm_vs_epsilon_ratio = sae_nfm_mean / epsilon_mean
        
        print(f"\nKey Ratios (vs. ε-random baseline):")
        print(f"  SAE only vs ε-random:        {sae_vs_epsilon_ratio:.3f}x")
        print(f"  SAE+Linear vs ε-random:      {sae_linear_vs_epsilon_ratio:.3f}x")
        print(f"  SAE+Interaction vs ε-random: {sae_interaction_vs_epsilon_ratio:.3f}x")
        print(f"  SAE+NFM full vs ε-random:    {sae_nfm_vs_epsilon_ratio:.3f}x")
        
        # Calculate improvement ratios (vs. SAE only baseline)
        sae_linear_vs_sae_ratio = sae_linear_mean / sae_mean
        sae_interaction_vs_sae_ratio = sae_interaction_mean / sae_mean
        sae_nfm_vs_sae_ratio = sae_nfm_mean / sae_mean
        
        print(f"\nImprovement Ratios (vs. SAE only baseline):")
        print(f"  SAE+Linear vs SAE only:      {sae_linear_vs_sae_ratio:.3f}x ({(1-sae_linear_vs_sae_ratio)*100:.1f}% improvement)")
        print(f"  SAE+Interaction vs SAE only: {sae_interaction_vs_sae_ratio:.3f}x ({(1-sae_interaction_vs_sae_ratio)*100:.1f}% improvement)")
        print(f"  SAE+NFM full vs SAE only:    {sae_nfm_vs_sae_ratio:.3f}x ({(1-sae_nfm_vs_sae_ratio)*100:.1f}% improvement)")
        
        # Statistical tests
        epsilon_kls = results_df[results_df['intervention_type'] == 'epsilon_random']['kl_divergence']
        sae_kls = results_df[results_df['intervention_type'] == 'SAE_only']['kl_divergence']
        sae_linear_kls = results_df[results_df['intervention_type'] == 'SAE_linear_only']['kl_divergence']
        sae_interaction_kls = results_df[results_df['intervention_type'] == 'SAE_interaction_only']['kl_divergence']
        sae_nfm_kls = results_df[results_df['intervention_type'] == 'SAE_NFM_full']['kl_divergence']
        
        # T-tests vs. ε-random
        sae_vs_epsilon_stat, sae_vs_epsilon_p = stats.ttest_ind(sae_kls, epsilon_kls)
        sae_linear_vs_epsilon_stat, sae_linear_vs_epsilon_p = stats.ttest_ind(sae_linear_kls, epsilon_kls)
        sae_interaction_vs_epsilon_stat, sae_interaction_vs_epsilon_p = stats.ttest_ind(sae_interaction_kls, epsilon_kls)
        sae_nfm_vs_epsilon_stat, sae_nfm_vs_epsilon_p = stats.ttest_ind(sae_nfm_kls, epsilon_kls)
        
        # T-tests vs. SAE only
        sae_linear_vs_sae_stat, sae_linear_vs_sae_p = stats.ttest_ind(sae_linear_kls, sae_kls)
        sae_interaction_vs_sae_stat, sae_interaction_vs_sae_p = stats.ttest_ind(sae_interaction_kls, sae_kls)
        sae_nfm_vs_sae_stat, sae_nfm_vs_sae_p = stats.ttest_ind(sae_nfm_kls, sae_kls)
        
        print(f"\nStatistical Tests vs. ε-random (t-tests):")
        print(f"  SAE only: t={sae_vs_epsilon_stat:.3f}, p={sae_vs_epsilon_p:.6f}")
        print(f"  SAE+Linear: t={sae_linear_vs_epsilon_stat:.3f}, p={sae_linear_vs_epsilon_p:.6f}")
        print(f"  SAE+Interaction: t={sae_interaction_vs_epsilon_stat:.3f}, p={sae_interaction_vs_epsilon_p:.6f}")
        print(f"  SAE+NFM full: t={sae_nfm_vs_epsilon_stat:.3f}, p={sae_nfm_vs_epsilon_p:.6f}")
        
        print(f"\nStatistical Tests vs. SAE only (t-tests):")
        print(f"  SAE+Linear vs SAE: t={sae_linear_vs_sae_stat:.3f}, p={sae_linear_vs_sae_p:.6f}")
        print(f"  SAE+Interaction vs SAE: t={sae_interaction_vs_sae_stat:.3f}, p={sae_interaction_vs_sae_p:.6f}")
        print(f"  SAE+NFM full vs SAE: t={sae_nfm_vs_sae_stat:.3f}, p={sae_nfm_vs_sae_p:.6f}")
        
        # Interpretation
        print(f"\n=== COMPONENT ABLATION HYPOTHESIS TESTING ===")
        
        # Which component drives the improvement?
        linear_improvement = (1 - sae_linear_vs_sae_ratio) * 100
        interaction_improvement = (1 - sae_interaction_vs_sae_ratio) * 100
        full_improvement = (1 - sae_nfm_vs_sae_ratio) * 100
        
        print(f"\nComponent Contribution Analysis:")
        print(f"  Linear component alone:      {linear_improvement:.1f}% improvement over SAE")
        print(f"  Interaction component alone: {interaction_improvement:.1f}% improvement over SAE")
        print(f"  Both components together:    {full_improvement:.1f}% improvement over SAE")
        
        # Determine which component is more important
        if interaction_improvement > linear_improvement:
            print(f"  🎯 INTERACTION COMPONENT is more important ({interaction_improvement:.1f}% vs {linear_improvement:.1f}%)")
        elif linear_improvement > interaction_improvement:
            print(f"  🎯 LINEAR COMPONENT is more important ({linear_improvement:.1f}% vs {interaction_improvement:.1f}%)")
        else:
            print(f"  🤝 Both components contribute equally")
        
        # Check for synergy
        expected_combined = linear_improvement + interaction_improvement
        actual_combined = full_improvement
        synergy = actual_combined - expected_combined
        
        
        # Statistical significance checks
        significant_tests = []
        if sae_linear_vs_sae_p < 0.05:
            significant_tests.append("Linear component")
        if sae_interaction_vs_sae_p < 0.05:
            significant_tests.append("Interaction component")
        if sae_nfm_vs_sae_p < 0.05:
            significant_tests.append("Full NFM")
        
        if significant_tests:
            print(f"\n✓ STATISTICALLY SIGNIFICANT improvements: {', '.join(significant_tests)}")
        else:
            print(f"\n⚠️  No statistically significant improvements detected")
        
        # Overall conclusions
        total_measurements = len(results_df)
        
        print(f"\nSample Size:")
        print(f"  Total measurements: {total_measurements:,}")
        print(f"  Measurements per condition: {total_measurements // 5:,}")
        
        return {
            'summary_stats': summary_stats.to_dict(),
            'ratios_vs_epsilon': {
                'sae_vs_epsilon': sae_vs_epsilon_ratio,
                'sae_linear_vs_epsilon': sae_linear_vs_epsilon_ratio,
                'sae_interaction_vs_epsilon': sae_interaction_vs_epsilon_ratio,
                'sae_nfm_vs_epsilon': sae_nfm_vs_epsilon_ratio
            },
            'ratios_vs_sae': {
                'sae_linear_vs_sae': sae_linear_vs_sae_ratio,
                'sae_interaction_vs_sae': sae_interaction_vs_sae_ratio,
                'sae_nfm_vs_sae': sae_nfm_vs_sae_ratio
            },
            'improvements': {
                'linear_improvement_pct': linear_improvement,
                'interaction_improvement_pct': interaction_improvement,
                'full_improvement_pct': full_improvement,
                'synergy_pct': synergy
            },
            'statistical_tests': {
                'sae_linear_vs_sae': {'statistic': sae_linear_vs_sae_stat, 'p_value': sae_linear_vs_sae_p},
                'sae_interaction_vs_sae': {'statistic': sae_interaction_vs_sae_stat, 'p_value': sae_interaction_vs_sae_p},
                'sae_nfm_vs_sae': {'statistic': sae_nfm_vs_sae_stat, 'p_value': sae_nfm_vs_sae_p}
            },
            'conclusions': {
                'interaction_more_important': interaction_improvement > linear_improvement,
                'linear_more_important': linear_improvement > interaction_improvement,
                'positive_synergy': synergy > 1.0,
                'statistically_significant': len(significant_tests) > 0,
                'total_measurements': total_measurements
            }
        }

def create_component_ablation_plots(results_df, analysis_results, output_dir):
    """Create visualizations of component ablation KL divergence results."""
    output_dir = Path(output_dir)
    plots_dir = output_dir / "component_ablation_plots"
    plots_dir.mkdir(exist_ok=True)
    
    plt.style.use('default')
    sns.set_palette("Set2")
    
    # 1. Box plot comparing all conditions
    fig, ax = plt.subplots(figsize=(14, 8))
    
    intervention_order = ['epsilon_random', 'SAE_only', 'SAE_linear_only', 'SAE_interaction_only', 'SAE_NFM_full']
    intervention_labels = ['ε-random', 'SAE only', 'SAE + Linear', 'SAE + Interaction', 'SAE + NFM full']
    
    # Create box plot
    box_data = [results_df[results_df['intervention_type'] == itype]['kl_divergence'] 
                for itype in intervention_order]
    
    bp = ax.boxplot(box_data, labels=intervention_labels, patch_artist=True)
    
    # Color the boxes with a progression
    colors = ['lightblue', 'lightcoral', 'lightyellow', 'lightgreen', 'darkgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('KL Divergence', fontsize=14)
    ax.set_title('Component Ablation: Linear vs. Interaction Effects on KL Divergence\n'
                 f'WikiText-103 Analysis with {len(results_df):,} measurements', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add improvement annotations
    improvements = analysis_results['improvements']
    ax.text(0.02, 0.98, f"Improvements over SAE only:\n"
                         f"Linear: {improvements['linear_improvement_pct']:.1f}%\n"
                         f"Interaction: {improvements['interaction_improvement_pct']:.1f}%\n"
                         f"Full NFM: {improvements['full_improvement_pct']:.1f}%\n"
                         f"Synergy: {improvements['synergy_pct']:.1f}%", 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xticks(rotation=15)
    plt.tight_layout()
    boxplot_path = plots_dir / "component_ablation_boxplot.png"
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Bar plot of mean improvements
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ratios = analysis_results['ratios_vs_epsilon']
    conditions = ['SAE only', 'SAE + Linear', 'SAE + Interaction', 'SAE + NFM full']
    ratio_values = [
        ratios['sae_vs_epsilon'],
        ratios['sae_linear_vs_epsilon'], 
        ratios['sae_interaction_vs_epsilon'],
        ratios['sae_nfm_vs_epsilon']
    ]
    
    bars = ax.bar(conditions, ratio_values, color=['lightcoral', 'lightyellow', 'lightgreen', 'darkgreen'])
    
    # Add value labels on bars
    for bar, val in zip(bars, ratio_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}x', ha='center', va='bottom', fontweight='bold')
    
    ax.axhline(y=1.0, color='blue', linestyle='--', alpha=0.7, label='ε-random baseline')
    ax.set_ylabel('KL Divergence Ratio (vs. ε-random)', fontsize=12)
    ax.set_title('Component Contributions: How Much Each Addition Helps\n'
                 'Lower is better (closer to random baseline)', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=15)
    plt.tight_layout()
    barplot_path = plots_dir / "component_improvements_bar.png"
    plt.savefig(barplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Progression line plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    conditions_short = ['SAE', 'SAE+Lin', 'SAE+Int', 'SAE+Full']
    x_positions = range(len(conditions_short))
    
    ax.plot(x_positions, ratio_values, marker='o', linewidth=3, markersize=8, color='red')
    ax.axhline(y=1.0, color='blue', linestyle='--', alpha=0.7, label='ε-random baseline')
    
    # Annotate points with values
    for i, (pos, val) in enumerate(zip(x_positions, ratio_values)):
        ax.annotate(f'{val:.3f}x', (pos, val), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontweight='bold')
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(conditions_short)
    ax.set_ylabel('KL Divergence Ratio (vs. ε-random)', fontsize=12)
    ax.set_title('Progressive Component Addition: Path to Better Behavioral Fidelity\n'
                 'Shows cumulative effect of adding NFM components', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    progression_path = plots_dir / "component_progression_line.png"
    plt.savefig(progression_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Component ablation plots saved to: {plots_dir}")
    return {
        'plots_directory': str(plots_dir),
        'boxplot': str(boxplot_path),
        'barplot': str(barplot_path),
        'progression_plot': str(progression_path)
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
    print(f"NFM: {num_features} features → {k_dim} embedding dim → {output_dim} output")
    
    return tokenizer, base_model, primary_sae, nfm_model

def save_component_results(results_df, analysis_results, output_dir):
    """Save component ablation results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw results
    results_csv_path = output_dir / "component_ablation_kl_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"Component ablation results saved to: {results_csv_path}")
    
    # Save analysis summary
    analysis_json_path = output_dir / "component_ablation_analysis.json"
    
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
    print(f"Component ablation analysis saved to: {analysis_json_path}")

def main():
    parser = argparse.ArgumentParser(description="Component Ablation KL Divergence Test: Linear vs. Interaction Effects")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base LLM model")
    parser.add_argument("--primary_sae_path", type=str, required=True, help="Path to Primary TopK SAE model")
    parser.add_argument("--nfm_path", type=str, required=True, help="Path to NFM model")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    
    parser.add_argument("--num_wiki_samples", type=int, default=5000,
                       help="Number of WikiText-103 chunks to analyze (default: 5000 for faster ablation)")
    parser.add_argument("--num_epsilon_samples", type=int, default=3,
                       help="Number of ε-random samples per text")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for processing")
    parser.add_argument("--max_token_length", type=int, default=128,
                       help="Maximum token length for sequences")
    
    # TopK parameters
    parser.add_argument("--primary_k", type=int, default=1024, 
                       help="TopK parameter for Primary SAE")
    
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
            logging.FileHandler(output_dir / "component_ablation_analysis.log"),
            logging.StreamHandler()
        ]
    )
    
    print(f"Starting Component Ablation KL Divergence Analysis")
    print(f"Testing which NFM components drive pathological error reduction:")
    print(f"  • SAE only (baseline)")
    print(f"  • SAE + Linear component")  
    print(f"  • SAE + Interaction component")
    print(f"  • SAE + NFM full")
    print(f"  • ε-random (reference)")
    print(f"Primary SAE TopK: {args.primary_k}")
    print(f"WikiText-103 samples: {args.num_wiki_samples:,}")
    print(f"Max token length: {args.max_token_length}")
    
    # Load models
    tokenizer, base_model, primary_sae, nfm_model = load_models(args)
    
    # Initialize analyzer
    analyzer = ComponentKLDivergenceAnalyzer(
        primary_sae, nfm_model, tokenizer, base_model, 
        args.device, args.target_layer, args.max_token_length
    )
    
    # Load WikiText-103 dataset
    print("\n" + "="*60)
    print("LOADING WIKITEXT-103 DATASET")
    print("="*60)
    
    wiki_texts = analyzer.load_wiki_dataset(args.num_wiki_samples)
    
    # Run component ablation experiment
    print("\n" + "="*60)
    print("RUNNING COMPONENT ABLATION EXPERIMENT")
    print("="*60)
    
    results_df = analyzer.run_component_ablation_kl_experiment(
        wiki_texts, args.num_epsilon_samples, args.batch_size
    )
    
    if len(results_df) == 0:
        print("Error: No results collected. Check your model paths and data.")
        return
    
    # Analyze results
    print("\n" + "="*60)
    print("ANALYZING COMPONENT ABLATION RESULTS")
    print("="*60)
    
    analysis_results = analyzer.analyze_component_kl_results(results_df)
    
    # Create plots
    print("\n" + "="*60)
    print("CREATING COMPONENT ABLATION PLOTS")
    print("="*60)
    
    plot_results = create_component_ablation_plots(results_df, analysis_results, args.output_dir)
    
    # Save results
    print("\n" + "="*60)
    print("SAVING COMPONENT ABLATION RESULTS")
    print("="*60)
    
    save_component_results(results_df, analysis_results, args.output_dir)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL COMPONENT ABLATION SUMMARY")
    print("="*60)
    
    improvements = analysis_results['improvements']
    conclusions = analysis_results['conclusions']
    
    print(f"\nCOMPONENT EFFECTIVENESS:")
    print(f"  • Linear component:      {improvements['linear_improvement_pct']:.1f}% improvement")
    print(f"  • Interaction component: {improvements['interaction_improvement_pct']:.1f}% improvement")
    print(f"  • Full NFM:              {improvements['full_improvement_pct']:.1f}% improvement")

    
    print(f"\nKEY FINDINGS:")
    
    if conclusions['statistically_significant']:
        print(f"  ✓ Results are statistically significant")
    else:
        print(f"  ⚠️  Results not statistically significant - consider larger sample")
    
    print(f"\nTotal measurements: {conclusions['total_measurements']:,}")
    print(f"All component ablation results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()