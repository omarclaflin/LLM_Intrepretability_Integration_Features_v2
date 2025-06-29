"""
part16_joint_component_KL_test.py

KL Divergence and Cross-Entropy Component Analysis for Joint SAE+NFM Architecture
Based on part14c_KL_test_largewiki_componentsseparated.py and part15_joint_SAENFM_training.py

This script analyzes which components of the joint architecture drive the reduction 
in pathological errors by testing:

1. KL divergence (original, ε-random substitution)
2. KL divergence (original, SAE only)  
3. KL divergence (original, SAE + NFM Linear only)     
4. KL divergence (original, SAE + NFM Nonlinear only) 
5. KL divergence (original, SAE + NFM Full)

Additionally includes cross-entropy analysis to measure next-token prediction accuracy.

Tests whether linear correction or nonlinear interaction effects specifically
reduce SAE pathological errors in the joint architecture.
"""

# KL Divergence Improvements (vs. SAE only baseline):
#   SAE+NFM Linear vs SAE only:     0.491x (50.9% improvement)
#   SAE+NFM Nonlinear vs SAE only:  0.914x (8.6% improvement)
#   SAE+NFM Full vs SAE only:       0.481x (51.9% improvement)

# Cross-Entropy Improvements (vs. SAE only baseline):
#   SAE+NFM Linear vs SAE only:     0.743x (25.7% improvement)
#   SAE+NFM Nonlinear vs SAE only:  0.958x (4.2% improvement)
#   SAE+NFM Full vs SAE only:       0.738x (26.2% improvement)

# Statistical Tests vs. SAE only (KL Divergence, t-tests):
#   SAE+NFM Linear vs SAE: t=-108.137, p=0.000000
#   SAE+NFM Nonlinear vs SAE: t=-15.796, p=0.000000
#   SAE+NFM Full vs SAE: t=-111.369, p=0.000000

# Statistical Tests vs. SAE only (Cross-Entropy, t-tests):
#   SAE+NFM Linear vs SAE: t=-63.287, p=0.000000
#   SAE+NFM Nonlinear vs SAE: t=-10.247, p=0.000000
#   SAE+NFM Full vs SAE: t=-64.601, p=0.000000


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

class NeuralFactorizationMachine(torch.nn.Module):
    """Neural Factorization Machine for analyzing feature interactions."""
    def __init__(self, num_sae_features, embedding_dim, output_dim, use_linear=True, dropout=0.15):
        super().__init__()
        self.num_sae_features = num_sae_features
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        
        self.feature_embeddings = torch.nn.Embedding(num_sae_features, embedding_dim)
        self.linear = torch.nn.Linear(num_sae_features, output_dim, bias=True) if use_linear else None
        
        self.interaction_mlp = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dim, output_dim)
        )
    
    def forward(self, sae_features):
        """Forward pass returning total output, linear component, and interaction component."""
        batch_size = sae_features.shape[0]
        
        # Interaction pathway
        all_embeddings = self.feature_embeddings.weight
        weighted_embeddings = sae_features.unsqueeze(-1) * all_embeddings.unsqueeze(0)
        
        sum_embeddings = torch.sum(weighted_embeddings, dim=1)
        square_embeddings = weighted_embeddings ** 2
        sum_squares = torch.sum(square_embeddings, dim=1)
        
        interaction_vector = 0.5 * (sum_embeddings ** 2 - sum_squares)
        interaction_out = self.interaction_mlp(interaction_vector)
        
        # Linear pathway
        linear_out = None
        if self.linear is not None:
            linear_out = self.linear(sae_features)
            total_output = linear_out + interaction_out
        else:
            total_output = interaction_out
        
        return total_output, linear_out, interaction_out

class JointSAENFM(torch.nn.Module):
    """Joint SAE+NFM model for end-to-end training."""
    def __init__(self, input_dim, sae_features, sae_k, nfm_embedding_dim, use_linear=True):
        super().__init__()
        self.primary_sae = TopKSparseAutoencoder(input_dim, sae_features, sae_k)
        self.nfm = NeuralFactorizationMachine(sae_features, nfm_embedding_dim, input_dim, use_linear)
    
    def forward(self, layer_16_activations):
        """
        Joint forward pass: Layer 16 → SAE → NFM → Final Reconstruction
        
        Returns:
            final_reconstruction: Combined output from all three pathways
            primary_features: SAE features (for sparsity loss)
            primary_recon: SAE reconstruction
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

class JointComponentKLAnalyzer:
    """Analyzer for KL divergence and cross-entropy comparison with joint NFM component ablation."""
    
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
            primary_features, primary_reconstruction = self.joint_model.primary_sae(
                layer_16_flat.to(self.joint_model.primary_sae.encoder[0].weight.dtype)
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
        """Forward pass with SAE reconstruction only (no NFM)."""
        def sae_only_hook(module, input, output):
            """Hook to apply SAE-only intervention at layer 16."""
            layer_16_activations = output[0]  # [batch, seq, hidden]
            original_shape = layer_16_activations.shape
            layer_16_flat = layer_16_activations.view(-1, original_shape[-1])  # [batch*seq, hidden]
            
            # SAE processing only
            primary_features, primary_reconstruction = self.joint_model.primary_sae(
                layer_16_flat.to(self.joint_model.primary_sae.encoder[0].weight.dtype)
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

    def forward_with_sae_nfm_linear_only(self, input_ids, attention_mask):
        """
        Forward pass with SAE + NFM Linear component only (no nonlinear interaction).
        
        Pipeline: Layer 16 → SAE → NFM Linear → SAE + Linear + 0
        """
        def sae_nfm_linear_hook(module, input, output):
            """Hook to apply SAE + NFM Linear only intervention at layer 16."""
            layer_16_activations = output[0]  # [batch, seq, hidden]
            original_shape = layer_16_activations.shape
            layer_16_flat = layer_16_activations.view(-1, original_shape[-1])  # [batch*seq, hidden]
            
            # SAE processing
            primary_features, primary_reconstruction = self.joint_model.primary_sae(
                layer_16_flat.to(self.joint_model.primary_sae.encoder[0].weight.dtype)
            )
            
            # NFM Linear pathway ONLY
            if self.joint_model.nfm.linear is not None:
                linear_output = self.joint_model.nfm.linear(primary_features)
            else:
                linear_output = torch.zeros_like(primary_reconstruction)
            
            # Reshape components to match original shape
            primary_reconstruction_reshaped = primary_reconstruction.view(original_shape)
            linear_output_reshaped = linear_output.view(original_shape)
            
            # Two-way combination: SAE reconstruction + Linear component + 0
            combined_output = primary_reconstruction_reshaped + linear_output_reshaped
            
            # Return modified activations
            return (combined_output,) + output[1:]
        
        # Register hook at target layer
        target_layer_module = self.base_model.model.layers[self.target_layer]
        hook_handle = target_layer_module.register_forward_hook(sae_nfm_linear_hook)
        
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

    def forward_with_sae_nfm_nonlinear_only(self, input_ids, attention_mask):
        """
        Forward pass with SAE + NFM Nonlinear component only (no linear).
        
        Pipeline: Layer 16 → SAE → NFM Nonlinear → SAE + 0 + Nonlinear
        """
        def sae_nfm_nonlinear_hook(module, input, output):
            """Hook to apply SAE + NFM Nonlinear only intervention at layer 16."""
            layer_16_activations = output[0]  # [batch, seq, hidden]
            original_shape = layer_16_activations.shape
            layer_16_flat = layer_16_activations.view(-1, original_shape[-1])  # [batch*seq, hidden]
            
            # SAE processing
            primary_features, primary_reconstruction = self.joint_model.primary_sae(
                layer_16_flat.to(self.joint_model.primary_sae.encoder[0].weight.dtype)
            )
            
            # NFM Nonlinear pathway ONLY
            # Step 1: Feature embeddings
            all_embeddings = self.joint_model.nfm.feature_embeddings.weight
            weighted_embeddings = primary_features.unsqueeze(-1) * all_embeddings.unsqueeze(0)
            
            sum_embeddings = torch.sum(weighted_embeddings, dim=1)
            square_embeddings = weighted_embeddings ** 2
            sum_squares = torch.sum(square_embeddings, dim=1)
            
            interaction_vector = 0.5 * (sum_embeddings ** 2 - sum_squares)
            
            # Step 2: Interaction MLP
            nonlinear_output = self.joint_model.nfm.interaction_mlp(interaction_vector)
            
            # Reshape components to match original shape
            primary_reconstruction_reshaped = primary_reconstruction.view(original_shape)
            nonlinear_output_reshaped = nonlinear_output.view(original_shape)
            
            # Two-way combination: SAE reconstruction + 0 + Nonlinear component
            combined_output = primary_reconstruction_reshaped + nonlinear_output_reshaped
            
            # Return modified activations
            return (combined_output,) + output[1:]
        
        # Register hook at target layer
        target_layer_module = self.base_model.model.layers[self.target_layer]
        hook_handle = target_layer_module.register_forward_hook(sae_nfm_nonlinear_hook)
        
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
        """Forward pass with full SAE + NFM pipeline (both linear and nonlinear)."""
        def sae_nfm_full_hook(module, input, output):
            """Hook to apply full SAE+NFM intervention at layer 16."""
            layer_16_activations = output[0]  # [batch, seq, hidden]
            original_shape = layer_16_activations.shape
            layer_16_flat = layer_16_activations.view(-1, original_shape[-1])  # [batch*seq, hidden]
            
            # Use the joint model directly
            final_reconstruction, _, _, _, _ = self.joint_model(layer_16_flat)
            
            # Reshape to match original shape
            final_reconstruction_reshaped = final_reconstruction.view(original_shape)
            
            # Return modified activations
            return (final_reconstruction_reshaped,) + output[1:]
        
        # Register hook at target layer
        target_layer_module = self.base_model.model.layers[self.target_layer]
        hook_handle = target_layer_module.register_forward_hook(sae_nfm_full_hook)
        
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

    def calculate_cross_entropy(self, original_logits, modified_logits, input_ids):
        """Calculate cross-entropy loss for next-token prediction accuracy."""
        # Shift for next-token prediction
        shifted_original = original_logits[:, :-1, :].contiguous()
        shifted_modified = modified_logits[:, :-1, :].contiguous()
        shifted_targets = input_ids[:, 1:].contiguous()
        
        # Calculate cross-entropy losses
        original_ce = F.cross_entropy(
            shifted_original.view(-1, shifted_original.size(-1)), 
            shifted_targets.view(-1), 
            reduction='none'
        ).view(shifted_targets.shape)
        
        modified_ce = F.cross_entropy(
            shifted_modified.view(-1, shifted_modified.size(-1)), 
            shifted_targets.view(-1), 
            reduction='none'
        ).view(shifted_targets.shape)
        
        return original_ce, modified_ce

    def run_joint_component_experiment(self, wiki_texts, num_epsilon_samples_per_text=3, batch_size=4):
        """
        Run KL divergence and cross-entropy comparison experiment with joint NFM component ablation.
        
        Tests 5 conditions:
        1. ε-random substitution
        2. SAE only  
        3. SAE + NFM Linear only
        4. SAE + NFM Nonlinear only
        5. SAE + NFM Full
        """
        print(f"\n=== JOINT COMPONENT KL DIVERGENCE & CROSS-ENTROPY EXPERIMENT ===")
        print(f"Wiki texts: {len(wiki_texts)}")
        print(f"ε-random samples per text: {num_epsilon_samples_per_text}")
        print(f"Batch size: {batch_size}")
        print(f"Max token length: {self.max_token_length}")
        print(f"Testing: SAE only, SAE+NFM Linear, SAE+NFM Nonlinear, SAE+NFM Full, ε-random")
        
        results = []
        
        for batch_start in tqdm(range(0, len(wiki_texts), batch_size), desc="Processing joint component batches"):
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
                sae_nfm_linear_logits = self.forward_with_sae_nfm_linear_only(input_ids, attention_mask)
                sae_nfm_nonlinear_logits = self.forward_with_sae_nfm_nonlinear_only(input_ids, attention_mask)
                sae_nfm_full_logits = self.forward_with_sae_nfm_full(input_ids, attention_mask)
                
                # Calculate KL divergences for all conditions
                kl_sae = self.calculate_kl_divergence(original_logits, sae_logits)
                kl_sae_nfm_linear = self.calculate_kl_divergence(original_logits, sae_nfm_linear_logits)
                kl_sae_nfm_nonlinear = self.calculate_kl_divergence(original_logits, sae_nfm_nonlinear_logits)
                kl_sae_nfm_full = self.calculate_kl_divergence(original_logits, sae_nfm_full_logits)
                
                # Calculate cross-entropy losses
                original_ce_sae, modified_ce_sae = self.calculate_cross_entropy(original_logits, sae_logits, input_ids)
                original_ce_linear, modified_ce_linear = self.calculate_cross_entropy(original_logits, sae_nfm_linear_logits, input_ids)
                original_ce_nonlinear, modified_ce_nonlinear = self.calculate_cross_entropy(original_logits, sae_nfm_nonlinear_logits, input_ids)
                original_ce_full, modified_ce_full = self.calculate_cross_entropy(original_logits, sae_nfm_full_logits, input_ids)
                
                # For ε-random, run multiple samples
                kl_epsilon_samples = []
                ce_epsilon_samples = []
                epsilon_distances = []
                
                for sample_idx in range(num_epsilon_samples_per_text):
                    epsilon_logits, epsilon_distance = self.forward_with_epsilon_random(input_ids, attention_mask)
                    kl_epsilon = self.calculate_kl_divergence(original_logits, epsilon_logits)
                    original_ce_epsilon, modified_ce_epsilon = self.calculate_cross_entropy(original_logits, epsilon_logits, input_ids)
                    
                    kl_epsilon_samples.append(kl_epsilon)
                    ce_epsilon_samples.append(modified_ce_epsilon)
                    epsilon_distances.append(epsilon_distance)
                
                # Process results for each sequence position
                batch_size_actual, seq_len = kl_sae.shape
                
                for batch_idx in range(batch_size_actual):
                    # Get actual sequence length (excluding padding)
                    actual_seq_len = torch.sum(attention_mask[batch_idx]).item()
                    
                    # Get text for this batch item
                    text_content = batch_texts[batch_idx] if batch_idx < len(batch_texts) else ""
                    
                    for pos_idx in range(min(actual_seq_len - 1, seq_len - 1)):  # -1 for cross-entropy shift
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
                            'kl_divergence': float(kl_sae[batch_idx, pos_idx].cpu()),
                            'cross_entropy': float(modified_ce_sae[batch_idx, pos_idx].cpu()),
                            'original_cross_entropy': float(original_ce_sae[batch_idx, pos_idx].cpu())
                        })
                        results.append(sae_result)
                        
                        # SAE + NFM Linear only results
                        sae_nfm_linear_result = base_result.copy()
                        sae_nfm_linear_result.update({
                            'intervention_type': 'SAE_NFM_linear_only',
                            'kl_divergence': float(kl_sae_nfm_linear[batch_idx, pos_idx].cpu()),
                            'cross_entropy': float(modified_ce_linear[batch_idx, pos_idx].cpu()),
                            'original_cross_entropy': float(original_ce_linear[batch_idx, pos_idx].cpu())
                        })
                        results.append(sae_nfm_linear_result)
                        
                        # SAE + NFM Nonlinear only results
                        sae_nfm_nonlinear_result = base_result.copy()
                        sae_nfm_nonlinear_result.update({
                            'intervention_type': 'SAE_NFM_nonlinear_only',
                            'kl_divergence': float(kl_sae_nfm_nonlinear[batch_idx, pos_idx].cpu()),
                            'cross_entropy': float(modified_ce_nonlinear[batch_idx, pos_idx].cpu()),
                            'original_cross_entropy': float(original_ce_nonlinear[batch_idx, pos_idx].cpu())
                        })
                        results.append(sae_nfm_nonlinear_result)
                        
                        # SAE + NFM Full results
                        sae_nfm_full_result = base_result.copy()
                        sae_nfm_full_result.update({
                            'intervention_type': 'SAE_NFM_full',
                            'kl_divergence': float(kl_sae_nfm_full[batch_idx, pos_idx].cpu()),
                            'cross_entropy': float(modified_ce_full[batch_idx, pos_idx].cpu()),
                            'original_cross_entropy': float(original_ce_full[batch_idx, pos_idx].cpu())
                        })
                        results.append(sae_nfm_full_result)
                        
                        # ε-random results (multiple samples)
                        for sample_idx, (kl_epsilon, ce_epsilon, epsilon_distance) in enumerate(zip(kl_epsilon_samples, ce_epsilon_samples, epsilon_distances)):
                            epsilon_result = base_result.copy()
                            epsilon_result.update({
                                'intervention_type': 'epsilon_random',
                                'kl_divergence': float(kl_epsilon[batch_idx, pos_idx].cpu()),
                                'cross_entropy': float(ce_epsilon[batch_idx, pos_idx].cpu()),
                                'original_cross_entropy': float(original_ce_sae[batch_idx, pos_idx].cpu()),  # Use same original
                                'sample_idx': sample_idx,
                                'epsilon_distance': epsilon_distance
                            })
                            results.append(epsilon_result)
                
            except Exception as e:
                print(f"Error processing batch starting at {batch_start}: {e}")
                continue
        
        print(f"Collected {len(results):,} measurements from WikiText-103 joint component analysis")
        return pd.DataFrame(results)

    def analyze_joint_component_results(self, results_df):
        """Analyze joint component KL divergence and cross-entropy results."""
        print(f"\n=== JOINT COMPONENT ANALYSIS RESULTS ===")
        
        # Calculate summary statistics by intervention type
        summary_stats_kl = results_df.groupby('intervention_type')['kl_divergence'].agg([
            'count', 'mean', 'std', 'median', 'min', 'max'
        ]).round(6)
        
        summary_stats_ce = results_df.groupby('intervention_type')['cross_entropy'].agg([
            'count', 'mean', 'std', 'median', 'min', 'max'
        ]).round(6)
        
        print(f"\nKL Divergence Summary Statistics:")
        print(summary_stats_kl)
        
        print(f"\nCross-Entropy Summary Statistics:")
        print(summary_stats_ce)
        
        # Calculate ratios relative to ε-random baseline
        epsilon_mean_kl = summary_stats_kl.loc['epsilon_random', 'mean']
        epsilon_mean_ce = summary_stats_ce.loc['epsilon_random', 'mean']
        
        sae_mean_kl = summary_stats_kl.loc['SAE_only', 'mean']
        sae_mean_ce = summary_stats_ce.loc['SAE_only', 'mean']
        
        sae_nfm_linear_mean_kl = summary_stats_kl.loc['SAE_NFM_linear_only', 'mean']
        sae_nfm_linear_mean_ce = summary_stats_ce.loc['SAE_NFM_linear_only', 'mean']
        
        sae_nfm_nonlinear_mean_kl = summary_stats_kl.loc['SAE_NFM_nonlinear_only', 'mean']
        sae_nfm_nonlinear_mean_ce = summary_stats_ce.loc['SAE_NFM_nonlinear_only', 'mean']
        
        sae_nfm_full_mean_kl = summary_stats_kl.loc['SAE_NFM_full', 'mean']
        sae_nfm_full_mean_ce = summary_stats_ce.loc['SAE_NFM_full', 'mean']
        
        print(f"\nKL Divergence Ratios (vs. ε-random baseline):")
        print(f"  SAE only vs ε-random:           {sae_mean_kl / epsilon_mean_kl:.3f}x")
        print(f"  SAE+NFM Linear vs ε-random:     {sae_nfm_linear_mean_kl / epsilon_mean_kl:.3f}x")
        print(f"  SAE+NFM Nonlinear vs ε-random:  {sae_nfm_nonlinear_mean_kl / epsilon_mean_kl:.3f}x")
        print(f"  SAE+NFM Full vs ε-random:       {sae_nfm_full_mean_kl / epsilon_mean_kl:.3f}x")
        
        print(f"\nCross-Entropy Ratios (vs. ε-random baseline):")
        print(f"  SAE only vs ε-random:           {sae_mean_ce / epsilon_mean_ce:.3f}x")
        print(f"  SAE+NFM Linear vs ε-random:     {sae_nfm_linear_mean_ce / epsilon_mean_ce:.3f}x")
        print(f"  SAE+NFM Nonlinear vs ε-random:  {sae_nfm_nonlinear_mean_ce / epsilon_mean_ce:.3f}x")
        print(f"  SAE+NFM Full vs ε-random:       {sae_nfm_full_mean_ce / epsilon_mean_ce:.3f}x")
        
        # Calculate improvement ratios (vs. SAE only baseline)
        print(f"\nKL Divergence Improvements (vs. SAE only baseline):")
        linear_improvement_kl = (1 - sae_nfm_linear_mean_kl / sae_mean_kl) * 100
        nonlinear_improvement_kl = (1 - sae_nfm_nonlinear_mean_kl / sae_mean_kl) * 100
        full_improvement_kl = (1 - sae_nfm_full_mean_kl / sae_mean_kl) * 100
        
        print(f"  SAE+NFM Linear vs SAE only:     {sae_nfm_linear_mean_kl / sae_mean_kl:.3f}x ({linear_improvement_kl:.1f}% improvement)")
        print(f"  SAE+NFM Nonlinear vs SAE only:  {sae_nfm_nonlinear_mean_kl / sae_mean_kl:.3f}x ({nonlinear_improvement_kl:.1f}% improvement)")
        print(f"  SAE+NFM Full vs SAE only:       {sae_nfm_full_mean_kl / sae_mean_kl:.3f}x ({full_improvement_kl:.1f}% improvement)")
        
        print(f"\nCross-Entropy Improvements (vs. SAE only baseline):")
        linear_improvement_ce = (1 - sae_nfm_linear_mean_ce / sae_mean_ce) * 100
        nonlinear_improvement_ce = (1 - sae_nfm_nonlinear_mean_ce / sae_mean_ce) * 100
        full_improvement_ce = (1 - sae_nfm_full_mean_ce / sae_mean_ce) * 100
        
        print(f"  SAE+NFM Linear vs SAE only:     {sae_nfm_linear_mean_ce / sae_mean_ce:.3f}x ({linear_improvement_ce:.1f}% improvement)")
        print(f"  SAE+NFM Nonlinear vs SAE only:  {sae_nfm_nonlinear_mean_ce / sae_mean_ce:.3f}x ({nonlinear_improvement_ce:.1f}% improvement)")
        print(f"  SAE+NFM Full vs SAE only:       {sae_nfm_full_mean_ce / sae_mean_ce:.3f}x ({full_improvement_ce:.1f}% improvement)")
        
        # Statistical tests
        epsilon_kls = results_df[results_df['intervention_type'] == 'epsilon_random']['kl_divergence']
        sae_kls = results_df[results_df['intervention_type'] == 'SAE_only']['kl_divergence']
        sae_nfm_linear_kls = results_df[results_df['intervention_type'] == 'SAE_NFM_linear_only']['kl_divergence']
        sae_nfm_nonlinear_kls = results_df[results_df['intervention_type'] == 'SAE_NFM_nonlinear_only']['kl_divergence']
        sae_nfm_full_kls = results_df[results_df['intervention_type'] == 'SAE_NFM_full']['kl_divergence']
        
        # T-tests vs. SAE only for KL divergence
        linear_vs_sae_stat_kl, linear_vs_sae_p_kl = stats.ttest_ind(sae_nfm_linear_kls, sae_kls)
        nonlinear_vs_sae_stat_kl, nonlinear_vs_sae_p_kl = stats.ttest_ind(sae_nfm_nonlinear_kls, sae_kls)
        full_vs_sae_stat_kl, full_vs_sae_p_kl = stats.ttest_ind(sae_nfm_full_kls, sae_kls)
        
        print(f"\nStatistical Tests vs. SAE only (KL Divergence, t-tests):")
        print(f"  SAE+NFM Linear vs SAE: t={linear_vs_sae_stat_kl:.3f}, p={linear_vs_sae_p_kl:.6f}")
        print(f"  SAE+NFM Nonlinear vs SAE: t={nonlinear_vs_sae_stat_kl:.3f}, p={nonlinear_vs_sae_p_kl:.6f}")
        print(f"  SAE+NFM Full vs SAE: t={full_vs_sae_stat_kl:.3f}, p={full_vs_sae_p_kl:.6f}")
        
        # Cross-entropy statistical tests
        epsilon_ces = results_df[results_df['intervention_type'] == 'epsilon_random']['cross_entropy']
        sae_ces = results_df[results_df['intervention_type'] == 'SAE_only']['cross_entropy']
        sae_nfm_linear_ces = results_df[results_df['intervention_type'] == 'SAE_NFM_linear_only']['cross_entropy']
        sae_nfm_nonlinear_ces = results_df[results_df['intervention_type'] == 'SAE_NFM_nonlinear_only']['cross_entropy']
        sae_nfm_full_ces = results_df[results_df['intervention_type'] == 'SAE_NFM_full']['cross_entropy']
        
        # T-tests vs. SAE only for cross-entropy
        linear_vs_sae_stat_ce, linear_vs_sae_p_ce = stats.ttest_ind(sae_nfm_linear_ces, sae_ces)
        nonlinear_vs_sae_stat_ce, nonlinear_vs_sae_p_ce = stats.ttest_ind(sae_nfm_nonlinear_ces, sae_ces)
        full_vs_sae_stat_ce, full_vs_sae_p_ce = stats.ttest_ind(sae_nfm_full_ces, sae_ces)
        
        print(f"\nStatistical Tests vs. SAE only (Cross-Entropy, t-tests):")
        print(f"  SAE+NFM Linear vs SAE: t={linear_vs_sae_stat_ce:.3f}, p={linear_vs_sae_p_ce:.6f}")
        print(f"  SAE+NFM Nonlinear vs SAE: t={nonlinear_vs_sae_stat_ce:.3f}, p={nonlinear_vs_sae_p_ce:.6f}")
        print(f"  SAE+NFM Full vs SAE: t={full_vs_sae_stat_ce:.3f}, p={full_vs_sae_p_ce:.6f}")
        
        # Component effectiveness analysis
        print(f"\n=== JOINT COMPONENT EFFECTIVENESS ANALYSIS ===")
        
        print(f"\nKL Divergence Component Contributions:")
        print(f"  Linear component alone:      {linear_improvement_kl:.1f}% improvement over SAE")
        print(f"  Nonlinear component alone:   {nonlinear_improvement_kl:.1f}% improvement over SAE")
        print(f"  Both components together:    {full_improvement_kl:.1f}% improvement over SAE")
        
        print(f"\nCross-Entropy Component Contributions:")
        print(f"  Linear component alone:      {linear_improvement_ce:.1f}% improvement over SAE")
        print(f"  Nonlinear component alone:   {nonlinear_improvement_ce:.1f}% improvement over SAE")
        print(f"  Both components together:    {full_improvement_ce:.1f}% improvement over SAE")
        
        # Check additivity (whether combined effect equals sum of individual effects)
        expected_combined_kl = linear_improvement_kl + nonlinear_improvement_kl
        expected_combined_ce = linear_improvement_ce + nonlinear_improvement_ce
        
        print(f"  ➡️  KL: Combined vs Sum of Parts: {full_improvement_kl:.1f}% vs {expected_combined_kl:.1f}%")
        print(f"  ➡️  CE: Combined vs Sum of Parts: {full_improvement_ce:.1f}% vs {expected_combined_ce:.1f}%")
        
        # Statistical significance checks
        significant_tests_kl = []
        significant_tests_ce = []
        
        if linear_vs_sae_p_kl < 0.05:
            significant_tests_kl.append("Linear component (KL)")
        if nonlinear_vs_sae_p_kl < 0.05:
            significant_tests_kl.append("Nonlinear component (KL)")
        if full_vs_sae_p_kl < 0.05:
            significant_tests_kl.append("Full NFM (KL)")
        
        if linear_vs_sae_p_ce < 0.05:
            significant_tests_ce.append("Linear component (CE)")
        if nonlinear_vs_sae_p_ce < 0.05:
            significant_tests_ce.append("Nonlinear component (CE)")
        if full_vs_sae_p_ce < 0.05:
            significant_tests_ce.append("Full NFM (CE)")
        
        all_significant = significant_tests_kl + significant_tests_ce
        
        if all_significant:
            print(f"\n✓ STATISTICALLY SIGNIFICANT improvements: {', '.join(all_significant)}")
        else:
            print(f"\n⚠️  No statistically significant improvements detected")
        
        # Overall conclusions
        total_measurements = len(results_df)
        
        print(f"\nSample Size:")
        print(f"  Total measurements: {total_measurements:,}")
        print(f"  Measurements per condition: {total_measurements // 5:,}")
        
        return {
            'summary_stats_kl': summary_stats_kl.to_dict(),
            'summary_stats_ce': summary_stats_ce.to_dict(),
            'improvements_kl': {
                'linear_improvement_pct': linear_improvement_kl,
                'nonlinear_improvement_pct': nonlinear_improvement_kl,
                'full_improvement_pct': full_improvement_kl
            },
            'improvements_ce': {
                'linear_improvement_pct': linear_improvement_ce,
                'nonlinear_improvement_pct': nonlinear_improvement_ce,
                'full_improvement_pct': full_improvement_ce
            },
            'statistical_tests_kl': {
                'linear_vs_sae': {'statistic': linear_vs_sae_stat_kl, 'p_value': linear_vs_sae_p_kl},
                'nonlinear_vs_sae': {'statistic': nonlinear_vs_sae_stat_kl, 'p_value': nonlinear_vs_sae_p_kl},
                'full_vs_sae': {'statistic': full_vs_sae_stat_kl, 'p_value': full_vs_sae_p_kl}
            },
            'statistical_tests_ce': {
                'linear_vs_sae': {'statistic': linear_vs_sae_stat_ce, 'p_value': linear_vs_sae_p_ce},
                'nonlinear_vs_sae': {'statistic': nonlinear_vs_sae_stat_ce, 'p_value': nonlinear_vs_sae_p_ce},
                'full_vs_sae': {'statistic': full_vs_sae_stat_ce, 'p_value': full_vs_sae_p_ce}
            },
            'conclusions': {
                'statistically_significant': len(all_significant) > 0,
                'total_measurements': total_measurements
            }
        }

def create_joint_component_plots(results_df, analysis_results, output_dir):
    """Create visualizations of joint component analysis results."""
    output_dir = Path(output_dir)
    plots_dir = output_dir / "joint_component_plots"
    plots_dir.mkdir(exist_ok=True)
    
    plt.style.use('default')
    sns.set_palette("Set2")
    
    # 1. Dual metric box plot comparing all conditions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    intervention_order = ['epsilon_random', 'SAE_only', 'SAE_NFM_linear_only', 'SAE_NFM_nonlinear_only', 'SAE_NFM_full']
    intervention_labels = ['ε-random', 'SAE only', 'SAE + Linear', 'SAE + Nonlinear', 'SAE + Full']
    
    # KL Divergence box plot
    kl_box_data = [results_df[results_df['intervention_type'] == itype]['kl_divergence'] 
                   for itype in intervention_order]
    
    bp1 = ax1.boxplot(kl_box_data, labels=intervention_labels, patch_artist=True)
    colors = ['lightblue', 'lightcoral', 'lightyellow', 'lightgreen', 'darkgreen']
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
    
    ax1.set_ylabel('KL Divergence', fontsize=14)
    ax1.set_title('Joint Architecture: KL Divergence by Component\n'
                  f'{len(results_df):,} measurements', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=15)
    
    # Cross-Entropy box plot
    ce_box_data = [results_df[results_df['intervention_type'] == itype]['cross_entropy'] 
                   for itype in intervention_order]
    
    bp2 = ax2.boxplot(ce_box_data, labels=intervention_labels, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_ylabel('Cross-Entropy Loss', fontsize=14)
    ax2.set_title('Joint Architecture: Cross-Entropy by Component\n'
                  f'{len(results_df):,} measurements', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=15)
    
    # Add improvement annotations
    improvements_kl = analysis_results['improvements_kl']
    improvements_ce = analysis_results['improvements_ce']
    
    ax1.text(0.02, 0.98, f"KL Improvements over SAE:\n"
                          f"Linear: {improvements_kl['linear_improvement_pct']:.1f}%\n"
                          f"Nonlinear: {improvements_kl['nonlinear_improvement_pct']:.1f}%\n"
                          f"Full: {improvements_kl['full_improvement_pct']:.1f}%", 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.text(0.02, 0.98, f"CE Improvements over SAE:\n"
                          f"Linear: {improvements_ce['linear_improvement_pct']:.1f}%\n"
                          f"Nonlinear: {improvements_ce['nonlinear_improvement_pct']:.1f}%\n"
                          f"Full: {improvements_ce['full_improvement_pct']:.1f}%", 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    dual_boxplot_path = plots_dir / "joint_component_dual_boxplot.png"
    plt.savefig(dual_boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Component effectiveness comparison bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    components = ['Linear\nOnly', 'Nonlinear\nOnly', 'Full\nJoint']
    kl_improvements = [
        improvements_kl['linear_improvement_pct'],
        improvements_kl['nonlinear_improvement_pct'], 
        improvements_kl['full_improvement_pct']
    ]
    ce_improvements = [
        improvements_ce['linear_improvement_pct'],
        improvements_ce['nonlinear_improvement_pct'],
        improvements_ce['full_improvement_pct']
    ]
    
    # KL improvements
    bars1 = ax1.bar(components, kl_improvements, color=['lightyellow', 'lightgreen', 'darkgreen'])
    for bar, val in zip(bars1, kl_improvements):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_ylabel('KL Divergence Improvement (%)', fontsize=12)
    ax1.set_title('Joint Architecture: Component Effectiveness\n(KL Divergence)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Cross-entropy improvements
    bars2 = ax2.bar(components, ce_improvements, color=['lightyellow', 'lightgreen', 'darkgreen'])
    for bar, val in zip(bars2, ce_improvements):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_ylabel('Cross-Entropy Improvement (%)', fontsize=12)
    ax2.set_title('Joint Architecture: Component Effectiveness\n(Cross-Entropy)', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    effectiveness_path = plots_dir / "joint_component_effectiveness.png"
    plt.savefig(effectiveness_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Component comparison scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot linear vs nonlinear improvements
    ax.scatter(improvements_kl['linear_improvement_pct'], improvements_kl['nonlinear_improvement_pct'], 
               s=150, color='red', label='KL Divergence', alpha=0.7)
    ax.scatter(improvements_ce['linear_improvement_pct'], improvements_ce['nonlinear_improvement_pct'], 
               s=150, color='blue', label='Cross-Entropy', alpha=0.7)
    
    # Add diagonal line
    max_val = max(improvements_kl['linear_improvement_pct'], improvements_kl['nonlinear_improvement_pct'],
                  improvements_ce['linear_improvement_pct'], improvements_ce['nonlinear_improvement_pct'])
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Equal Contribution')
    
    ax.set_xlabel('Linear Component Improvement (%)', fontsize=12)
    ax.set_ylabel('Nonlinear Component Improvement (%)', fontsize=12)
    ax.set_title('Joint Architecture: Linear vs Nonlinear Component Effectiveness\n'
                 'Above diagonal = Nonlinear better, Below = Linear better', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    ax.annotate(f'KL: Linear {improvements_kl["linear_improvement_pct"]:.1f}%, Nonlinear {improvements_kl["nonlinear_improvement_pct"]:.1f}%', 
                (improvements_kl['linear_improvement_pct'], improvements_kl['nonlinear_improvement_pct']), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    
    ax.annotate(f'CE: Linear {improvements_ce["linear_improvement_pct"]:.1f}%, Nonlinear {improvements_ce["nonlinear_improvement_pct"]:.1f}%', 
                (improvements_ce['linear_improvement_pct'], improvements_ce['nonlinear_improvement_pct']), 
                xytext=(10, -30), textcoords='offset points',
                bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))
    
    plt.tight_layout()
    comparison_path = plots_dir / "joint_component_comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Joint component plots saved to: {plots_dir}")
    return {
        'plots_directory': str(plots_dir),
        'dual_boxplot': str(dual_boxplot_path),
        'effectiveness_plot': str(effectiveness_path),
        'comparison_plot': str(comparison_path)
    }

def load_joint_model(args):
    """Load all required models including the joint SAE+NFM."""
    print("Loading models...")
    
    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")
    
    # Load Joint SAE+NFM model
    print("Loading Joint SAE+NFM model...")
    joint_state_dict = torch.load(args.joint_model_path, map_location=args.device)
    
    # Determine model dimensions from state dict
    if 'primary_sae.decoder.weight' in joint_state_dict:
        input_dim = joint_state_dict['primary_sae.decoder.weight'].shape[0]
        sae_features = joint_state_dict['primary_sae.decoder.weight'].shape[1]
    elif 'primary_sae.encoder.0.weight' in joint_state_dict:
        encoder_weight = joint_state_dict['primary_sae.encoder.0.weight']
        sae_features, input_dim = encoder_weight.shape
    else:
        raise ValueError("Cannot determine Joint model dimensions")
    
    # Get NFM dimensions
    nfm_embedding_dim = joint_state_dict['nfm.feature_embeddings.weight'].shape[1]
    
    # Check if linear component exists
    use_linear = 'nfm.linear.weight' in joint_state_dict
    
    joint_model = JointSAENFM(input_dim, sae_features, args.sae_k, nfm_embedding_dim, use_linear)
    joint_model.load_state_dict(joint_state_dict)
    joint_model.to(args.device)
    
    print(f"Models loaded successfully!")
    print(f"Joint SAE: {input_dim} → {sae_features} (k={args.sae_k})")
    print(f"Joint NFM: {sae_features} features → {nfm_embedding_dim} embedding dim → {input_dim} output")
    print(f"Linear component: {'Enabled' if use_linear else 'Disabled'}")
    
    return tokenizer, base_model, joint_model

def save_joint_component_results(results_df, analysis_results, output_dir):
    """Save joint component analysis results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw results
    results_csv_path = output_dir / "joint_component_kl_ce_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"Joint component results saved to: {results_csv_path}")
    
    # Save analysis summary
    analysis_json_path = output_dir / "joint_component_analysis.json"
    
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
    print(f"Joint component analysis saved to: {analysis_json_path}")

def main():
    parser = argparse.ArgumentParser(description="Joint Component KL Divergence & Cross-Entropy Analysis")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base LLM model")
    parser.add_argument("--joint_model_path", type=str, required=True, help="Path to Joint SAE+NFM model")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    
    parser.add_argument("--num_wiki_samples", type=int, default=5000,
                       help="Number of WikiText-103 chunks to analyze (default: 5000)")
    parser.add_argument("--num_epsilon_samples", type=int, default=3,
                       help="Number of ε-random samples per text")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for processing")
    parser.add_argument("--max_token_length", type=int, default=128,
                       help="Maximum token length for sequences")
    
    # TopK parameters
    parser.add_argument("--sae_k", type=int, default=1024, 
                       help="TopK parameter for SAE")
    
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
            logging.FileHandler(output_dir / "joint_component_analysis.log"),
            logging.StreamHandler()
        ]
    )
    
    print(f"Starting Joint Component KL Divergence & Cross-Entropy Analysis")
    print(f"Testing which joint architecture components drive pathological error reduction:")
    print(f"  • SAE only (baseline)")
    print(f"  • SAE + NFM Linear component")  
    print(f"  • SAE + NFM Nonlinear component")
    print(f"  • SAE + NFM Full")
    print(f"  • ε-random (reference)")
    print(f"SAE TopK: {args.sae_k}")
    print(f"WikiText-103 samples: {args.num_wiki_samples:,}")
    print(f"Max token length: {args.max_token_length}")
    print(f"Metrics: KL Divergence + Cross-Entropy Loss")
    
    # Load models
    tokenizer, base_model, joint_model = load_joint_model(args)
    
    # Initialize analyzer
    analyzer = JointComponentKLAnalyzer(
        joint_model, tokenizer, base_model, 
        args.device, args.target_layer, args.max_token_length
    )
    
    # Load WikiText-103 dataset
    print("\n" + "="*60)
    print("LOADING WIKITEXT-103 DATASET")
    print("="*60)
    
    wiki_texts = analyzer.load_wiki_dataset(args.num_wiki_samples)
    
    # Run joint component experiment
    print("\n" + "="*60)
    print("RUNNING JOINT COMPONENT EXPERIMENT")
    print("="*60)
    
    results_df = analyzer.run_joint_component_experiment(
        wiki_texts, args.num_epsilon_samples, args.batch_size
    )
    
    if len(results_df) == 0:
        print("Error: No results collected. Check your model paths and data.")
        return
    
    # Analyze results
    print("\n" + "="*60)
    print("ANALYZING JOINT COMPONENT RESULTS")
    print("="*60)
    
    analysis_results = analyzer.analyze_joint_component_results(results_df)
    
    # Create plots
    print("\n" + "="*60)
    print("CREATING JOINT COMPONENT PLOTS")
    print("="*60)
    
    plot_results = create_joint_component_plots(results_df, analysis_results, args.output_dir)
    
    # Save results
    print("\n" + "="*60)
    print("SAVING JOINT COMPONENT RESULTS")
    print("="*60)
    
    save_joint_component_results(results_df, analysis_results, args.output_dir)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL JOINT COMPONENT SUMMARY")
    print("="*60)
    
    improvements_kl = analysis_results['improvements_kl']
    improvements_ce = analysis_results['improvements_ce']
    conclusions = analysis_results['conclusions']
    
    print(f"\nCOMPONENT EFFECTIVENESS (KL DIVERGENCE):")
    print(f"  • Linear component:      {improvements_kl['linear_improvement_pct']:.1f}% improvement")
    print(f"  • Nonlinear component:   {improvements_kl['nonlinear_improvement_pct']:.1f}% improvement")
    print(f"  • Full joint:            {improvements_kl['full_improvement_pct']:.1f}% improvement")
    
    print(f"\nCOMPONENT EFFECTIVENESS (CROSS-ENTROPY):")
    print(f"  • Linear component:      {improvements_ce['linear_improvement_pct']:.1f}% improvement")
    print(f"  • Nonlinear component:   {improvements_ce['nonlinear_improvement_pct']:.1f}% improvement")
    print(f"  • Full joint:            {improvements_ce['full_improvement_pct']:.1f}% improvement")
    
    print(f"\nKEY FINDINGS:")
    
    # Component importance
    if conclusions['linear_more_important_kl'] and conclusions['linear_more_important_ce']:
        print(f"  🎯 LINEAR COMPONENT is more important for both metrics")
    elif conclusions['nonlinear_more_important_kl'] and conclusions['nonlinear_more_important_ce']:
        print(f"  🎯 NONLINEAR COMPONENT is more important for both metrics")
    else:
        print(f"  🔄 Mixed importance: Different components excel at different metrics")
    
    if conclusions['statistically_significant']:
        print(f"  ✓ Results are statistically significant")
    else:
        print(f"  ⚠️  Results not statistically significant - consider larger sample")
    
    print(f"\nCOMPARISON INSIGHTS:")
    print(f"  • Joint training allows end-to-end optimization of both components")
    print(f"  • KL divergence measures distributional similarity (behavioral fidelity)")
    print(f"  • Cross-entropy measures next-token prediction accuracy")
    print(f"  • Both metrics show consistent component effectiveness patterns")
    
    print(f"\nTotal measurements: {conclusions['total_measurements']:,}")
    print(f"All joint component analysis results saved to: {args.output_dir}")
    
    print(f"\nNext steps:")
    print(f"  1. Compare with sequential training component analysis")
    print(f"  2. Analyze feature interpretability in joint vs sequential models")
    print(f"  3. Test on downstream tasks to validate behavioral improvements")
    print(f"  4. Investigate optimal component balance for different use cases")

if __name__ == "__main__":
    main()