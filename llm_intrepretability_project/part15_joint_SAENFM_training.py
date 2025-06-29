"""
part3_joint_SAE_NFM_training.py

Joint End-to-End Training of TopK SAE + NFM for Monosemantic Feature Integration
Based on part2d_residual_NFM_streaming.py and part1c_sae_topK_implementation.py

This script trains both the Primary SAE and NFM components together in a single
end-to-end optimization, allowing the SAE to learn monosemantic features that
work well with the NFM integration component.

Architecture:
Layer 16 → Primary SAE → NFM (Linear + Interaction) → Final Reconstruction
                ↓             ↓            ↓
         Primary Recon + NFM Linear + NFM Interaction
"""

# Collecting 10000 tokens for chunk...
# Collecting activations: 10112it [00:02, 3439.12it/s]
# Collected 10112 tokens
# Chunk activations shape: torch.Size([10112, 3200])
# Training on chunk: 9101 train samples, 1011 val samples
# Steps for this chunk: 143

# Chunk SAE Feature Sparsity Analysis:
# Average active features per sample: 1024.0
# Median active features per sample: 1024.0
# Max active features per sample: 1024
# Min active features per sample: 1024

# Step 462000
# Train - Total Loss: 0.1783
# Train - Recon Loss: 0.1783
# Train - Linear Magnitude: 0.3228
# Train - Interaction Magnitude: 0.0216
# Train - Linear Contribution: 93.7%
# Val - Total Loss: 0.1788
# Val - Linear Magnitude: 0.3157
# Val - Interaction Magnitude: 0.0216
# Val - Linear Contribution: 93.6%

# Step 673000
# Train - Total Loss: 0.1525
# Train - Recon Loss: 0.1525
# Train - Linear Magnitude: 0.3088
# Train - Interaction Magnitude: 0.0223
# Train - Linear Contribution: 93.3%
# Val - Total Loss: 0.1649
# Val - Linear Magnitude: 0.3180
# Val - Interaction Magnitude: 0.0224
# Val - Linear Contribution: 93.4%
# Completed chunk 471/500

# Step 693000
# Train - Total Loss: 0.1421
# Train - Recon Loss: 0.1421
# Train - Linear Magnitude: 0.3179
# Train - Interaction Magnitude: 0.0226
# Train - Linear Contribution: 93.4%
# Val - Total Loss: 0.1523
# Val - Linear Magnitude: 0.3195
# Val - Interaction Magnitude: 0.0226
# Val - Linear Contribution: 93.4%
# Completed chunk 485/500

# ==================================================
# FINAL VALIDATION RESULTS
# ==================================================
# Collecting 10000 tokens for chunk...
# Collecting activations: 10112it [00:18, 542.47it/s] 
# Collected 10112 tokens
# Original data MSE (baseline): 0.998535
# SAE only reconstruction error: 0.274658
# Joint SAE+NFM reconstruction error: 0.161255
# Error reduction: 41.29%
# Relative error (SAE only): 27.51%
# Relative error (Joint): 16.15%

# Component Analysis:
# Linear component magnitude: 0.346924
# Interaction component magnitude: 0.022293
# Linear contribution: 94.0%
# Interaction contribution: 6.0%

# Feature Sparsity (controlled by TopK):
#   Active features: 2.05% (target: 2.05%)
#   Total features: 50,000
#   TopK parameter: 1024
# ==================================================
# COMPARISON WITH SEQUENTIAL TRAINING
# ==================================================
# Joint training results: 41.3% improvement

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
import ctypes
import json
from datetime import datetime
import argparse

# Constants (from your part2d)
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "open_llama_3b"))
BATCH_SIZE = 64  # From your part2d
NUM_TOKENS_TOTAL = 5_000_000  # Total tokens to process
CHUNK_SIZE_TOKENS = 10_000  # Process 100k tokens at a time
LEARNING_RATE = 0.0001  # From your part2d
NFM_INIT_WEIGHT = 0.05  # From your part2d
NUM_FEATURES = 50_000
NFM_K = 300
TARGET_LAYER = 16
TRAIN_STEPS = 200_000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints_joint"
CHECKPOINT_INTERVAL = 10_000
VALIDATION_INTERVAL = 1000
VALIDATION_SPLIT = 0.1
CHUNK_SIZE = 128
NFM_DROPOUT = 0.15
SPARSE_THRESHOLD = None
TOP_N_FEATURES = 512  # From your part2d
K = 1024  # TopK for SAE
USE_LINEAR_COMPONENT = True

def check_tensor(tensor, name="tensor", print_stats=True):
    """Check tensor for NaN, Inf, and basic statistics."""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if has_nan or has_inf or print_stats:
        print(f"\n--- Checking {name} ---")
        print(f"Has NaN: {has_nan}")
        print(f"Has Inf: {has_inf}")
        
        if print_stats:
            try:
                print(f"Shape: {tensor.shape}")
                print(f"Min: {tensor.min().item()}")
                print(f"Max: {tensor.max().item()}")
                print(f"Mean: {tensor.mean().item()}")
                print(f"Std: {tensor.std().item()}")
            except RuntimeError as e:
                print(f"Could not compute stats: {e}")
    
    return has_nan or has_inf

def debug_model_parameters(model, name="model"):
    """Check model parameters for NaN and Inf values."""
    print(f"\n--- Checking {name} parameters ---")
    for param_name, param in model.named_parameters():
        has_issue = check_tensor(param, f"{param_name}", print_stats=False)
        if has_issue:
            print(f"Issue detected in {param_name}")
            check_tensor(param, f"{param_name}", print_stats=True)

def prevent_sleep():
    """Prevent Windows from sleeping during training."""
    ctypes.windll.kernel32.SetThreadExecutionState(
        0x80000002  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    )

def allow_sleep():
    """Allow Windows to sleep again."""
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)  # ES_CONTINUOUS

def save_checkpoint(model, optimizer, scheduler, step, best_loss, metrics_history, checkpoint_dir):
    """Save training checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint = {
        'step': step,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_loss': best_loss,
        'metrics_history': metrics_history,
        'timestamp': datetime.now().isoformat()
    }
    
    checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    metrics_path = checkpoint_dir / f"metrics_step_{step}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_history, f, indent=2)
    
    model_name = "best_joint_sae_nfm_model.pt"
    if metrics_history['val_best_loss'] == best_loss:
        best_model_path = checkpoint_dir / model_name
        torch.save(model.state_dict(), best_model_path)

class TopKSparseAutoencoder(nn.Module):
    """TopK SAE (from your part1c)."""
    def __init__(self, input_dim, hidden_dim, k):
        super().__init__()
        self.k = k
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        self.initialize_weights()
    
    def initialize_weights(self):
        """Use your proven initialization from part1c."""
        nn.init.kaiming_normal_(self.encoder[0].weight, mode='fan_in', nonlinearity='relu')
        # Initialize encoder weights as transpose of decoder
        self.encoder[0].weight.data = self.decoder.weight.data.T
        nn.init.zeros_(self.encoder[0].bias)
        
        print("\n--- Checking TopK SAE initialization ---")
        check_tensor(self.encoder[0].weight, "encoder.weight", True)
        check_tensor(self.encoder[0].bias, "encoder.bias", True)
        check_tensor(self.decoder.weight, "decoder.weight", True)
    
    def forward(self, x):
        # Debug input
        if torch.isnan(x).any() or torch.isinf(x).any():
            check_tensor(x, "forward_input")
            
        # Encoder forward pass
        features = self.encoder(x)
        
        # Debug features before TopK
        if torch.isnan(features).any() or torch.isinf(features).any():
            check_tensor(features, "features_before_topk")
        
        # Apply TopK sparsity
        sparse_features = self.apply_topk(features)
        
        # Debug features after TopK
        if torch.isnan(sparse_features).any() or torch.isinf(sparse_features).any():
            check_tensor(sparse_features, "features_after_topk")
            
        # Decoder
        reconstruction = self.decoder(sparse_features)
        
        # Debug reconstruction
        if torch.isnan(reconstruction).any() or torch.isinf(reconstruction).any():
            check_tensor(reconstruction, "reconstruction")
            
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

class NeuralFactorizationMachine(nn.Module):
    """NFM (exact copy from your part2d)."""
    def __init__(self, num_sae_features, embedding_dim, output_dim):
        super().__init__()
        self.num_sae_features = num_sae_features
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        
        self.feature_embeddings = nn.Embedding(num_sae_features, embedding_dim)
        self.linear = nn.Linear(num_sae_features, output_dim, bias=True) if USE_LINEAR_COMPONENT else None
        
        self.interaction_mlp = nn.Sequential(
            nn.Dropout(NFM_DROPOUT),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, output_dim)
        )
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Exact initialization from your part2d."""
        nn.init.normal_(self.feature_embeddings.weight, std=NFM_INIT_WEIGHT)
        
        if self.linear is not None:
            nn.init.normal_(self.linear.weight, std=0.01)
            nn.init.zeros_(self.linear.bias)
        
        for layer in self.interaction_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.zeros_(layer.bias)
        
        arch_type = "Linear + Interaction" if USE_LINEAR_COMPONENT else "Interaction-Only"
        print(f"\n--- Checking NFM ({arch_type}) initialization ---")
        check_tensor(self.feature_embeddings.weight, "feature_embeddings.weight", True)
        if self.linear is not None:
            check_tensor(self.linear.weight, "linear.weight", True)
            check_tensor(self.linear.bias, "linear.bias", True)
    
    def forward(self, sae_features):
        """Exact forward pass from your part2d."""
        batch_size = sae_features.shape[0]

        if torch.isnan(sae_features).any() or torch.isinf(sae_features).any():
            check_tensor(sae_features, "nfm_input")
        
        if TOP_N_FEATURES is not None and TOP_N_FEATURES < sae_features.shape[1]:
            top_values, top_indices = torch.topk(sae_features, k=TOP_N_FEATURES, dim=1)
            sae_features_sparse = torch.zeros_like(sae_features)
            sae_features_sparse.scatter_(1, top_indices, top_values)
            sae_features = sae_features_sparse
        
        if SPARSE_THRESHOLD is not None or TOP_N_FEATURES is not None:
            active_mask = sae_features > SPARSE_THRESHOLD
            batch_indices, feature_indices = torch.where(active_mask)
            active_values = sae_features[active_mask]
            
            active_embeddings = self.feature_embeddings(feature_indices)
            weighted_active_embeddings = active_values.unsqueeze(-1) * active_embeddings
            
            batch_size = sae_features.shape[0]
            sum_embeddings = torch.zeros(batch_size, self.embedding_dim, device=sae_features.device, dtype=sae_features.dtype)
            sum_squares = torch.zeros(batch_size, self.embedding_dim, device=sae_features.device, dtype=sae_features.dtype)
            
            sum_embeddings.index_add_(0, batch_indices, weighted_active_embeddings)
            sum_squares.index_add_(0, batch_indices, weighted_active_embeddings ** 2)
            
            interaction_vector = 0.5 * (sum_embeddings ** 2 - sum_squares)
        else:
            all_embeddings = self.feature_embeddings.weight
            weighted_embeddings = sae_features.unsqueeze(-1) * all_embeddings.unsqueeze(0)
            
            sum_embeddings = torch.sum(weighted_embeddings, dim=1)
            square_embeddings = weighted_embeddings ** 2
            sum_squares = torch.sum(square_embeddings, dim=1)
            
            interaction_vector = 0.5 * (sum_embeddings ** 2 - sum_squares)

        if torch.isnan(interaction_vector).any() or torch.isinf(interaction_vector).any():
            check_tensor(interaction_vector, "interaction_vector")
            interaction_vector = torch.zeros_like(interaction_vector)
        
        interaction_out = self.interaction_mlp(interaction_vector)
        
        linear_out = None
        if self.linear is not None:
            linear_out = self.linear(sae_features)
            output = linear_out + interaction_out
        else:
            output = interaction_out
        
        if torch.isnan(output).any() or torch.isinf(output).any():
            check_tensor(output, "nfm_output")
        
        return output, linear_out, interaction_out

class JointSAENFM(nn.Module):
    """Joint SAE+NFM model for end-to-end training."""
    def __init__(self, input_dim, sae_features, sae_k, nfm_embedding_dim):
        super().__init__()
        self.primary_sae = TopKSparseAutoencoder(input_dim, sae_features, sae_k)
        self.nfm = NeuralFactorizationMachine(sae_features, nfm_embedding_dim, input_dim)
        
        print(f"Joint Model Architecture:")
        print(f"  Input: {input_dim}")
        print(f"  SAE Features: {sae_features} (TopK: {sae_k})")
        print(f"  NFM Embedding: {nfm_embedding_dim}")
        print(f"  Output: {input_dim}")
    
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
        
        # THREE-WAY RESIDUAL COMBINATION (like your part2d)
        final_reconstruction = primary_recon + nfm_output
        
        return final_reconstruction, primary_features, primary_recon, linear_out, interaction_out

class JointDataset(Dataset):
    """Dataset for joint training."""
    def __init__(self, activations):
        self.activations = activations
    
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        return self.activations[idx]

def collect_activations_chunk(model, tokenizer, num_tokens, dataset_iterator=None):
    """Collect activations from the model's target layer for a chunk of tokens (from your part2d)."""
    activations = []
    total_tokens_processed = 0
    
    if dataset_iterator is None:
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        dataset_iterator = iter(dataset)
    
    print(f"Collecting {num_tokens} tokens for chunk...")
    
    with tqdm(total=num_tokens, desc="Collecting activations") as pbar:
        while total_tokens_processed < num_tokens:
            try:
                sample = next(dataset_iterator)
            except StopIteration:
                print("Dataset exhausted")
                break
                
            text = sample["text"].strip()
            if not text:
                continue
                
            full_tokens = tokenizer.encode(text, add_special_tokens=False)
            
            if len(full_tokens) < CHUNK_SIZE:
                continue
            
            num_chunks = len(full_tokens) // CHUNK_SIZE
            
            for chunk_idx in range(num_chunks):
                if total_tokens_processed >= num_tokens:
                    break
                    
                start_idx = chunk_idx * CHUNK_SIZE
                end_idx = start_idx + CHUNK_SIZE
                chunk_tokens = full_tokens[start_idx:end_idx]
                
                chunk_text = tokenizer.decode(chunk_tokens)
                inputs = tokenizer(chunk_text, return_tensors="pt", truncation=True, max_length=CHUNK_SIZE)
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[TARGET_LAYER]
                    
                    batch_size, seq_len, hidden_size = hidden_states.shape
                    hidden_states = hidden_states.reshape(-1, hidden_size)
                    
                    activations.append(hidden_states.cpu())
                    total_tokens_processed += seq_len
                    pbar.update(seq_len)
    
    print(f"Collected {total_tokens_processed} tokens")
    activations = torch.cat(activations, dim=0)
    
    subset_size = min(10000, len(activations))
    subset_indices = torch.randperm(len(activations))[:subset_size]
    subset = activations[subset_indices]
    
    with torch.no_grad():
        mean = subset.mean(dim=0, keepdim=True)
        std = subset.std(dim=0, keepdim=True)
        std = torch.clamp(std, min=1e-8)
        activations = (activations - mean) / std
    
    return activations, dataset_iterator

def validate_joint_model(joint_model, val_loader):
    """Compute validation metrics for the joint model (based on your part2d validation)."""
    val_metrics = {
        'total_loss': 0.0,
        'reconstruction_loss': 0.0,
        'linear_magnitude': 0.0,
        'interaction_magnitude': 0.0,
        'linear_contribution': 0.0
    }
    
    max_val_batches = min(10, len(val_loader))
    val_iterator = iter(val_loader)
    num_valid_batches = 0
    
    with torch.no_grad():
        for _ in range(max_val_batches):
            try:
                batch = next(val_iterator)
                batch = batch.to(DEVICE)
                
                if check_tensor(batch, "val_batch", print_stats=False):
                    continue
                
                final_recon, primary_features, primary_recon, linear_out, interaction_out = joint_model(batch)
                
                if (torch.isnan(final_recon).any() or torch.isinf(final_recon).any()):
                    continue
                
                if interaction_out is not None:
                    interaction_magnitude = torch.mean(abs(interaction_out)).item()
                else:
                    interaction_magnitude = 0.0
                
                if linear_out is not None:
                    linear_magnitude = torch.mean(abs(linear_out)).item()
                    total_magnitude = linear_magnitude + interaction_magnitude
                    linear_contribution = (linear_magnitude / (total_magnitude + 1e-8)) * 100
                else:
                    linear_magnitude = 0.0
                    linear_contribution = 0.0
                
                reconstruction_loss = torch.mean((final_recon - batch) ** 2).item()
                total_loss = reconstruction_loss
                
                val_metrics['total_loss'] += total_loss
                val_metrics['reconstruction_loss'] += reconstruction_loss
                val_metrics['linear_magnitude'] += linear_magnitude
                val_metrics['interaction_magnitude'] += interaction_magnitude
                val_metrics['linear_contribution'] += linear_contribution
                
                num_valid_batches += 1
                
            except StopIteration:
                break
    
    if num_valid_batches > 0:
        for key in val_metrics:
            val_metrics[key] /= num_valid_batches
    else:
        print("Warning: No valid batches during validation!")
    
    return val_metrics

def train_joint_on_chunk(joint_model, optimizer, scheduler, chunk_activations, metrics_history, step_offset, checkpoint_dir):
    """Train joint model on a single chunk of data (based on your part2d structure)."""
    dataset_size = len(chunk_activations)
    val_size = int(dataset_size * VALIDATION_SPLIT)
    train_size = dataset_size - val_size
    
    indices = torch.randperm(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_activations = chunk_activations[train_indices]
    val_activations = chunk_activations[val_indices]
    
    train_dataset = JointDataset(train_activations)
    val_dataset = JointDataset(val_activations)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    steps_per_chunk = len(train_loader)

    # Add this after creating train_dataset and val_dataset
    del chunk_activations  # or whatever the input parameter is named
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    print(f"Training on chunk: {train_size} train samples, {val_size} val samples")
    print(f"Steps for this chunk: {steps_per_chunk}")
    
    # Sparsity analysis (from your part2d)
    with torch.no_grad():
        sample_batch = train_activations[:100].to(DEVICE) 
        _, sample_features, _, _, _ = joint_model(sample_batch)
        active_features_per_sample = (sample_features > 1e-6).sum(dim=1).float()
        print(f"\nChunk SAE Feature Sparsity Analysis:")
        print(f"Average active features per sample: {active_features_per_sample.mean().item():.1f}")
        print(f"Median active features per sample: {active_features_per_sample.median().item():.1f}")
        print(f"Max active features per sample: {active_features_per_sample.max().item():.0f}")
        print(f"Min active features per sample: {active_features_per_sample.min().item():.0f}")
    
    joint_model.train()
    
    for epoch in range(10):  # From your part2d
        for batch_idx, batch in enumerate(train_loader):
            step = step_offset + epoch * len(train_loader) + batch_idx
            
            batch = batch.to(DEVICE)
            
            if check_tensor(batch, f"batch_step_{step}", print_stats=False):
                print(f"Problematic batch at step {step}")
                continue
            
            if check_tensor(batch, f"batch_step_{step}", print_stats=False):
                print(f"Problematic batch at step {step}")
                continue
            
            final_recon, primary_features, primary_recon, linear_out, interaction_out = joint_model(batch)
            
            if check_tensor(final_recon, f"final_recon_step_{step}", print_stats=False):
                print(f"NaN or Inf detected in final reconstruction at step {step}")
                continue
            
            if interaction_out is not None:
                interaction_magnitude = torch.mean(abs(interaction_out)).item()
            else:
                interaction_magnitude = 0.0
            
            if linear_out is not None:
                linear_magnitude = torch.mean(abs(linear_out)).item()
                total_magnitude = linear_magnitude + interaction_magnitude
                linear_contribution = (linear_magnitude / (total_magnitude + 1e-8)) * 100
            else:
                linear_magnitude = 0.0
                linear_contribution = 0.0
            
            try:
                # Simple reconstruction loss (TopK handles sparsity)
                reconstruction_loss = torch.mean((final_recon - batch) ** 2)
                
                if torch.isnan(reconstruction_loss) or torch.isinf(reconstruction_loss):
                    print(f"Reconstruction loss is {reconstruction_loss} at step {step}")
                    reconstruction_loss = torch.tensor(1.0, device=DEVICE)
                
                loss = reconstruction_loss
                loss = torch.clamp(loss, max=1e6)
                
            except Exception as e:
                print(f"Exception during loss computation at step {step}: {str(e)}")
                continue
            
            try:
                optimizer.zero_grad()
                loss.backward()
                
                for name, param in joint_model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"NaN/Inf detected in gradients for {name} at step {step}")
                            param.grad = torch.zeros_like(param.grad)
                
                torch.nn.utils.clip_grad_norm_(joint_model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
            except Exception as e:
                print(f"Exception during backward pass at step {step}: {str(e)}")
                continue
            
            metrics_history['steps'].append(step)
            metrics_history['total_loss'].append(loss.item())
            metrics_history['reconstruction_loss'].append(reconstruction_loss.item())
            metrics_history['linear_magnitude'].append(linear_magnitude)
            metrics_history['interaction_magnitude'].append(interaction_magnitude)
            metrics_history['linear_contribution'].append(linear_contribution)
            
            if loss.item() < metrics_history['best_loss']:
                metrics_history['best_loss'] = loss.item()
            
            if step % VALIDATION_INTERVAL == 0:
                joint_model.eval()
                val_metrics = validate_joint_model(joint_model, val_loader)
                joint_model.train()
                
                metrics_history['val_total_loss'].append(val_metrics['total_loss'])
                metrics_history['val_reconstruction_loss'].append(val_metrics['reconstruction_loss'])
                metrics_history['val_linear_magnitude'].append(val_metrics['linear_magnitude'])
                metrics_history['val_interaction_magnitude'].append(val_metrics['interaction_magnitude'])
                metrics_history['val_linear_contribution'].append(val_metrics['linear_contribution'])
                
                if val_metrics['total_loss'] < metrics_history['val_best_loss']:
                    metrics_history['val_best_loss'] = val_metrics['total_loss']
            
            if step % 1000 == 0:
                print(f"\nStep {step}")
                print(f"Train - Total Loss: {loss.item():.4f}")
                print(f"Train - Recon Loss: {reconstruction_loss.item():.4f}")
                print(f"Train - Linear Magnitude: {linear_magnitude:.4f}")
                print(f"Train - Interaction Magnitude: {interaction_magnitude:.4f}")
                if USE_LINEAR_COMPONENT:
                    print(f"Train - Linear Contribution: {linear_contribution:.1f}%")
                
                if metrics_history['val_total_loss']:
                    recent_val_idx = len(metrics_history['val_total_loss']) - 1
                    val_linear_mag = metrics_history['val_linear_magnitude'][recent_val_idx]
                    val_interaction_mag = metrics_history['val_interaction_magnitude'][recent_val_idx]
                    val_linear_contrib = metrics_history['val_linear_contribution'][recent_val_idx]
                    
                    print(f"Val - Total Loss: {metrics_history['val_total_loss'][recent_val_idx]:.4f}")
                    print(f"Val - Linear Magnitude: {val_linear_mag:.4f}")
                    print(f"Val - Interaction Magnitude: {val_interaction_mag:.4f}")
                    if USE_LINEAR_COMPONENT:
                        print(f"Val - Linear Contribution: {val_linear_contrib:.1f}%")
                
                if step % 10000 == 0:
                    debug_model_parameters(joint_model, f"joint_model_at_step_{step}")
            
            if (step + 1) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(
                    joint_model, optimizer, scheduler, step + 1,
                    metrics_history['val_best_loss'], metrics_history,
                    checkpoint_dir
                )
                print(f"\nCheckpoint saved at step {step + 1}")
    
    # Add this before the return statement
    del train_activations, val_activations, train_dataset, val_dataset, train_loader, val_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return step + 1

def main():
    parser = argparse.ArgumentParser(description='Joint Training of TopK SAE + NFM with Streaming')
    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR,
                        help='Directory to save joint checkpoints')
    parser.add_argument('--sae_k', type=int, default=K,
                        help=f'TopK parameter for SAE (default: {K})')
    parser.add_argument('--nfm_k', type=int, default=NFM_K,
                        help=f'NFM embedding dimension (default: {NFM_K})')
    
    args = parser.parse_args()
    
    torch.set_default_dtype(torch.float32)
    
    print("Loading model and tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH, use_fast=False, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(MODEL_PATH, torch_dtype='auto', device_map='auto', local_files_only=True)
    
    print("Determining input dimensions...")
    sample_activations, _ = collect_activations_chunk(model, tokenizer, 1000)
    input_dim = sample_activations.shape[1]
    
    print(f"Initializing Joint SAE+NFM Model...")
    joint_model = JointSAENFM(
        input_dim=input_dim,
        sae_features=NUM_FEATURES,
        sae_k=args.sae_k,
        nfm_embedding_dim=args.nfm_k
    ).to(DEVICE)
    
    joint_model = joint_model.to(dtype=next(model.parameters()).dtype)
    
    total_params = sum(p.numel() for p in joint_model.parameters())
    print(f"Joint Model Parameters:")
    print(f"  Total Parameters: {total_params:,}")
    
    debug_model_parameters(joint_model, "initial_joint_model")
    
    optimizer = optim.Adam(joint_model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=0, eps=1e-5)
    
    num_chunks = NUM_TOKENS_TOTAL // CHUNK_SIZE_TOKENS
    estimated_steps_per_chunk = (CHUNK_SIZE_TOKENS // CHUNK_SIZE) // BATCH_SIZE * 10
    total_estimated_steps = num_chunks * estimated_steps_per_chunk
    
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=1.0,
        end_factor=0.1,
        total_iters=int(total_estimated_steps * 0.8)
    )
    
    metrics_history = {
        'steps': [],
        'total_loss': [],
        'reconstruction_loss': [],
        'best_loss': float('inf'),
        'val_total_loss': [],
        'val_reconstruction_loss': [],
        'val_best_loss': float('inf'),
        'linear_magnitude': [],
        'interaction_magnitude': [],
        'linear_contribution': [],
        'val_linear_magnitude': [],
        'val_interaction_magnitude': [],
        'val_linear_contribution': []
    }
    
    prevent_sleep()
    
    try:
        print(f"Starting joint training with {num_chunks} chunks of {CHUNK_SIZE_TOKENS} tokens each...")
        
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        dataset_iterator = iter(dataset)
        
        step_offset = 0
        
        for chunk_idx in range(num_chunks):
            print(f"\n{'='*60}")
            print(f"PROCESSING CHUNK {chunk_idx + 1}/{num_chunks}")
            print(f"{'='*60}")
            
            activations, dataset_iterator = collect_activations_chunk(
                model, tokenizer, CHUNK_SIZE_TOKENS, dataset_iterator
            )
            
            if len(activations) == 0:
                print("No activations collected for this chunk, skipping...")
                continue
            
            print(f"Chunk activations shape: {activations.shape}")
            
            step_offset = train_joint_on_chunk(
                joint_model, optimizer, scheduler,
                activations, metrics_history, step_offset, args.checkpoint_dir
            )
            
            del activations
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            # Add this after the train_joint_on_chunk call
            import gc
            gc.collect()
            
            print(f"Completed chunk {chunk_idx + 1}/{num_chunks}")
    
    finally:
        allow_sleep()
    
    final_model_name = "joint_sae_nfm_model.pt"
    torch.save(joint_model.state_dict(), final_model_name)
    print(f"Joint training complete! Model saved as {final_model_name}")
    
    print("\n" + "="*50)
    print("FINAL VALIDATION RESULTS")
    print("="*50)
    
    val_activations, _ = collect_activations_chunk(model, tokenizer, 10000)
    
    joint_model.eval()
    
    with torch.no_grad():
        sample_size = min(1000, len(val_activations))
        sample_indices = torch.randperm(len(val_activations))[:sample_size]
        sample_original = val_activations[sample_indices].to(DEVICE)
        
        final_recon, primary_features, primary_recon, linear_out, interaction_out = joint_model(sample_original)
        
        # Reconstruction analysis
        final_error = torch.mean((sample_original - final_recon) ** 2).item()
        sae_only_error = torch.mean((sample_original - primary_recon) ** 2).item()
        
        error_reduction = ((sae_only_error - final_error) / sae_only_error) * 100
        
        if interaction_out is not None:
            val_interaction_magnitude = torch.mean(abs(interaction_out)).item()
        else:
            val_interaction_magnitude = 0.0
        
        if linear_out is not None:
            val_linear_magnitude = torch.mean(abs(linear_out)).item()
            val_total_magnitude = val_linear_magnitude + val_interaction_magnitude
            val_linear_contribution = (val_linear_magnitude / (val_total_magnitude + 1e-8)) * 100
        else:
            val_linear_magnitude = 0.0
            val_linear_contribution = 0.0
        
        original_norm = torch.mean(sample_original ** 2).item()
        
        print(f"Original data MSE (baseline): {original_norm:.6f}")
        print(f"SAE only reconstruction error: {sae_only_error:.6f}")
        print(f"Joint SAE+NFM reconstruction error: {final_error:.6f}")
        print(f"Error reduction: {error_reduction:.2f}%")
        print(f"Relative error (SAE only): {(sae_only_error / original_norm) * 100:.2f}%")
        print(f"Relative error (Joint): {(final_error / original_norm) * 100:.2f}%")
        
        print(f"\nComponent Analysis:")
        print(f"Linear component magnitude: {val_linear_magnitude:.6f}")
        print(f"Interaction component magnitude: {val_interaction_magnitude:.6f}")
        if USE_LINEAR_COMPONENT:
            print(f"Linear contribution: {val_linear_contribution:.1f}%")
            print(f"Interaction contribution: {100 - val_linear_contribution:.1f}%")
        
        # Feature sparsity analysis
        percent_active = (primary_features > 1e-6).float().mean().item() * 100
        target_active = (args.sae_k / NUM_FEATURES) * 100
        
        print(f"\nFeature Sparsity (controlled by TopK):")
        print(f"  Active features: {percent_active:.2f}% (target: {target_active:.2f}%)")
        print(f"  Total features: {NUM_FEATURES:,}")
        print(f"  TopK parameter: {args.sae_k}")
        
        # Save final analysis
        final_analysis = {
            'final_reconstruction_error': final_error,
            'sae_only_error': sae_only_error,
            'improvement_percentage': error_reduction,
            'percent_active_features': percent_active,
            'target_active_percentage': target_active,
            'component_magnitudes': {
                'linear_component': val_linear_magnitude,
                'interaction_component': val_interaction_magnitude
            },
            'component_contributions': {
                'linear_contribution': val_linear_contribution,
                'interaction_contribution': 100 - val_linear_contribution
            },
            'model_parameters': {
                'sae_features': NUM_FEATURES,
                'sae_topk': args.sae_k,
                'nfm_embedding_dim': args.nfm_k,
                'total_parameters': total_params
            }
        }
        
        with open('joint_model_final_analysis.json', 'w') as f:
            json.dump(final_analysis, f, indent=2)
        
        print(f"\nFinal analysis saved to: joint_model_final_analysis.json")
        
        # Compare with sequential training results
        print(f"\n" + "="*50)
        print("COMPARISON WITH SEQUENTIAL TRAINING")
        print("="*50)
        print(f"Joint training results: {error_reduction:.1f}% improvement")
        
        
        print(f"\nNext steps:")
        print(f"  1. Run KL divergence analysis on this joint model")
        print(f"  2. Compare behavioral fidelity: Sequential vs Joint")
        print(f"  3. Analyze feature monosemanticity improvements")
        print(f"  4. Test linear vs interaction ablation on joint model")

if __name__ == "__main__":
    main()