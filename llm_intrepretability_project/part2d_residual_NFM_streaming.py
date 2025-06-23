#This version has streaming training data (so we can exceed 100-200k tokens)

#250/250 200 0.05 0.0001
# Step 101000
# Train - Total Loss: 0.2930
# Train - Recon Loss: 0.2930
# Train - SAE1 Only Error: 0.3813
# Train - Combined (SAE1+NFM) Error: 0.2930
# Train - Error Reduction: 23.18%
# Train - Linear Magnitude: 0.2773
# Train - Interaction Magnitude: 0.0130
# Train - Linear Contribution: 95.5%
# Val - Total Loss: 0.2811
# Val - SAE1 Only Error: 0.3672
# Val - Combined (SAE1+NFM) Error: 0.2811
# Val - Error Reduction: 23.43%
# Val - Linear Magnitude: 0.2771
# Val - Interaction Magnitude: 0.0129
# Val - Linear Contribution: 95.5%


import torch
import torch.nn as nn
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

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "open_llama_3b"))
BATCH_SIZE = 130
NUM_TOKENS_TOTAL = 5_000_000  # Total tokens to process
CHUNK_SIZE_TOKENS = 100_000  # Process 100k tokens at a time
LEARNING_RATE = 0.0001
NFM_INIT_WEIGHT = 0.05
NUM_FEATURES = 50_000
NFM_K = 300
TARGET_LAYER = 16
TRAIN_STEPS = 300_000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints_nfm"
CHECKPOINT_INTERVAL = 10_000
VALIDATION_INTERVAL = 1000
VALIDATION_SPLIT = 0.1
CHUNK_SIZE = 128
NFM_DROPOUT = 0.15
SPARSE_THRESHOLD = None
TOP_N_FEATURES = 500
K = 500
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
    
    model_name = "best_nfm_linear_interaction_model.pt" if USE_LINEAR_COMPONENT else "best_nfm_interaction_only_model.pt"
    if metrics_history['val_best_loss'] == best_loss:
        best_model_path = checkpoint_dir / model_name
        torch.save(model.state_dict(), best_model_path)

class TopKSparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, k):
        super().__init__()
        self.k = k
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
    
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

class NeuralFactorizationMachine(nn.Module):
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
        batch_size = sae_features.shape[0]
        
        if torch.isnan(sae_features).any() or torch.isinf(sae_features).any():
            check_tensor(sae_features, "nfm_input")
        
        if TOP_N_FEATURES is not None and TOP_N_FEATURES < sae_features.shape[1]:
            top_values, top_indices = torch.topk(sae_features, k=TOP_N_FEATURES, dim=1)
            sae_features_sparse = torch.zeros_like(sae_features)
            sae_features_sparse.scatter_(1, top_indices, top_values)
            sae_features = sae_features_sparse
        
        if SPARSE_THRESHOLD is not None:
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

class ResidualDataset(Dataset):
    def __init__(self, original_activations, sae_features, residual_errors):
        self.original_activations = original_activations
        self.sae_features = sae_features
        self.residual_errors = residual_errors
        assert len(original_activations) == len(sae_features) == len(residual_errors)
    
    def __len__(self):
        return len(self.residual_errors)
    
    def __getitem__(self, idx):
        return {
            'original': self.original_activations[idx],
            'sae_features': self.sae_features[idx],
            'residual': self.residual_errors[idx]
        }

def collect_activations_chunk(model, tokenizer, num_tokens, dataset_iterator=None):
    """Collect activations from the model's target layer for a chunk of tokens."""
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

def compute_residual_errors_and_features(original_activations, sae_model):
    """Compute residual errors and SAE features using the trained TopK SAE."""
    residual_errors = []
    sae_features_list = []
    
    batch_size = BATCH_SIZE
    num_batches = (len(original_activations) + batch_size - 1) // batch_size
    
    sae_model.eval()
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Computing residual errors and SAE features"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(original_activations))
            batch = original_activations[start_idx:end_idx].to(DEVICE)
            
            sae_features, sae_reconstruction = sae_model(batch)
            residual = batch - sae_reconstruction
            
            residual_errors.append(residual.cpu())
            sae_features_list.append(sae_features.cpu())
    
    return torch.cat(residual_errors, dim=0), torch.cat(sae_features_list, dim=0)

def validate_nfm(nfm_model, val_loader):
    """Compute validation metrics for the NFM."""
    val_metrics = {
        'total_loss': 0.0,
        'reconstruction_loss': 0.0,
        'combined_reconstruction_error': 0.0,
        'sae1_only_error': 0.0,
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
                original_activations = batch['original'].to(DEVICE)
                sae_features = batch['sae_features'].to(DEVICE)
                residual_errors = batch['residual'].to(DEVICE)
                
                if check_tensor(residual_errors, "val_residual_batch", print_stats=False):
                    continue
                
                if check_tensor(sae_features, "val_sae_features_batch", print_stats=False):
                    continue
                
                nfm_prediction, linear_out, interaction_out = nfm_model(sae_features)
                
                if (torch.isnan(nfm_prediction).any() or torch.isinf(nfm_prediction).any()):
                    continue
                
                interaction_magnitude = torch.mean(abs(interaction_out)).item()
                
                if linear_out is not None:
                    linear_magnitude = torch.mean(abs(linear_out)).item()
                    total_magnitude = linear_magnitude + interaction_magnitude
                    linear_contribution = (linear_magnitude / (total_magnitude + 1e-8)) * 100
                else:
                    linear_magnitude = 0.0
                    linear_contribution = 0.0
                
                reconstruction_loss = torch.mean((nfm_prediction - residual_errors) ** 2).item()
                total_loss = reconstruction_loss
                
                sae1_only_error = torch.mean(residual_errors ** 2).item()
                combined_error = torch.mean((residual_errors - nfm_prediction) ** 2).item()
                
                val_metrics['total_loss'] += total_loss
                val_metrics['reconstruction_loss'] += reconstruction_loss
                val_metrics['combined_reconstruction_error'] += combined_error
                val_metrics['sae1_only_error'] += sae1_only_error
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

def train_nfm_on_chunk(nfm_model, optimizer, scheduler, chunk_original, chunk_sae_features, chunk_residual, metrics_history, step_offset, checkpoint_dir):
    """Train NFM on a single chunk of data."""
    dataset_size = len(chunk_original)
    val_size = int(dataset_size * VALIDATION_SPLIT)
    train_size = dataset_size - val_size
    
    indices = torch.randperm(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_original = chunk_original[train_indices]
    train_sae_features = chunk_sae_features[train_indices]
    train_residual = chunk_residual[train_indices]
    
    val_original = chunk_original[val_indices]
    val_sae_features = chunk_sae_features[val_indices]
    val_residual = chunk_residual[val_indices]
    
    train_dataset = ResidualDataset(train_original, train_sae_features, train_residual)
    val_dataset = ResidualDataset(val_original, val_sae_features, val_residual)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    steps_per_chunk = len(train_loader)
    
    print(f"Training on chunk: {train_size} train samples, {val_size} val samples")
    print(f"Steps for this chunk: {steps_per_chunk}")
    
    with torch.no_grad():
        active_features_per_sample = (chunk_sae_features > 1e-6).sum(dim=1).float()
        print(f"\nChunk SAE Feature Sparsity Analysis:")
        print(f"Average active features per sample: {active_features_per_sample.mean().item():.1f}")
        print(f"Median active features per sample: {active_features_per_sample.median().item():.1f}")
        print(f"Max active features per sample: {active_features_per_sample.max().item():.0f}")
        print(f"Min active features per sample: {active_features_per_sample.min().item():.0f}")
    
    nfm_model.train()
    
    for epoch in range(10):
        for batch_idx, batch in enumerate(train_loader):
            step = step_offset + epoch * len(train_loader) + batch_idx
            
            original_activations = batch['original'].to(DEVICE)
            sae_features = batch['sae_features'].to(DEVICE)
            residual_errors = batch['residual'].to(DEVICE)
            
            if check_tensor(residual_errors, f"residual_batch_step_{step}", print_stats=False):
                print(f"Problematic residual batch at step {step}")
                continue
            
            if check_tensor(sae_features, f"sae_features_batch_step_{step}", print_stats=False):
                print(f"Problematic SAE features batch at step {step}")
                continue
            
            nfm_prediction, linear_out, interaction_out = nfm_model(sae_features)
            
            if check_tensor(nfm_prediction, f"nfm_prediction_step_{step}", print_stats=False):
                print(f"NaN or Inf detected in NFM prediction at step {step}")
                continue
            
            interaction_magnitude = torch.mean(abs(interaction_out)).item()
            
            if linear_out is not None:
                linear_magnitude = torch.mean(abs(linear_out)).item()
                total_magnitude = linear_magnitude + interaction_magnitude
                linear_contribution = (linear_magnitude / (total_magnitude + 1e-8)) * 100
            else:
                linear_magnitude = 0.0
                linear_contribution = 0.0
            
            try:
                reconstruction_diff = (nfm_prediction - residual_errors)
                reconstruction_loss = torch.mean(reconstruction_diff ** 2)
                
                if torch.isnan(reconstruction_loss) or torch.isinf(reconstruction_loss):
                    print(f"Reconstruction loss is {reconstruction_loss} at step {step}")
                    reconstruction_loss = torch.tensor(1.0, device=DEVICE)
                
                loss = reconstruction_loss
                loss = torch.clamp(loss, max=1e6)
                
                sae1_only_error = torch.mean(residual_errors ** 2).item()
                combined_error = torch.mean((residual_errors - nfm_prediction) ** 2).item()
                
            except Exception as e:
                print(f"Exception during loss computation at step {step}: {str(e)}")
                continue
            
            try:
                optimizer.zero_grad()
                loss.backward()
                
                for name, param in nfm_model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"NaN/Inf detected in gradients for {name} at step {step}")
                            param.grad = torch.zeros_like(param.grad)
                
                torch.nn.utils.clip_grad_norm_(nfm_model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
            except Exception as e:
                print(f"Exception during backward pass at step {step}: {str(e)}")
                continue
            
            metrics_history['steps'].append(step)
            metrics_history['total_loss'].append(loss.item())
            metrics_history['reconstruction_loss'].append(reconstruction_loss.item())
            metrics_history['combined_reconstruction_error'].append(combined_error)
            metrics_history['sae1_only_error'].append(sae1_only_error)
            metrics_history['linear_magnitude'].append(linear_magnitude)
            metrics_history['interaction_magnitude'].append(interaction_magnitude)
            metrics_history['linear_contribution'].append(linear_contribution)
            
            if loss.item() < metrics_history['best_loss']:
                metrics_history['best_loss'] = loss.item()
            
            if step % VALIDATION_INTERVAL == 0:
                nfm_model.eval()
                val_metrics = validate_nfm(nfm_model, val_loader)
                nfm_model.train()
                
                metrics_history['val_total_loss'].append(val_metrics['total_loss'])
                metrics_history['val_reconstruction_loss'].append(val_metrics['reconstruction_loss'])
                metrics_history['val_combined_reconstruction_error'].append(val_metrics['combined_reconstruction_error'])
                metrics_history['val_sae1_only_error'].append(val_metrics['sae1_only_error'])
                metrics_history['val_linear_magnitude'].append(val_metrics['linear_magnitude'])
                metrics_history['val_interaction_magnitude'].append(val_metrics['interaction_magnitude'])
                metrics_history['val_linear_contribution'].append(val_metrics['linear_contribution'])
                
                if val_metrics['total_loss'] < metrics_history['val_best_loss']:
                    metrics_history['val_best_loss'] = val_metrics['total_loss']
            
            if step % 1000 == 0:
                print(f"\nStep {step}")
                print(f"Train - Total Loss: {loss.item():.4f}")
                print(f"Train - Recon Loss: {reconstruction_loss.item():.4f}")
                print(f"Train - SAE1 Only Error: {sae1_only_error:.4f}")
                print(f"Train - Combined (SAE1+NFM) Error: {combined_error:.4f}")
                print(f"Train - Error Reduction: {((sae1_only_error - combined_error) / sae1_only_error * 100):.2f}%")
                print(f"Train - Linear Magnitude: {linear_magnitude:.4f}")
                print(f"Train - Interaction Magnitude: {interaction_magnitude:.4f}")
                if USE_LINEAR_COMPONENT:
                    print(f"Train - Linear Contribution: {linear_contribution:.1f}%")
                
                if metrics_history['val_total_loss']:
                    recent_val_idx = len(metrics_history['val_total_loss']) - 1
                    val_sae1_error = metrics_history['val_sae1_only_error'][recent_val_idx]
                    val_combined_error = metrics_history['val_combined_reconstruction_error'][recent_val_idx]
                    val_linear_mag = metrics_history['val_linear_magnitude'][recent_val_idx]
                    val_interaction_mag = metrics_history['val_interaction_magnitude'][recent_val_idx]
                    val_linear_contrib = metrics_history['val_linear_contribution'][recent_val_idx]
                    
                    print(f"Val - Total Loss: {metrics_history['val_total_loss'][recent_val_idx]:.4f}")
                    print(f"Val - SAE1 Only Error: {val_sae1_error:.4f}")
                    print(f"Val - Combined (SAE1+NFM) Error: {val_combined_error:.4f}")
                    print(f"Val - Error Reduction: {((val_sae1_error - val_combined_error) / val_sae1_error * 100):.2f}%")
                    print(f"Val - Linear Magnitude: {val_linear_mag:.4f}")
                    print(f"Val - Interaction Magnitude: {val_interaction_mag:.4f}")
                    if USE_LINEAR_COMPONENT:
                        print(f"Val - Linear Contribution: {val_linear_contrib:.1f}%")
                
                if step % 10000 == 0:
                    debug_model_parameters(nfm_model, f"nfm_at_step_{step}")
            
            if (step + 1) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(
                    nfm_model, optimizer, scheduler, step + 1,
                    metrics_history['val_best_loss'], metrics_history,
                    checkpoint_dir
                )
                print(f"\nCheckpoint saved at step {step + 1}")
    
    return step + 1

def main():
    parser = argparse.ArgumentParser(description='Train Neural Factorization Machine for Residual Modeling with Streaming')
    parser.add_argument('--sae1_model_path', type=str, default='checkpoints_topk/best_model.pt',
                        help='Path to the trained TopK SAE from part1c (default: checkpoints_topk/best_model.pt)')
    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR,
                        help='Directory to save NFM checkpoints')
    parser.add_argument('--nfm_k', type=int, default=NFM_K,
                        help=f'NFM embedding dimension (default: {NFM_K})')
    
    args = parser.parse_args()
    
    torch.set_default_dtype(torch.float32)
    
    print("Loading model and tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH, use_fast=False, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(MODEL_PATH, torch_dtype='auto', device_map='auto', local_files_only=True)
    
    print(f"Loading trained TopK SAE from {args.sae1_model_path}...")
    
    sample_activations, _ = collect_activations_chunk(model, tokenizer, 1000)
    input_dim = sample_activations.shape[1]
    
    sae1 = TopKSparseAutoencoder(
        input_dim=input_dim,
        hidden_dim=NUM_FEATURES,
        k=K
    ).to(DEVICE)
    
    sae1 = sae1.to(dtype=next(model.parameters()).dtype)
    
    if not os.path.exists(args.sae1_model_path):
        raise FileNotFoundError(f"TopK SAE model not found at {args.sae1_model_path}. Please run part1c first.")
    
    try:
        checkpoint = torch.load(args.sae1_model_path, map_location=DEVICE)
        if 'model_state' in checkpoint:
            sae1.load_state_dict(checkpoint['model_state'])
        else:
            sae1.load_state_dict(checkpoint)
    except:
        sae1.load_state_dict(torch.load(args.sae1_model_path, map_location=DEVICE))
    
    print("TopK SAE loaded successfully!")
    
    arch_type = "Linear + Interaction" if USE_LINEAR_COMPONENT else "Interaction-Only"
    print(f"Initializing Neural Factorization Machine ({arch_type}) with K={NFM_K}...")
    nfm_model = NeuralFactorizationMachine(
        num_sae_features=NUM_FEATURES,
        embedding_dim=NFM_K,
        output_dim=input_dim
    ).to(DEVICE)
    
    nfm_model = nfm_model.to(dtype=next(model.parameters()).dtype)
    
    total_params = sum(p.numel() for p in nfm_model.parameters())
    print(f"NFM ({arch_type}) Model Parameters:")
    print(f"  Feature Embeddings: {nfm_model.feature_embeddings.weight.numel():,}")
    if nfm_model.linear is not None:
        print(f"  Linear Layer: {nfm_model.linear.weight.numel() + nfm_model.linear.bias.numel():,}")
    print(f"  Interaction MLP: {sum(p.numel() for p in nfm_model.interaction_mlp.parameters()):,}")
    print(f"  Total Parameters: {total_params:,}")
    
    debug_model_parameters(nfm_model, f"initial_nfm_{arch_type.lower().replace(' + ', '_')}")
    
    optimizer = optim.Adam(nfm_model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=0, eps=1e-5)
    
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
        'combined_reconstruction_error': [],
        'sae1_only_error': [],
        'val_combined_reconstruction_error': [],
        'val_sae1_only_error': [],
        'linear_magnitude': [],
        'interaction_magnitude': [],
        'linear_contribution': [],
        'val_linear_magnitude': [],
        'val_interaction_magnitude': [],
        'val_linear_contribution': []
    }
    
    prevent_sleep()
    
    try:
        print(f"Starting streaming training with {num_chunks} chunks of {CHUNK_SIZE_TOKENS} tokens each...")
        
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
            
            print("Computing residual errors and SAE features for chunk...")
            residual_errors, sae_features = compute_residual_errors_and_features(activations, sae1)
            
            print(f"Chunk residual errors computed. Shape: {residual_errors.shape}")
            print(f"Chunk SAE features extracted. Shape: {sae_features.shape}")
            
            step_offset = train_nfm_on_chunk(
                nfm_model, optimizer, scheduler,
                activations, sae_features, residual_errors,
                metrics_history, step_offset, args.checkpoint_dir
            )
            
            del activations, residual_errors, sae_features
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            print(f"Completed chunk {chunk_idx + 1}/{num_chunks}")
    
    finally:
        allow_sleep()
    
    final_model_name = "nfm_linear_interaction_residual_model.pt" if USE_LINEAR_COMPONENT else "nfm_interaction_only_residual_model.pt"
    torch.save(nfm_model.state_dict(), final_model_name)
    print(f"Training complete! NFM ({arch_type}) saved as {final_model_name}")
    
    print("\n" + "="*50)
    print("FINAL VALIDATION RESULTS")
    print("="*50)
    
    val_activations, _ = collect_activations_chunk(model, tokenizer, 10000)
    val_residual_errors, val_sae_features = compute_residual_errors_and_features(val_activations, sae1)
    
    sae1.eval()
    nfm_model.eval()
    
    with torch.no_grad():
        sample_size = min(1000, len(val_activations))
        sample_indices = torch.randperm(len(val_activations))[:sample_size]
        sample_original = val_activations[sample_indices].to(DEVICE)
        sample_sae_features = val_sae_features[sample_indices].to(DEVICE)
        sample_residual = val_residual_errors[sample_indices].to(DEVICE)
        
        _, sae1_reconstruction = sae1(sample_original)
        sae1_error = torch.mean((sample_original - sae1_reconstruction) ** 2).item()
        
        nfm_prediction, val_linear_out, val_interaction_out = nfm_model(sample_sae_features)
        
        combined_reconstruction = sae1_reconstruction + nfm_prediction
        combined_error = torch.mean((sample_original - combined_reconstruction) ** 2).item()
        
        error_reduction = ((sae1_error - combined_error) / sae1_error) * 100
        
        val_interaction_magnitude = torch.mean(abs(val_interaction_out)).item()
        
        if val_linear_out is not None:
            val_linear_magnitude = torch.mean(abs(val_linear_out)).item()
            val_total_magnitude = val_linear_magnitude + val_interaction_magnitude
            val_linear_contribution = (val_linear_magnitude / (val_total_magnitude + 1e-8)) * 100
        else:
            val_linear_magnitude = 0.0
            val_linear_contribution = 0.0
        
        original_norm = torch.mean(sample_original ** 2).item()
        
        print(f"Original data MSE (baseline): {original_norm:.6f}")
        print(f"SAE1 only reconstruction error: {sae1_error:.6f}")
        print(f"SAE1 + NFM ({arch_type}) reconstruction error: {combined_error:.6f}")
        print(f"Error reduction: {error_reduction:.2f}%")
        print(f"Relative error (SAE1 only): {(sae1_error / original_norm) * 100:.2f}%")
        print(f"Relative error (Combined): {(combined_error / original_norm) * 100:.2f}%")
        print(f"\nNFM Component Analysis:")
        print(f"Linear component magnitude: {val_linear_magnitude:.6f}")
        print(f"Interaction component magnitude: {val_interaction_magnitude:.6f}")
        if USE_LINEAR_COMPONENT:
            print(f"Linear contribution: {val_linear_contribution:.1f}%")
            print(f"Interaction contribution: {100 - val_linear_contribution:.1f}%")

if __name__ == "__main__":
    main()