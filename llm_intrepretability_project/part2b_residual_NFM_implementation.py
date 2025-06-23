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

# Constants - keeping same as original for consistency -- MODIFIED to be able to run 128 GB memory/24 GB GPU memory
#tons of optimization still to do -- just proof of concept
# we had to modify the initialization to 0.01 from (0.0001), along with a bias and ampltidue, to get a signal w/ interaction only
# NFMs typically operate on sparse space, with a ~0.02 learning rate, but we're using lot less sparsity (250/50k) more training runs
# k =150   
# top 250
# dropout 0.2
# tokensize 500_000

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "open_llama_3b"))
BATCH_SIZE = 100
NUM_TOKENS = 180_000  # Same as part1 5_000_000
LEARNING_RATE = 0.00005  # More conservative - NFM literature might assume different data scales  
NUM_FEATURES = 50_000  # Same feature size as part1
NFM_K = 150  # NFM embedding dimension - modifiable parameter
TARGET_LAYER = 16  # Same target layer as part1
TRAIN_STEPS = 100_000  # Same number of training steps
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints_nfm"  # NFM checkpoint directory
CHECKPOINT_INTERVAL = 10_000
VALIDATION_INTERVAL = 1000
VALIDATION_SPLIT = 0.2
CHUNK_SIZE = 128  # Same as SAE
#for interaction only
NFM_DROPOUT = 0.3  # Dropout rate for NFM interaction MLP - modifiable parameter
SPARSE_THRESHOLD = None  # Set to float value (e.g. 0.4), hard filter on SAE activations > threshold, to enable sparse interactions, None for dense
TOP_N_FEATURES = 20  # Only keep top N features per sample, zero out the rest -- use this instead
K = 1024  # TopK parameter (for SAE) -- we use masking on SAE inference
USE_LINEAR_COMPONENT = True  # Set to True to add linear component alongside interactions


# Reuse utility functions from original script
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
    
    # Save checkpoint
    checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Save metrics separately for easy access
    metrics_path = checkpoint_dir / f"metrics_step_{step}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_history, f, indent=2)
    
    # Save best model if this is the best loss
    model_name = "best_nfm_linear_interaction_model.pt" if USE_LINEAR_COMPONENT else "best_nfm_interaction_only_model.pt"
    if metrics_history['val_best_loss'] == best_loss:
        best_model_path = checkpoint_dir / model_name
        torch.save(model.state_dict(), best_model_path)

def compute_l0_sparsity(features, threshold=1e-6):
    """Compute L0 'norm' (count of non-zero elements) for features."""
    with torch.no_grad():
        zeros = (torch.abs(features) < threshold).float().mean().item()
        return zeros

# TopK SAE class - same as part1c
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

# IMPROVED Neural Factorization Machine for modeling residuals
class ImprovedNeuralFactorizationMachine(nn.Module):
    def __init__(self, num_sae_features, embedding_dim, output_dim, dropout_rate=0.3):
        super().__init__()
        self.num_sae_features = num_sae_features
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # Feature embeddings for interaction modeling
        self.feature_embeddings = nn.Embedding(num_sae_features, embedding_dim)
        
        # Optional linear component (first-order effects)
        self.linear = nn.Linear(num_sae_features, output_dim, bias=True) if USE_LINEAR_COMPONENT else None
        
        # Global bias (standard in NFM literature)
        self.global_bias = nn.Parameter(torch.zeros(output_dim))
        
        # Batch normalization for bi-interaction output (crucial for NFM stability)
        self.bi_interaction_bn = nn.BatchNorm1d(embedding_dim)
        
        # IMPROVED MLP for processing interaction vector with BatchNorm
        self.interaction_mlp = nn.Sequential(
            nn.Dropout(dropout_rate),  # NFM-specific: dropout on interaction vector before MLP
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),  # BatchNorm after first linear layer
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),  # Reduced dropout for deeper layers
            nn.Linear(embedding_dim, output_dim)
        )
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """IMPROVED weight initialization following NFM best practices."""
        
        # Xavier initialization for embeddings (better variance scaling)
        nn.init.xavier_uniform_(self.feature_embeddings.weight)
        
        # Xavier initialization for linear layer if present
        if self.linear is not None:
            nn.init.xavier_uniform_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
        
        # Initialize global bias to zero
        nn.init.zeros_(self.global_bias)
        
        # IMPROVED MLP initialization
        for layer in self.interaction_mlp:
            if isinstance(layer, nn.Linear):
                # Kaiming initialization for ReLU layers (better than normal)
                nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(layer.bias)
        
        arch_type = "Linear + Interaction" if USE_LINEAR_COMPONENT else "Interaction-Only"
        print(f"\n--- Checking Improved NFM ({arch_type}) initialization ---")
        check_tensor(self.feature_embeddings.weight, "feature_embeddings.weight", True)
        if self.linear is not None:
            check_tensor(self.linear.weight, "linear.weight", True)
            check_tensor(self.linear.bias, "linear.bias", True)
        check_tensor(self.global_bias, "global_bias", True)
    
    def bi_interaction_pooling(self, sae_features):
        """
        Proper Bi-Interaction pooling operation from NFM paper.
        
        This is the core innovation of NFM - efficient computation of all pairwise 
        feature interactions using the polynomial expansion trick.
        """
        batch_size = sae_features.shape[0]
        
        # Keep only top N features per sample, zero out the rest
        if TOP_N_FEATURES is not None and TOP_N_FEATURES < sae_features.shape[1]:
            # Get top N indices per sample
            top_values, top_indices = torch.topk(sae_features, k=TOP_N_FEATURES, dim=1)
            # Create sparse version by zeroing out non-top features
            sae_features_sparse = torch.zeros_like(sae_features)
            sae_features_sparse.scatter_(1, top_indices, top_values)
            sae_features = sae_features_sparse
        
        if SPARSE_THRESHOLD is not None:
            # Sparse computation - only process active features
            active_mask = sae_features > SPARSE_THRESHOLD
            
            # Get batch indices and feature indices for active features
            batch_indices, feature_indices = torch.where(active_mask)
            active_values = sae_features[active_mask]
            
            # Get embeddings only for active features
            active_embeddings = self.feature_embeddings(feature_indices)
            weighted_active_embeddings = active_values.unsqueeze(-1) * active_embeddings
            
            # Group by batch and sum embeddings per batch
            batch_size = sae_features.shape[0]
            # Fix: Match dtype to the model's dtype
            sum_embeddings = torch.zeros(batch_size, self.embedding_dim, device=sae_features.device, dtype=sae_features.dtype)
            sum_squares = torch.zeros(batch_size, self.embedding_dim, device=sae_features.device, dtype=sae_features.dtype)
            
            sum_embeddings.index_add_(0, batch_indices, weighted_active_embeddings)
            sum_squares.index_add_(0, batch_indices, weighted_active_embeddings ** 2)
            
            # Compute interaction vector: 0.5 * (sum^2 - sum_of_squares)
            bi_interaction = 0.5 * (sum_embeddings ** 2 - sum_squares)
        else:
            # Dense computation (standard NFM approach)
            all_embeddings = self.feature_embeddings.weight  # [num_features, embedding_dim]
            weighted_embeddings = sae_features.unsqueeze(-1) * all_embeddings.unsqueeze(0)  # [batch, num_features, embedding_dim]
            
            # Standard NFM bi-interaction pooling formula:
            # 0.5 * (sum_of_embeddings^2 - sum_of_squared_embeddings)
            
            # Sum of weighted embeddings: sum_i(x_i * v_i)
            sum_embeddings = torch.sum(weighted_embeddings, dim=1)  # [batch, embedding_dim]
            
            # Sum of squared weighted embeddings: sum_i((x_i * v_i)^2)
            squared_weighted_embeddings = weighted_embeddings ** 2
            sum_squared_embeddings = torch.sum(squared_weighted_embeddings, dim=1)  # [batch, embedding_dim]
            
            # Bi-interaction pooling: captures all pairwise interactions efficiently
            # This is equivalent to sum_i(sum_j!=i(x_i * v_i * x_j * v_j)) but computed in O(k*d) time
            bi_interaction = 0.5 * (sum_embeddings ** 2 - sum_squared_embeddings)  # [batch, embedding_dim]        

        return bi_interaction
    
    def forward(self, sae_features):
        """
        IMPROVED forward pass with proper NFM architecture.
        
        Args:
            sae_features: [batch_size, num_sae_features] - activated SAE features
            
        Returns:
            output: [batch_size, output_dim] - predicted residual
            linear_out: [batch_size, output_dim] - linear component (None if not used)
            interaction_out: [batch_size, output_dim] - interaction component
        """
        batch_size = sae_features.shape[0]
        
        # Check for NaN/Inf in input
        if torch.isnan(sae_features).any() or torch.isinf(sae_features).any():
            check_tensor(sae_features, "nfm_input")
        
        # 1. Global bias (standard NFM component)
        output = self.global_bias.unsqueeze(0).expand(batch_size, -1)
        
        # 2. Linear component (first-order effects) if enabled
        linear_out = None
        if self.linear is not None:
            linear_out = self.linear(sae_features)
            output = output + linear_out
        
        # 3. Interaction component (second-order effects)
        # Use proper bi-interaction pooling
        interaction_vector = self.bi_interaction_pooling(sae_features)
        
        # Check for issues in interaction computation
        if torch.isnan(interaction_vector).any() or torch.isinf(interaction_vector).any():
            check_tensor(interaction_vector, "interaction_vector")
            # Fallback to zeros if computation fails
            interaction_vector = torch.zeros_like(interaction_vector)
        
        # Apply batch normalization (helps with training stability)
        if interaction_vector.shape[0] > 1:  # Skip BN for batch size 1
            interaction_vector = self.bi_interaction_bn(interaction_vector)
        
        # Process interaction vector through improved MLP
        interaction_out = self.interaction_mlp(interaction_vector)
        
        # Add interaction component to output
        output = output + interaction_out
        
        # Final check
        if torch.isnan(output).any() or torch.isinf(output).any():
            check_tensor(output, "nfm_output")
            # Fallback to linear-only if interaction computation fails
            print("Warning: Interaction computation failed, falling back to linear-only")
            if linear_out is not None:
                output = self.global_bias.unsqueeze(0).expand(batch_size, -1) + linear_out
            else:
                output = self.global_bias.unsqueeze(0).expand(batch_size, -1)
            interaction_out = torch.zeros_like(interaction_out)
        
        return output, linear_out, interaction_out

class ResidualDataset(Dataset):
    """Dataset that holds original activations, SAE features, and residual errors."""
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

def collect_activations(model, tokenizer, num_tokens):
    """Collect activations from the model's target layer - same as part1d."""
    activations = []
    total_tokens_processed = 0
    
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    
    print(f"Collecting {num_tokens} tokens for evaluation...")
    
    for sample in tqdm(dataset, desc="Collecting eval activations"):
        if total_tokens_processed >= num_tokens:
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
    
    print(f"Collected {total_tokens_processed} tokens")
    activations = torch.cat(activations, dim=0)
    
    # Apply same normalization as part1d
    subset_size = min(10000, len(activations))
    subset_indices = torch.randperm(len(activations))[:subset_size]
    subset = activations[subset_indices]
    
    with torch.no_grad():
        mean = subset.mean(dim=0, keepdim=True)
        std = subset.std(dim=0, keepdim=True)
        std = torch.clamp(std, min=1e-8)
        activations = (activations - mean) / std
    
    return activations

def compute_residual_errors_and_features(original_activations, sae_model):
    """Compute residual errors and SAE features using the trained TopK SAE from part1c."""
    residual_errors = []
    sae_features_list = []
    
    # Process in batches to avoid memory issues
    batch_size = BATCH_SIZE
    num_batches = (len(original_activations) + batch_size - 1) // batch_size
    
    sae_model.eval()
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Computing residual errors and SAE features"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(original_activations))
            batch = original_activations[start_idx:end_idx].to(DEVICE)
            
            # Get SAE features and reconstruction
            sae_features, sae_reconstruction = sae_model(batch)
            
            # Compute residual error
            residual = batch - sae_reconstruction
            
            residual_errors.append(residual.cpu())
            sae_features_list.append(sae_features.cpu())
    
    return torch.cat(residual_errors, dim=0), torch.cat(sae_features_list, dim=0)

def create_component_specific_optimizer(nfm_model, base_lr=LEARNING_RATE):
    """
    Create optimizer with component-specific learning rates.
    
    Based on NFM best practices and the need to balance component contributions.
    """
    
    # Component-specific learning rates
    embedding_lr = base_lr * 2.0       # Boost embedding learning (interactions need help)
    mlp_lr = base_lr * 1.5            # Moderate boost for MLP
    
    if USE_LINEAR_COMPONENT:
        linear_lr = base_lr * 0.5          # Reduce linear learning (often dominates)
        print(f"\nComponent-specific learning rates:")
        print(f"  Linear components: {linear_lr}")
        print(f"  Embedding components: {embedding_lr}")
        print(f"  MLP components: {mlp_lr}")
        print(f"  Global bias: {base_lr}")
        
        optimizer = optim.Adam([
            {
                'params': nfm_model.linear.parameters(), 
                'lr': linear_lr,
                'weight_decay': 1e-6  # Small L2 regularization for linear
            },
            {
                'params': nfm_model.feature_embeddings.parameters(), 
                'lr': embedding_lr,
                'weight_decay': 1e-5  # Stronger regularization for embeddings
            },
            {
                'params': list(nfm_model.interaction_mlp.parameters()) + 
                         list(nfm_model.bi_interaction_bn.parameters()),
                'lr': mlp_lr,
                'weight_decay': 1e-6
            },
            {
                'params': [nfm_model.global_bias],
                'lr': base_lr,
                'weight_decay': 0  # No regularization on bias
            }
        ], betas=(0.9, 0.999), eps=1e-8)
    else:
        print(f"\nComponent-specific learning rates (Interaction-Only):")
        print(f"  Embedding components: {embedding_lr}")
        print(f"  MLP components: {mlp_lr}")
        print(f"  Global bias: {base_lr}")
        
        optimizer = optim.Adam([
            {
                'params': nfm_model.feature_embeddings.parameters(), 
                'lr': embedding_lr,
                'weight_decay': 1e-5
            },
            {
                'params': list(nfm_model.interaction_mlp.parameters()) + 
                         list(nfm_model.bi_interaction_bn.parameters()),
                'lr': mlp_lr,
                'weight_decay': 1e-6
            },
            {
                'params': [nfm_model.global_bias],
                'lr': base_lr,
                'weight_decay': 0
            }
        ], betas=(0.9, 0.999), eps=1e-8)
    
    return optimizer

def improved_train_nfm(nfm_model, train_loader, val_loader, num_steps, checkpoint_dir):
    """IMPROVED training function with component-specific optimization."""
    
    # Use component-specific optimizer
    optimizer = create_component_specific_optimizer(nfm_model, base_lr=LEARNING_RATE)
    
    # Better learning rate scheduler (cosine annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_steps,
        eta_min=LEARNING_RATE * 0.01  # Don't let LR go to zero
    )
    
    # Enhanced metrics tracking for NFM
    metrics_history = {
        'steps': [],
        'total_loss': [],
        'reconstruction_loss': [],
        'best_loss': float('inf'),
        'val_total_loss': [],
        'val_reconstruction_loss': [],
        'val_best_loss': float('inf'),
        # Metrics for tracking combined reconstruction quality
        'combined_reconstruction_error': [],  # Error with SAE1 + NFM
        'sae1_only_error': [],  # Error with SAE1 only
        'val_combined_reconstruction_error': [],
        'val_sae1_only_error': [],
        # Component analysis metrics
        'linear_magnitude': [],  # RMS magnitude of linear component
        'interaction_magnitude': [],  # RMS magnitude of interaction component
        'linear_contribution': [],  # Percentage contribution of linear component
        'val_linear_magnitude': [],
        'val_interaction_magnitude': [],
        'val_linear_contribution': [],
        # New gradient monitoring
        'embedding_grad_norm': [],
        'linear_grad_norm': [],
        'mlp_grad_norm': []
    }
    
    prevent_sleep()
    
    try:
        train_iterator = iter(train_loader)
        
        arch_type = "Linear + Interaction" if USE_LINEAR_COMPONENT else "Interaction-Only"
        for step in tqdm(range(num_steps), desc=f"Training Improved NFM ({arch_type})"):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
            
            # Batch contains original activations, SAE features, and residual errors
            original_activations = batch['original'].to(DEVICE)
            sae_features = batch['sae_features'].to(DEVICE)
            residual_errors = batch['residual'].to(DEVICE)
            
            if check_tensor(residual_errors, f"residual_batch_step_{step}", print_stats=False):
                print(f"Problematic residual batch at step {step}")
                continue
            
            if check_tensor(sae_features, f"sae_features_batch_step_{step}", print_stats=False):
                print(f"Problematic SAE features batch at step {step}")
                continue
            
            # Forward pass on SAE features to predict residual
            nfm_prediction, linear_out, interaction_out = nfm_model(sae_features)
            
            # Check outputs
            if check_tensor(nfm_prediction, f"nfm_prediction_step_{step}", print_stats=False):
                print(f"NaN or Inf detected in NFM prediction at step {step}")
                continue
            
            # Compute component magnitudes and contributions
            interaction_magnitude = torch.sqrt(torch.mean(interaction_out ** 2)).item()
            
            if linear_out is not None:
                linear_magnitude = torch.sqrt(torch.mean(linear_out ** 2)).item()
                total_magnitude = linear_magnitude + interaction_magnitude
                linear_contribution = (linear_magnitude / (total_magnitude + 1e-8)) * 100
            else:
                linear_magnitude = 0.0
                linear_contribution = 0.0
            
            # Compute losses
            try:
                # Reconstruction loss for NFM
                reconstruction_diff = (nfm_prediction - residual_errors)
                reconstruction_loss = torch.mean(reconstruction_diff ** 2)
                
                if torch.isnan(reconstruction_loss) or torch.isinf(reconstruction_loss):
                    print(f"Reconstruction loss is {reconstruction_loss} at step {step}")
                    reconstruction_loss = torch.tensor(1.0, device=DEVICE)
                
                # Total loss (no regularization for NFM)
                loss = reconstruction_loss
                loss = torch.clamp(loss, max=1e6)
                
                # Compute additional error metrics
                # SAE1 only error (this is just the residual error we're trying to model)
                sae1_only_error = torch.mean(residual_errors ** 2).item()
                
                # Combined error (original - (sae1_reconstruction + nfm_prediction))
                # Since residual_errors = original - sae1_reconstruction, 
                # the combined error is: original - (sae1_reconstruction + nfm_prediction)
                # = original - sae1_reconstruction - nfm_prediction
                # = residual_errors - nfm_prediction
                combined_error = torch.mean((residual_errors - nfm_prediction) ** 2).item()
                
            except Exception as e:
                print(f"Exception during loss computation at step {step}: {str(e)}")
                continue
            
            # IMPROVED backward pass with gradient monitoring
            try:
                optimizer.zero_grad()
                loss.backward()
                
                # Monitor gradient norms for each component
                embedding_grad_norm = 0.0
                linear_grad_norm = 0.0
                mlp_grad_norm = 0.0
                
                for name, param in nfm_model.named_parameters():
                    if param.grad is not None:
                        # Check for NaN/Inf gradients
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"NaN/Inf detected in gradients for {name} at step {step}")
                            param.grad = torch.zeros_like(param.grad)
                        
                        # Accumulate gradient norms by component
                        grad_norm = param.grad.norm().item()
                        if 'feature_embeddings' in name:
                            embedding_grad_norm += grad_norm ** 2
                        elif 'linear' in name and 'interaction' not in name:
                            linear_grad_norm += grad_norm ** 2
                        elif ('interaction_mlp' in name or 'bi_interaction' in name or 
                              'global_bias' in name):
                            mlp_grad_norm += grad_norm ** 2
                
                # Convert to RMS values
                embedding_grad_norm = np.sqrt(embedding_grad_norm)
                linear_grad_norm = np.sqrt(linear_grad_norm)
                mlp_grad_norm = np.sqrt(mlp_grad_norm)
                
                # Gradient clipping for stability (more conservative)
                torch.nn.utils.clip_grad_norm_(nfm_model.parameters(), max_norm=0.5)
                
                optimizer.step()
                scheduler.step()
                
            except Exception as e:
                print(f"Exception during backward pass at step {step}: {str(e)}")
                continue
            
            # Track metrics
            metrics_history['steps'].append(step)
            metrics_history['total_loss'].append(loss.item())
            metrics_history['reconstruction_loss'].append(reconstruction_loss.item())
            metrics_history['combined_reconstruction_error'].append(combined_error)
            metrics_history['sae1_only_error'].append(sae1_only_error)
            metrics_history['linear_magnitude'].append(linear_magnitude)
            metrics_history['interaction_magnitude'].append(interaction_magnitude)
            metrics_history['linear_contribution'].append(linear_contribution)
            metrics_history['embedding_grad_norm'].append(embedding_grad_norm)
            metrics_history['linear_grad_norm'].append(linear_grad_norm)
            metrics_history['mlp_grad_norm'].append(mlp_grad_norm)
            
            if loss.item() < metrics_history['best_loss']:
                metrics_history['best_loss'] = loss.item()
            
            # Validation
            if step % VALIDATION_INTERVAL == 0:
                nfm_model.eval()
                val_metrics = validate_nfm(nfm_model, val_loader)
                nfm_model.train()
                
                # Record validation metrics
                metrics_history['val_total_loss'].append(val_metrics['total_loss'])
                metrics_history['val_reconstruction_loss'].append(val_metrics['reconstruction_loss'])
                metrics_history['val_combined_reconstruction_error'].append(val_metrics['combined_reconstruction_error'])
                metrics_history['val_sae1_only_error'].append(val_metrics['sae1_only_error'])
                metrics_history['val_linear_magnitude'].append(val_metrics['linear_magnitude'])
                metrics_history['val_interaction_magnitude'].append(val_metrics['interaction_magnitude'])
                metrics_history['val_linear_contribution'].append(val_metrics['linear_contribution'])
                
                if val_metrics['total_loss'] < metrics_history['val_best_loss']:
                    metrics_history['val_best_loss'] = val_metrics['total_loss']
            
            # ENHANCED progress reporting
            if step % 1000 == 0:
                # Get current learning rates for each component
                if USE_LINEAR_COMPONENT:
                    current_lr_linear = optimizer.param_groups[0]['lr']
                    current_lr_embedding = optimizer.param_groups[1]['lr']
                    current_lr_mlp = optimizer.param_groups[2]['lr']
                    current_lr_bias = optimizer.param_groups[3]['lr']
                else:
                    current_lr_linear = 0.0
                    current_lr_embedding = optimizer.param_groups[0]['lr']
                    current_lr_mlp = optimizer.param_groups[1]['lr']
                    current_lr_bias = optimizer.param_groups[2]['lr']
                
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
                
                # Learning rate and gradient info
                if USE_LINEAR_COMPONENT:
                    print(f"Learning Rates - Linear: {current_lr_linear:.2e}, Embedding: {current_lr_embedding:.2e}, MLP: {current_lr_mlp:.2e}, Bias: {current_lr_bias:.2e}")
                else:
                    print(f"Learning Rates - Embedding: {current_lr_embedding:.2e}, MLP: {current_lr_mlp:.2e}, Bias: {current_lr_bias:.2e}")
                print(f"Gradient Norms - Linear: {linear_grad_norm:.4f}, Embedding: {embedding_grad_norm:.4f}, MLP: {mlp_grad_norm:.4f}")
                
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
                    debug_model_parameters(nfm_model, f"improved_nfm_at_step_{step}")
            
            # Save checkpoint
            if (step + 1) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(
                    nfm_model, optimizer, scheduler, step + 1,
                    metrics_history['val_best_loss'], metrics_history,
                    checkpoint_dir
                )
                print(f"\nCheckpoint saved at step {step + 1}")
    
    finally:
        allow_sleep()

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
                
                # Forward pass
                nfm_prediction, linear_out, interaction_out = nfm_model(sae_features)
                
                if (torch.isnan(nfm_prediction).any() or torch.isinf(nfm_prediction).any()):
                    continue
                
                # Compute component magnitudes and contributions
                interaction_magnitude = torch.sqrt(torch.mean(interaction_out ** 2)).item()
                
                if linear_out is not None:
                    linear_magnitude = torch.sqrt(torch.mean(linear_out ** 2)).item()
                    total_magnitude = linear_magnitude + interaction_magnitude
                    linear_contribution = (linear_magnitude / (total_magnitude + 1e-8)) * 100
                else:
                    linear_magnitude = 0.0
                    linear_contribution = 0.0
                
                # Compute losses
                reconstruction_loss = torch.mean((nfm_prediction - residual_errors) ** 2).item()
                total_loss = reconstruction_loss
                
                # Error metrics
                sae1_only_error = torch.mean(residual_errors ** 2).item()
                combined_error = torch.mean((residual_errors - nfm_prediction) ** 2).item()
                
                # Accumulate metrics
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
    
    # Average metrics
    if num_valid_batches > 0:
        for key in val_metrics:
            val_metrics[key] /= num_valid_batches
    else:
        print("Warning: No valid batches during validation!")
    
    return val_metrics

def main():
    parser = argparse.ArgumentParser(description='Train IMPROVED Neural Factorization Machine for Residual Modeling')
    parser.add_argument('--sae1_model_path', type=str, default='checkpoints_topk/best_model.pt',
                        help='Path to the trained TopK SAE from part1c (default: checkpoints_topk/best_model.pt)')
    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR,
                        help='Directory to save NFM checkpoints')
    parser.add_argument('--nfm_k', type=int, default=NFM_K,
                        help=f'NFM embedding dimension (default: {NFM_K})')
    parser.add_argument('--dropout_rate', type=float, default=NFM_DROPOUT,
                        help=f'Dropout rate for NFM (default: {NFM_DROPOUT})')
    parser.add_argument('--top_n_features', type=int, default=TOP_N_FEATURES,
                        help=f'Keep only top N features per sample (default: {TOP_N_FEATURES})')
    
    args = parser.parse_args()
    
    # Update global variables with command line arguments
    global NFM_K, NFM_DROPOUT, TOP_N_FEATURES
    NFM_K = args.nfm_k
    NFM_DROPOUT = args.dropout_rate
    TOP_N_FEATURES = args.top_n_features
    
    # Set numerical stability
    torch.set_default_dtype(torch.float32)
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH, use_fast=False, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(MODEL_PATH, torch_dtype='auto', device_map='auto', local_files_only=True)
    
    # Collect activations (same as part1d)
    print("Collecting activations...")
    activations = collect_activations(model, tokenizer, NUM_TOKENS)
    
    # Load the trained TopK SAE from part1c
    print(f"Loading trained TopK SAE from {args.sae1_model_path}...")
    
    # Create TopK SAE with correct parameters
    sae1 = TopKSparseAutoencoder(
        input_dim=activations.shape[1],
        hidden_dim=NUM_FEATURES,
        k=K  # Same K as part1c
    ).to(DEVICE)
    
    # Convert to same dtype as model
    sae1 = sae1.to(dtype=next(model.parameters()).dtype)
    
    # Load the trained weights - handle both checkpoint and direct state dict
    if not os.path.exists(args.sae1_model_path):
        raise FileNotFoundError(f"TopK SAE model not found at {args.sae1_model_path}. Please run part1c first.")
    
    try:
        # Try loading as checkpoint first
        checkpoint = torch.load(args.sae1_model_path, map_location=DEVICE)
        if 'model_state' in checkpoint:
            sae1.load_state_dict(checkpoint['model_state'])
        else:
            # Direct state dict
            sae1.load_state_dict(checkpoint)
    except:
        # Try loading as direct state dict
        sae1.load_state_dict(torch.load(args.sae1_model_path, map_location=DEVICE))
    
    print("TopK SAE loaded successfully!")
    
    # Compute residual errors and extract SAE features
    print("Computing residual errors and SAE features...")
    residual_errors, sae_features = compute_residual_errors_and_features(activations, sae1)
    
    print(f"Residual errors computed. Shape: {residual_errors.shape}")
    print(f"SAE features extracted. Shape: {sae_features.shape}")
    check_tensor(residual_errors, "residual_errors", True)
    check_tensor(sae_features, "sae_features", True)
    
    # Analyze SAE feature sparsity
    with torch.no_grad():
        active_features_per_sample = (sae_features > 1e-6).sum(dim=1).float()
        print(f"\nSAE Feature Sparsity Analysis:")
        print(f"Average active features per sample: {active_features_per_sample.mean().item():.1f}")
        print(f"Median active features per sample: {active_features_per_sample.median().item():.1f}")
        print(f"Max active features per sample: {active_features_per_sample.max().item():.0f}")
        print(f"Min active features per sample: {active_features_per_sample.min().item():.0f}")
        print(f"Using top {TOP_N_FEATURES} features per sample for NFM")
    
    # Create train/validation split
    print("Creating train/validation split...")
    dataset_size = len(activations)
    val_size = int(dataset_size * VALIDATION_SPLIT)
    train_size = dataset_size - val_size
    
    indices = torch.randperm(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_original = activations[train_indices]
    train_sae_features = sae_features[train_indices]
    train_residual = residual_errors[train_indices]
    
    val_original = activations[val_indices]
    val_sae_features = sae_features[val_indices]
    val_residual = residual_errors[val_indices]
    
    print(f"Train set size: {train_size} samples")
    print(f"Validation set size: {val_size} samples")
    
    # Create datasets
    train_dataset = ResidualDataset(train_original, train_sae_features, train_residual)
    val_dataset = ResidualDataset(val_original, val_sae_features, val_residual)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize IMPROVED NFM
    arch_type = "Linear + Interaction" if USE_LINEAR_COMPONENT else "Interaction-Only"
    print(f"Initializing IMPROVED Neural Factorization Machine ({arch_type}) with K={NFM_K}, dropout={NFM_DROPOUT}...")
    nfm_model = ImprovedNeuralFactorizationMachine(
        num_sae_features=NUM_FEATURES,  # 50K SAE features
        embedding_dim=NFM_K,  # Embedding dimension
        output_dim=activations.shape[1],  # Output dimension (same as input activations)
        dropout_rate=NFM_DROPOUT  # Configurable dropout rate
    ).to(DEVICE)
    
    # Convert to same dtype as model
    nfm_model = nfm_model.to(dtype=next(model.parameters()).dtype)
    
    # Calculate and print parameter count
    total_params = sum(p.numel() for p in nfm_model.parameters())
    print(f"IMPROVED NFM ({arch_type}) Model Parameters:")
    print(f"  Feature Embeddings: {nfm_model.feature_embeddings.weight.numel():,}")
    if nfm_model.linear is not None:
        print(f"  Linear Layer: {nfm_model.linear.weight.numel() + nfm_model.linear.bias.numel():,}")
    print(f"  Global Bias: {nfm_model.global_bias.numel():,}")
    print(f"  Interaction MLP: {sum(p.numel() for p in nfm_model.interaction_mlp.parameters()):,}")
    print(f"  Batch Norm Layers: {sum(p.numel() for p in nfm_model.bi_interaction_bn.parameters()):,}")
    print(f"  Total Parameters: {total_params:,}")
    
    # Debug initial parameters
    debug_model_parameters(nfm_model, f"initial_improved_nfm_{arch_type.lower().replace(' + ', '_')}")
    
    print(f"Training IMPROVED Neural Factorization Machine ({arch_type})...")
    improved_train_nfm(nfm_model, train_loader, val_loader, TRAIN_STEPS, args.checkpoint_dir)
    
    # Save the final trained NFM
    final_model_name = "improved_nfm_linear_interaction_residual_model.pt" if USE_LINEAR_COMPONENT else "improved_nfm_interaction_only_residual_model.pt"
    torch.save(nfm_model.state_dict(), final_model_name)
    print(f"Training complete! IMPROVED NFM ({arch_type}) saved as {final_model_name}")
    
    # Compute final error comparison on validation set
    print("\n" + "="*60)
    print("FINAL VALIDATION RESULTS - IMPROVED NFM")
    print("="*60)
    
    sae1.eval()
    nfm_model.eval()
    
    with torch.no_grad():
        # Take a sample of validation data for final comparison
        sample_size = min(1000, len(val_original))
        sample_indices = torch.randperm(len(val_original))[:sample_size]
        sample_original = val_original[sample_indices].to(DEVICE)
        sample_sae_features = val_sae_features[sample_indices].to(DEVICE)
        sample_residual = val_residual[sample_indices].to(DEVICE)
        
        # SAE1 reconstruction
        _, sae1_reconstruction = sae1(sample_original)
        sae1_error = torch.mean((sample_original - sae1_reconstruction) ** 2).item()
        
        # NFM prediction of residual
        nfm_prediction, val_linear_out, val_interaction_out = nfm_model(sample_sae_features)
        
        # Combined reconstruction (SAE1 + NFM)
        combined_reconstruction = sae1_reconstruction + nfm_prediction
        combined_error = torch.mean((sample_original - combined_reconstruction) ** 2).item()
        
        # Calculate improvement
        error_reduction = ((sae1_error - combined_error) / sae1_error) * 100
        
        # Component analysis
        val_interaction_magnitude = torch.sqrt(torch.mean(val_interaction_out ** 2)).item()
        
        if val_linear_out is not None:
            val_linear_magnitude = torch.sqrt(torch.mean(val_linear_out ** 2)).item()
            val_total_magnitude = val_linear_magnitude + val_interaction_magnitude
            val_linear_contribution = (val_linear_magnitude / (val_total_magnitude + 1e-8)) * 100
        else:
            val_linear_magnitude = 0.0
            val_linear_contribution = 0.0
        
        # Additional analysis
        original_norm = torch.mean(sample_original ** 2).item()
        
        print(f"Original data MSE (baseline): {original_norm:.6f}")
        print(f"SAE1 only reconstruction error: {sae1_error:.6f}")
        print(f"SAE1 + IMPROVED NFM ({arch_type}) reconstruction error: {combined_error:.6f}")
        print(f"Error reduction: {error_reduction:.2f}%")
        print(f"Relative error (SAE1 only): {(sae1_error / original_norm) * 100:.2f}%")
        print(f"Relative error (Combined): {(combined_error / original_norm) * 100:.2f}%")
        print(f"\nIMPROVED NFM Component Analysis:")
        print(f"Linear component magnitude: {val_linear_magnitude:.6f}")
        print(f"Interaction component magnitude: {val_interaction_magnitude:.6f}")
        if USE_LINEAR_COMPONENT:
            print(f"Linear contribution: {val_linear_contribution:.1f}%")
            print(f"Interaction contribution: {100 - val_linear_contribution:.1f}%")
        print(f"\nKey Improvements Made:")
        print("- ✅ Proper bi-interaction pooling with NFM formula")
        print("- ✅ BatchNorm for interaction vector stability")
        print("- ✅ Component-specific learning rates")
        print("- ✅ Xavier/Kaiming initialization")
        print("- ✅ Global bias term")
        print("- ✅ Enhanced gradient monitoring")
        print("- ✅ Cosine annealing scheduler")

if __name__ == "__main__":
    main()