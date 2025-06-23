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

# K=150, top_n/K = 250, batch=130, init std = 0.05, lt = 5e-4
# Step 12000
# Train - Total Loss: 0.1700
# Train - Recon Loss: 0.1700
# Train - SAE1 Only Error: 0.3252
# Train - Combined (SAE1+NFM) Error: 0.1700
# Train - Error Reduction: 47.71%
# Train - Linear Magnitude: 0.3428
# Train - Interaction Magnitude: 0.0313
# Train - Linear Contribution: 91.6%
# Val - Total Loss: 0.2659
# Val - SAE1 Only Error: 0.3346
# Val - Combined (SAE1+NFM) Error: 0.2659
# Val - Error Reduction: 20.54%
# Val - Linear Magnitude: 0.3482
# Val - Interaction Magnitude: 0.0312
# Val - Linear Contribution: 91.8%
# Step 32000
# Train - Total Loss: 0.1445
# Train - Recon Loss: 0.1445
# Train - SAE1 Only Error: 0.3169
# Train - Combined (SAE1+NFM) Error: 0.1445
# Train - Error Reduction: 54.39%
# Train - Linear Magnitude: 0.3481
# Train - Interaction Magnitude: 0.0505
# Train - Linear Contribution: 87.3%
# Val - Total Loss: 0.2857
# Val - SAE1 Only Error: 0.3346
# Val - Combined (SAE1+NFM) Error: 0.2857
# Val - Error Reduction: 14.62%
# Val - Linear Magnitude: 0.3712
# Val - Interaction Magnitude: 0.0502
# Val - Linear Contribution: 88.1%

# K=300, top_n/K = 500, batch=100, init std = 0.01, lr = 5e-5
# Step 245000
# Train - Total Loss: 0.1050
# Train - Recon Loss: 0.1050
# Train - SAE1 Only Error: 0.1898
# Train - Combined (SAE1+NFM) Error: 0.1050
# Train - Error Reduction: 44.69%
# Train - Linear Magnitude: 0.2178
# Train - Interaction Magnitude: 0.0233
# Train - Linear Contribution: 90.3%
# Val - Total Loss: 0.1881
# Val - SAE1 Only Error: 0.1975
# Val - Combined (SAE1+NFM) Error: 0.1881
# Val - Error Reduction: 4.73%
# Val - Linear Magnitude: 0.2301
# Val - Interaction Magnitude: 0.0233
# Val - Linear Contribution: 90.8%

#K = 100, top n/topk = 300, 5e-4, 0.05
# Step 7000
# Train - Total Loss: 0.1580
# Train - Recon Loss: 0.1580
# Train - SAE1 Only Error: 0.2964
# Train - Combined (SAE1+NFM) Error: 0.1580
# Train - Error Reduction: 46.71%
# Train - Linear Magnitude: 0.3130
# Train - Interaction Magnitude: 0.0217
# Train - Linear Contribution: 93.5%
# Val - Total Loss: 0.2345
# Val - SAE1 Only Error: 0.2871
# Val - Combined (SAE1+NFM) Error: 0.2345
# Val - Error Reduction: 18.33%
# Val - Linear Magnitude: 0.3067
# Val - Interaction Magnitude: 0.0217
# Val - Linear Contribution: 93.4%

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "open_llama_3b"))
BATCH_SIZE = 130
NUM_TOKENS = 200_000  # Same as part1 5_000_000
LEARNING_RATE = 0.0005  # More conservative - NFM literature might assume different data scales  
NUM_FEATURES = 50_000  # Same feature size as part1
NFM_K = 100  # NFM embedding dimension - modifiable parameter
TARGET_LAYER = 16  # Same target layer as part1
TRAIN_STEPS = 300_000  # Same number of training steps
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints_nfm"  # NFM checkpoint directory
CHECKPOINT_INTERVAL = 10_000
VALIDATION_INTERVAL = 1000
VALIDATION_SPLIT = 0.1
CHUNK_SIZE = 128  # Same as SAE
#for interaction only
NFM_DROPOUT = 0.15  # Dropout rate for NFM interaction MLP
SPARSE_THRESHOLD = None  # Set to float value (e.g. 0.4), hard filter on SAE activations > threshold, to enable sparse interactions, None for dense
TOP_N_FEATURES = 300  # Only keep top N features per sample, zero out the rest -- use this instead
K = 300  # TopK parameter (for SAE) -- we use masking on SAE inference
USE_LINEAR_COMPONENT = True  # Set to True to add linear component alongside interactions
NFM_INIT_WEIGHT = 0.05

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

# Neural Factorization Machine for modeling residuals
class NeuralFactorizationMachine(nn.Module):
    def __init__(self, num_sae_features, embedding_dim, output_dim):
        super().__init__()
        self.num_sae_features = num_sae_features
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        
        # Feature embeddings for interaction modeling
        self.feature_embeddings = nn.Embedding(num_sae_features, embedding_dim)
        
        # Optional linear component (first-order effects)
        self.linear = nn.Linear(num_sae_features, output_dim, bias=True) if USE_LINEAR_COMPONENT else None
        
        # MLP for processing interaction vector
        self.interaction_mlp = nn.Sequential(
            nn.Dropout(NFM_DROPOUT),  # NFM-specific: dropout=0.5 on interaction vector before MLP
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, output_dim)
        )
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize NFM weights."""
        # Initialize embeddings with small random values
        nn.init.normal_(self.feature_embeddings.weight, std=NFM_INIT_WEIGHT)
        
        # Initialize linear layer if present
        if self.linear is not None:
            nn.init.normal_(self.linear.weight, std=0.01)
            nn.init.zeros_(self.linear.bias)
        
        # Initialize MLP layers
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
        """
        Forward pass of NFM.
        
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
        
        # Keep only top N features per sample, zero out the rest
        if TOP_N_FEATURES is not None and TOP_N_FEATURES < sae_features.shape[1]:
            # Get top N indices per sample
            top_values, top_indices = torch.topk(sae_features, k=TOP_N_FEATURES, dim=1)
            # Create sparse version by zeroing out non-top features
            sae_features_sparse = torch.zeros_like(sae_features)
            sae_features_sparse.scatter_(1, top_indices, top_values)
            sae_features = sae_features_sparse
        
        # Interaction component (second-order)
        # Create interaction vector efficiently using the polynomial expansion trick
        # This computes all pairwise interactions implicitly
        
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
            
            # Compute interaction vector: 0.5 * (sum^2 - sum_of_squares) - heavily scaled for stability with dense SAE features
            interaction_vector = 0.5 * (sum_embeddings ** 2 - sum_squares)
        else:
            # Dense computation (current approach)
            all_embeddings = self.feature_embeddings.weight  # [num_features, embedding_dim]
            weighted_embeddings = sae_features.unsqueeze(-1) * all_embeddings.unsqueeze(0)  # [batch, num_features, embedding_dim]
            
            # Sum of weighted embeddings
            sum_embeddings = torch.sum(weighted_embeddings, dim=1)  # [batch, embedding_dim]
            
            # Sum of squares of weighted embeddings
            square_embeddings = weighted_embeddings ** 2
            sum_squares = torch.sum(square_embeddings, dim=1)  # [batch, embedding_dim]
            
            # Compute interaction vector: 0.5 * (sum^2 - sum_of_squares) - heavily scaled for stability with dense SAE features
            interaction_vector = 0.5 * (sum_embeddings ** 2 - sum_squares)  # [batch, embedding_dim]        

        # Check for issues in interaction computation
        if torch.isnan(interaction_vector).any() or torch.isinf(interaction_vector).any():
            check_tensor(interaction_vector, "interaction_vector")
            # Fallback to zeros if computation fails
            interaction_vector = torch.zeros_like(interaction_vector)
        
        # Process interaction vector through MLP
        interaction_out = self.interaction_mlp(interaction_vector)
        
        # Compute linear component if present
        linear_out = None
        if self.linear is not None:
            linear_out = self.linear(sae_features)
            output = linear_out + interaction_out
        else:
            output = interaction_out  # Interaction-only
        
        # Final check
        if torch.isnan(output).any() or torch.isinf(output).any():
            check_tensor(output, "nfm_output")
        
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

def train_nfm(nfm_model, train_loader, val_loader, num_steps, checkpoint_dir):
    """Train the Neural Factorization Machine."""
    optimizer = optim.Adam(nfm_model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=0, eps=1e-5)
    
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=1.0,
        end_factor=0.1,
        total_iters=int(num_steps * 0.8)
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
        'val_linear_contribution': []
    }
    
    prevent_sleep()
    
    try:
        train_iterator = iter(train_loader)
        
        arch_type = "Linear + Interaction" if USE_LINEAR_COMPONENT else "Interaction-Only"
        for step in tqdm(range(num_steps), desc=f"Training NFM ({arch_type})"):
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
            interaction_magnitude = torch.mean(abs(interaction_out)).item()
            
            if linear_out is not None:
                linear_magnitude = torch.mean(abs(linear_out)).item()
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
            
            # Backward pass
            try:
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient checking and clipping
                for name, param in nfm_model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"NaN/Inf detected in gradients for {name} at step {step}")
                            param.grad = torch.zeros_like(param.grad)
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(nfm_model.parameters(), max_norm=1.0)
                
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
            
            # Print metrics
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
                interaction_magnitude = torch.mean(abs(interaction_out)).item()
                
                if linear_out is not None:
                    linear_magnitude = torch.mean(abs(linear_out) ).item()
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
    parser = argparse.ArgumentParser(description='Train Neural Factorization Machine for Residual Modeling')
    parser.add_argument('--sae1_model_path', type=str, default='checkpoints_topk/best_model.pt',
                        help='Path to the trained TopK SAE from part1c (default: checkpoints_topk/best_model.pt)')
    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR,
                        help='Directory to save NFM checkpoints')
    parser.add_argument('--nfm_k', type=int, default=NFM_K,
                        help=f'NFM embedding dimension (default: {NFM_K})')
    
    args = parser.parse_args()
    
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
    
    # Initialize NFM
    arch_type = "Linear + Interaction" if USE_LINEAR_COMPONENT else "Interaction-Only"
    print(f"Initializing Neural Factorization Machine ({arch_type}) with K={NFM_K}...")
    nfm_model = NeuralFactorizationMachine(
        num_sae_features=NUM_FEATURES,  # 50K SAE features
        embedding_dim=NFM_K,  # Embedding dimension
        output_dim=activations.shape[1]  # Output dimension (same as input activations)
    ).to(DEVICE)
    
    # Convert to same dtype as model
    nfm_model = nfm_model.to(dtype=next(model.parameters()).dtype)
    
    # Calculate and print parameter count
    total_params = sum(p.numel() for p in nfm_model.parameters())
    print(f"NFM ({arch_type}) Model Parameters:")
    print(f"  Feature Embeddings: {nfm_model.feature_embeddings.weight.numel():,}")
    if nfm_model.linear is not None:
        print(f"  Linear Layer: {nfm_model.linear.weight.numel() + nfm_model.linear.bias.numel():,}")
    print(f"  Interaction MLP: {sum(p.numel() for p in nfm_model.interaction_mlp.parameters()):,}")
    print(f"  Total Parameters: {total_params:,}")
    
    # Debug initial parameters
    debug_model_parameters(nfm_model, f"initial_nfm_{arch_type.lower().replace(' + ', '_')}")
    
    print(f"Training Neural Factorization Machine ({arch_type})...")
    train_nfm(nfm_model, train_loader, val_loader, TRAIN_STEPS, args.checkpoint_dir)
    
    # Save the final trained NFM
    final_model_name = "nfm_linear_interaction_residual_model.pt" if USE_LINEAR_COMPONENT else "nfm_interaction_only_residual_model.pt"
    torch.save(nfm_model.state_dict(), final_model_name)
    print(f"Training complete! NFM ({arch_type}) saved as {final_model_name}")
    
    # Compute final error comparison on validation set
    print("\n" + "="*50)
    print("FINAL VALIDATION RESULTS")
    print("="*50)
    
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
        val_interaction_magnitude = torch.mean(abs(val_interaction_out )).item()
        
        if val_linear_out is not None:
            val_linear_magnitude = torch.mean(abs(val_linear_out)).item()
            val_total_magnitude = val_linear_magnitude + val_interaction_magnitude
            val_linear_contribution = (val_linear_magnitude / (val_total_magnitude + 1e-8)) * 100
        else:
            val_linear_magnitude = 0.0
            val_linear_contribution = 0.0
        
        # Additional analysis
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