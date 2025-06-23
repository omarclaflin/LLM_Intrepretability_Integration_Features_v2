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

# Constants
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "open_llama_3b"))
BATCH_SIZE = 4096
NUM_TOKENS = 1_000_000  # Start with 1M-5M tokens
LEARNING_RATE = 1e-5  
NUM_FEATURES = 50_000  # Number of features in SAE
L1_LAMBDA = 50.0  # Increased from 1.0 to get 1-5% sparsity instead of 50-80%
TARGET_LAYER = 16  # LLaMA 3B has 32 layers, we target the middle layer
TRAIN_STEPS = 100_000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints_improved_4"
CHECKPOINT_INTERVAL = 10_000
VALIDATION_INTERVAL = 1000  # How often to compute validation metrics
VALIDATION_SPLIT = 0.1  # 10% of data used for validation
CHUNK_SIZE = 128  # Changed from 512 to 100 tokens per chunk
DEAD_NEURON_THRESHOLD = 1e-6  # Threshold below which a feature is considered "dead"

# Add debugging functions
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
    if metrics_history['val_best_loss'] == best_loss:
        best_model_path = checkpoint_dir / "best_model.pt"
        torch.save(model.state_dict(), best_model_path)

def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    
    return checkpoint['step'], checkpoint['best_loss'], checkpoint['metrics_history']

def compute_variance_explained(original_activations, reconstructed_activations):
    """
    Compute variance explained by SAE reconstructions.
    
    Returns:
        - Total variance explained (scalar)
        - Per-dimension variance explained (vector)
    """
    with torch.no_grad():
        # Center the data (subtract mean)
        original_centered = original_activations - original_activations.mean(dim=0, keepdim=True)
        reconstructed_centered = reconstructed_activations - reconstructed_activations.mean(dim=0, keepdim=True)
        
        # Total variance in original data
        total_variance = torch.var(original_centered, dim=0, unbiased=False)  # Per dimension
        
        # Variance of reconstruction errors
        residual = original_centered - reconstructed_centered
        residual_variance = torch.var(residual, dim=0, unbiased=False)
        
        # Variance explained per dimension
        var_explained_per_dim = 1 - (residual_variance / (total_variance + 1e-8))
        
        # Overall variance explained (weighted by original variance)
        total_var_explained = torch.sum(var_explained_per_dim * total_variance) / torch.sum(total_variance)
        
        return total_var_explained.item(), var_explained_per_dim

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        self.initialize_weights()
    
    def initialize_weights(self):
        # Use a much smaller standard deviation for initialization
        nn.init.normal_(self.decoder.weight, std=0.01)
        # Initialize encoder weights as transpose of decoder
        self.encoder[0].weight.data = self.decoder.weight.data.T
        # Initialize biases to zero
        nn.init.zeros_(self.encoder[0].bias)
        
        # Debug initialization
        print("\n--- Checking initialization ---")
        check_tensor(self.encoder[0].weight, "encoder.weight", True)
        check_tensor(self.encoder[0].bias, "encoder.bias", True)
        check_tensor(self.decoder.weight, "decoder.weight", True)
    
    def forward(self, x):
        # Debug input
        if torch.isnan(x).any() or torch.isinf(x).any():
            check_tensor(x, "forward_input")
            
        features = self.encoder(x)
        
        # Debug features
        if torch.isnan(features).any() or torch.isinf(features).any():
            check_tensor(features, "features")
            
        reconstruction = self.decoder(features)
        
        # Debug reconstruction
        if torch.isnan(reconstruction).any() or torch.isinf(reconstruction).any():
            check_tensor(reconstruction, "reconstruction")
            
        return features, reconstruction

class ActivationDataset(Dataset):
    def __init__(self, activations):
        self.activations = activations
    
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        return self.activations[idx]

def collect_activations(model, tokenizer, num_tokens):
    """Collect activations from the model's target layer using multiple non-overlapping chunks."""
    activations = []
    total_tokens_processed = 0
    
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    
    print(f"Collecting activations using {CHUNK_SIZE}-token chunks...")
    
    for sample in tqdm(dataset, desc="Collecting activations"):
        if total_tokens_processed >= num_tokens:
            break
            
        text = sample["text"].strip()
        if not text:  # Skip empty samples
            continue
            
        # Tokenize the full sample first to see how long it is
        full_tokens = tokenizer.encode(text, add_special_tokens=False)
        
        if len(full_tokens) < CHUNK_SIZE:
            continue  # Skip samples that are too short
        
        # Extract multiple non-overlapping chunks
        num_chunks = len(full_tokens) // CHUNK_SIZE
        
        for chunk_idx in range(num_chunks):
            if total_tokens_processed >= num_tokens:
                break
                
            start_idx = chunk_idx * CHUNK_SIZE
            end_idx = start_idx + CHUNK_SIZE
            chunk_tokens = full_tokens[start_idx:end_idx]
            
            # Convert back to text and tokenize for model input
            chunk_text = tokenizer.decode(chunk_tokens)
            inputs = tokenizer(chunk_text, return_tensors="pt", truncation=True, max_length=CHUNK_SIZE)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            # Get activations from target layer
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[TARGET_LAYER]
                
                # Debug hidden states
                if check_tensor(hidden_states, "hidden_states_chunk", print_stats=False):
                    print(f"Found problematic hidden states in chunk")
                    continue
                    
                # Reshape to (batch_size * seq_len, hidden_size)
                batch_size, seq_len, hidden_size = hidden_states.shape
                hidden_states = hidden_states.reshape(-1, hidden_size)
                
                # Store activations
                activations.append(hidden_states.cpu())
                total_tokens_processed += seq_len
    
    print(f"Collected {total_tokens_processed} tokens from {len(activations)} chunks")
    
    # Concatenate all activations
    activations = torch.cat(activations, dim=0)
    
    # Check raw activations before normalization
    print("\n--- Pre-normalization activations ---")
    check_tensor(activations, "raw_activations", True)
    
    # Compute normalization statistics on a subset to save memory
    subset_size = min(10000, len(activations))
    subset_indices = torch.randperm(len(activations))[:subset_size]
    subset = activations[subset_indices]
    
    # Use standard z-score normalization
    with torch.no_grad():
        mean = subset.mean(dim=0, keepdim=True)
        std = subset.std(dim=0, keepdim=True)
        
        # Prevent division by zero
        std = torch.clamp(std, min=1e-8)
        
        print(f"\nNormalization stats computed from {subset_size} samples:")
        print(f"Mean magnitude: {mean.norm().item():.6f}")
        print(f"Std magnitude: {std.norm().item():.6f}")
        
        # Apply z-score normalization
        activations = (activations - mean) / std
    
    # Check normalized activations
    print("\n--- Post-normalization activations ---")
    check_tensor(activations, "normalized_activations", True)
    
    return activations

def process_batch(model, tokenizer, batch_texts, activations):
    """Process a batch of texts and return number of tokens processed."""
    # Tokenize batch
    inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=CHUNK_SIZE, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    # Get activations from target layer
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Use activations from the target layer
        hidden_states = outputs.hidden_states[TARGET_LAYER]
        
        # Debug hidden states
        if check_tensor(hidden_states, "hidden_states_batch", print_stats=False):
            print(f"Found problematic hidden states in batch")
            
        # Reshape to (batch_size * seq_len, hidden_size)
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_size)
        
        # Store activations
        activations.append(hidden_states.cpu())
        
        # Return number of tokens processed
        return batch_size * seq_len

def compute_dead_neurons(features, threshold=DEAD_NEURON_THRESHOLD):
    """
    Compute percentage of dead neurons (features that never activate above threshold)
    """
    with torch.no_grad():
        max_activations = torch.max(torch.abs(features), dim=0)[0]
        dead_mask = max_activations < threshold
        dead_percentage = dead_mask.float().mean().item() * 100
        return dead_percentage

def train_sae(model, train_loader, val_loader, num_steps, checkpoint_dir=CHECKPOINT_DIR):
    """Train the sparse autoencoder."""
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=0, eps=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=1.0,
        end_factor=0.1,
        total_iters=int(num_steps * 0.8)
    )
    
    # L1 lambda scheduler - more gradual ramp up
    l1_scheduler = lambda step: min(L1_LAMBDA * (step / (num_steps * 0.1)), L1_LAMBDA)
    
    # Initialize metrics tracking
    metrics_history = {
        'steps': [],
        'total_loss': [],
        'reconstruction_loss': [],
        'l1_loss': [],
        'best_loss': float('inf'),
        'l0_sparsity': [],  # Percentage of features that are zero
        'percent_active': [],  # Percentage of features that are active (1 - l0_sparsity)
        'dead_neurons': [],  # Percentage of neurons that never activate
        'variance_explained': [],  # Variance explained by reconstructions
        'val_total_loss': [],
        'val_reconstruction_loss': [],
        'val_l1_loss': [],
        'val_best_loss': float('inf'),
        'val_l0_sparsity': [],
        'val_percent_active': [],
        'val_dead_neurons': [],
        'val_variance_explained': []  # Validation variance explained
    }
    
    # Track feature activations across batches for dead neuron analysis
    feature_max_activations = torch.zeros(NUM_FEATURES, device=DEVICE)
    
    # Prevent system sleep
    prevent_sleep()
    
    try:
        # Create iterator once outside the loop
        train_iterator = iter(train_loader)
        
        for step in tqdm(range(num_steps), desc="Training SAE"):
            # Get next batch, restart iterator if needed
            try:
                batch = next(train_iterator)
            except StopIteration:
                # Dataset exhausted, create new iterator
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
            
            batch = batch.to(DEVICE)
            
            # Check batch for NaN/Inf
            if check_tensor(batch, f"batch_step_{step}", print_stats=False):
                print(f"Problematic batch at step {step}")
                check_tensor(batch, f"batch_step_{step}", print_stats=True)
                print("Skipping this batch")
                continue
            
            # Forward pass
            features, reconstruction = model(batch)
            
            # Update feature activation tracking for dead neuron analysis
            with torch.no_grad():
                batch_max_activations = torch.max(torch.abs(features), dim=0)[0]
                feature_max_activations = torch.max(feature_max_activations, batch_max_activations)
            
            # Check outputs for NaN/Inf
            features_issue = check_tensor(features, f"features_step_{step}", print_stats=False)
            recon_issue = check_tensor(reconstruction, f"reconstruction_step_{step}", print_stats=False)
            
            if features_issue or recon_issue:
                print(f"NaN or Inf detected in forward pass at step {step}")
                if features_issue:
                    check_tensor(features, f"features_step_{step}", print_stats=True)
                if recon_issue:
                    check_tensor(reconstruction, f"reconstruction_step_{step}", print_stats=True)
                
                # Debug model parameters
                debug_model_parameters(model, f"model_at_step_{step}")
                
                print("Skipping this batch")
                continue
            
            # Compute losses with safeguards
            try:
                # Compute reconstruction loss with clipping to prevent extremely large values
                reconstruction_diff = (reconstruction - batch)
                reconstruction_loss = torch.mean(reconstruction_diff ** 2)
                
                if torch.isnan(reconstruction_loss) or torch.isinf(reconstruction_loss):
                    print(f"Reconstruction loss is {reconstruction_loss} at step {step}")
                    # More detailed debugging
                    print("Reconstruction diff stats:")
                    check_tensor(reconstruction_diff, "reconstruction_diff")
                    reconstruction_loss = torch.tensor(1.0, device=DEVICE)  # Use a safe fallback
                
                # Compute L1 loss with safeguards
                current_l1_lambda = l1_scheduler(step)
                decoder_norms = torch.norm(model.decoder.weight, p=2, dim=0)
                if check_tensor(decoder_norms, f"decoder_norms_step_{step}", print_stats=False):
                    print(f"Issue with decoder norms at step {step}")
                    check_tensor(decoder_norms, f"decoder_norms_step_{step}")
                    # Use safe fallback
                    decoder_norms = torch.ones_like(decoder_norms)
                
                abs_features = torch.abs(features)
                if check_tensor(abs_features, f"abs_features_step_{step}", print_stats=False):
                    print(f"Issue with abs(features) at step {step}")
                    check_tensor(abs_features, f"abs_features_step_{step}")
                    # Use safe fallback
                    abs_features = torch.ones_like(abs_features)
                
                l1_loss = current_l1_lambda * torch.mean(abs_features * decoder_norms)
                
                if torch.isnan(l1_loss) or torch.isinf(l1_loss):
                    print(f"L1 loss is {l1_loss} at step {step}")
                    l1_loss = torch.tensor(0.0, device=DEVICE)  # Use a safe fallback
                
                # Calculate sparsity metrics
                l0_sparsity = compute_l0_sparsity(features)
                percent_active = (1.0 - l0_sparsity) * 100  # Convert to percentage
                
                # Calculate variance explained
                var_explained, _ = compute_variance_explained(batch, reconstruction)
                
                # Total loss with clipping
                loss = reconstruction_loss + l1_loss
                loss = torch.clamp(loss, max=1e6)  # Prevent truly infinite values
                
            except Exception as e:
                print(f"Exception during loss computation at step {step}: {str(e)}")
                # Skip this iteration
                continue
            
            # Backward pass with exception handling
            try:
                optimizer.zero_grad()
                loss.backward()
                
                # Check for NaN/Inf in gradients
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"NaN/Inf detected in gradients for {name} at step {step}")
                            param.grad = torch.zeros_like(param.grad)  # Zero out problematic gradients
                
                # Gradient clipping
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
            except Exception as e:
                print(f"Exception during backward pass at step {step}: {str(e)}")
                continue
            
            # Track metrics
            metrics_history['steps'].append(step)
            metrics_history['total_loss'].append(loss.item())
            metrics_history['reconstruction_loss'].append(reconstruction_loss.item())
            metrics_history['l1_loss'].append(l1_loss.item())
            metrics_history['l0_sparsity'].append(l0_sparsity)
            metrics_history['percent_active'].append(percent_active)
            metrics_history['variance_explained'].append(var_explained)
            
            # Compute dead neurons percentage
            if step % 1000 == 0:  # Don't compute every step for efficiency
                dead_neuron_pct = (feature_max_activations < DEAD_NEURON_THRESHOLD).float().mean().item() * 100
                metrics_history['dead_neurons'].append(dead_neuron_pct)
            
            # Use full lambda loss for best model comparison (fixes early training bias)
            full_lambda_loss = reconstruction_loss + L1_LAMBDA * torch.mean(abs_features * decoder_norms)
            if full_lambda_loss.item() < metrics_history['best_loss']:
                metrics_history['best_loss'] = full_lambda_loss.item()
            
            # Validation - run periodically to monitor generalization
            if step % VALIDATION_INTERVAL == 0:
                model.eval()  # Set model to evaluation mode
                val_metrics = validate_sae(model, val_loader, current_l1_lambda)
                model.train()  # Set model back to training mode
                
                # Record validation metrics
                metrics_history['val_total_loss'].append(val_metrics['total_loss'])
                metrics_history['val_reconstruction_loss'].append(val_metrics['reconstruction_loss'])
                metrics_history['val_l1_loss'].append(val_metrics['l1_loss'])
                metrics_history['val_l0_sparsity'].append(val_metrics['l0_sparsity'])
                metrics_history['val_percent_active'].append(val_metrics['percent_active'])
                metrics_history['val_dead_neurons'].append(val_metrics['dead_neurons'])
                metrics_history['val_variance_explained'].append(val_metrics['variance_explained'])
                
                # Use full lambda loss for validation best model comparison too
                val_full_lambda_loss = val_metrics['reconstruction_loss'] + L1_LAMBDA * val_metrics['raw_l1']
                if val_full_lambda_loss < metrics_history['val_best_loss']:
                    metrics_history['val_best_loss'] = val_full_lambda_loss
            
            # Print metrics every 1000 steps
            if step % 1000 == 0:
                print(f"\nStep {step}")
                print(f"Train - Total Loss: {loss.item():.4f}")
                print(f"Train - Recon Loss: {reconstruction_loss.item():.4f}")
                print(f"Train - L1 Loss: {l1_loss.item():.4f}")
                print(f"Train - % Active Features: {percent_active:.2f}%")
                print(f"Train - Variance Explained: {var_explained:.4f}")
                print(f"Train - Best Loss (full λ): {metrics_history['best_loss']:.4f}")
                
                if len(metrics_history['dead_neurons']) > 0:
                    print(f"Train - Dead Neurons: {metrics_history['dead_neurons'][-1]:.2f}%")
                
                # Print validation metrics if available
                if metrics_history['val_total_loss']:
                    recent_val_idx = len(metrics_history['val_total_loss']) - 1
                    print(f"Val - Total Loss: {metrics_history['val_total_loss'][recent_val_idx]:.4f}")
                    print(f"Val - Recon Loss: {metrics_history['val_reconstruction_loss'][recent_val_idx]:.4f}")
                    print(f"Val - L1 Loss: {metrics_history['val_l1_loss'][recent_val_idx]:.4f}")
                    print(f"Val - % Active Features: {metrics_history['val_percent_active'][recent_val_idx]:.2f}%")
                    print(f"Val - Variance Explained: {metrics_history['val_variance_explained'][recent_val_idx]:.4f}")
                    print(f"Val - Best Loss (full λ): {metrics_history['val_best_loss']:.4f}")
                    print(f"Val - Dead Neurons: {metrics_history['val_dead_neurons'][recent_val_idx]:.2f}%")
                
                print(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
                print(f"Current L1 Lambda: {current_l1_lambda:.4f}")
                
                # Check model parameters periodically
                if step % 10000 == 0:
                    debug_model_parameters(model, f"model_at_step_{step}")
            
            # Save checkpoint every CHECKPOINT_INTERVAL steps
            if (step + 1) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(
                    model, optimizer, scheduler, step + 1,
                    metrics_history['val_best_loss'], metrics_history,
                    checkpoint_dir
                )
                print(f"\nCheckpoint saved at step {step + 1}")
    
    finally:
        # Allow system sleep again
        allow_sleep()

def validate_sae(model, val_loader, current_l1_lambda):
    """
    Compute validation metrics for the SAE
    Returns a dictionary with validation metrics
    """
    val_metrics = {
        'total_loss': 0.0,
        'reconstruction_loss': 0.0,
        'l1_loss': 0.0,
        'raw_l1': 0.0,  # L1 loss without current lambda scaling
        'l0_sparsity': 0.0,
        'percent_active': 0.0,
        'dead_neurons': 0.0,
        'variance_explained': 0.0
    }
    
    # Number of validation batches to sample (use a maximum to keep validation fast)
    max_val_batches = min(10, len(val_loader))
    val_iterator = iter(val_loader)
    
    num_valid_batches = 0
    feature_max_activations = torch.zeros(NUM_FEATURES, device=DEVICE)
    
    with torch.no_grad():  # No gradients needed for validation
        for _ in range(max_val_batches):
            try:
                batch = next(val_iterator)
                batch = batch.to(DEVICE)
                
                # Skip problematic batches
                if check_tensor(batch, "val_batch", print_stats=False):
                    continue
                
                # Forward pass
                features, reconstruction = model(batch)
                
                # Skip if outputs have issues
                if (torch.isnan(features).any() or torch.isinf(features).any() or
                    torch.isnan(reconstruction).any() or torch.isinf(reconstruction).any()):
                    continue
                
                # Update feature activation tracking
                batch_max_activations = torch.max(torch.abs(features), dim=0)[0]
                feature_max_activations = torch.max(feature_max_activations, batch_max_activations)
                
                # Compute losses
                reconstruction_loss = torch.mean((reconstruction - batch) ** 2).item()
                
                # Calculate L1 loss components
                decoder_norms = torch.norm(model.decoder.weight, p=2, dim=0)
                abs_features = torch.abs(features)
                raw_l1 = torch.mean(abs_features * decoder_norms).item()
                l1_loss = current_l1_lambda * raw_l1
                
                # Calculate total loss
                total_loss = reconstruction_loss + l1_loss
                
                # Calculate sparsity metrics
                l0_sparsity = compute_l0_sparsity(features)
                percent_active = (1.0 - l0_sparsity) * 100
                
                # Calculate variance explained
                var_explained, _ = compute_variance_explained(batch, reconstruction)
                
                # Accumulate metrics
                val_metrics['total_loss'] += total_loss
                val_metrics['reconstruction_loss'] += reconstruction_loss
                val_metrics['l1_loss'] += l1_loss
                val_metrics['raw_l1'] += raw_l1
                val_metrics['l0_sparsity'] += l0_sparsity
                val_metrics['percent_active'] += percent_active
                val_metrics['variance_explained'] += var_explained
                
                num_valid_batches += 1
                
            except StopIteration:
                break
    
    # Compute dead neurons for validation
    if num_valid_batches > 0:
        dead_neuron_pct = (feature_max_activations < DEAD_NEURON_THRESHOLD).float().mean().item() * 100
        val_metrics['dead_neurons'] = dead_neuron_pct
        
        # Compute average metrics
        for key in val_metrics:
            if key != 'dead_neurons':  # Don't average dead neurons, it's already a percentage
                val_metrics[key] /= num_valid_batches
    else:
        print("Warning: No valid batches during validation!")
    
    return val_metrics

# Add L0 sparsity calculation function
def compute_l0_sparsity(features, threshold=1e-6):
    """
    Compute L0 'norm' (count of zero elements) for features
    threshold: values below this are considered zero
    Returns: fraction of elements that are zero (higher = more sparse)
    """
    with torch.no_grad():
        # Count elements close to zero
        zeros = (torch.abs(features) < threshold).float().mean().item()
        return zeros

def main():
    # Set stronger numerical stability settings
    torch.set_default_dtype(torch.float32)  # Use at least float32 precision
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH, use_fast=False, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(MODEL_PATH, torch_dtype='auto', device_map='auto', local_files_only=True)
    
    # Collect activations
    print("Collecting activations...")
    activations = collect_activations(model, tokenizer, NUM_TOKENS)
    
    # Create train/validation split
    print("Creating train/validation split...")
    dataset_size = len(activations)
    val_size = int(dataset_size * VALIDATION_SPLIT)
    train_size = dataset_size - val_size
    
    # Randomly permute indices to create the split
    indices = torch.randperm(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_activations = activations[train_indices]
    val_activations = activations[val_indices]
    
    print(f"Train set size: {train_size} samples")
    print(f"Validation set size: {val_size} samples")
    
    # Create dataset and dataloader for both train and validation
    train_dataset = ActivationDataset(train_activations)
    val_dataset = ActivationDataset(val_activations)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize and train SAE
    print("Initializing SAE...")
    sae = SparseAutoencoder(
        input_dim=activations.shape[1],
        hidden_dim=NUM_FEATURES
    ).to(DEVICE)
    
    # Convert SAE to same dtype as model
    sae = sae.to(dtype=next(model.parameters()).dtype)
    
    # Debug initial SAE parameters
    debug_model_parameters(sae, "initial_sae")
    
    print("Training SAE...")
    train_sae(sae, train_loader, val_loader, TRAIN_STEPS)
    
    # Post-process: normalize decoder columns to unit norm
    with torch.no_grad():
        decoder_norms = torch.norm(sae.decoder.weight, p=2, dim=0)
        # Prevent division by zero
        decoder_norms = torch.clamp(decoder_norms, min=1e-8)
        sae.decoder.weight.data = sae.decoder.weight.data / decoder_norms
    
    # Save the final trained SAE
    torch.save(sae.state_dict(), "sae_model.pt")
    print("Training complete! Model saved as sae_model.pt")

if __name__ == "__main__":
    main()