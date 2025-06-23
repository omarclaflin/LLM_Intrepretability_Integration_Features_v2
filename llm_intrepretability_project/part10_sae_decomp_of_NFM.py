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

# Constants - matching original SAE script
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "open_llama_3b"))
BATCH_SIZE = 2048  # Same as original SAE
NFM_BATCH_SIZE = 64  # Smaller batch size for memory-intensive NFM operations
NUM_TOKENS = 5_000_000  # Same as original SAE
LEARNING_RATE = 1e-4  # Same as TopK SAE
TARGET_LAYER = 16  # Same target layer
TRAIN_STEPS = 200_000  # Same as original SAE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints_interaction_sae_topk"
CHECKPOINT_INTERVAL = 10_000
VALIDATION_INTERVAL = 1000
VALIDATION_SPLIT = 0.1

# TopK Constants (from part1c)
# K will be set dynamically based on NFM dimensions
DEAD_NEURON_THRESHOLD = 1e-6

# Interaction SAE constants
INTERACTION_SAE_EXPANSION = 25  # 15x expansion: K -> K*15 features

# Utility functions from original scripts
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
        best_model_path = checkpoint_dir / "best_interaction_sae_topk_model.pt"
        torch.save(model.state_dict(), best_model_path)

def compute_l0_sparsity(features, threshold=1e-6):
    """Compute L0 'norm' (count of non-zero elements) for features."""
    with torch.no_grad():
        zeros = (torch.abs(features) < threshold).float().mean().item()
        return zeros

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

# TopK SAE class (from part1c, adapted for interaction SAE)
class TopKSparseAutoencoder(nn.Module):
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
        # Use Kaiming initialization for encoder (from part1c)
        nn.init.kaiming_normal_(self.encoder[0].weight, mode='fan_in', nonlinearity='relu')
        # Initialize encoder weights as transpose of decoder
        self.encoder[0].weight.data = self.decoder.weight.data.T
        # Initialize biases to zero
        nn.init.zeros_(self.encoder[0].bias)
        
        # Debug initialization
        print("\n--- Checking TopK Interaction SAE initialization ---")
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

# Modified NFM class to expose post-MLP1 vectors
class NeuralFactorizationMachine(nn.Module):
    def __init__(self, num_sae_features, embedding_dim, output_dim, use_linear_component=True):
        super().__init__()
        self.num_sae_features = num_sae_features
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.use_linear_component = use_linear_component
        
        # Feature embeddings for interaction modeling
        self.feature_embeddings = nn.Embedding(num_sae_features, embedding_dim)
        
        # Optional linear component
        self.linear = nn.Linear(num_sae_features, output_dim, bias=True) if use_linear_component else None
        
        # MLP for processing interaction vector
        self.interaction_mlp = nn.Sequential(
            nn.Dropout(0.4),  # Use default dropout
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, output_dim)
        )
    
    def get_interaction_vector(self, sae_features):
        """
        Extract just the interaction vector (BEFORE MLP).
        This is the original interaction computation.
        """
        batch_size = sae_features.shape[0]
        
        # Keep only top N features per sample, zero out the rest
        # Detect this from the loaded model or use all features
        top_n_features = getattr(self, 'top_n_features', None)
        if top_n_features is not None and top_n_features < sae_features.shape[1]:
            top_values, top_indices = torch.topk(sae_features, k=top_n_features, dim=1)
            sae_features_sparse = torch.zeros_like(sae_features)
            sae_features_sparse.scatter_(1, top_indices, top_values)
            sae_features = sae_features_sparse
        
        # Compute interaction vector - same as original NFM
        all_embeddings = self.feature_embeddings.weight
        weighted_embeddings = sae_features.unsqueeze(-1) * all_embeddings.unsqueeze(0)
        
        # Sum of weighted embeddings
        sum_embeddings = torch.sum(weighted_embeddings, dim=1)
        
        # Sum of squares of weighted embeddings
        square_embeddings = weighted_embeddings ** 2
        sum_squares = torch.sum(square_embeddings, dim=1)
        
        # Compute interaction vector: 0.5 * (sum^2 - sum_of_squares)
        interaction_vector = 0.5 * (sum_embeddings ** 2 - sum_squares)
        
        return interaction_vector
    
    def get_post_mlp1_vector(self, sae_features):
        """
        Extract vector AFTER Interaction MLP Matrix 1 but BEFORE ReLU.
        This is what the NFM SAE should learn to decompose.
        """
        # Get the original interaction vector
        interaction_vector = self.get_interaction_vector(sae_features)  # [batch_size, embedding_dim]
        
        # Apply ONLY the first MLP layer (Matrix 1) with dropout
        # Extract components from interaction_mlp
        dropout_layer = self.interaction_mlp[0]  # Dropout
        mlp1_layer = self.interaction_mlp[1]     # Linear [embedding_dim x embedding_dim]
        
        # Apply dropout and first linear layer
        post_dropout = dropout_layer(interaction_vector)    # [batch_size, embedding_dim]
        post_mlp1 = mlp1_layer(post_dropout)               # [batch_size, embedding_dim]
        
        return post_mlp1  # This is what NFM SAE should learn on
    
    def forward(self, sae_features):
        """Original forward pass - kept for compatibility."""
        interaction_vector = self.get_interaction_vector(sae_features)
        
        # Process interaction vector through MLP
        interaction_out = self.interaction_mlp(interaction_vector)
        
        # Combine with linear component if present
        if self.linear is not None:
            linear_out = self.linear(sae_features)
            output = linear_out + interaction_out
        else:
            output = interaction_out
        
        return output

class PostMLP1Dataset(Dataset):
    def __init__(self, post_mlp1_vectors):
        self.post_mlp1_vectors = post_mlp1_vectors
    
    def __len__(self):
        return len(self.post_mlp1_vectors)
    
    def __getitem__(self, idx):
        return self.post_mlp1_vectors[idx]

def collect_activations(model, tokenizer, num_tokens):
    """Collect activations from the model's target layer - same as original."""
    activations = []
    
    # Same target indices as original
    target_indices = set()
    target_chunks = [
        (14000, 15000), (16000, 17000), (66000, 67000), (111000, 112000), (147000, 148000),
        (165000, 166000), (182000, 183000), (187000, 188000), (251000, 252000), (290000, 291000),
        (295000, 296000), (300000, 301000), (313000, 314000), (343000, 344000), (366000, 367000),
        (367000, 368000), (380000, 381000), (400000, 401000), (407000, 408000), (420000, 421000),
        (440000, 441000), (443000, 444000), (479000, 480000), (480000, 481000), (514000, 515000),
        (523000, 524000), (552000, 553000), (579000, 580000), (583000, 584000), (616000, 617000),
        (659000, 660000), (663000, 664000), (690000, 691000), (810000, 811000), (824000, 825000),
        (876000, 877000), (881000, 882000), (908000, 909000), (969000, 970000), (970000, 971000),
        (984000, 985000), (990000, 991000), (995000, 996000), (997000, 998000), (1000000, 1001000),
        (1024000, 1025000), (1099000, 1100000), (1127000, 1128000), (1163000, 1164000), (1182000, 1183000),
        (1209000, 1210000), (1253000, 1254000), (1266000, 1267000), (1270000, 1271000), (1276000, 1277000),
        (1290000, 1291000), (1307000, 1308000), (1326000, 1327000), (1345000, 1346000), (1359000, 1360000),
        (1364000, 1365000), (1367000, 1368000), (1385000, 1386000), (1391000, 1392000), (1468000, 1469000),
        (1508000, 1509000), (1523000, 1524000), (1539000, 1540000), (1574000, 1575000), (1583000, 1584000),
        (1590000, 1591000), (1593000, 1594000), (1599000, 1600000), (1627000, 1628000), (1679000, 1680000),
        (1690000, 1691000), (1691000, 1692000), (1782000, 1783000), (1788000, 1789000)
    ]
    
    for start, end in target_chunks:
        target_indices.update(range(start, end))
    
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    
    batch_count = 0
    for idx, sample in enumerate(tqdm(dataset, desc="Collecting activations")):
        if idx not in target_indices:
            continue
            
        inputs = tokenizer(sample["text"], padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[TARGET_LAYER]
            
            if check_tensor(hidden_states, "hidden_states_batch", print_stats=False):
                print(f"Found problematic hidden states in batch")
                
            batch_size, seq_len, hidden_size = hidden_states.shape
            hidden_states = hidden_states.reshape(-1, hidden_size)
            activations.append(hidden_states.cpu())
        
        batch_count += 1
        if batch_count >= num_tokens // BATCH_SIZE:
            break
    
    # Same processing as original
    activations = torch.cat(activations, dim=0)
    
    print("\n--- Pre-normalization activations ---")
    check_tensor(activations, "raw_activations", True)
    
    # Same clipping and normalization as original
    with torch.no_grad():
        norm_values = torch.norm(activations, dim=1).to(torch.float32)
        print("\n--- Activation norms percentiles ---")
        for p in [0, 0.1, 1, 5, 50, 95, 99, 99.9, 100]:
            percentile = torch.quantile(norm_values, torch.tensor(p/100, dtype=torch.float32)).item()
            print(f"Percentile {p}%: {percentile:.6f}")
    
    with torch.no_grad():
        mean = activations.mean()
        std = activations.std()
        n_std = 6.0
        lower_bound = mean - n_std * std
        upper_bound = mean + n_std * std
        
        below_count = (activations < lower_bound).sum().item()
        above_count = (activations > upper_bound).sum().item()
        total_elements = activations.numel()
        print(f"\nClipping bounds: {lower_bound.item()} to {upper_bound.item()}")
        print(f"Values below lower bound: {below_count} ({100.0 * below_count / total_elements:.6f}%)")
        print(f"Values above upper bound: {above_count} ({100.0 * above_count / total_elements:.6f}%)")
        
        activations = torch.clamp(activations, min=lower_bound, max=upper_bound)
        print("\n--- After clipping extreme values ---")
        check_tensor(activations, "clipped_activations")
    
    with torch.no_grad():
        mean_norm = torch.norm(activations, dim=1).mean()
        if mean_norm > 0:
            scale = np.sqrt(activations.shape[1]) / mean_norm
            activations = activations * scale
        else:
            print("WARNING: Mean norm is zero or negative, skipping normalization")
    
    print("\n--- Post-normalization activations ---")
    check_tensor(activations, "normalized_activations", True)
    
    return activations

def collect_sae_features_and_post_mlp1_vectors(original_activations, sae1_model, nfm_model):
    """
    Collect SAE features and post-MLP1 vectors using pre-trained models.
    """
    sae_features_list = []
    post_mlp1_vectors_list = []
    
    # Use smaller batch size to avoid memory issues
    batch_size = NFM_BATCH_SIZE  # Use parameter from top of script
    num_batches = (len(original_activations) + batch_size - 1) // batch_size
    
    sae1_model.eval()
    nfm_model.eval()
    
    total_processed = 0
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Collecting SAE features and post-MLP1 vectors"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(original_activations))
            batch = original_activations[start_idx:end_idx].to(DEVICE)
            
            # Get SAE features
            sae_features, _ = sae1_model(batch)
            
            # Get post-MLP1 vectors from NFM (AFTER Matrix 1, BEFORE ReLU)
            post_mlp1_vectors = nfm_model.get_post_mlp1_vector(sae_features)
            
            sae_features_list.append(sae_features.cpu())
            post_mlp1_vectors_list.append(post_mlp1_vectors.cpu())
            
            total_processed += batch.shape[0]
    
    final_sae_features = torch.cat(sae_features_list, dim=0)
    final_post_mlp1_vectors = torch.cat(post_mlp1_vectors_list, dim=0)
    
    return final_sae_features, final_post_mlp1_vectors

def train_interaction_sae_topk(model, train_loader, val_loader, num_steps, checkpoint_dir):
    """Train the TopK sparse autoencoder on post-MLP1 vectors."""
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=0, eps=1e-5)
    
    # Learning rate scheduler (same as part1c)
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=1.0,
        end_factor=0.1,
        total_iters=int(num_steps * 0.8)
    )
    
    # Initialize metrics tracking (TopK version - no L1 loss)
    metrics_history = {
        'steps': [],
        'total_loss': [],
        'reconstruction_loss': [],
        'best_loss': float('inf'),
        'l0_sparsity': [],  # Percentage of features that are zero
        'percent_active': [],  # Percentage of features that are active (1 - l0_sparsity)
        'dead_neurons': [],  # Percentage of neurons that never activate
        'variance_explained': [],  # Variance explained by reconstructions
        'val_total_loss': [],
        'val_reconstruction_loss': [],
        'val_best_loss': float('inf'),
        'val_l0_sparsity': [],
        'val_percent_active': [],
        'val_dead_neurons': [],
        'val_variance_explained': []  # Validation variance explained
    }
    
    # Track feature activations across batches for dead neuron analysis
    feature_max_activations = torch.zeros(model.encoder[0].out_features, device=DEVICE)
    
    prevent_sleep()
    
    try:
        train_iterator = iter(train_loader)
        
        for step in tqdm(range(num_steps), desc="Training TopK Interaction SAE"):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
            
            batch = batch.to(DEVICE)
            
            # Check batch for NaN/Inf
            if check_tensor(batch, f"post_mlp1_batch_step_{step}", print_stats=False):
                print(f"Problematic post-MLP1 batch at step {step}")
                check_tensor(batch, f"post_mlp1_batch_step_{step}", print_stats=True)
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
                
                debug_model_parameters(model, f"interaction_sae_topk_at_step_{step}")
                print("Skipping this batch")
                continue
            
            # Compute loss - TopK SAE only uses reconstruction loss
            try:
                # Only reconstruction loss - no L1 penalty needed with TopK
                reconstruction_loss = torch.mean((reconstruction - batch) ** 2)
                
                if torch.isnan(reconstruction_loss) or torch.isinf(reconstruction_loss):
                    print(f"Reconstruction loss is {reconstruction_loss} at step {step}")
                    reconstruction_loss = torch.tensor(1.0, device=DEVICE)  # Use a safe fallback
                
                # Total loss is just reconstruction loss
                loss = reconstruction_loss
                
                # Calculate sparsity metrics
                l0_sparsity = compute_l0_sparsity(features)
                percent_active = (1.0 - l0_sparsity) * 100  # Convert to percentage
                
                # Calculate variance explained
                var_explained, _ = compute_variance_explained(batch, reconstruction)
                
            except Exception as e:
                print(f"Exception during loss computation at step {step}: {str(e)}")
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
                
                # Gradient clipping for TopK stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
            except Exception as e:
                print(f"Exception during backward pass at step {step}: {str(e)}")
                continue
            
            # Track metrics
            metrics_history['steps'].append(step)
            metrics_history['total_loss'].append(loss.item())
            metrics_history['reconstruction_loss'].append(reconstruction_loss.item())
            metrics_history['l0_sparsity'].append(l0_sparsity)
            metrics_history['percent_active'].append(percent_active)
            metrics_history['variance_explained'].append(var_explained)
            
            # Compute dead neurons percentage
            if step % 1000 == 0:  # Don't compute every step for efficiency
                dead_neuron_pct = (feature_max_activations < DEAD_NEURON_THRESHOLD).float().mean().item() * 100
                metrics_history['dead_neurons'].append(dead_neuron_pct)
            
            # Use reconstruction loss for best model comparison (no L1 component)
            if reconstruction_loss.item() < metrics_history['best_loss']:
                metrics_history['best_loss'] = reconstruction_loss.item()
            
            # Validation - run periodically to monitor generalization
            if step % VALIDATION_INTERVAL == 0:
                model.eval()  # Set model to evaluation mode
                val_metrics = validate_interaction_sae_topk(model, val_loader)
                model.train()  # Set model back to training mode
                
                # Record validation metrics
                metrics_history['val_total_loss'].append(val_metrics['total_loss'])
                metrics_history['val_reconstruction_loss'].append(val_metrics['reconstruction_loss'])
                metrics_history['val_l0_sparsity'].append(val_metrics['l0_sparsity'])
                metrics_history['val_percent_active'].append(val_metrics['percent_active'])
                metrics_history['val_dead_neurons'].append(val_metrics['dead_neurons'])
                metrics_history['val_variance_explained'].append(val_metrics['variance_explained'])
                
                # Use reconstruction loss for validation best model comparison
                if val_metrics['reconstruction_loss'] < metrics_history['val_best_loss']:
                    metrics_history['val_best_loss'] = val_metrics['reconstruction_loss']
            
            # Print metrics every 1000 steps
            if step % 1000 == 0:
                print(f"\nStep {step}")
                print(f"Train - Reconstruction Loss: {reconstruction_loss.item():.4f}")
                print(f"Train - % Active Features: {percent_active:.2f}% (Target: {model.k/model.encoder[0].out_features*100:.2f}%)")
                print(f"Train - Variance Explained: {var_explained:.4f}")
                print(f"Train - Best Loss: {metrics_history['best_loss']:.4f}")
                
                if len(metrics_history['dead_neurons']) > 0:
                    print(f"Train - Dead Neurons: {metrics_history['dead_neurons'][-1]:.2f}%")
                
                # Print validation metrics if available
                if metrics_history['val_total_loss']:
                    recent_val_idx = len(metrics_history['val_total_loss']) - 1
                    print(f"Val - Reconstruction Loss: {metrics_history['val_reconstruction_loss'][recent_val_idx]:.4f}")
                    print(f"Val - % Active Features: {metrics_history['val_percent_active'][recent_val_idx]:.2f}%")
                    print(f"Val - Variance Explained: {metrics_history['val_variance_explained'][recent_val_idx]:.4f}")
                    print(f"Val - Best Loss: {metrics_history['val_best_loss']:.4f}")
                    print(f"Val - Dead Neurons: {metrics_history['val_dead_neurons'][recent_val_idx]:.2f}%")
                
                print(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
                
                # Check model parameters periodically
                if step % 10000 == 0:
                    debug_model_parameters(model, f"interaction_sae_topk_at_step_{step}")
            
            # Save checkpoint every CHECKPOINT_INTERVAL steps
            if (step + 1) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(
                    model, optimizer, scheduler, step + 1,
                    metrics_history['val_best_loss'], metrics_history,
                    checkpoint_dir
                )
                print(f"\nCheckpoint saved at step {step + 1}")
    
    finally:
        allow_sleep()

def validate_interaction_sae_topk(model, val_loader):
    """
    Compute validation metrics for the TopK interaction SAE
    Returns a dictionary with validation metrics
    """
    val_metrics = {
        'total_loss': 0.0,
        'reconstruction_loss': 0.0,
        'l0_sparsity': 0.0,
        'percent_active': 0.0,
        'dead_neurons': 0.0,
        'variance_explained': 0.0
    }
    
    # Number of validation batches to sample (use a maximum to keep validation fast)
    max_val_batches = min(10, len(val_loader))
    val_iterator = iter(val_loader)
    
    num_valid_batches = 0
    feature_max_activations = torch.zeros(model.encoder[0].out_features, device=DEVICE)
    
    with torch.no_grad():  # No gradients needed for validation
        for _ in range(max_val_batches):
            try:
                batch = next(val_iterator)
                batch = batch.to(DEVICE)
                
                # Skip problematic batches
                if check_tensor(batch, "val_post_mlp1_batch", print_stats=False):
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
                
                # Compute loss (only reconstruction loss for TopK)
                reconstruction_loss = torch.mean((reconstruction - batch) ** 2).item()
                
                # Calculate total loss (same as reconstruction loss for TopK)
                total_loss = reconstruction_loss
                
                # Calculate sparsity metrics
                l0_sparsity = compute_l0_sparsity(features)
                percent_active = (1.0 - l0_sparsity) * 100
                
                # Calculate variance explained
                var_explained, _ = compute_variance_explained(batch, reconstruction)
                
                # Accumulate metrics
                val_metrics['total_loss'] += total_loss
                val_metrics['reconstruction_loss'] += reconstruction_loss
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

def main():
    parser = argparse.ArgumentParser(description='Train TopK Sparse SAE on NFM Post-MLP1 Vectors')
    parser.add_argument('--sae1_model_path', type=str, default='checkpoints/best_model.pt',
                        help='Path to the trained SAE from part1')
    parser.add_argument('--nfm_model_path', type=str, default='nfm_linear_interaction_residual_model.pt',
                        help='Path to the trained NFM from part2')
    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR,
                        help='Directory to save interaction SAE checkpoints')
    parser.add_argument('--expansion_factor', type=int, default=INTERACTION_SAE_EXPANSION,
                        help=f'SAE expansion factor (default: {INTERACTION_SAE_EXPANSION}x)')
    parser.add_argument('--topk_ratio', type=float, default=0.02,
                        help='TopK ratio - fraction of features to keep active (default: 0.02 = 2%)')
    
    args = parser.parse_args()
    
    # Set numerical stability
    torch.set_default_dtype(torch.float32)
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH, use_fast=False, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(MODEL_PATH, torch_dtype='auto', device_map='auto', local_files_only=True)
    
    # Collect activations (same as original)
    print("Collecting activations...")
    activations = collect_activations(model, tokenizer, NUM_TOKENS)
    
    # Load the trained SAE from part1 and detect dimensions
    print(f"Loading trained SAE from {args.sae1_model_path}...")
    
    # First load to detect architecture
    if not os.path.exists(args.sae1_model_path):
        raise FileNotFoundError(f"SAE model not found at {args.sae1_model_path}. Please run part1 first.")
    
    sae_checkpoint = torch.load(args.sae1_model_path, map_location='cpu')
    
    # Detect SAE dimensions from the model state
    sae_encoder_weight_shape = sae_checkpoint['encoder.0.weight'].shape  # [hidden_dim, input_dim]
    sae_input_dim = sae_encoder_weight_shape[1]  # input dimension
    sae_hidden_dim = sae_encoder_weight_shape[0]  # hidden dimension (number of SAE features)
    
    print(f"Detected SAE dimensions: {sae_input_dim} -> {sae_hidden_dim}")
    
    # Verify this matches our activations
    if sae_input_dim != activations.shape[1]:
        raise ValueError(f"SAE input dim {sae_input_dim} doesn't match activation dim {activations.shape[1]}")
    
    # For loading SAE1, we need to check if it's TopK or regular SAE
    # If it has 'k' parameter in state dict, it's TopK, otherwise it's regular
    if any('k' in key for key in sae_checkpoint.keys()) or 'TopK' in str(type(sae_checkpoint)):
        # It's a TopK SAE - need to determine K
        # We'll assume it's a TopK SAE but create a regular one for loading
        from part1c_sae_topK_implementation import TopKSparseAutoencoder as SAE1TopK
        # Since we can't import, we'll create a regular SAE and load weights
        pass
    
    # Create regular SAE for SAE1 (we just need its forward pass)
    class RegularSAE(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            )
            self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        
        def forward(self, x):
            features = self.encoder(x)
            reconstruction = self.decoder(features)
            return features, reconstruction
    
    sae1 = RegularSAE(
        input_dim=sae_input_dim,
        hidden_dim=sae_hidden_dim
    ).to(DEVICE)
    
    # Convert to same dtype as model
    sae1 = sae1.to(dtype=next(model.parameters()).dtype)
    
    # Load the trained weights
    sae1.load_state_dict(sae_checkpoint)
    print("SAE1 loaded successfully!")
    
    # Freeze SAE1
    sae1.eval()
    for param in sae1.parameters():
        param.requires_grad = False
    
    # Load the trained NFM from part2 and detect K automatically
    print(f"Loading trained NFM from {args.nfm_model_path}...")
    
    # First load to detect architecture
    nfm_checkpoint = torch.load(args.nfm_model_path, map_location='cpu')
    
    # Detect NFM_K from the embedding layer shape
    embedding_weight_shape = nfm_checkpoint['feature_embeddings.weight'].shape
    NFM_K = embedding_weight_shape[1]  # [num_features, embedding_dim] -> get embedding_dim
    INTERACTION_SAE_FEATURES = NFM_K * args.expansion_factor
    
    # Calculate TopK K parameter
    TOPK_K = int(INTERACTION_SAE_FEATURES * args.topk_ratio)
    
    # Detect if NFM has linear component
    has_linear_component = 'linear.weight' in nfm_checkpoint
    
    print(f"Detected NFM embedding dimension K = {NFM_K}")
    print(f"NFM has linear component: {has_linear_component}")
    print(f"Will train TopK Interaction SAE: {NFM_K} -> {INTERACTION_SAE_FEATURES} (expansion: {args.expansion_factor}x)")
    print(f"TopK K = {TOPK_K} ({args.topk_ratio*100:.1f}% of {INTERACTION_SAE_FEATURES} features)")
    
    # Now create NFM model with correct dimensions and linear component
    nfm_model = NeuralFactorizationMachine(
        num_sae_features=sae_hidden_dim,  # Use detected SAE hidden dim
        embedding_dim=NFM_K,
        output_dim=activations.shape[1],
        use_linear_component=has_linear_component  # Pass this as parameter
    ).to(DEVICE)
    
    # Convert to same dtype as model
    nfm_model = nfm_model.to(dtype=next(model.parameters()).dtype)
    
    # Load the trained weights
    if not os.path.exists(args.nfm_model_path):
        raise FileNotFoundError(f"NFM model not found at {args.nfm_model_path}. Please run part2 first.")
    
    nfm_model.load_state_dict(nfm_checkpoint)
    nfm_model = nfm_model.to(DEVICE)
    print("NFM loaded successfully!")
    
    # Freeze NFM
    nfm_model.eval()
    for param in nfm_model.parameters():
        param.requires_grad = False
    
    # Extract SAE features and post-MLP1 vectors
    print("Extracting SAE features and post-MLP1 vectors...")
    sae_features, post_mlp1_vectors = collect_sae_features_and_post_mlp1_vectors(activations, sae1, nfm_model)
    
    print(f"Post-MLP1 vectors extracted. Shape: {post_mlp1_vectors.shape}")
    check_tensor(post_mlp1_vectors, "post_mlp1_vectors", True)
    
    # Analyze post-MLP1 vector statistics
    with torch.no_grad():
        print(f"\nPost-MLP1 Vector Analysis:")
        print(f"Shape: {post_mlp1_vectors.shape}")
        print(f"Dimension: {post_mlp1_vectors.shape[1]} (should be {NFM_K})")
        
        # Check for sparsity in post-MLP1 vectors (they should be dense but may have different patterns than interaction vectors)
        non_zero_per_sample = (torch.abs(post_mlp1_vectors) > 1e-6).sum(dim=1).float()
        print(f"Average non-zero elements per sample: {non_zero_per_sample.mean().item():.1f}")
        print(f"Max non-zero elements per sample: {non_zero_per_sample.max().item():.0f}")
        print(f"Min non-zero elements per sample: {non_zero_per_sample.min().item():.0f}")
        
        # Analyze magnitude distribution
        norms = torch.norm(post_mlp1_vectors, dim=1)
        print(f"Post-MLP1 vector norms - Mean: {norms.mean().item():.4f}, Std: {norms.std().item():.4f}")
    
    # Normalize post-MLP1 vectors using z-score normalization (from part1c)
    print("Normalizing post-MLP1 vectors using z-score normalization...")
    with torch.no_grad():
        # Use standard z-score normalization (like part1c)
        subset_size = min(10000, len(post_mlp1_vectors))
        subset_indices = torch.randperm(len(post_mlp1_vectors))[:subset_size]
        subset = post_mlp1_vectors[subset_indices]
        
        # Compute normalization statistics on a subset to save memory
        mean = subset.mean(dim=0, keepdim=True)
        std = subset.std(dim=0, keepdim=True)
        
        # Prevent division by zero
        std = torch.clamp(std, min=1e-8)
        
        print(f"\nNormalization stats computed from {subset_size} samples:")
        print(f"Mean magnitude: {mean.norm().item():.6f}")
        print(f"Std magnitude: {std.norm().item():.6f}")
        
        # Apply z-score normalization
        post_mlp1_vectors = (post_mlp1_vectors - mean) / std
    
    print("\n--- Post-normalization post-MLP1 vectors ---")
    check_tensor(post_mlp1_vectors, "normalized_post_mlp1_vectors", True)
    
    # Create train/validation split
    print("Creating train/validation split...")
    dataset_size = len(post_mlp1_vectors)
    val_size = int(dataset_size * VALIDATION_SPLIT)
    train_size = dataset_size - val_size
    
    indices = torch.randperm(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_post_mlp1_vectors = post_mlp1_vectors[train_indices]
    val_post_mlp1_vectors = post_mlp1_vectors[val_indices]
    
    print(f"Train set size: {train_size} samples")
    print(f"Validation set size: {val_size} samples")
    
    # Create datasets and dataloaders
    train_dataset = PostMLP1Dataset(train_post_mlp1_vectors)
    val_dataset = PostMLP1Dataset(val_post_mlp1_vectors)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize TopK Interaction SAE
    print(f"Initializing TopK Interaction SAE...")
    print(f"  Input dimension: {NFM_K}")
    print(f"  Hidden dimension: {INTERACTION_SAE_FEATURES}")
    print(f"  TopK K: {TOPK_K}")
    print(f"  Expansion factor: {args.expansion_factor}x")
    print(f"  Target sparsity: {TOPK_K}/{INTERACTION_SAE_FEATURES} = {TOPK_K/INTERACTION_SAE_FEATURES*100:.2f}% active features")
    
    interaction_sae = TopKSparseAutoencoder(
        input_dim=NFM_K,
        hidden_dim=INTERACTION_SAE_FEATURES,
        k=TOPK_K
    ).to(DEVICE)
    
    # Convert to same dtype as model
    interaction_sae = interaction_sae.to(dtype=next(model.parameters()).dtype)
    
    # Calculate and print parameter count
    total_params = sum(p.numel() for p in interaction_sae.parameters())
    encoder_params = interaction_sae.encoder[0].weight.numel() + interaction_sae.encoder[0].bias.numel()
    decoder_params = interaction_sae.decoder.weight.numel()
    
    print(f"TopK Interaction SAE Model Parameters:")
    print(f"  Encoder: {encoder_params:,}")
    print(f"  Decoder: {decoder_params:,}")
    print(f"  Total Parameters: {total_params:,}")
    
    # Debug initial parameters
    debug_model_parameters(interaction_sae, "initial_topk_interaction_sae")
    
    print("Training TopK Interaction SAE...")
    train_interaction_sae_topk(interaction_sae, train_loader, val_loader, TRAIN_STEPS, args.checkpoint_dir)
    
    # Post-process: normalize decoder columns to unit norm (from part1c)
    with torch.no_grad():
        decoder_norms = torch.norm(interaction_sae.decoder.weight, p=2, dim=0)
        # Prevent division by zero
        decoder_norms = torch.clamp(decoder_norms, min=1e-8)
        interaction_sae.decoder.weight.data = interaction_sae.decoder.weight.data / decoder_norms
    
    # Save the final trained TopK Interaction SAE
    final_model_name = f"interaction_sae_topk_post_mlp1_k{NFM_K}_features{INTERACTION_SAE_FEATURES}_topk{TOPK_K}.pt"
    torch.save(interaction_sae.state_dict(), final_model_name)
    print(f"Training complete! TopK Interaction SAE saved as {final_model_name}")
    
    # Final analysis - use the trained model directly
    print("\n" + "="*60)
    print("FINAL ANALYSIS: TopK Interaction SAE Performance on Post-MLP1 Vectors")
    print("="*60)
    
    # Keep model in eval mode
    interaction_sae.eval()
    with torch.no_grad():
        # Take a sample for final analysis
        sample_size = min(1000, len(val_post_mlp1_vectors))
        sample_indices = torch.randperm(len(val_post_mlp1_vectors))[:sample_size]
        sample_vectors = val_post_mlp1_vectors[sample_indices].to(DEVICE)
        
        # Check for any problematic values in input
        if torch.isnan(sample_vectors).any() or torch.isinf(sample_vectors).any():
            print("ERROR - Input contains NaN or Inf values!")
            check_tensor(sample_vectors, "sample_vectors_with_issues")
        
        # Get SAE reconstruction
        sae_features, sae_reconstruction = interaction_sae(sample_vectors)
        
        # Check outputs immediately
        if torch.isnan(sae_features).any() or torch.isinf(sae_features).any():
            print("ERROR - SAE features contain NaN or Inf!")
            check_tensor(sae_features, "problematic_sae_features")
        
        if torch.isnan(sae_reconstruction).any() or torch.isinf(sae_reconstruction).any():
            print("ERROR - SAE reconstruction contains NaN or Inf!")
            check_tensor(sae_reconstruction, "problematic_sae_reconstruction")
        
        # Compute reconstruction error
        reconstruction_error = torch.mean((sample_vectors - sae_reconstruction) ** 2).item()
        original_norm = torch.mean(sample_vectors ** 2).item()
        
        # Compute variance explained
        var_explained, _ = compute_variance_explained(sample_vectors, sae_reconstruction)
        
        # Compute sparsity metrics
        l0_sparsity = compute_l0_sparsity(sae_features)
        active_features_per_sample = (torch.abs(sae_features) > 1e-6).sum(dim=1).float()
        
        print(f"Original post-MLP1 vector MSE: {original_norm:.6f}")
        print(f"Reconstruction error: {reconstruction_error:.6f}")
        print(f"Relative reconstruction error: {(reconstruction_error / original_norm) * 100:.2f}%")
        print(f"Variance explained: {var_explained:.4f}")
        print(f"L0 sparsity (fraction zeros): {l0_sparsity:.4f}")
        print(f"Average active features per sample: {active_features_per_sample.mean().item():.1f} / {INTERACTION_SAE_FEATURES}")
        print(f"Target active features: {TOPK_K} / {INTERACTION_SAE_FEATURES}")
        print(f"Actual sparsity ratio: {(active_features_per_sample.mean().item() / INTERACTION_SAE_FEATURES) * 100:.2f}%")
        print(f"Target sparsity ratio: {(TOPK_K / INTERACTION_SAE_FEATURES) * 100:.2f}%")
        
        # Feature activation analysis
        feature_activation_counts = (torch.abs(sae_features) > 1e-6).sum(dim=0)
        print(f"\nFeature activation analysis:")
        print(f"Most active feature used: {feature_activation_counts.max().item()} times")
        print(f"Least active feature used: {feature_activation_counts.min().item()} times")
        print(f"Dead features (never activated): {(feature_activation_counts == 0).sum().item()} / {INTERACTION_SAE_FEATURES}")
        print(f"Dead feature percentage: {(feature_activation_counts == 0).float().mean().item() * 100:.2f}%")

if __name__ == "__main__":
    main()