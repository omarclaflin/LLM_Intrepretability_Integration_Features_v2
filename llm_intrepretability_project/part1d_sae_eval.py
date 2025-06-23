# k =1024 (used during inference)
# === EVALUATION RESULTS ===
# Reconstruction Loss: 0.114220
# Variance Explained: 0.8835
# % Active Features: 2.05% (Target: 2.05%)
# Dead Neurons: 0.00%
# Evaluated on 49 batches

#trying with no 'top feature' k=50000
# === EVALUATION RESULTS ===
# Reconstruction Loss: 1.670301
# Variance Explained: -0.6299
# % Active Features: 49.20% (Target: 100.00%)
# Dead Neurons: 0.00%
# Evaluated on 49 batches

#trying now with k=None (modded code to accept)
# === EVALUATION RESULTS ===
# Reconstruction Loss: 1.673828
# Variance Explained: -0.6296
# % Active Features: 49.19% (Threshold: 1e-06)
# Dead Neurons: 0.00%
# Evaluated on 49 batches


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
import argparse
import glob

# Constants
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "open_llama_3b"))
BATCH_SIZE = 1024
NUM_EVAL_TOKENS = 50_000  # Small evaluation set
NUM_FEATURES = 50_000
K = 1024
TARGET_LAYER = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHUNK_SIZE = 128
DEAD_NEURON_THRESHOLD = 1e-6

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

class ActivationDataset(Dataset):
    def __init__(self, activations):
        self.activations = activations
    
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        return self.activations[idx]

def collect_activations(model, tokenizer, num_tokens):
    """Collect activations for evaluation."""
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
    
    # Apply same normalization as training
    subset_size = min(10000, len(activations))
    subset_indices = torch.randperm(len(activations))[:subset_size]
    subset = activations[subset_indices]
    
    with torch.no_grad():
        mean = subset.mean(dim=0, keepdim=True)
        std = subset.std(dim=0, keepdim=True)
        std = torch.clamp(std, min=1e-8)
        activations = (activations - mean) / std
    
    return activations

def compute_variance_explained(original_activations, reconstructed_activations):
    with torch.no_grad():
        original_centered = original_activations - original_activations.mean(dim=0, keepdim=True)
        reconstructed_centered = reconstructed_activations - reconstructed_activations.mean(dim=0, keepdim=True)
        
        total_variance = torch.var(original_centered, dim=0, unbiased=False)
        residual = original_centered - reconstructed_centered
        residual_variance = torch.var(residual, dim=0, unbiased=False)
        
        var_explained_per_dim = 1 - (residual_variance / (total_variance + 1e-8))
        total_var_explained = torch.sum(var_explained_per_dim * total_variance) / torch.sum(total_variance)
        
        return total_var_explained.item(), var_explained_per_dim

def compute_l0_sparsity(features, threshold=1e-6):
    with torch.no_grad():
        zeros = (torch.abs(features) < threshold).float().mean().item()
        return zeros

def find_checkpoint(checkpoint_dir, step=None, best=False):
    """Find specific checkpoint, latest, or best model."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    if best:
        # Look for best model
        best_model_path = checkpoint_dir / "best_model.pt"
        return str(best_model_path) if best_model_path.exists() else None
    elif step is not None:
        # Look for specific step
        checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
        return str(checkpoint_path) if checkpoint_path.exists() else None
    else:
        # Find latest
        checkpoint_pattern = str(checkpoint_dir / "checkpoint_step_*.pt")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if not checkpoint_files:
            return None
        
        latest_step = -1
        latest_checkpoint = None
        
        for checkpoint_file in checkpoint_files:
            filename = os.path.basename(checkpoint_file)
            try:
                step_str = filename.replace("checkpoint_step_", "").replace(".pt", "")
                step_num = int(step_str)
                if step_num > latest_step:
                    latest_step = step_num
                    latest_checkpoint = checkpoint_file
            except ValueError:
                continue
        
        return latest_checkpoint

def evaluate_checkpoint(checkpoint_path, eval_loader, is_best_model=False):
    """Evaluate a checkpoint or best model."""
    print(f"Loading {'best model' if is_best_model else 'checkpoint'}: {checkpoint_path}")
    
    if is_best_model:
        # Load just the model state dict
        model_state = torch.load(checkpoint_path)
        print("Loaded best model weights")
    else:
        # Load full checkpoint
        checkpoint = torch.load(checkpoint_path)
        model_state = checkpoint['model_state']
        
        # Print checkpoint info
        print(f"Checkpoint step: {checkpoint['step']}")
        print(f"Checkpoint timestamp: {checkpoint.get('timestamp', 'Unknown')}")
        print(f"Checkpoint best loss: {checkpoint['best_loss']:.6f}")
    
    # Load model
    input_dim = next(iter(eval_loader)).shape[1]
    sae = TopKSparseAutoencoder(input_dim, NUM_FEATURES, K).to(DEVICE)
    sae.load_state_dict(model_state)
    
    # Match dtype to the LLaMA model
    sae = sae.to(dtype=torch.float16)
    sae.eval()
    
    # Evaluate
    total_loss = 0.0
    total_l0_sparsity = 0.0
    total_variance_explained = 0.0
    num_batches = 0
    feature_max_activations = torch.zeros(NUM_FEATURES, device=DEVICE)
    
    print("Evaluating...")
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluation"):
            batch = batch.to(DEVICE)
            
            features, reconstruction = sae(batch)
            
            # Update feature tracking
            batch_max_activations = torch.max(torch.abs(features), dim=0)[0]
            feature_max_activations = torch.max(feature_max_activations, batch_max_activations)
            
            # Compute metrics
            reconstruction_loss = torch.mean((reconstruction - batch) ** 2).item()
            l0_sparsity = compute_l0_sparsity(features)
            var_explained, _ = compute_variance_explained(batch, reconstruction)
            
            total_loss += reconstruction_loss
            total_l0_sparsity += l0_sparsity
            total_variance_explained += var_explained
            num_batches += 1
    
    # Compute averages
    avg_loss = total_loss / num_batches
    avg_l0_sparsity = total_l0_sparsity / num_batches
    avg_variance_explained = total_variance_explained / num_batches
    percent_active = (1.0 - avg_l0_sparsity) * 100
    dead_neurons = (feature_max_activations < DEAD_NEURON_THRESHOLD).float().mean().item() * 100
    
    # Print results
    print(f"\n=== EVALUATION RESULTS ===")
    print(f"Reconstruction Loss: {avg_loss:.6f}")
    print(f"Variance Explained: {avg_variance_explained:.4f}")
    if K is not None:
        print(f"% Active Features: {percent_active:.2f}% (Target: {K/NUM_FEATURES*100:.2f}%)")
    else:
        print(f"% Active Features: {percent_active:.2f}% (Threshold: {DEAD_NEURON_THRESHOLD})")
    print(f"Dead Neurons: {dead_neurons:.2f}%")
    print(f"Evaluated on {num_batches} batches")

def main():
    parser = argparse.ArgumentParser(description='Evaluate TopK SAE checkpoint')
    parser.add_argument('--checkpoint-dir', default='checkpoints_topk', 
                       help='Directory containing checkpoints')
    parser.add_argument('--step', type=int, default=None,
                       help='Specific checkpoint step to evaluate (default: latest)')
    parser.add_argument('--best', action='store_true',
                       help='Evaluate best_model.pt instead of checkpoint')
    args = parser.parse_args()
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH, use_fast=False, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(MODEL_PATH, torch_dtype='auto', device_map='auto', local_files_only=True)
    
    # Collect evaluation data
    activations = collect_activations(model, tokenizer, NUM_EVAL_TOKENS)
    eval_dataset = ActivationDataset(activations)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Find checkpoint or best model
    if args.best:
        checkpoint_path = find_checkpoint(args.checkpoint_dir, best=True)
        if checkpoint_path is None:
            print("best_model.pt not found!")
            return
        evaluate_checkpoint(checkpoint_path, eval_loader, is_best_model=True)
    else:
        checkpoint_path = find_checkpoint(args.checkpoint_dir, args.step)
        if checkpoint_path is None:
            if args.step is not None:
                print(f"Checkpoint for step {args.step} not found!")
            else:
                print("No checkpoints found!")
            return
        evaluate_checkpoint(checkpoint_path, eval_loader, is_best_model=False)

if __name__ == "__main__":
    main()