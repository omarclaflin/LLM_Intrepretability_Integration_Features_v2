"""
Simple SAE vs No-SAE Comparison Script

Compares layer 16 activations in 4 conditions:
1. Static + No SAE
2. Static + SAE  
3. Generation + No SAE
4. Generation + SAE

Just reports means for each condition.
"""

import torch
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

class TopKSparseAutoencoder(torch.nn.Module):
    """TopK Sparse Autoencoder module that can handle k=None."""
    def __init__(self, input_dim, hidden_dim, k):
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

def load_sae(sae_path, device="cuda"):
    """Load SAE from checkpoint."""
    sae_state_dict = torch.load(sae_path, map_location=device)
    
    if 'model_state' in sae_state_dict:
        actual_state_dict = sae_state_dict['model_state']
    else:
        actual_state_dict = sae_state_dict
    
    if 'decoder.weight' in actual_state_dict:
        input_dim = actual_state_dict['decoder.weight'].shape[0]
        hidden_dim = actual_state_dict['decoder.weight'].shape[1]
    elif 'encoder.0.weight' in actual_state_dict:
        encoder_weight = actual_state_dict['encoder.0.weight']
        hidden_dim, input_dim = encoder_weight.shape
    else:
        raise ValueError(f"Cannot determine SAE dimensions")
    
    # Create TopK SAE with k=None (no built-in top-k filtering)
    sae = TopKSparseAutoencoder(input_dim, hidden_dim, k=None)
    sae.load_state_dict(actual_state_dict)
    sae.to(device)
    
    return sae

class SimpleComparator:
    def __init__(self, model, tokenizer, sae, device="cuda", target_layer=16):
        self.model = model
        self.tokenizer = tokenizer
        self.sae = sae
        self.device = device
        self.target_layer = target_layer
        
        self.model.eval()
        self.sae.eval()

    def collect_static_activations(self, prompts, use_sae=False, top_k=None):
        """Collect activations during static forward passes."""
        activations = []
        
        def hook_fn(module, input, output):
            acts = output[0].detach()  # [batch, seq, hidden]
            flat_acts = acts.view(-1, acts.shape[-1])
            
            if use_sae:
                # Apply SAE (which has k=None, so no built-in top-k)
                features, reconstruction = self.sae(flat_acts.to(self.sae.encoder[0].weight.dtype))
                
                if top_k is not None:
                    # Apply external top-k filtering to the features
                    features_topk = torch.zeros_like(features)
                    top_indices = torch.topk(features, top_k, dim=1).indices
                    features_topk.scatter_(1, top_indices, features.gather(1, top_indices))
                    reconstruction = self.sae.decoder(features_topk)
                
                activations.append(reconstruction.cpu())
            else:
                # No SAE
                activations.append(flat_acts.cpu())
        
        target_module = self.model.model.layers[self.target_layer]
        hook_handle = target_module.register_forward_hook(hook_fn)
        
        try:
            for prompt in prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
                with torch.no_grad():
                    self.model(**inputs)
        finally:
            hook_handle.remove()
            
        if activations:
            return torch.cat(activations, dim=0)
        return torch.empty(0, 3200)

    def collect_generation_activations(self, prompts, max_tokens=20, use_sae=False, top_k=None):
        """Collect activations during generation."""
        activations = []
        
        def hook_fn(module, input, output):
            acts = output[0].detach()  # [batch, seq, hidden]
            last_token_act = acts[:, -1:, :].view(-1, acts.shape[-1])  # [1, hidden]
            
            if use_sae:
                # Apply SAE (which has k=None, so no built-in top-k)
                features, reconstruction = self.sae(last_token_act.to(self.sae.encoder[0].weight.dtype))
                
                if top_k is not None:
                    # Apply external top-k filtering to the features
                    features_topk = torch.zeros_like(features)
                    top_indices = torch.topk(features, top_k, dim=1).indices
                    features_topk.scatter_(1, top_indices, features.gather(1, top_indices))
                    reconstruction = self.sae.decoder(features_topk)
                
                activations.append(reconstruction.cpu())
            else:
                # No SAE
                activations.append(last_token_act.cpu())
        
        target_module = self.model.model.layers[self.target_layer]
        hook_handle = target_module.register_forward_hook(hook_fn)
        
        try:
            for prompt in prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                input_ids = inputs["input_ids"]
                
                for step in range(max_tokens):
                    with torch.no_grad():
                        outputs = self.model(input_ids, use_cache=False)
                        next_token = torch.argmax(outputs.logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
                        input_ids = torch.cat([input_ids, next_token], dim=-1)
                        
                        if next_token.item() == self.tokenizer.eos_token_id:
                            break
                            
        finally:
            hook_handle.remove()
            
        if activations:
            return torch.cat(activations, dim=0)
        return torch.empty(0, 3200)

def main():
    parser = argparse.ArgumentParser(description="Simple SAE vs No-SAE Comparison")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--sae_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top_k", type=int, default=1024, help="Number of top features to keep for top-k comparisons")
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")
    sae = load_sae(args.sae_path, args.device)
    
    comparator = SimpleComparator(model, tokenizer, sae, args.device)
    
    # Test prompts
    test_prompts = [
        "The quick brown fox",
        "I think that", 
        "Hello world"
    ]
    
    print("Running comparisons...")
    print(f"Using top_k = {args.top_k} for top-k comparisons")
    
    # 1. Static + No SAE
    print("1. Static + No SAE")
    static_no_sae = comparator.collect_static_activations(test_prompts, use_sae=False)
    static_no_sae_mean = static_no_sae.mean().item() if len(static_no_sae) > 0 else 0.0
    
    # 2. Static + SAE (full)
    print("2. Static + SAE (full)")
    static_sae = comparator.collect_static_activations(test_prompts, use_sae=True)
    static_sae_mean = static_sae.mean().item() if len(static_sae) > 0 else 0.0
    
    # 3. Static + SAE (top k)
    print(f"3. Static + SAE (top {args.top_k})")
    static_sae_topk = comparator.collect_static_activations(test_prompts, use_sae=True, top_k=args.top_k)
    static_sae_topk_mean = static_sae_topk.mean().item() if len(static_sae_topk) > 0 else 0.0
    
    # 4. Generation + No SAE  
    print("4. Generation + No SAE")
    gen_no_sae = comparator.collect_generation_activations(test_prompts[:2], use_sae=False)
    gen_no_sae_mean = gen_no_sae.mean().item() if len(gen_no_sae) > 0 else 0.0
    
    # 5. Generation + SAE (full)
    print("5. Generation + SAE (full)")
    gen_sae = comparator.collect_generation_activations(test_prompts[:2], use_sae=True)
    gen_sae_mean = gen_sae.mean().item() if len(gen_sae) > 0 else 0.0
    
    # 6. Generation + SAE (top k)
    print(f"6. Generation + SAE (top {args.top_k})")
    gen_sae_topk = comparator.collect_generation_activations(test_prompts[:2], use_sae=True, top_k=args.top_k)
    gen_sae_topk_mean = gen_sae_topk.mean().item() if len(gen_sae_topk) > 0 else 0.0
    
    # Results
    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    print(f"Static + No SAE:        {static_no_sae_mean:.6f}")
    print(f"Static + SAE (full):    {static_sae_mean:.6f}")
    print(f"Static + SAE (top{args.top_k}): {static_sae_topk_mean:.6f}")
    print(f"Generation + No SAE:    {gen_no_sae_mean:.6f}")
    print(f"Generation + SAE (full): {gen_sae_mean:.6f}")
    print(f"Generation + SAE (top{args.top_k}): {gen_sae_topk_mean:.6f}")
    
    print(f"\nCOMPARISONS:")
    print(f"Static SAE (full) effect:     {static_sae_mean - static_no_sae_mean:.6f}")
    print(f"Static SAE (top{args.top_k}) effect:  {static_sae_topk_mean - static_no_sae_mean:.6f}")
    print(f"Generation SAE (full) effect: {gen_sae_mean - gen_no_sae_mean:.6f}")
    print(f"Generation SAE (top{args.top_k}) effect: {gen_sae_topk_mean - gen_no_sae_mean:.6f}")
    
    # Save results
    results_file = output_dir / "simple_comparison.txt"
    with open(results_file, 'w') as f:
        f.write("Simple SAE vs No-SAE Comparison\n")
        f.write("="*40 + "\n\n")
        f.write(f"Top-k value used: {args.top_k}\n\n")
        f.write(f"Static + No SAE:        {static_no_sae_mean:.6f}\n")
        f.write(f"Static + SAE (full):    {static_sae_mean:.6f}\n")
        f.write(f"Static + SAE (top{args.top_k}): {static_sae_topk_mean:.6f}\n")
        f.write(f"Generation + No SAE:    {gen_no_sae_mean:.6f}\n")
        f.write(f"Generation + SAE (full): {gen_sae_mean:.6f}\n")
        f.write(f"Generation + SAE (top{args.top_k}): {gen_sae_topk_mean:.6f}\n\n")
        f.write(f"Static SAE (full) effect:     {static_sae_mean - static_no_sae_mean:.6f}\n")
        f.write(f"Static SAE (top{args.top_k}) effect:  {static_sae_topk_mean - static_no_sae_mean:.6f}\n")
        f.write(f"Generation SAE (full) effect: {gen_sae_mean - gen_no_sae_mean:.6f}\n")
        f.write(f"Generation SAE (top{args.top_k}) effect: {gen_sae_topk_mean - gen_no_sae_mean:.6f}\n")
    
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()