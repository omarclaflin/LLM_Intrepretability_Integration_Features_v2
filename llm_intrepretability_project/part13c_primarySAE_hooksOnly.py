"""
Primary SAE Hook Test Script

This script tests ONLY the Primary SAE reconstruction hook to isolate where the pipeline breaks.
No NFM, no Secondary SAE, no clamping - just Primary SAE encode/decode at Layer 16.

Pipeline: Layer 16 → Primary SAE → Primary SAE Reconstruction → Continue

This will help determine if the issue is in:
1. The Primary SAE itself
2. The hook mechanism  
3. Or the more complex NFM/Secondary SAE parts
"""

import torch
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

# Same prompt sets as part13 and baseline
PROMPT_SETS = {
    "neutral": [
        "I need to inform everyone about",
        "When someone asks me about",
        "The reason this happened is",
        "My thoughts on this matter are",
        "The best way to handle this is"
    ],
    "casual_emotional": [
        "OMG I can't believe how excited I am about",
        "Honestly, I'm so frustrated with",
        "I'm literally freaking out about",
        "This is seriously the most amazing thing about",
        "I'm totally devastated by"
    ],
    "formal_neutral": [
        "According to the documentation, the procedure requires",
        "The analysis indicates that the optimal approach involves",
        "Pursuant to the established guidelines, it is recommended that",
        "The evaluation demonstrates that the most effective method is",
        "Based on the available evidence, the conclusion suggests that"
    ]
}

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

class PrimarySAEHookTester:
    """Test Primary SAE hook in isolation."""
    
    def __init__(self, primary_sae, tokenizer, base_model, device="cuda", target_layer=16):
        self.primary_sae = primary_sae
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.device = device
        self.target_layer = target_layer
        
        # Set models to eval mode
        self.primary_sae.eval()
        self.base_model.eval()

    def forward_with_primary_sae_hook(self, input_ids, attention_mask):
        """
        Forward pass with PRIMARY SAE ONLY hook at Layer 16.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
        
        Returns:
            logits: Final logits after Primary SAE reconstruction
            primary_features: Primary SAE features
            primary_reconstruction: Primary SAE reconstruction
        """
        # Store hook data
        hook_data = {}
        
        def primary_sae_hook(module, input, output):
            """Hook to apply Primary SAE at layer 16."""
            layer_16_activations = output[0]  # [batch, seq, hidden]
            original_shape = layer_16_activations.shape
            layer_16_flat = layer_16_activations.view(-1, original_shape[-1])  # [batch*seq, hidden]
            
            # Primary SAE processing ONLY
            primary_features, primary_reconstruction = self.primary_sae(
                layer_16_flat.to(self.primary_sae.encoder[0].weight.dtype)
            )
            
            # Store for later return
            hook_data['primary_features'] = primary_features
            hook_data['primary_reconstruction'] = primary_reconstruction
            
            # Reshape back to original dimensions
            primary_reconstruction_reshaped = primary_reconstruction.view(original_shape)
            
            # Return ONLY Primary SAE reconstruction (no NFM, no other pathways)
            return (primary_reconstruction_reshaped,) + output[1:]
        
        # Register hook at target layer
        target_layer_module = self.base_model.model.layers[self.target_layer]
        hook_handle = target_layer_module.register_forward_hook(primary_sae_hook)
        
        try:
            with torch.no_grad():
                # Forward pass with Primary SAE hook
                outputs = self.base_model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    use_cache=False
                )
                logits = outputs.logits
                
        finally:
            # Remove hook
            hook_handle.remove()
        
        return (logits, 
                hook_data.get('primary_features', None),
                hook_data.get('primary_reconstruction', None))

    def generate_with_primary_sae_hook(self, prompt, max_length=50):
        """Generate text with Primary SAE hook only."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        
        generated_tokens = []
        
        for step in range(max_length):
            # Create attention mask
            attention_mask = torch.ones_like(input_ids)
            
            # Get logits with Primary SAE hook
            logits, primary_features, primary_reconstruction = self.forward_with_primary_sae_hook(
                input_ids, attention_mask
            )
            
            # Sample next token (using greedy decoding for consistency)
            next_token_id = torch.argmax(logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
            
            # Check for EOS
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
                
            generated_tokens.append(next_token_id.item())
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            
            # Debug: Print first few tokens to catch issues early
            if step < 5:
                token_text = self.tokenizer.decode([next_token_id.item()], skip_special_tokens=True)
                print(f"Step {step}: Generated token '{token_text}' (ID: {next_token_id.item()})")
        
        # Decode the generated part only
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_text

    def test_single_forward_pass(self, prompt):
        """Test a single forward pass to check logits."""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        
        # Get logits with Primary SAE hook
        logits, primary_features, primary_reconstruction = self.forward_with_primary_sae_hook(
            inputs["input_ids"], inputs["attention_mask"]
        )
        
        # Check reconstruction quality
        if primary_features is not None and primary_reconstruction is not None:
            print(f"Primary features shape: {primary_features.shape}")
            print(f"Primary features sparsity: {(primary_features > 0).float().mean():.4f}")
            print(f"Primary features range: [{primary_features.min():.4f}, {primary_features.max():.4f}]")
            print(f"Primary reconstruction range: [{primary_reconstruction.min():.4f}, {primary_reconstruction.max():.4f}]")
        
        # Extract logits for the last token
        last_token_logits = logits[0, -1, :]  # [vocab_size]
        print(f"Logits range: [{last_token_logits.min():.4f}, {last_token_logits.max():.4f}]")
        
        # Get top 5 predicted tokens
        top_logits, top_indices = torch.topk(last_token_logits, 5)
        print("Top 5 predicted tokens:")
        for i, (logit, token_id) in enumerate(zip(top_logits, top_indices)):
            token_text = self.tokenizer.decode([token_id.item()], skip_special_tokens=True)
            print(f"  {i+1}. '{token_text}' (ID: {token_id.item()}, logit: {logit:.4f})")
        
        return logits

def load_models(args):
    """Load Primary SAE and base model."""
    print("Loading models...")
    
    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")
    
    # Load Primary SAE
    primary_sae_state_dict = torch.load(args.primary_sae_path, map_location=args.device)
    
    # Debug: Print available keys
    print(f"Primary SAE checkpoint keys: {list(primary_sae_state_dict.keys())}")
    
    # Handle different checkpoint formats
    if 'model_state' in primary_sae_state_dict:
        actual_state_dict = primary_sae_state_dict['model_state']
        print(f"Primary SAE model state keys: {list(actual_state_dict.keys())}")
    else:
        actual_state_dict = primary_sae_state_dict
    
    # Try different possible key patterns
    if 'decoder.weight' in actual_state_dict:
        input_dim = actual_state_dict['decoder.weight'].shape[0]
        hidden_dim = actual_state_dict['decoder.weight'].shape[1]
        print(f"Found decoder.weight: {actual_state_dict['decoder.weight'].shape}")
    elif 'encoder.0.weight' in actual_state_dict:
        encoder_weight = actual_state_dict['encoder.0.weight']
        hidden_dim, input_dim = encoder_weight.shape
        print(f"Found encoder.0.weight: {encoder_weight.shape}")
    elif 'encoder.weight' in actual_state_dict:
        encoder_weight = actual_state_dict['encoder.weight']
        hidden_dim, input_dim = encoder_weight.shape
        print(f"Found encoder.weight: {encoder_weight.shape}")
    else:
        # Try to infer from any available weight keys
        available_keys = [k for k in actual_state_dict.keys() if 'weight' in k]
        print(f"Available weight keys: {available_keys}")
        
        # Look for encoder-like and decoder-like keys
        encoder_keys = [k for k in available_keys if 'encoder' in k.lower()]
        decoder_keys = [k for k in available_keys if 'decoder' in k.lower()]
        
        print(f"Encoder-like keys: {encoder_keys}")
        print(f"Decoder-like keys: {decoder_keys}")
        
        if encoder_keys:
            encoder_weight = actual_state_dict[encoder_keys[0]]
            if len(encoder_weight.shape) == 2:
                hidden_dim, input_dim = encoder_weight.shape
                print(f"Using {encoder_keys[0]}: {encoder_weight.shape}")
            else:
                raise ValueError(f"Unexpected encoder weight shape: {encoder_weight.shape}")
        elif decoder_keys:
            decoder_weight = actual_state_dict[decoder_keys[0]]
            if len(decoder_weight.shape) == 2:
                input_dim, hidden_dim = decoder_weight.shape
                print(f"Using {decoder_keys[0]}: {decoder_weight.shape}")
            else:
                raise ValueError(f"Unexpected decoder weight shape: {decoder_weight.shape}")
        else:
            raise ValueError(f"Cannot determine SAE dimensions from available keys: {available_keys}")
    
    # Create TopK SAE with specified k parameter
    primary_sae = TopKSparseAutoencoder(input_dim, hidden_dim, k=args.k)
    primary_sae.load_state_dict(actual_state_dict)
    primary_sae.to(args.device)
    
    print(f"Models loaded successfully!")
    print(f"Primary SAE: {input_dim} → {hidden_dim} (k={args.k})")
    
    return tokenizer, base_model, primary_sae

def main():
    parser = argparse.ArgumentParser(description="Primary SAE Hook Test")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base LLM model")
    parser.add_argument("--primary_sae_path", type=str, required=True, help="Path to Primary SAE model")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--generation_length", type=int, default=50, help="Number of tokens to generate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--k", type=int, default=1024, help="TopK parameter for Primary SAE (default: 1024)")
    
    args = parser.parse_args()
    
    print(f"Using TopK parameter k={args.k}")
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    tokenizer, base_model, primary_sae = load_models(args)
    
    # Initialize tester
    tester = PrimarySAEHookTester(primary_sae, tokenizer, base_model, args.device)
    
    # Test 1: Single forward pass analysis
    print("\n" + "="*60)
    print("SINGLE FORWARD PASS TEST")
    print("="*60)
    
    test_prompt = "I think that"
    print(f"Testing prompt: '{test_prompt}'")
    tester.test_single_forward_pass(test_prompt)
    
    # Test 2: Quick generation test
    print("\n" + "="*60)
    print("QUICK GENERATION TEST")
    print("="*60)
    
    quick_prompts = ["Hello", "The cat", "I think"]
    quick_results = []
    
    for prompt in quick_prompts:
        print(f"\nGenerating for: '{prompt}'")
        try:
            generated = tester.generate_with_primary_sae_hook(prompt, max_length=10)
            full_text = prompt + " " + generated
            quick_results.append({'prompt': prompt, 'generated': generated, 'full_text': full_text})
            print(f"Result: '{full_text}'")
        except Exception as e:
            print(f"ERROR: {e}")
            quick_results.append({'prompt': prompt, 'generated': f"ERROR: {str(e)}", 'full_text': f"{prompt} ERROR: {str(e)}"})
    
    # Test 3: Full generation analysis (subset)
    print("\n" + "="*60)
    print("GENERATION ANALYSIS (PRIMARY SAE HOOK ONLY)")
    print("="*60)
    
    generation_results = {}
    
    # Test a subset of prompts
    test_categories = ["neutral", "casual_emotional"]
    
    for prompt_type in test_categories:
        print(f"\nProcessing {prompt_type} prompts...")
        generation_results[prompt_type] = []
        
        # Test first 2 prompts from each category
        prompts = PROMPT_SETS[prompt_type][:2]
        
        for prompt in tqdm(prompts, desc=f"Generating for {prompt_type}"):
            try:
                generated = tester.generate_with_primary_sae_hook(
                    prompt, args.generation_length
                )
                
                generation_results[prompt_type].append({
                    'prompt': prompt,
                    'generated': generated,
                    'full_text': prompt + " " + generated
                })
            except Exception as e:
                print(f"Error generating for prompt '{prompt}': {e}")
                generation_results[prompt_type].append({
                    'prompt': prompt,
                    'generated': f"ERROR: {str(e)}",
                    'full_text': f"{prompt} ERROR: {str(e)}"
                })
    
    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Save quick test results
    quick_test_path = output_dir / "primary_sae_quick_test.txt"
    with open(quick_test_path, 'w', encoding='utf-8') as f:
        f.write("=== PRIMARY SAE HOOK QUICK TEST ===\n\n")
        for result in quick_results:
            f.write(f"Prompt: {result['prompt']}\n")
            f.write(f"Generated: {result['generated']}\n")
            f.write(f"Full text: {result['full_text']}\n\n")
    
    # Save generation results
    gen_dir = output_dir / "primary_sae_generations"
    gen_dir.mkdir(exist_ok=True)
    
    for prompt_type, generations in generation_results.items():
        filename = f"primary_sae_generations_{prompt_type}.txt"
        filepath = gen_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"=== PRIMARY SAE HOOK {prompt_type.upper()} PROMPTS ===\n\n")
            for i, gen in enumerate(generations):
                f.write(f"--- Generation {i+1} ---\n")
                f.write(f"Prompt: {gen['prompt']}\n")
                f.write(f"Generated: {gen['generated']}\n")
                f.write(f"Full text: {gen['full_text']}\n\n")
    
    print(f"Primary SAE hook test results saved to: {output_dir}")
    print(f"Quick test: {quick_test_path}")
    print(f"Generations: {gen_dir}")
    
    print(f"\n{'='*60}")
    print("PRIMARY SAE HOOK TEST COMPLETE")
    print(f"{'='*60}")
    print("Compare these results with:")
    print("1. Baseline (no hooks) - should be identical if Primary SAE is perfect")
    print("2. Part13 full pipeline - to isolate where the JR JR JR issue starts")

if __name__ == "__main__":
    main()