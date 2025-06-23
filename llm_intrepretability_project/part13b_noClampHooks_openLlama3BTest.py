"""
Baseline Generation Test Script

This script generates text using the exact same prompts and settings as part13,
but with NO hooks, NO clamping, NO interventions - just pure model generation.

Use this to compare against the intervention results to identify if the issue
is in the intervention pipeline or elsewhere.
"""

# Example results
# ============================================================
# QUICK SANITY CHECK
# ============================================================
# 'Hello' → 'Hello , I am interested about your Toyota Corolla car'
# 'The cat' → 'The cat is out of the bag.
# The cat is'
# 'I think' → 'I think I've been a little bit too quiet lately'
# 'Yesterday' → 'Yesterday , I was in the middle of a conversation with'


import torch
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

# Modified prompt sets with tone analysis prefix
TONE_PREFIX = "Describe the tone of this sentence: '"

# Original prompt sets - these will be prefixed
ORIGINAL_PROMPT_SETS = {
    "neutral": [
        "I need to inform everyone about",
        "When someone asks me about",
        "The reason this happened is",
        "My thoughts on this matter are",
        "The best way to handle this is"
    ],
    "formal_emotional": [
        "I must express my profound concern regarding",
        "It is with great disappointment that I must address",
        "I am deeply troubled by the implications of",
        "With utmost urgency, I must convey that",
        "I feel compelled to voice my serious objections to"
    ],
    "formal_neutral": [
        "According to the documentation, the procedure requires",
        "The analysis indicates that the optimal approach involves",
        "Pursuant to the established guidelines, it is recommended that",
        "The evaluation demonstrates that the most effective method is",
        "Based on the available evidence, the conclusion suggests that"
    ],
    "casual_emotional": [
        "OMG I can't believe how excited I am about",
        "Honestly, I'm so frustrated with",
        "I'm literally freaking out about",
        "This is seriously the most amazing thing about",
        "I'm totally devastated by"
    ],
    "casual_neutral": [
        "So anyway, I was thinking about",
        "Yeah, the thing is that",
        "Basically what happened was",
        "I guess the main point is",
        "You know, it's kinda interesting how"
    ]
}

# Create the modified prompt sets with the tone analysis prefix
PROMPT_SETS = {}
for category, prompts in ORIGINAL_PROMPT_SETS.items():
    PROMPT_SETS[category] = [TONE_PREFIX + prompt for prompt in prompts]

# Same logit test prompts as part13 - also modified with prefix
ORIGINAL_LOGIT_TEST_PROMPTS = [
    "I think that",
    "The situation is", 
    "My opinion on this is",
    "It seems to me that",
    "The best approach would be"
]

LOGIT_TEST_PROMPTS = [TONE_PREFIX + prompt for prompt in ORIGINAL_LOGIT_TEST_PROMPTS]

def generate_baseline_text(model, tokenizer, prompt, max_length=50, device="cuda"):
    """Generate text with the base model - no interventions."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(max_length):
            # Standard forward pass
            outputs = model(input_ids=input_ids, use_cache=False)
            logits = outputs.logits
            
            # Sample next token (greedy decoding for consistency with part13)
            next_token_id = torch.argmax(logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
            
            # Check for EOS
            if next_token_id.item() == tokenizer.eos_token_id:
                break
                
            generated_tokens.append(next_token_id.item())
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
    
    # Decode the generated part only
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text

def get_baseline_logits(model, tokenizer, prompt, target_words, device="cuda"):
    """Get logit values for target words - no interventions."""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False)
        logits = outputs.logits
        
        # Extract logits for the last token
        last_token_logits = logits[0, -1, :]  # [vocab_size]
        
        # Get logits for target words
        word_logits = {}
        for word in target_words:
            # Try different tokenization approaches (same as part13)
            tokens_simple = tokenizer.encode(word, add_special_tokens=False)
            tokens_space = tokenizer.encode(" " + word, add_special_tokens=False)
            
            token_id = None
            if len(tokens_simple) == 1:
                token_id = tokens_simple[0]
            elif len(tokens_space) == 1:
                token_id = tokens_space[0]
            elif len(tokens_space) >= 2:
                token_id = tokens_space[1]
            elif len(tokens_simple) >= 1:
                token_id = tokens_simple[0]
            
            if token_id is not None:
                word_logits[word] = {
                    'token_id': token_id,
                    'logit_value': float(last_token_logits[token_id].cpu())
                }
    
    return word_logits

def main():
    parser = argparse.ArgumentParser(description="Baseline Generation Test")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base LLM model")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--generation_length", type=int, default=50, help="Number of tokens to generate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading model...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Generation length: {args.generation_length}")
    print(f"Using tone analysis prefix: '{TONE_PREFIX}'")
    
    # Test 1: Generation Analysis (same as part13 clamping analysis)
    print("\n" + "="*60)
    print("BASELINE GENERATION ANALYSIS")
    print("="*60)
    
    generation_results = {}
    
    for prompt_type, prompts in PROMPT_SETS.items():
        print(f"\nProcessing {prompt_type} prompts...")
        generation_results[prompt_type] = []
        
        for prompt in tqdm(prompts, desc=f"Generating for {prompt_type}"):
            try:
                generated = generate_baseline_text(
                    model, tokenizer, prompt, args.generation_length, args.device
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
    
    # Save generation results
    gen_dir = output_dir / "baseline_generations"
    gen_dir.mkdir(exist_ok=True)
    
    for prompt_type, generations in generation_results.items():
        filename = f"baseline_generations_{prompt_type}.txt"
        filepath = gen_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"=== BASELINE {prompt_type.upper()} PROMPTS ===\n\n")
            for i, gen in enumerate(generations):
                f.write(f"--- Generation {i+1} ---\n")
                f.write(f"Prompt: {gen['prompt']}\n")
                f.write(f"Generated: {gen['generated']}\n")
                f.write(f"Full text: {gen['full_text']}\n\n")
    
    print(f"Baseline generations saved to: {gen_dir}")
    
    # Test 2: Logit Analysis (same prompts as part13)
    print("\n" + "="*60)
    print("BASELINE LOGIT ANALYSIS")
    print("="*60)
    
    # Use same target words as part13 defaults
    target_words = [
        # Formal + low emotion
        "perhaps", "may", "however", "therefore", "consequently", "moreover", "accordingly", "furthermore",
        # Formal + high emotion  
        "profoundly", "devastated", "magnificent", "appalling", "extraordinary", "overwhelming", "catastrophic", "triumphant",
        # Casual + low emotion
        "yeah", "okay", "kinda", "basically", "anyway", "whatever", "guess", "stuff",
        # Casual + high emotion
        "OMG", "totally", "literally", "seriously", "honestly", "really", "absolutely", "incredibly"
    ]
    
    logit_results = []
    
    for prompt in tqdm(LOGIT_TEST_PROMPTS, desc="Processing logit prompts"):
        word_logits = get_baseline_logits(model, tokenizer, prompt, target_words, args.device)
        
        for word, logit_data in word_logits.items():
            logit_results.append({
                'prompt': prompt,
                'word': word,
                'token_id': logit_data['token_id'],
                'logit_value': logit_data['logit_value']
            })
    
    # Save logit results
    logit_dir = output_dir / "baseline_logits"
    logit_dir.mkdir(exist_ok=True)
    
    import csv
    logit_csv_path = logit_dir / "baseline_logit_analysis.csv"
    with open(logit_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['prompt', 'word', 'token_id', 'logit_value'])
        writer.writeheader()
        writer.writerows(logit_results)
    
    print(f"Baseline logit analysis saved to: {logit_csv_path}")
    
    # Test 3: Single token generation for quick sanity check
    print("\n" + "="*60)
    print("QUICK SANITY CHECK")
    print("="*60)
    
    # Also add tone prefix to sanity check prompts
    original_test_prompts = ["Hello", "The cat", "I think", "Yesterday"]
    test_prompts = [TONE_PREFIX + prompt for prompt in original_test_prompts]
    
    sanity_results = []
    for prompt in test_prompts:
        generated = generate_baseline_text(model, tokenizer, prompt, max_length=10, device=args.device)
        full_text = prompt + " " + generated
        sanity_results.append({'prompt': prompt, 'generated': generated, 'full_text': full_text})
        print(f"'{prompt}' → '{full_text}'")
    
    # Save sanity check
    sanity_path = output_dir / "baseline_sanity_check.txt"
    with open(sanity_path, 'w', encoding='utf-8') as f:
        f.write("=== BASELINE SANITY CHECK ===\n\n")
        for result in sanity_results:
            f.write(f"Prompt: {result['prompt']}\n")
            f.write(f"Generated: {result['generated']}\n")
            f.write(f"Full text: {result['full_text']}\n\n")
    
    print(f"\n{'='*60}")
    print("BASELINE ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"Compare these baseline results with your part13 intervention results.")
    print(f"If baseline looks normal but part13 produces nonsense, the issue is in the intervention pipeline.")

if __name__ == "__main__":
    main()