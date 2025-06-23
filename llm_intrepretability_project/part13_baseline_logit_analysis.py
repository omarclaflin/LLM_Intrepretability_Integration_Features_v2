"""
part13_baseline_logit_analysis.py

Baseline Logit Analysis WITHOUT any SAE interventions

This script performs the exact same logit analysis as part13_logit_and_clamping_analysis_on_interaction_feature.py
but with NO hooks, NO SAE processing, NO interventions - just pure model logit analysis.

Use this to establish baseline logit values for target words, then compare against
the intervention results to see the actual effects of clamping Secondary SAE features.

Pipeline: Input → Base LLM → Logits (NO MODIFICATIONS)
"""

import torch
import numpy as np
import argparse
import pandas as pd
from pathlib import Path
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
import logging
from typing import List, Tuple, Dict, Any
import csv

# Default word lists for logit lens analysis - organized by both dimensions
DEFAULT_FORMAL_LOW_EMOTION_WORDS = ["perhaps", "may", "however", "therefore", "consequently", "moreover", "accordingly", "furthermore"]
DEFAULT_FORMAL_HIGH_EMOTION_WORDS = ["profoundly", "devastated", "magnificent", "appalling", "extraordinary", "overwhelming", "catastrophic", "triumphant"]
DEFAULT_CASUAL_LOW_EMOTION_WORDS = ["yeah", "okay", "kinda", "basically", "anyway", "whatever", "guess", "stuff"]
DEFAULT_CASUAL_HIGH_EMOTION_WORDS = ["OMG", "totally", "literally", "seriously", "honestly", "really", "absolutely", "incredibly"]

# Legacy word lists for backward compatibility
DEFAULT_FORMAL_WORDS = DEFAULT_FORMAL_LOW_EMOTION_WORDS + DEFAULT_FORMAL_HIGH_EMOTION_WORDS
DEFAULT_CASUAL_WORDS = DEFAULT_CASUAL_LOW_EMOTION_WORDS + DEFAULT_CASUAL_HIGH_EMOTION_WORDS

# Prompt sets for different formality/emotion combinations
PROMPT_SETS = {
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

class BaselineLogitAnalyzer:
    """Baseline logit analyzer with NO interventions."""
    
    def __init__(self, tokenizer, base_model, device="cuda"):
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.device = device
        
        # Set model to eval mode
        self.base_model.eval()
        
        # Get vocabulary for logit analysis
        self.vocab_size = len(tokenizer.vocab)

    def get_word_token_ids_by_dimensions(self, formal_low_emotion, formal_high_emotion, casual_low_emotion, casual_high_emotion):
        """Convert words to their token IDs, organized by formality and emotion dimensions."""
        word_categories = {
            'formal_low_emotion': formal_low_emotion,
            'formal_high_emotion': formal_high_emotion,
            'casual_low_emotion': casual_low_emotion,
            'casual_high_emotion': casual_high_emotion
        }
        
        categorized_tokens = {}
        
        for category, words in word_categories.items():
            word_token_map = {}
            for word in words:
                # Try different tokenization approaches
                tokens_simple = self.tokenizer.encode(word, add_special_tokens=False)
                tokens_space = self.tokenizer.encode(" " + word, add_special_tokens=False)
                
                # Use the single token if available, otherwise take the first token
                if len(tokens_simple) == 1:
                    word_token_map[word] = tokens_simple[0]
                elif len(tokens_space) == 1:
                    word_token_map[word] = tokens_space[0]
                elif len(tokens_space) >= 2:
                    # If " word" gives multiple tokens, take the second one (after space)
                    word_token_map[word] = tokens_space[1]
                elif len(tokens_simple) >= 1:
                    # Fallback to first token of simple tokenization
                    word_token_map[word] = tokens_simple[0]
                else:
                    print(f"Warning: Could not tokenize word '{word}' in category '{category}'")
                    continue
                    
            categorized_tokens[category] = word_token_map
                
        return categorized_tokens

    def baseline_forward_pass(self, input_ids, attention_mask):
        """
        Standard forward pass with NO interventions.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
        
        Returns:
            logits: Final logits from base model
        """
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                use_cache=False
            )
            logits = outputs.logits
        
        return logits

    def logit_lens_analysis(self, formal_low_emotion, formal_high_emotion, casual_low_emotion, casual_high_emotion):
        """
        Analyze baseline logit values for words across both formality and emotion dimensions.
        
        Args:
            formal_low_emotion: List of formal + low emotion words
            formal_high_emotion: List of formal + high emotion words  
            casual_low_emotion: List of casual + low emotion words
            casual_high_emotion: List of casual + high emotion words
        
        Returns:
            DataFrame with baseline logit analysis results
        """
        print(f"\n=== BASELINE LOGIT LENS ANALYSIS (2D: Formality × Emotion) ===")
        
        # Get token IDs for all word categories
        categorized_tokens = self.get_word_token_ids_by_dimensions(
            formal_low_emotion, formal_high_emotion, casual_low_emotion, casual_high_emotion
        )
        
        print(f"Word categories:")
        for category, word_map in categorized_tokens.items():
            print(f"  {category}: {word_map}")
        
        # Use a simple prompt that could go either direction
        test_prompts = [
            "I think that",
            "The situation is", 
            "My opinion on this is",
            "It seems to me that",
            "The best approach would be"
        ]
        
        results = []
        
        for prompt in tqdm(test_prompts, desc="Processing prompts"):
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
            
            # Get baseline logits (no intervention)
            logits = self.baseline_forward_pass(inputs["input_ids"], inputs["attention_mask"])
            
            # Extract logits for the last token (where we'd generate next)
            last_token_logits = logits[0, -1, :]  # [vocab_size]
            
            # Record logits for all word categories
            for category, word_token_map in categorized_tokens.items():
                for word, token_id in word_token_map.items():
                    results.append({
                        'prompt': prompt,
                        'word_category': category,
                        'word': word,
                        'token_id': token_id,
                        'logit_value': float(last_token_logits[token_id].cpu()),
                        'analysis_type': 'baseline'
                    })
        
        return pd.DataFrame(results)

    def generate_baseline_text(self, prompt, max_length=50):
        """Generate text with NO interventions."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        
        generated_tokens = []
        
        for _ in range(max_length):
            # Create attention mask
            attention_mask = torch.ones_like(input_ids)
            
            # Get logits with no intervention
            logits = self.baseline_forward_pass(input_ids, attention_mask)
            
            # Sample next token (using simple greedy decoding for consistency)
            next_token_id = torch.argmax(logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
            
            # Check for EOS
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
                
            generated_tokens.append(next_token_id.item())
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        
        # Decode the generated part only
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_text

    def baseline_generation_analysis(self, generation_length):
        """
        Generate text completions with NO interventions.
        
        Args:
            generation_length: Number of tokens to generate
        
        Returns:
            Dictionary with baseline generation results
        """
        print(f"\n=== BASELINE GENERATION ANALYSIS ===")
        print(f"Generation length: {generation_length}")
        
        results = {}
        
        for prompt_type, prompts in PROMPT_SETS.items():
            print(f"\nProcessing {prompt_type} prompts...")
            results[prompt_type] = []
            
            for prompt in tqdm(prompts, desc=f"Baseline generation"):
                try:
                    generated = self.generate_baseline_text(prompt, generation_length)
                    
                    results[prompt_type].append({
                        'prompt': prompt,
                        'generated': generated,
                        'full_text': prompt + " " + generated
                    })
                except Exception as e:
                    print(f"Error generating for prompt '{prompt}': {e}")
                    results[prompt_type].append({
                        'prompt': prompt,
                        'generated': f"ERROR: {str(e)}",
                        'full_text': f"{prompt} ERROR: {str(e)}"
                    })
        
        return results

def load_model(args):
    """Load base model only."""
    print("Loading base model...")
    
    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")
    
    print(f"Base model loaded successfully!")
    
    return tokenizer, base_model

def save_results(logit_df, generation_results, output_dir, args):
    """Save baseline analysis results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save logit analysis
    logit_dir = output_dir / "baseline_logit_analysis"
    logit_dir.mkdir(exist_ok=True)
    
    logit_csv_path = logit_dir / "baseline_logit_analysis.csv"
    logit_df.to_csv(logit_csv_path, index=False)
    print(f"Baseline logit analysis saved to: {logit_csv_path}")
    
    # Create and save logit summary statistics
    logit_summary_df = create_baseline_logit_summary(logit_df)
    logit_summary_path = logit_dir / "baseline_logit_summary_stats.csv"
    logit_summary_df.to_csv(logit_summary_path, index=False)
    print(f"Baseline logit summary saved to: {logit_summary_path}")
    
    # Create and save dimensional analysis
    dimensional_df = create_baseline_dimensional_summary(logit_df)
    dimensional_path = logit_dir / "baseline_logit_dimensional_analysis.csv"
    dimensional_df.to_csv(dimensional_path, index=False)
    print(f"Baseline dimensional analysis saved to: {dimensional_path}")
    
    # Save generation analysis
    gen_dir = output_dir / "baseline_generation_analysis"
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
    
    print(f"Baseline generation analysis saved to: {gen_dir}")
    
    # Save summary
    summary = {
        'analysis_type': 'baseline',
        'generation_length': args.generation_length,
        'formal_words': args.formal_words,
        'casual_words': args.casual_words,
        'logit_analysis_file': str(logit_csv_path),
        'logit_summary_file': str(logit_summary_path),
        'generation_analysis_dir': str(gen_dir),
        'model_info': {
            'model_path': args.model_path
        }
    }
    
    summary_path = output_dir / "baseline_analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Baseline analysis summary saved to: {summary_path}")

def create_baseline_logit_summary(logit_df):
    """
    Create summary statistics for baseline logit analysis.
    
    For each prompt and word combination, calculate basic statistics.
    """
    summary_stats = []
    
    # Group by prompt, word_category, and word
    grouped = logit_df.groupby(['prompt', 'word_category', 'word'])
    
    for (prompt, word_category, word), group in grouped:
        # Since this is baseline (no interventions), each group should have only one row
        row = group.iloc[0]
        
        summary_stats.append({
            'prompt': prompt,
            'word_category': word_category,
            'word': word,
            'token_id': row['token_id'],
            'baseline_logit': row['logit_value'],
            'analysis_type': 'baseline'
        })
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Sort by logit value (highest first)
    summary_df = summary_df.sort_values('baseline_logit', ascending=False)
    
    return summary_df

def create_baseline_dimensional_summary(logit_df):
    """
    Create summary statistics grouped by word category (formality × emotion dimensions).
    """
    dimensional_stats = []
    
    # Group by word_category
    grouped = logit_df.groupby(['word_category'])
    
    for word_category, group in grouped:
        mean_logit = group['logit_value'].mean()
        std_logit = group['logit_value'].std()
        median_logit = group['logit_value'].median()
        num_words = len(group['word'].unique())
        num_prompts = len(group['prompt'].unique())
        
        dimensional_stats.append({
            'word_category': word_category,
            'mean_logit': mean_logit,
            'std_logit': std_logit,
            'median_logit': median_logit,
            'num_words': num_words,
            'num_prompts': num_prompts,
            'total_observations': len(group),
            'analysis_type': 'baseline'
        })
    
    dimensional_df = pd.DataFrame(dimensional_stats)
    
    return dimensional_df

def main():
    parser = argparse.ArgumentParser(description="Baseline Logit Analysis (No Interventions)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base LLM model")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    
    parser.add_argument("--generation_length", type=int, default=50,
                       help="Number of tokens to generate")
    
    parser.add_argument("--formal_low_emotion_words", type=str, nargs="+", default=DEFAULT_FORMAL_LOW_EMOTION_WORDS,
                       help="Formal + low emotion words for logit analysis")
    parser.add_argument("--formal_high_emotion_words", type=str, nargs="+", default=DEFAULT_FORMAL_HIGH_EMOTION_WORDS,
                       help="Formal + high emotion words for logit analysis")
    parser.add_argument("--casual_low_emotion_words", type=str, nargs="+", default=DEFAULT_CASUAL_LOW_EMOTION_WORDS,
                       help="Casual + low emotion words for logit analysis")
    parser.add_argument("--casual_high_emotion_words", type=str, nargs="+", default=DEFAULT_CASUAL_HIGH_EMOTION_WORDS,
                       help="Casual + high emotion words for logit analysis")
    
    # Legacy support
    parser.add_argument("--formal_words", type=str, nargs="+", default=DEFAULT_FORMAL_WORDS,
                       help="Legacy: Formal words for logit analysis")
    parser.add_argument("--casual_words", type=str, nargs="+", default=DEFAULT_CASUAL_WORDS,
                       help="Legacy: Casual words for logit analysis")
    
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / "baseline_analysis.log"),
            logging.StreamHandler()
        ]
    )
    
    print(f"Starting BASELINE logit analysis (no interventions)")
    print(f"Generation length: {args.generation_length}")
    
    # Load model
    tokenizer, base_model = load_model(args)
    
    # Initialize analyzer
    analyzer = BaselineLogitAnalyzer(tokenizer, base_model, args.device)
    
    # Run logit lens analysis
    print("\n" + "="*60)
    print("RUNNING BASELINE LOGIT LENS ANALYSIS")
    print("="*60)
    
    # Use new dimensional word lists if available, otherwise fall back to legacy
    if hasattr(args, 'formal_low_emotion_words'):
        logit_df = analyzer.logit_lens_analysis(
            args.formal_low_emotion_words, args.formal_high_emotion_words,
            args.casual_low_emotion_words, args.casual_high_emotion_words
        )
    else:
        # Legacy support - treat formal_words as formal+low_emotion, casual_words as casual+high_emotion  
        logit_df = analyzer.logit_lens_analysis(
            args.formal_words, [],  # formal_low_emotion, formal_high_emotion
            [], args.casual_words  # casual_low_emotion, casual_high_emotion
        )
    
    # Run generation analysis
    print("\n" + "="*60) 
    print("RUNNING BASELINE GENERATION ANALYSIS")
    print("="*60)
    
    generation_results = analyzer.baseline_generation_analysis(args.generation_length)
    
    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    save_results(logit_df, generation_results, args.output_dir, args)
    
    print(f"\nBaseline Analysis complete! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()