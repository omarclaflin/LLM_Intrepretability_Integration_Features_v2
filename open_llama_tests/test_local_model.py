#Basically, ran this to see how good/bad the open llama model was with various prompts we wanted to use (as research tools, not the research subject matter)
#Can't follow directions (rate on a rubric, find a common semantic pattern, etc)
#Did debug some text formatting issues.

import torch
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Configuration ---
MODEL_PATH = "../models/open_llama_3b" # 

# Define a default query file name if none is provided via command line
DEFAULT_QUERY_FILENAME = "query_from_file.txt"

# --- Path Validation (Optional but Recommended) ---
if not os.path.isdir(MODEL_PATH):
    print(f"Error: The specified MODEL_PATH '{MODEL_PATH}' is not a valid directory.")
    print("Please update the MODEL_PATH variable in the script to point to your local model files.")
    exit()
# --------------------------------------------------

# Set the device (cuda if available, otherwise cpu)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- Load Model and Tokenizer ---
print(f"Attempting to load tokenizer from local path: {MODEL_PATH}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=False, # Use Python implementation as in your original script
        padding_side="right", # Recommended for causal models as in your original script
        local_files_only=True # <-- Ensures no internet access for model files
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

except Exception as e:
    print(f"Error loading tokenizer from local path '{MODEL_PATH}': {e}")
    print("Please ensure the path is correct and contains tokenizer files (e.g., tokenizer.json).")
    exit()

print(f"Attempting to load model from local path: {MODEL_PATH}...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto", # Automatically distribute model layers
        torch_dtype=torch.float16, # Use half-precision
        local_files_only=True # <-- Ensures no internet access for model files
    )
    model.eval() # Set model to evaluation mode
except Exception as e:
    print(f"Error loading model from local path '{MODEL_PATH}': {e}")
    print("Please ensure the path is correct and contains model files (e.g., pytorch_model.bin).")
    exit()

print("Model loaded successfully from local files.")

# --- Run Query ---
def run_query(query: str):
    """Runs a query against the loaded model and prints the response."""
    if not query or not query.strip():
        print("Warning: Query read from file is empty or only contains whitespace. Skipping generation.")
        return

    print(f"\n--- Running Query ---\nQuery: {query.strip()}") # Print stripped query for clarity

    inputs = tokenizer(query, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(
        output_tokens[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    print("\n--- Model Response ---")
    print(generated_text.strip())
    print("----------------------")


# --- Script Entry Point ---
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run a query from a text file against a local LLM.")
    parser.add_argument(
        "--query_file",
        type=str,
        required=False, # Make the query file argument optional
        default=DEFAULT_QUERY_FILENAME, # Set the default filename
        help=f"Path to the text file containing the query (defaults to '{DEFAULT_QUERY_FILENAME}')."
    )

    args = parser.parse_args()

    # Get the query file path (either from arg or default)
    query_file_path = args.query_file

    # Validate that the query file exists before trying to read
    if not os.path.isfile(query_file_path):
        if query_file_path == DEFAULT_QUERY_FILENAME:
             print(f"Error: Default query file '{DEFAULT_QUERY_FILENAME}' not found.")
             print(f"Please create a file named '{DEFAULT_QUERY_FILENAME}' in the current directory with your query,")
             print("or specify a different file using the --query_file argument.")
        else:
            print(f"Error: The specified query file '{query_file_path}' does not exist.")
        exit()

    print(f"Reading query from file: {query_file_path}")
    try:
        with open(query_file_path, 'r', encoding='utf-8') as f:
            query_content = f.read()
    except Exception as e:
        print(f"Error reading query file '{query_file_path}': {e}")
        exit()

    # Run the query using the content read from the file
    # run_query("Write a brief description of the planet Mars.")
    run_query(query_content.strip()) 