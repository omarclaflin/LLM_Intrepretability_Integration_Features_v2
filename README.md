# Feature Integration Beyond Sparse Coding

This project implements and validates a dual encoding hypothesis for neural network interpretability: that networks encode both **feature identity** (what concepts are present) and **feature integration** (how concepts combine computationally) in the same representational neural space. Previous work in this field primarily focuses on feature identity and treats feature packing as pure noise/intererence minimization rather than rich, even as a non-linear feature binding space, also encoded with identity.

## Paper

Full paper: [Feature Integration Beyond Sparse Coding: Evidence for Non-Linear Computation Spaces in Neural Networks](https://github.com/omarclaflin/LLM_Intrepretability_Integration_Features_v2/blob/main/Feature%20Integration%20Beyond%20Sparse%20Coding_%20Evidence%20for%20Non-Linear%20Computation%20Spaces%20in%20Neural%20Networks.pdf)

Blog: [https://omarclaflin.com/2025/06/23/llm-intervention-experiments-with-integrated-features-part-3/](https://omarclaflin.com/2025/06/23/llm-intervention-experiments-with-integrated-features-part-3/)
[Original Blog post/project idea and initial attempt](https://omarclaflin.com/2025/06/14/information-space-contains-computations-not-just-features/)

## Key Results

- **3-23% reconstruction improvement** over baseline TopK SAE using Neural Factorization Machines
- **4-20% contribution** from non-linear interaction components
- **Selective intervention effects** demonstrating functional significance of integration features
- **Statistical validation** of 2×2 semantic interaction patterns (formality × emotion)

## Architecture

```
Raw Activations → Primary SAE → NFM → Secondary SAE
        ↓           ↓      ↓      ↑         ↓
      Feature Identity    Residuals    Feature Integration
                              
```

The pipeline decomposes neural representations into:
1. **Primary SAE**: Sparse feature extraction (50k features, TopK=1024)
2. **Neural Factorization Machine**: Captures feature integration patterns in SAE residuals
3. **Secondary SAE**: Decomposes NFM interaction embeddings for interpretability

## Setup

### Prerequisites
- NVIDIA GPU with 24GB+ VRAM
- Python 3.8+
- PyTorch with CUDA support

### Installation
```bash
# Setup environment (Windows)
.\setup-windows\setup-llm-intrepretability.ps1

# Download model
.\setup-windows\download_openLlama3B.ps1

# Activate environment
cd llm_intrepretability_project
..\llm-venv\Scripts\activate.bat
```

## Workflow

### 1. Train Primary SAE
```bash
python part1c_sae_topK_implementation.py
```
Trains 50k-feature TopK SAE on OpenLLaMA-3B layer 16 activations.

### 2. Train NFM on Residuals
```bash
python part2d_residual_NFM_streaming.py --sae1_model_path checkpoints_topk/best_model.pt --checkpoint_dir NFM_300_300_200
```
Trains Neural Factorization Machine to predict SAE reconstruction residuals.

### 3. Train Secondary SAE on NFM Embeddings
```bash
python part10_sae_decomp_of_NFM.py --nfm_model_path checkpoints_nfm/best_nfm_linear_interaction_model.pt --sae1_model_path checkpoints_topk/best_model.pt --checkpoint_dir ../interaction_sae
```
Decomposes NFM interaction embeddings using secondary SAE.

### 4. Feature Discovery and Analysis

#### Stimulus-based discovery:
```bash
python part9b_stimulus_response_listOfDiscoveredFeatures_analysis.py --model_path ../models/open_llama_3b --sae_path checkpoints_topk/best_model.pt --nfm_path checkpoints_nfm/best_nfm_linear_interaction_model.pt --output_dir ./stimulus_response_results_discovered --n_features 1 --sae_k 500
```

#### Sparsity-based discovery:
```bash
python part4_NFM_sparsity_inspector.py --model_path checkpoints_nfm/best_nfm_linear_interaction_model.pt --component embeddings
```

### 5. Feature Interpretation
```bash
python part12_NFMSAE_feature_meaning_large_wiki.py --model_path ../models/open_llama_3b --primary_sae_path ./checkpoints_topk/best_model.pt --nfm_path ./checkpoints_nfm/best_nfm_linear_interaction_model.pt --secondary_sae_path ./interaction_sae/best_interaction_sae_topk_model.pt --features "4067,4022,3520,899,2020" --output_dir ./secondary_feature_analysis --max_token_length 10 --claude_examples 20
```

### 6. Intervention Experiments
```bash
python part13_logit_and_clamping_analysis_on_interaction_feature.py --model_path ../models/open_llama_3b --primary_sae_path checkpoints_topk/best_model.pt --nfm_path checkpoints_nfm/best_nfm_linear_interaction_model.pt --secondary_sae_path ./interaction_sae/best_interaction_sae_topk_model.pt --output_dir ./ablationAndLogitResults --target_features 4067 --clamp_multipliers -4.0 0.0 1.0 4.0 --generation_length 50
```

## Key Scripts

| Script | Purpose |
|--------|---------|
| `part1c_sae_topK_implementation.py` | Train primary TopK SAE |
| `part2d_residual_NFM_streaming.py` | Train NFM on SAE residuals |
| `part10_sae_decomp_of_NFM.py` | Train secondary SAE on NFM embeddings |
| `part4_NFM_sparsity_inspector.py` | Find high-Gini coefficient features |
| `part9b_stimulus_response_*` | Stimulus-based feature discovery |
| `part12_NFMSAE_feature_meaning_*` | Feature interpretation pipeline |
| `part13_logit_and_clamping_analysis_*` | Intervention experiments |

## Debugging and Testing

Model behavior validation:
```bash
# Test baseline model
python part13b_noClampHooks_openLlama3BTest.py --model_path ../models/open_llama_3b --output_dir ./baseline_results --generation_length 50

# Test SAE hooks only
python part13c_primarySAE_hooksOnly.py --model_path ../models/open_llama_3b --primary_sae_path checkpoints_topk/best_model.pt --output_dir ./primary_sae_test --generation_length 50

# Debug SAE outputs
python part13d_primarySAEdebugging.py --model_path ../models/open_llama_3b --sae_path checkpoints_topk/checkpoint_step_200000.pt --output_dir ./sae_debug_results --device cuda --top_k 1024
```

## Results Structure

```
./checkpoints_topk/          # Primary SAE checkpoints
./checkpoints_nfm/           # NFM model checkpoints  
./interaction_sae/           # Secondary SAE checkpoints
./stimulus_response_results_discovered/  # Feature discovery results
./secondary_feature_analysis/            # Feature interpretation
./ablationAndLogitResults/              # Intervention experiments
```

## Model Requirements

- **Model**: OpenLLaMA-3B
- **Layer**: 16 (middle layer)
- **Dataset**: WikiText-103
- **Hardware**: Single RTX 3090 (24GB VRAM), 128GB RAM

## Paper

Full paper: [Feature Integration Beyond Sparse Coding: Evidence for Non-Linear Computation Spaces in Neural Networks](https://github.com/omarclaflin/LLM_Intrepretability_Integration_Features_v2/blob/main/Feature%20Integration%20Beyond%20Sparse%20Coding_%20Evidence%20for%20Non-Linear%20Computation%20Spaces%20in%20Neural%20Networks.pdf)

Blog: [https://omarclaflin.com/2025/06/23/llm-intervention-experiments-with-integrated-features-part-3/](https://omarclaflin.com/2025/06/23/llm-intervention-experiments-with-integrated-features-part-3/)

## Citation

```bibtex
@article{claflin2024feature,
  title={Feature Integration Beyond Sparse Coding: Evidence for Non-Linear Computation Spaces in Neural Networks},
  author={Claflin, Omar},
  year={2024}
}
```
