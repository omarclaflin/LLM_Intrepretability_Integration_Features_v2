"""
This script performs stimulus-response analysis on identified SAE features by creating controlled
2x2 stimulus sets and measuring their activations through different model components.

This study analyzes Feature 32806 (our "Paris" detector) and Feature 23360 (our "Folk/Released" detector) across four interaction conditions:

Feature1 High + Feature2 High: Paris + Folk
Feature1 High + Feature2 Low: Paris + Released
Feature1 Low + Feature2 High: @-@ + Folk
Feature1 Low + Feature2 Low: @-@ + Released

[OLD below -- not good SAE modulation on Feature 2]
It analyzes Feature 32806 (Paris detector) and Feature 40595 (Bidding detector) across four
interaction conditions:
- Feature1 High + Feature2 High: Paris + Bidding
- Feature1 High + Feature2 Low: Paris + Nicknames
- Feature1 Low + Feature2 High: @-@ + Bidding  
- Feature1 Low + Feature2 Low: @-@ + Nicknames


The script measures SAE activations, NFM linear contributions, and NFM interaction strengths
across these conditions and creates visualization plots.
"""

#part 7 was running the part_5 script to label the identified features
#  python part5_find_feature_meaning_large_wiki.py --sae_path ../checkpoints/best_model.pt --model_path ../models/open_llama_3b --features "32806" --output_dir ./NFM_feature_analysis_results_lowest/ --config_dir ../config --find_weakest

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from pathlib import Path
import json
from tqdm.auto import tqdm
import random
from typing import List, Tuple, Dict, Any
import pandas as pd

# # Hardcoded stimulus sets - genrated by Claude 40

# feature1name = 'Paris'
# feature1bname = '@-@'
# feature2name = 'Bidding'
# feature2bname = 'Nicknames'
# feature1ID = 32806
# feature2ID = 40595
# STIMULUS_SETS = {
#     # Feature1 High + Feature2 High: Paris + Bidding
#     "F1_high_F2_high": [
#         "The art auction house in Paris announced bidding would begin at noon.",
#         "Bidding for the Louvre restoration contract reached record highs in Paris.",
#         "The Paris stock exchange saw intense bidding activity this morning.",
#         "Real estate bidding wars have intensified across Paris neighborhoods.",
#         "The antique auction in central Paris attracted competitive bidding.",
#         "Bidding on Parisian apartments has reached unprecedented levels.",
#         "The wine auction in Paris featured aggressive bidding from collectors.",
#         "Construction companies are bidding for the new Paris metro expansion.",
#         "The art gallery in Paris opened bidding for the impressionist collection.",
#         "Bidding for parking permits in Paris has become increasingly competitive."
#     ],
    
#     # Feature1 High + Feature2 Low: Paris + Nicknames
#     "F1_high_F2_low": [
#         "The Eiffel Tower (\"The Iron Lady\") attracts millions of tourists to Paris.",
#         "Paris cafes serve \"The King\" (a special coffee blend) to visitors.",
#         "The Seine River flows through \"The City of Light\" (Paris) gracefully.",
#         "\"The Fashion Capital\" (Paris) showcases designer collections annually.",
#         "The Louvre in \"Paname\" (slang for Paris) houses artistic treasures.",
#         "Paris gardens provide retreats in \"The Heart of France\" (the capital).",
#         "The Metro efficiently connects \"Lutèce\" (ancient name for Paris) neighborhoods.",
#         "\"The Seine City\" (Paris) architecture blends historical and modern design.",
#         "The Champs-Élysées features luxury shopping in \"The Capital\" (Paris).",
#         "Museums in \"The City of Art\" (Paris) display world-renowned collections."
#     ],
    
#     # Feature1 Low + Feature2 High: @-@ + Bidding  
#     "F1_low_F2_high": [
#         "The online auction site crashed during heavy bidding at 3@-@30 PM.",
#         "Bidding increments were set at $10@-@$50 for most items.",
#         "The bidding war lasted from 9@-@15 AM until closing time.",
#         "Auto@-@bidding features helped users stay competitive.",
#         "The final bidding round went from lot 45@-@67.",
#         "Bidding activity peaked between sessions 12@-@18.",
#         "The platform showed bidding conflicts at time@-@stamps 14:30.",
#         "Silent bidding used codes like A@-@7 and B@-@12.",
#         "The bidding interface displayed ranges 100@-@500 clearly.",
#         "Phone bidding required PIN numbers like 789@-@456."
#     ],
    
#     # Feature1 Low + Feature2 Low: @-@ + Nicknames
#     "F1_low_F2_low": [
#         "The online forum featured \"The Professor\" (a user) posting at 9@-@15 AM.",
#         "Database entries showed \"Big Ben\" (clock ID) active from 12@-@18 hours.",
#         "The server logged \"The Boss\" (admin account) access at time@-@stamp 14:30.",
#         "Chat users called the moderator \"Captain America\" during 3@-@5 PM sessions.",
#         "System records show \"The King\" (user rank) online between 10@-@12 AM.",
#         "The gaming platform tracks \"The Duke\" (player handle) scores 100@-@500 range.",
#         "Network logs indicate \"Iron Lady\" (server name) downtime 2@-@4 hours.",
#         "Forum posts by \"Old Blue Eyes\" (username) peaked during 7@-@9 PM.",
#         "\"The Rocket\" (bot name) processed requests from codes A@-@7 to B@-@12.",
#         "Database shows \"The Great One\" (admin ID) login attempts 789@-@456 times."
#     ]
# }

# these ended up having low modulation of the SAE for Feature 2, picked a different Feature 2
#generated by Gemini 2.5 (ran out of Claude coins)

feature1name = 'Paris'
feature1bname = '@-@'
feature2name = 'Folk'
feature2bname = 'Released'
feature1ID = 32806
feature2ID = 23360

STIMULUS_SETS = {
    # Feature1 High + Feature2 High: Paris + Folk
    "F1_high_F2_high": [
        "Traditional folk music echoed through the streets of Paris.",
        "The museum in Paris showcased ancient folk tales and their origins.",
        "Paris is a hub where diverse folk cultures intertwine.",
        "A festival celebrating global folk traditions was held in Paris.",
        "Artists in Paris often draw inspiration from classic folk art.",
        "The Parisian art scene embraced new interpretations of folk dance.",
        "Researchers in Paris studied the evolution of European folk songs.",
        "Local artisans in Paris preserved traditional folk crafts.",
        "The spirit of folk storytelling is alive in Paris's literary circles.",
        "Paris's cultural centers host workshops on various folk practices."
    ],
    
    # Feature1 High + Feature2 Low: Paris + Released
    "F1_high_F2_low": [
        "A new film was recently released in cinemas across Paris.",
        "The Parisian author released her latest novel to critical acclaim.",
        "A major album was released by a French artist in Paris last week.",
        "The fashion house in Paris released its spring collection.",
        "Numerous digital art pieces were released online from galleries in Paris.",
        "The publishing company in Paris released a series of children's books.",
        "Concert organizers in Paris announced a newly released live album.",
        "The game developer based in Paris released their highly anticipated title.",
        "A collection of historical documents was released to the public in Paris.",
        "The French government released new tourism guidelines for Paris."
    ],
    
    # Feature1 Low + Feature2 High: @-@ + Folk
    "F1_low_F2_high": [
        "The band played folk music from 7@-@9 PM at the local fair.",
        "The ancient folk tales were passed down through generations 1@-@5.",
        "A study on folk culture surveyed individuals aged 18@-@35.",
        "The exhibition featured folk traditions from regions A@-@Z.",
        "Ethnomusicologists collected folk songs from periods 1900@-@1950.",
        "The community group organized a folk dance workshop for children 6@-@10.",
        "Historical records show folk practices between 1400@-@1600 AD.",
        "The article discussed the impact of globalization on folk art forms 1@-@20.",
        "Audience members enjoyed folk stories told from 8@-@10 PM.",
        "The collection included folk instruments from countries 3@-@7."
    ],
    
    # Feature1 Low + Feature2 Low: @-@ + Released
    "F1_low_F2_low": [
        "The software update was released on servers 1@-@5 between 2@-@4 AM.",
        "New academic papers were released online from subjects A@-@Z.",
        "The quarterly report was released, covering figures from Q1@-@Q2.",
        "Version 2.0 of the application was released to users 100@-@500.",
        "Data sets were released for public access between dates 2020@-@2023.",
        "Security patches were released for systems running OS versions 7@-@10.",
        "The official statement was released via channels 1@-@3 at 9@-@15 AM.",
        "Earnings reports were released for companies with IDs 123@-@456.",
        "The updated guidelines were released in stages 1@-@4.",
        "Patches were released to fix bugs in modules 8@-@12."
    ]
}


# feature1name = 'Paris'
# feature1bname = '@-@'
# feature2name = 'Folk'
# feature2bname = 'Modern'
# feature1ID = 32806
# feature2ID = 23360

# STIMULUS_SETS = {
#     # Feature1 High + Feature2 High: Paris + Folk
#     "F1_high_F2_high": [
#         "Traditional folk music echoed through the streets of Paris.",
#         "The museum in Paris showcased ancient folk tales and their origins.",
#         "Paris is a hub where diverse folk cultures intertwine.",
#         "A festival celebrating global folk traditions was held in Paris.",
#         "Artists in Paris often draw inspiration from classic folk art.",
#         "The Parisian art scene embraced new interpretations of folk dance.",
#         "Researchers in Paris studied the evolution of European folk songs.",
#         "Local artisans in Paris preserved traditional folk crafts.",
#         "The spirit of folk storytelling is alive in Paris's literary circles.",
#         "Paris's cultural centers host workshops on various folk practices."
#     ],
    
#     # Feature1 High + Feature2 Low: Paris + Modern, Commercial, Elite
#     "F1_high_F2_low": [
#         "A modern film premiered in cinemas across Paris.",
#         "The Parisian author published her commercial best-seller.",
#         "An elite fashion show captivated audiences in Paris last week.",
#         "The modern art gallery in Paris showcased its new collection.",
#         "Numerous commercial digital releases originated from galleries in Paris.",
#         "The elite publishing house in Paris launched a series of high-profile novels.",
#         "Concert organizers in Paris announced a highly anticipated modern opera.",
#         "The commercial game developer based in Paris unveiled their AAA title.",
#         "A collection of elite historical artifacts was displayed to the public in Paris.",
#         "The French government unveiled modern urban development plans for Paris."
#     ],
    
#     # Feature1 Low + Feature2 High: @-@ + Folk
#     "F1_low_F2_high": [
#         "The band played folk music from 7@-@9 PM at the local fair.",
#         "The ancient folk tales were passed down through generations 1@-@5.",
#         "A study on folk culture surveyed individuals aged 18@-@35.",
#         "The exhibition featured folk traditions from regions A@-@Z.",
#         "Ethnomusicologists collected folk songs from periods 1900@-@1950.",
#         "The community group organized a folk dance workshop for children 6@-@10.",
#         "Historical records show folk practices between 1400@-@1600 AD.",
#         "The article discussed the impact of globalization on folk art forms 1@-@20.",
#         "Audience members enjoyed folk stories told from 8@-@10 PM.",
#         "The collection included folk instruments from countries 3@-@7."
#     ],
    
#     # Feature1 Low + Feature2 Low: @-@ + Modern, Commercial, Elite
#     "F1_low_F2_low": [
#         "The modern software update was deployed on servers 1@-@5 between 2@-@4 AM.",
#         "New elite academic journals were published online from subjects A@-@Z.",
#         "The commercial quarterly report was issued, covering figures from Q1@-@Q2.",
#         "Version 2.0 of the modern application was distributed to users 100@-@500.",
#         "Commercial data sets were made available for public access between dates 2020@-@2023.",
#         "Elite security protocols were implemented for systems running OS versions 7@-@10.",
#         "The modern official announcement was broadcast via channels 1@-@3 at 9@-@15 AM.",
#         "Commercial earnings statements were issued for companies with IDs 123@-@456.",
#         "The elite updated guidelines were disseminated in stages 1@-@4.",
#         "Modern patches were installed to optimize modules 8@-@12."
#     ]
# }

class SparseAutoencoder(torch.nn.Module):
    """Simple Sparse Autoencoder module."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Linear(hidden_dim, input_dim, bias=False)

    def forward(self, x):
        features = self.encoder(x)
        reconstruction = self.decoder(features)
        return features, reconstruction

class NeuralFactorizationModel(torch.nn.Module):
    """Neural Factorization Model for analyzing feature interactions."""
    def __init__(self, num_features, k_dim, output_dim):
        super().__init__()
        # This is actually an embedding layer, not a linear layer
        self.feature_embeddings = torch.nn.Embedding(num_features, k_dim)
        self.linear = torch.nn.Linear(num_features, output_dim)
        # Match the checkpoint layer structure (layer 1 and 3, not 0 and 2)
        self.interaction_mlp = torch.nn.Sequential(
            torch.nn.Identity(),  # Layer 0 - placeholder
            torch.nn.Linear(k_dim, k_dim),  # Layer 1
            torch.nn.ReLU(),  # Layer 2  
            torch.nn.Linear(k_dim, output_dim)  # Layer 3
        )
    
    def forward(self, x):
        # Linear component
        linear_out = self.linear(x)
        
        # Interaction component - need to handle the embedding differently
        # Since we have continuous SAE activations, we need to multiply embeddings by activations
        # Get embeddings for all features and weight by activations
        embeddings = self.feature_embeddings.weight.T  # [k_dim, num_features]
        weighted_embeddings = torch.matmul(x, embeddings.T)  # [batch, k_dim]
        interaction_out = self.interaction_mlp(weighted_embeddings)
        
        return linear_out + interaction_out, linear_out, interaction_out

class StimulusResponseAnalyzer:
    """Analyzer for measuring SAE and NFM responses to controlled stimuli."""
    
    def __init__(self, sae_model, nfm_model, tokenizer, base_model, device="cuda", target_layer=16):
        self.sae_model = sae_model
        self.nfm_model = nfm_model
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.device = device
        self.target_layer = target_layer
        
        # Set models to eval mode
        self.sae_model.eval()
        self.nfm_model.eval()
        self.base_model.eval()
    
    def measure_activations(self, texts, feature_indices, batch_size=16):
        """
        Measure SAE activations for given texts and features.
        
        Returns:
            Dict with feature_idx -> list of max activations per text
        """
        feature_activations = {idx: [] for idx in feature_indices}
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Measuring SAE activations"):
            batch_texts = texts[i:i+batch_size]
            if not batch_texts:
                continue
                
            # Tokenize
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True,
                                   truncation=True, max_length=100).to(self.device)
            
            with torch.no_grad():
                # Get hidden states
                outputs = self.base_model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[self.target_layer]
                
                # Process through SAE
                features, _ = self.sae_model(hidden_states.to(self.sae_model.encoder[0].weight.dtype))
                
                # Extract activations for each text
                for b in range(features.shape[0]):
                    seq_len = torch.sum(inputs["attention_mask"][b]).item()
                    if seq_len > 0:
                        for feature_idx in feature_indices:
                            # Get max activation for this feature in this text
                            max_activation = torch.max(features[b, :seq_len, feature_idx]).item()
                            feature_activations[feature_idx].append(max_activation)
        
        return feature_activations
    
    def measure_nfm_contributions(self, sae_activations, feature_indices):
        """
        Measure NFM linear and interaction contributions by actually running the model components.
        Also measure embedding layer contributions.
        
        Args:
            sae_activations: Dict of feature_idx -> list of activations
            feature_indices: List of feature indices to analyze
            
        Returns:
            Tuple of (linear_contributions, interaction_contributions, embedding_contributions)
        """
        # Create feature vectors from SAE activations
        num_texts = len(next(iter(sae_activations.values())))
        num_features = self.sae_model.encoder[0].out_features
        
        # Per-feature measurements (same as before)
        linear_contributions = {idx: [] for idx in feature_indices}
        interaction_contributions = {idx: [] for idx in feature_indices}
        
        # Per-text measurements (not indexed by feature)
        top_k_embedding = []
        all_k_embedding = []
        random_k_embedding = []
        
        # Find the top K dimension that both features contribute to most strongly
        top_k_dim = 140  # From your previous analysis
        random_k_dim = 13  # Random dimension for comparison
        
        for text_idx in range(num_texts):
            # Create sparse feature vector for this text
            feature_vector = torch.zeros(num_features, device=self.device)
            for feature_idx, activations in sae_activations.items():
                if text_idx < len(activations):
                    feature_vector[feature_idx] = activations[text_idx]
            
            feature_vector = feature_vector.unsqueeze(0)  # Add batch dimension
            
            with torch.no_grad():
                # Get NFM outputs - this actually runs the components
                total_out, linear_out, interaction_out = self.nfm_model(feature_vector)
                
                # Get embedding layer activations
                embeddings = self.nfm_model.feature_embeddings.weight.T  # [k_dim, num_features]
                weighted_embeddings = torch.matmul(feature_vector, embeddings.T)  # [batch, k_dim]
                
                # DEBUG: Print embedding analysis (only once per condition)
                if text_idx == 0:
                    print(f"\n=== DEBUG INFO ===")
                    print(f"feature_vector shape: {feature_vector.shape}")
                    print(f"feature_vector non-zero count: {torch.count_nonzero(feature_vector).item()}")
                    print(f"feature_vector non-zero indices: {torch.nonzero(feature_vector).flatten().tolist()}")
                    print(f"embeddings shape: {embeddings.shape}")
                    print(f"weighted_embeddings shape: {weighted_embeddings.shape}")
                    print(f"weighted_embeddings mean: {weighted_embeddings.mean().item():.6f}")
                    print(f"weighted_embeddings std: {weighted_embeddings.std().item():.6f}")
                    print(f"weighted_embeddings min: {weighted_embeddings.min().item():.6f}")
                    print(f"weighted_embeddings max: {weighted_embeddings.max().item():.6f}")
                    print(f"weighted_embeddings[0, 140]: {weighted_embeddings[0, 140].item():.6f}")
                    print(f"weighted_embeddings[0, 13]: {weighted_embeddings[0, 13].item():.6f}")
                    print(f"torch.abs(weighted_embeddings).mean(): {torch.abs(weighted_embeddings).mean().item():.6f}")
                    print(f"abs(weighted_embeddings[0, 140]): {abs(weighted_embeddings[0, 140].item()):.6f}")
                    print(f"abs(weighted_embeddings[0, 13]): {abs(weighted_embeddings[0, 13].item()):.6f}")
                
                # Per-text embedding measurements (stored once per text, not per feature)
                top_k_contrib = weighted_embeddings[0, top_k_dim].item()
                top_k_embedding.append(abs(top_k_contrib))
                
                all_k_contrib = torch.abs(weighted_embeddings).mean().item()
                all_k_embedding.append(all_k_contrib)
                
                random_k_contrib = weighted_embeddings[0, random_k_dim].item()
                random_k_embedding.append(abs(random_k_contrib))
                
                # Per-feature measurements (for linear and interaction outputs)
                for feature_idx in feature_indices:
                    # Linear contribution: the actual linear output magnitude
                    linear_contrib = torch.abs(linear_out).mean().item()
                    linear_contributions[feature_idx].append(linear_contrib)
                    
                    # Interaction contribution: the actual interaction MLP output magnitude
                    interaction_contrib = torch.abs(interaction_out).mean().item()
                    interaction_contributions[feature_idx].append(interaction_contrib)
                
                # DEBUG: Print what gets stored (only for first text)
                if text_idx == 0:
                    print(f"\nPer-text embedding values stored:")
                    print(f"  top_k_embedding: {abs(top_k_contrib):.6f}")
                    print(f"  all_k_embedding: {all_k_contrib:.6f}")
                    print(f"  random_k_embedding: {abs(random_k_contrib):.6f}")
                    
                    print(f"\nPer-feature values (same for both features):")
                    for feature_idx in feature_indices:
                        print(f"  Feature {feature_idx} linear: {torch.abs(linear_out).mean().item():.6f}")
                        print(f"  Feature {feature_idx} interaction: {torch.abs(interaction_out).mean().item():.6f}")
        
        return linear_contributions, interaction_contributions, top_k_embedding, all_k_embedding, random_k_embedding

def create_analysis_plots(results, feature_indices, output_dir):
    """Create visualization plots for the stimulus-response analysis."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create first figure: Neural Network Layer Outputs
    fig1, axes1 = plt.subplots(3, 2, figsize=(15, 12))
    fig1.suptitle('SAE Feature Analysis - Neural Network Layer Outputs', fontsize=16, fontweight='bold')
    
    conditions = ['F1_high_F2_high', 'F1_high_F2_low', 'F1_low_F2_high', 'F1_low_F2_low']
    condition_labels = ['[1,1] '+feature1name+'+'+feature2name+'', '[1,0] '+feature1name+'+'+feature2bname, '[0,1] '+feature1bname+'+'+feature2name+'', '[0,0] '+feature1bname+'+'+feature2bname]
    
    feature_labels = [f'Feature {idx}' for idx in feature_indices]
    
    metrics = ['sae_activations', 'linear_contributions', 'interaction_contributions']
    metric_titles = ['SAE Activations', 'NFM Linear Layer Output', 'NFM Interaction MLP Output']
    
    for metric_idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
        # Left plot: Interaction conditions
        ax_left = axes1[metric_idx, 0]
        
        # Prepare data for interaction conditions
        interaction_data = []
        for cond in conditions:
            for feat_idx in feature_indices:
                values = results[cond][metric][feat_idx]
                for val in values:
                    interaction_data.append({
                        'Condition': condition_labels[conditions.index(cond)],
                        'Feature': f'Feature {feat_idx}',
                        'Value': val
                    })
        
        df_interaction = pd.DataFrame(interaction_data)
        
        # Create box plot for interaction conditions
        sns.boxplot(data=df_interaction, x='Condition', y='Value', hue='Feature', ax=ax_left)
        ax_left.set_title(f'{title} - Interaction Conditions')
        ax_left.set_xlabel('Condition')
        ax_left.set_ylabel('Value')
        ax_left.tick_params(axis='x', rotation=45)
        
        # Right plot: Feature categories (collapsed)
        ax_right = axes1[metric_idx, 1]
        
        # Prepare data for feature categories
        feature_data = []
        for feat_idx in feature_indices:
            # For Feature 32806 (Paris): High = F1_high_*, Low = F1_low_*
            # For Feature 23360 (Folk): High = *_F2_high, Low = *_F2_low
            if feat_idx == feature1ID:  # Paris feature
                high_conditions = ['F1_high_F2_high', 'F1_high_F2_low']
                low_conditions = ['F1_low_F2_high', 'F1_low_F2_low']
            else:  # Folk feature
                high_conditions = ['F1_high_F2_high', 'F1_low_F2_high']
                low_conditions = ['F1_high_F2_low', 'F1_low_F2_low']
            
            # Collect high values
            high_values = []
            for cond in high_conditions:
                high_values.extend(results[cond][metric][feat_idx])
            
            # Collect low values  
            low_values = []
            for cond in low_conditions:
                low_values.extend(results[cond][metric][feat_idx])
            
            # Add to data
            for val in high_values:
                feature_data.append({
                    'Feature': f'Feature {feat_idx}',
                    'Level': 'High',
                    'Value': val
                })
            for val in low_values:
                feature_data.append({
                    'Feature': f'Feature {feat_idx}',
                    'Level': 'Low', 
                    'Value': val
                })
        
        df_feature = pd.DataFrame(feature_data)
        
        # Create box plot for feature categories
        sns.boxplot(data=df_feature, x='Feature', y='Value', hue='Level', ax=ax_right)
        ax_right.set_title(f'{title} - Feature Categories')
        ax_right.set_xlabel('Feature')
        ax_right.set_ylabel('Value')
    
    plt.tight_layout()
    
    # Save first plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plot1_path = output_path / "neural_network_outputs.png"
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Neural network outputs plot saved to: {plot1_path}")
    
    # Create second figure: NFM Embedding Layer Analysis
    fig2, axes2 = plt.subplots(3, 2, figsize=(15, 12))
    fig2.suptitle('SAE Feature Analysis - NFM Embedding Layer', fontsize=16, fontweight='bold')
    
    # Three rows: Top K, All K, Random K
    embedding_metrics = ['top_k_embedding', 'all_k_embedding', 'random_k_embedding']
    embedding_titles = ['Top K Dimension 140 (Highest Combined Weight)', 'All K Dimensions (Mean)', 'Random K Dimension 13']
    
    for metric_idx, (metric, title) in enumerate(zip(embedding_metrics, embedding_titles)):
        # Left plot: Interaction conditions
        ax_left = axes2[metric_idx, 0]
        
        # Prepare data for interaction conditions - FIXED: per-text data, not per-feature
        interaction_data = []
        for cond in conditions:
            values = results[cond][metric]  # This is now a list, not a dict
            for val in values:
                interaction_data.append({
                    'Condition': condition_labels[conditions.index(cond)],
                    'Value': val
                })
        
        df_interaction = pd.DataFrame(interaction_data)
        
        # Create box plot for interaction conditions - no hue since it's per-text
        sns.boxplot(data=df_interaction, x='Condition', y='Value', ax=ax_left)
        ax_left.set_title(f'{title} - Interaction Conditions')
        ax_left.set_xlabel('Condition')
        ax_left.set_ylabel('Value')
        ax_left.tick_params(axis='x', rotation=45)
        
        # Right plot: Feature categories (collapsed) - using text indices to map to feature conditions
        ax_right = axes2[metric_idx, 1]
        
        # For embedding data, we need to map text indices back to feature conditions
        feature_data = []
        
        # High Feature 1 (Paris): F1_high_* conditions
        high_f1_values = []
        high_f1_values.extend(results['F1_high_F2_high'][metric])
        high_f1_values.extend(results['F1_high_F2_low'][metric])
        
        # Low Feature 1 (Paris): F1_low_* conditions  
        low_f1_values = []
        low_f1_values.extend(results['F1_low_F2_high'][metric])
        low_f1_values.extend(results['F1_low_F2_low'][metric])
        
        # High Feature 2 (Folk): *_F2_high conditions
        high_f2_values = []
        high_f2_values.extend(results['F1_high_F2_high'][metric])
        high_f2_values.extend(results['F1_low_F2_high'][metric])
        
        # Low Feature 2 (Folk): *_F2_low conditions
        low_f2_values = []
        low_f2_values.extend(results['F1_high_F2_low'][metric])
        low_f2_values.extend(results['F1_low_F2_low'][metric])
        
        # Add to data
        for val in high_f1_values:
            feature_data.append({'Feature': 'Feature '+str(feature1ID), 'Level': 'High', 'Value': val})
        for val in low_f1_values:
            feature_data.append({'Feature': 'Feature '+str(feature1ID), 'Level': 'Low', 'Value': val})
        for val in high_f2_values:
            feature_data.append({'Feature': 'Feature '+str(feature2ID), 'Level': 'High', 'Value': val})
        for val in low_f2_values:
            feature_data.append({'Feature': 'Feature '+str(feature2ID), 'Level': 'Low', 'Value': val})
        
        df_feature = pd.DataFrame(feature_data)
        
        # Create box plot for feature categories
        sns.boxplot(data=df_feature, x='Feature', y='Value', hue='Level', ax=ax_right)
        ax_right.set_title(f'{title} - Feature Categories')
        ax_right.set_xlabel('Feature')
        ax_right.set_ylabel('Value')
    
    plt.tight_layout()
    
    # Save second plot
    plot2_path = output_path / "nfm_embedding_layer.png"
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"NFM embedding layer plot saved to: {plot2_path}")

def print_summary_statistics(results, feature_indices):
    """Print detailed summary statistics for all conditions."""
    
    conditions = ['F1_high_F2_high', 'F1_high_F2_low', 'F1_low_F2_high', 'F1_low_F2_low']
    condition_names = ['[1,1] '+feature1name+'+' + feature2name, '[1,0] '+feature1name+'+'+feature2bname, '[0,1] '+feature1bname+'+'+feature2name, '[0,0] '+feature1bname+'+'+feature2bname]
    metrics = ['sae_activations', 'linear_contributions', 'interaction_contributions']
    metric_names = ['SAE Activations', 'Linear Contributions', 'Interaction Contributions']
    
    print("\n" + "="*80)
    print("DETAILED SUMMARY STATISTICS")
    print("="*80)
    
    for metric, metric_name in zip(metrics, metric_names):
        print(f"\n{metric_name}:")
        print("-" * 50)
        
        for cond, cond_name in zip(conditions, condition_names):
            print(f"\n{cond_name}:")
            for feat_idx in feature_indices:
                values = results[cond][metric][feat_idx]
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                
                feature_name = feature1name if feat_idx == feature1ID else feature2name
                print(f"  Feature {feat_idx} ({feature_name}): "
                      f"{mean_val:.4f} ± {std_val:.4f} "
                      f"(range: {min_val:.4f}-{max_val:.4f})")

def main():
    parser = argparse.ArgumentParser(description="Stimulus-Response Analysis for SAE Features")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base LLM model")
    parser.add_argument("--sae_path", type=str, required=True, help="Path to trained SAE model")
    parser.add_argument("--nfm_path", type=str, required=True, help="Path to trained NFM model")
    parser.add_argument("--output_dir", type=str, default="./stimulus_response_results", 
                        help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Feature indices we're analyzing
    feature_indices = [feature1ID, feature2ID]  # Paris and Folk features
    
    print("Loading models...")
    
    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")
    
    # Load SAE model
    sae_state_dict = torch.load(args.sae_path, map_location=args.device)
    if 'decoder.weight' in sae_state_dict:
        input_dim = sae_state_dict['decoder.weight'].shape[0]
        hidden_dim = sae_state_dict['decoder.weight'].shape[1]
    else:
        encoder_weight = sae_state_dict['encoder.0.weight']
        hidden_dim, input_dim = encoder_weight.shape
    
    sae_model = SparseAutoencoder(input_dim, hidden_dim)
    sae_model.load_state_dict(sae_state_dict)
    sae_model.to(args.device)
    
    # Load NFM model
    nfm_state_dict = torch.load(args.nfm_path, map_location=args.device)
    
    # Infer NFM dimensions
    num_features = nfm_state_dict['feature_embeddings.weight'].shape[0]
    k_dim = nfm_state_dict['feature_embeddings.weight'].shape[1]
    output_dim = nfm_state_dict['linear.weight'].shape[0]
    
    nfm_model = NeuralFactorizationModel(num_features, k_dim, output_dim)
    nfm_model.load_state_dict(nfm_state_dict)
    nfm_model.to(args.device)
    
    # Initialize analyzer
    analyzer = StimulusResponseAnalyzer(sae_model, nfm_model, tokenizer, base_model, args.device)
    
    print("Running stimulus-response analysis...")
    
    # Store results
    results = {}
    
    # Process each stimulus condition
    for condition_name, texts in STIMULUS_SETS.items():
        print(f"\nProcessing condition: {condition_name}")
        
        # Measure SAE activations
        sae_activations = analyzer.measure_activations(texts, feature_indices, args.batch_size)
        
        # Measure NFM contributions
        linear_contributions, interaction_contributions, top_k_embedding, all_k_embedding, random_k_embedding = analyzer.measure_nfm_contributions(
            sae_activations, feature_indices
        )
        
        # Store results
        results[condition_name] = {
            'sae_activations': sae_activations,
            'linear_contributions': linear_contributions,
            'interaction_contributions': interaction_contributions,
            'top_k_embedding': top_k_embedding,
            'all_k_embedding': all_k_embedding,
            'random_k_embedding': random_k_embedding
        }
    
    # Print detailed summary statistics
    print_summary_statistics(results, feature_indices)
    
    # Create visualization plots
    print("\nCreating analysis plots...")
    create_analysis_plots(results, feature_indices, args.output_dir)
    
    # Save detailed results
    results_path = Path(args.output_dir) / "detailed_results.json"
    
    # Convert results to JSON-serializable format
    json_results = {}
    for condition_name, condition_data in results.items():
        json_results[condition_name] = {}
        for metric_name, metric_values in condition_data.items():
            if isinstance(metric_values, dict):
                json_results[condition_name][metric_name] = {str(k): [float(v) for v in val_list]
                                                               for k, val_list in metric_values.items()}
            elif isinstance(metric_values, list):
                json_results[condition_name][metric_name] = [float(v) for v in metric_values]
            # No 'else' needed if all your metrics are either dicts or lists containing floats.

                #json_results[condition][metric][str(feat_idx)] = [float(v) for v in values]
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_path}")
    print("Analysis complete!")

if __name__ == "__main__":
    main()