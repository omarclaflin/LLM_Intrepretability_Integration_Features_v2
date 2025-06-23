"""
NFM Secondary SAE Analysis Script with Comprehensive Visualizations - TOPK VERSION

This script extends the NFM SAE analysis to include comprehensive histograms and bar charts
for all secondary features across all analyses. Provides complete visual overview of:

1. Secondary feature activation strengths for each primary feature
2. Primary feature contribution patterns  
3. Condition-based activation differences
4. ANOVA sensitivity across all features (with p<0.05 significance line)

All visualizations show the full feature space to reveal activation patterns and sparsity.
"""

import torch
import numpy as np
import argparse
from pathlib import Path
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
import logging
from scipy import stats
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

# Expanded stimulus sets with consistent token lengths across ALL categories (50 items each, ~10-12 tokens per item)
STIMULUS_SETS = {
    # Feature1 High (High Formality) + Feature2 High (High Emotion)
    "F1_high_F2_high": [
        "We profoundly regret the devastating error that has tarnished our reputation.",
        "It is with immense sorrow that I convey this tragic news.",
        "My heart is utterly broken by the egregious injustice perpetrated here.",
        "I hereby express my profound disgust at these deplorable findings today.",
        "The egregious suffering endured by citizens demands an immediate passionate response.",
        "With gravest apprehension, we anticipate the calamitous repercussions of this decision.",
        "An overwhelming sense of despair permeates the entire community currently.",
        "Our collective indignation at this flagrant disregard for truth is unyielding.",
        "I am compelled to voice my deepest anguish regarding this loss.",
        "The sheer terror of the unfolding catastrophe is beyond comprehension.",
        "We express our most sincere condolences for the devastating family loss.",
        "The profound injustice witnessed today fills me with insurmountable rage completely.",
        "With utmost reverence, we commemorate the heroic sacrifice of these souls.",
        "The excruciating pain of this betrayal cuts deeper than any wound.",
        "We are deeply mortified by the appalling negligence leading to tragedy.",
        "The overwhelming joy of this momentous achievement brings tears to eyes.",
        "I am absolutely devastated by the unconscionable cruelty displayed in proceedings.",
        "Our hearts overflow with gratitude for the extraordinary compassion you demonstrated.",
        "The horrific nature of these crimes demands swift and uncompromising justice.",
        "We are profoundly moved by the remarkable courage exhibited under circumstances.",
        "The intolerable anguish of separation from loved ones weighs heavily here.",
        "Our deepest sympathies extend to all those affected by this disaster.",
        "The magnificent triumph achieved today will be remembered for generations ahead.",
        "We are utterly appalled by the systematic oppression inflicted upon communities.",
        "The breathtaking beauty of this sacrifice illuminates the darkness of times.",
        "Our hearts are shattered by the senseless violence tearing families apart.",
        "The extraordinary resilience displayed by survivors inspires hope amidst overwhelming despair.",
        "We categorically condemn the heinous atrocities committed against innocent people here.",
        "The overwhelming gratitude we feel cannot adequately express our profound appreciation.",
        "Our souls are tormented by the knowledge of such preventable suffering.",
        "The triumphant victory achieved through sacrifice will never be forgotten completely.",
        "We are consumed with righteous fury at this blatant violation of rights.",
        "The exquisite agony of loss transforms grief into something almost sacred.",
        "Our hearts burst with pride witnessing such extraordinary acts of courage.",
        "The devastating impact of this betrayal will reverberate through future generations.",
        "We are eternally grateful for the supreme sacrifice made on behalf.",
        "The unspeakable horror of these events defies all rational human comprehension.",
        "Our collective mourning unites us in shared sorrow and common purpose.",
        "The brilliant triumph of justice brings hope to the darkest corners.",
        "We are profoundly disturbed by the systematic dehumanization we have witnessed.",
        "The overwhelming relief felt by families brings tears of pure joy.",
        "Our hearts ache with the unbearable weight of collective moral responsibility.",
        "The magnificent courage displayed inspires us all to achieve greater heights.",
        "We are devastated by the cruel indifference shown to human suffering.",
        "The extraordinary love demonstrated transcends all earthly boundaries and physical limitations.",
        "Our righteous indignation at this injustice burns like an eternal flame.",
        "The breathtaking sacrifice made will be honored throughout all of history.",
        "We are overcome with emotion at this unprecedented display of humanity.",
        "The terrifying implications of inaction weigh heavily upon our moral consciences.",
        "Our hearts overflow with compassion for all those who suffer needlessly."
    ],
    
    # Feature1 High (High Formality) + Feature2 Low (Low Emotion)
    "F1_high_F2_low": [
        "The aforementioned data substantiates the initial hypothesis with reasonable statistical confidence.",
        "It is imperative to review the procedural guidelines prior to implementation.",
        "The revised protocol stipulates adherence to stringent safety regulations throughout process.",
        "A comprehensive analysis of statistical variances was subsequently conducted by team.",
        "The committee's deliberations concluded without reaching a unanimous consensus on matter.",
        "This document delineates the parameters for forthcoming operational adjustments within department.",
        "The findings corroborate the established theoretical framework with acceptable precision levels.",
        "Please transmit the requisite documentation to the appropriate department in course.",
        "The stipulated deadlines must be strictly observed to ensure project continuity.",
        "Subsequent to evaluation, modifications to the existing infrastructure are recommended here.",
        "The methodology employed demonstrates adequate adherence to established research protocol standards.",
        "Pursuant to regulatory requirements, all documentation must be submitted by deadline.",
        "The analytical framework provides sufficient foundation for the proposed recommendations made.",
        "Implementation of these measures will require coordination across multiple organizational units.",
        "The assessment indicates satisfactory compliance with institutional standards and expectations currently.",
        "Preliminary observations suggest the need for additional data collection and analysis.",
        "The specified criteria have been met according to predetermined benchmark standards.",
        "Authorization for the proposed modifications must be obtained from appropriate authorities.",
        "The documentation reflects adherence to standard operating procedures throughout the process.",
        "Consultation with relevant stakeholders will be required before proceeding with implementation.",
        "The evaluation process has been completed in accordance with established guidelines.",
        "Distribution of resources will be allocated based on predetermined priority classifications.",
        "The systematic approach ensures consistency with organizational objectives and requirements established.",
        "Coordination between departments will facilitate efficient completion of assigned tasks here.",
        "The assessment protocol has been designed to meet institutional accreditation standards.",
        "Implementation will proceed according to the timeline established in preliminary planning.",
        "The documentation demonstrates compliance with regulatory standards and professional expectations currently.",
        "Evaluation of outcomes will be conducted using established measurement criteria systematically.",
        "The proposed framework aligns with institutional policies and procedural requirements established.",
        "Authorization protocols must be followed to ensure appropriate oversight and accountability.",
        "The methodology incorporates best practices from relevant professional literature and experience.",
        "Systematic documentation will be maintained throughout all phases of the process.",
        "The evaluation criteria reflect established standards for quality assurance and control.",
        "Implementation will require coordination with external agencies and regulatory bodies here.",
        "The assessment framework provides adequate foundation for informed decision-making processes currently.",
        "Documentation standards must be maintained to ensure compliance with archival requirements.",
        "The systematic approach facilitates efficient allocation of resources and personnel available.",
        "Evaluation protocols have been established to monitor progress and identify issues.",
        "The procedural framework ensures consistency with organizational policies and objectives established.",
        "Implementation will proceed in phases to minimize disruption to ongoing operations.",
        "The documentation reflects consideration of relevant factors and stakeholder input received.",
        "Assessment criteria have been developed to ensure objective evaluation of outcomes.",
        "The methodology incorporates standard practices for quality control and assurance measures.",
        "Coordination mechanisms will be established to facilitate communication between departments effectively.",
        "The evaluation framework provides structure for systematic assessment of results obtained.",
        "Implementation procedures have been designed to minimize risk and ensure compliance.",
        "The documentation standards reflect institutional requirements for record-keeping and accountability measures.",
        "Assessment protocols will be applied consistently across all relevant organizational units.",
        "The systematic approach ensures alignment with strategic objectives and operational requirements.",
        "Implementation will be monitored to ensure adherence to established timelines systematically."
    ],
    
    # Feature1 Low (Low Formality) + Feature2 High (High Emotion)  
    "F1_low_F2_high": [
        "OMG, I'm so hyped about that! This is seriously the best news!",
        "I literally can't even right now, this is just way too much!",
        "Dude, that was absolutely insane! I'm totally freaking out about it right!",
        "Seriously, I'm so mad about this I could just scream loudly!",
        "Holy cow, that's absolutely devastating! I feel so awful for them!",
        "I'm so thrilled about this, I might just cry the happiest tears!",
        "No way! This is like, completely mind-blowingly amazing and I can't believe!",
        "Ugh, I'm just so done with all this, it's making me furious!",
        "My heart just aches for everyone involved, it's such a tragic situation.",
        "I'm bouncing off the walls with excitement, you seriously have no idea!",
        "This is blowing my mind! I can't handle how awesome this thing is!",
        "I'm crying tears of joy right now, this is the most beautiful!",
        "Honestly, I'm so pissed off I can barely see straight right now!",
        "This is breaking my heart into a million tiny pieces, I'm totally devastated!",
        "I'm literally shaking with excitement! This is the coolest thing that happened!",
        "I could punch a wall right now, this whole thing makes me livid!",
        "Oh my gosh, I'm so happy I could burst! This is incredible news!",
        "This is making me sick to my stomach, I'm so disgusted right!",
        "I'm over the moon about this! Best day of my entire life!",
        "I'm so heartbroken I can't stop crying, this is absolutely terrible right!",
        "This is giving me chills! I'm so pumped up I could run!",
        "I'm seething with rage! This is the most infuriating thing I've ever seen!",
        "My heart is melting! This is the sweetest, most touching thing ever!",
        "I'm so frustrated I could scream! This is driving me absolutely crazy!",
        "This is making me giddy with excitement! I can't contain my joy!",
        "I'm boiling with anger! This is completely unacceptable and I'm totally furious!",
        "I'm glowing with happiness! This news has made my entire week perfect!",
        "This is crushing my soul! I'm in complete despair about this situation!",
        "I'm jumping for joy! This is the most exciting thing that's happened!",
        "I'm burning with indignation! This injustice is making my blood boil completely!",
        "This is filling me with pure bliss! I've never been happier in!",
        "I'm drowning in sorrow! This tragedy is tearing my heart apart completely!",
        "I'm electric with anticipation! This is going to be absolutely phenomenal and!",
        "I'm consumed with fury! This outrage is making me see red right!",
        "This is warming my heart! Such a beautiful moment that brings tears!",
        "I'm sick with worry! This terrifying situation is keeping me awake nights!",
        "I'm buzzing with energy! This fantastic news has me completely energized and!",
        "I'm choking with emotion! This heartbreaking story is overwhelming my senses completely!",
        "This is lighting me up! I'm radiating happiness and positive energy right!",
        "I'm trembling with fear! This scary situation is absolutely terrifying me completely!",
        "I'm beaming with pride! This achievement is making me glow with satisfaction!",
        "I'm aching with sympathy! This sad story is tugging at my heartstrings!",
        "This is energizing my spirit! I feel like I could conquer the world!",
        "I'm writhing in agony! This painful news is torturing my poor soul!",
        "I'm soaring with elation! This wonderful surprise has lifted my spirits high!",
        "I'm wilting with disappointment! This letdown is crushing all my hopes completely!",
        "This is intoxicating my senses! I'm drunk on happiness and pure euphoria!",
        "I'm withering with shame! This embarrassing situation is mortifying me completely today!",
        "I'm glowing with contentment! This peaceful moment is filling me with serenity!",
        "I'm collapsing with exhaustion! This stressful ordeal is draining all my energy!"
    ],
    
    # Feature1 Low (Low Formality) + Feature2 Low (Low Emotion)
    "F1_low_F2_low": [
        "Yeah, the light switch is on the wall over there by door.",
        "It's raining outside today. What's up with you these days?",
        "He's over there somewhere, I think near the back of room.",
        "I gotta go now, catch ya later. Bye for now everyone.",
        "The cat's asleep on the couch in the living room right now.",
        "Cool, got it. Thanks for letting me know about that thing.",
        "Just chilling at home, nothing much going on today around here.",
        "The food's in the fridge if you want to grab something good.",
        "It's kinda quiet around here today, not much happening at all.",
        "He said okay, I guess that works for everyone involved here.",
        "The meeting's at three, don't forget to bring your notes with.",
        "She's running a bit late but should be here soon enough.",
        "The weather's okay, not too hot and not too cold either today.",
        "I'll check my schedule and get back to you later this afternoon.",
        "The store closes at nine, so we have plenty of time left.",
        "That book's on the shelf, second row from the top there.",
        "The bus comes every twenty minutes or so during rush hour.",
        "I heard the news, seems like things are going alright these days.",
        "The coffee's ready if you want a cup right now this morning.",
        "My phone's charging, I'll call you back in a few minutes here.",
        "The game starts at seven, we should probably leave soon enough.",
        "She mentioned something about changing the plan slightly for us all.",
        "The parking lot's pretty full, but there are spots available still.",
        "I think the answer's in the back of the book somewhere.",
        "The train was on time, so everything worked out fine today.",
        "He's bringing lunch, so we don't need to worry about food.",
        "The computer's running slow, might need to restart it soon enough.",
        "I saw the email, looks like everything's set for tomorrow's meeting.",
        "The keys are on the kitchen counter by the fruit bowl.",
        "She's working late tonight, won't be home until after eight o'clock.",
        "The movie's okay, nothing special but not bad either I guess.",
        "I'll pick up milk on the way home from work today.",
        "The dog needs a walk, probably in an hour or so.",
        "He's got the information we need for the project we're doing.",
        "The internet's working fine, no problems with the connection today at.",
        "I'll send the file over once I finish making the changes.",
        "The package should arrive sometime this week, maybe Thursday or Friday.",
        "She's making dinner, should be ready in about thirty minutes from.",
        "The remote's somewhere on the coffee table under those magazines there.",
        "I'll check the calendar and see what day works best for.",
        "The library's open until six, so we have time to stop by.",
        "He mentioned meeting up for lunch sometime next week maybe Tuesday.",
        "The temperature's supposed to drop a bit tonight according to weather.",
        "I'll grab some snacks for the trip on the way out.",
        "The printer's out of paper, there's more in the supply closet.",
        "She's taking the early flight, should land around ten in morning.",
        "The restaurant's pretty good, we've been there a few times before.",
        "I'll finish this up and then we can head out together.",
        "The music's a bit loud, but it's not bothering me much.",
        "He's got the right idea about how to handle this situation."
    ]
}

# TopK Sparse Autoencoder (from part1c)
class TopKSparseAutoencoder(torch.nn.Module):
    """TopK Sparse Autoencoder module."""
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
        sparse_features = self.apply_topk(features)
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

class NeuralFactorizationModel(torch.nn.Module):
    """Neural Factorization Model for analyzing feature interactions."""
    def __init__(self, num_features, k_dim, output_dim):
        super().__init__()
        self.feature_embeddings = torch.nn.Embedding(num_features, k_dim)
        self.linear = torch.nn.Linear(num_features, output_dim)
        self.interaction_mlp = torch.nn.Sequential(
            torch.nn.Identity(),  # Layer 0 - placeholder
            torch.nn.Linear(k_dim, k_dim),  # Layer 1
            torch.nn.ReLU(),  # Layer 2  
            torch.nn.Linear(k_dim, output_dim)  # Layer 3
        )
    
    def forward(self, x):
        # Linear component
        linear_out = self.linear(x)
        
        # Interaction component - this gives us the embedding layer
        embeddings = self.feature_embeddings.weight.T  # [k_dim, num_features]
        weighted_embeddings = torch.matmul(x, embeddings.T)  # [batch, k_dim]
        interaction_out = self.interaction_mlp(weighted_embeddings)
        
        return linear_out + interaction_out, linear_out, interaction_out, weighted_embeddings

class NFMSAEAnalyzer:
    """Analyzer for NFM Secondary SAE relationships using COMPLETE pipeline with TopK SAEs."""
    
    def __init__(self, primary_sae, nfm_model, secondary_sae, tokenizer, base_model, device="cuda", target_layer=16):
        self.primary_sae = primary_sae
        self.nfm_model = nfm_model
        self.secondary_sae = secondary_sae
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.device = device
        self.target_layer = target_layer
        
        # Set models to eval mode
        self.primary_sae.eval()
        self.nfm_model.eval()
        self.secondary_sae.eval()
        self.base_model.eval()

    def get_complete_pipeline_activations(self, texts, batch_size=16):
        """
        Process texts through the COMPLETE pipeline: 
        Layer 16 → Primary TopK SAE → [Primary Reconstruction + NFM Linear + NFM Interaction] → Secondary TopK SAE
        
        Returns:
            primary_activations: [num_texts, primary_sae_features] 
            primary_reconstructions: [num_texts, layer_dim]
            nfm_linear_outputs: [num_texts, layer_dim]
            nfm_interaction_outputs: [num_texts, layer_dim]
            secondary_activations: [num_texts, secondary_sae_features]
        """
        primary_activations = []
        primary_reconstructions = []
        nfm_linear_outputs = []
        nfm_interaction_outputs = []
        secondary_activations = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing complete TopK pipeline"):
            batch_texts = texts[i:i+batch_size]
            if not batch_texts:
                continue
                
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True,
                                truncation=True, max_length=100).to(self.device)
            
            with torch.no_grad():
                # Step 1: Layer 16 → Primary TopK SAE
                outputs = self.base_model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[self.target_layer]
                
                # RESHAPE FOR TOPK SAE: 3D → 2D
                batch_size_inner, seq_len, hidden_dim = hidden_states.shape
                hidden_states_reshaped = hidden_states.reshape(-1, hidden_dim)
                
                primary_features, primary_reconstruction = self.primary_sae(
                    hidden_states_reshaped.to(self.primary_sae.encoder[0].weight.dtype)
                )
                
                # RESHAPE BACK: 2D → 3D
                primary_features = primary_features.reshape(batch_size_inner, seq_len, -1)
                primary_reconstruction = primary_reconstruction.reshape(batch_size_inner, seq_len, -1)
                
                # Step 2: NFM Linear Pathway
                nfm_linear_output = self.nfm_model.linear(primary_features)
                
                # Step 3: NFM Interaction Pathway
                # 3a: Feature embeddings
                embeddings = self.nfm_model.feature_embeddings.weight.T  # [k_dim, num_features]
                weighted_embeddings = torch.matmul(primary_features, embeddings.T)  # [batch, seq, k_dim]
                
                # 3b: Interaction MLP Layer 1 [500 x 500]
                mlp_layer1_output = self.nfm_model.interaction_mlp[1](weighted_embeddings)
                
                # 3c: Secondary TopK SAE (AFTER MLP Layer 1, BEFORE ReLU)
                original_shape = mlp_layer1_output.shape
                mlp_layer1_flat = mlp_layer1_output.view(-1, original_shape[-1])
                
                secondary_features, _ = self.secondary_sae(
                    mlp_layer1_flat.to(self.secondary_sae.encoder[0].weight.dtype)
                )
                secondary_reconstruction = self.secondary_sae.decoder(secondary_features)
                secondary_reconstruction_reshaped = secondary_reconstruction.view(original_shape)
                
                # 3d: Continue through remaining interaction MLP layers
                # ReLU activation (Layer 2)
                relu_output = self.nfm_model.interaction_mlp[2](secondary_reconstruction_reshaped)
                # Final linear layer (Layer 3)
                nfm_interaction_output = self.nfm_model.interaction_mlp[3](relu_output)
                
                # Get mean activation across sequence for each text
                for b in range(primary_features.shape[0]):
                    seq_len_actual = torch.sum(inputs["attention_mask"][b]).item()
                    if seq_len_actual > 0:
                        # Store all pathway outputs
                        mean_primary_features = torch.mean(primary_features[b, :seq_len_actual, :], dim=0)
                        mean_primary_reconstruction = torch.mean(primary_reconstruction[b, :seq_len_actual, :], dim=0)
                        mean_nfm_linear = torch.mean(nfm_linear_output[b, :seq_len_actual, :], dim=0)
                        mean_nfm_interaction = torch.mean(nfm_interaction_output[b, :seq_len_actual, :], dim=0)
                        
                        # For secondary features, we need to compute the mean correctly
                        # since secondary_features is already flattened [batch*seq, features]
                        start_idx = b * seq_len_actual
                        end_idx = start_idx + seq_len_actual
                        if end_idx <= secondary_features.shape[0]:
                            mean_secondary_features = torch.mean(secondary_features[start_idx:end_idx, :], dim=0)
                        else:
                            # Fallback if indexing doesn't work
                            secondary_features_reshaped = secondary_features.view(batch_size_inner, seq_len, -1)
                            mean_secondary_features = torch.mean(secondary_features_reshaped[b, :seq_len_actual, :], dim=0)
                        
                        primary_activations.append(mean_primary_features.cpu().numpy())
                        primary_reconstructions.append(mean_primary_reconstruction.cpu().numpy())
                        nfm_linear_outputs.append(mean_nfm_linear.cpu().numpy())
                        nfm_interaction_outputs.append(mean_nfm_interaction.cpu().numpy())
                        secondary_activations.append(mean_secondary_features.cpu().numpy())
        
        return (np.array(primary_activations), np.array(primary_reconstructions), 
                np.array(nfm_linear_outputs), np.array(nfm_interaction_outputs), 
                np.array(secondary_activations))

    def analyze_all_secondary_features_for_primary(self, primary_feature_idx, output_dir):
        """
        Analysis 1 Visualization: Create histogram of ALL secondary feature activations for a single primary feature.
        Shows complete activation landscape and sparsity patterns.
        """
        print(f"\n=== ANALYSIS 1 VISUALIZATION: ALL SECONDARY FEATURE ACTIVATIONS FOR PRIMARY FEATURE {primary_feature_idx} ===")
        
        # Collect all texts for analysis
        all_texts = []
        for condition_texts in STIMULUS_SETS.values():
            all_texts.extend(condition_texts)
        
        # Get complete pipeline activations
        (primary_activations, primary_reconstructions, nfm_linear_outputs, 
         nfm_interaction_outputs, secondary_activations) = self.get_complete_pipeline_activations(all_texts)
        
        # Focus on the specific primary feature
        primary_feature_activations = primary_activations[:, primary_feature_idx]
        
        # For each text, isolate this primary feature and process through complete pipeline
        secondary_isolated_activations = []
        
        for text_idx in range(len(all_texts)):
            # Get primary feature activation for this text
            primary_activation = primary_feature_activations[text_idx]
            
            # Create a sparse vector with only this primary feature activated
            sparse_primary = np.zeros(primary_activations.shape[1])
            sparse_primary[primary_feature_idx] = primary_activation
            sparse_primary_tensor = torch.tensor(sparse_primary, device=self.device, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                # Process through complete NFM pipeline
                # Linear pathway
                linear_output = self.nfm_model.linear(sparse_primary_tensor)
                
                # Interaction pathway
                embeddings = self.nfm_model.feature_embeddings.weight.T
                weighted_embeddings = torch.matmul(sparse_primary_tensor, embeddings.T)
                
                # MLP Layer 1
                mlp_layer1_output = self.nfm_model.interaction_mlp[1](weighted_embeddings)
                
                # Secondary TopK SAE
                secondary_features, _ = self.secondary_sae(
                    mlp_layer1_output.to(self.secondary_sae.encoder[0].weight.dtype)
                )
                
                secondary_isolated_activations.append(secondary_features.squeeze().cpu().numpy())
        
        # Convert to array and find mean activation for each secondary feature
        secondary_isolated_activations = np.array(secondary_isolated_activations)
        mean_secondary_activations = np.mean(secondary_isolated_activations, axis=0)
        
        # Define threshold for "non-zero" (very small number)
        threshold = 1e-6
        
        # Create comprehensive histogram
        plt.figure(figsize=(15, 10))
        
        # Main histogram
        plt.subplot(2, 2, 1)
        plt.hist(mean_secondary_activations, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Mean Secondary Feature Activation')
        plt.ylabel('Number of Features')
        plt.title(f'Distribution of Secondary Feature Activations\nfor Primary Feature {primary_feature_idx}')
        plt.grid(True, alpha=0.3)
        
        # Bar chart of top 20 features
        plt.subplot(2, 2, 2)
        top_20_indices = np.argsort(mean_secondary_activations)[-20:][::-1]
        top_20_values = mean_secondary_activations[top_20_indices]
        bars = plt.bar(range(len(top_20_values)), top_20_values, color='coral')
        plt.xlabel('Top 20 Secondary Features (Rank)')
        plt.ylabel('Mean Activation')
        plt.title(f'Top 20 Secondary Features for Primary {primary_feature_idx}')
        plt.xticks(range(len(top_20_values)), [f'#{i}' for i in top_20_indices], rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, top_20_values)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8)
        
        # Log-scale histogram to see sparsity pattern
        plt.subplot(2, 2, 3)
        # Remove zeros for log scale
        non_zero_activations = mean_secondary_activations[mean_secondary_activations > threshold]
        if len(non_zero_activations) > 0:
            plt.hist(non_zero_activations, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
            plt.xlabel('Mean Secondary Feature Activation (Non-zero only)')
            plt.ylabel('Number of Features')
            plt.title(f'Non-zero Activations (Log Scale)\n{len(non_zero_activations)}/{len(mean_secondary_activations)} features active')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No non-zero activations', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('No Non-zero Activations')
        
        # Activation pattern across all features
        plt.subplot(2, 2, 4)
        plt.plot(mean_secondary_activations, 'o', markersize=2, alpha=0.6, color='purple')
        plt.xlabel('Secondary Feature Index')
        plt.ylabel('Mean Activation')
        plt.title(f'Activation Pattern Across All Secondary Features')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / f'analysis1_primary_{primary_feature_idx}_all_secondary_features.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # EXPORT NON-ZERO ONLY VERSION
        if len(non_zero_activations) > 0:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.hist(non_zero_activations, bins=40, alpha=0.7, color='lightgreen', edgecolor='black')
            plt.xlabel('Mean Secondary Feature Activation (Non-zero only)')
            plt.ylabel('Number of Features')
            plt.title(f'Non-zero Secondary Feature Activations\nfor Primary Feature {primary_feature_idx}\n({len(non_zero_activations)}/{len(mean_secondary_activations)} features)')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 2)
            plt.hist(non_zero_activations, bins=40, alpha=0.7, color='lightgreen', edgecolor='black')
            plt.xlabel('Mean Secondary Feature Activation (Non-zero only)')
            plt.ylabel('Number of Features (Log Scale)')
            plt.title(f'Non-zero Activations - Log Y Scale')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            
            # Box plot for distribution details
            plt.subplot(2, 2, 3)
            plt.boxplot(non_zero_activations, vert=True)
            plt.ylabel('Mean Activation')
            plt.title(f'Non-zero Activation Distribution\nMedian: {np.median(non_zero_activations):.6f}')
            plt.grid(True, alpha=0.3)
            
            # Statistics text
            plt.subplot(2, 2, 4)
            stats_text = f"""Non-zero Activation Statistics:
            
Count: {len(non_zero_activations)}/{len(mean_secondary_activations)}
Percentage: {(len(non_zero_activations)/len(mean_secondary_activations)*100):.2f}%

Min: {np.min(non_zero_activations):.6f}
Max: {np.max(non_zero_activations):.6f}
Mean: {np.mean(non_zero_activations):.6f}
Median: {np.median(non_zero_activations):.6f}
Std: {np.std(non_zero_activations):.6f}

25th percentile: {np.percentile(non_zero_activations, 25):.6f}
75th percentile: {np.percentile(non_zero_activations, 75):.6f}
95th percentile: {np.percentile(non_zero_activations, 95):.6f}"""
            
            plt.text(0.1, 0.5, stats_text, transform=plt.gca().transAxes, fontsize=10, 
                    verticalalignment='center', fontfamily='monospace')
            plt.axis('off')
            
            plt.tight_layout()
            output_path_nonzero = output_dir / f'analysis1_primary_{primary_feature_idx}_nonzero_only.png'
            plt.savefig(output_path_nonzero, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved non-zero only visualization: {output_path_nonzero}")
        
        print(f"  Saved visualization: {output_path}")
        print(f"  Total secondary features: {len(mean_secondary_activations)}")
        print(f"  Non-zero features (> {threshold}): {np.sum(mean_secondary_activations > threshold)}")
        print(f"  Max activation: {np.max(mean_secondary_activations):.6f}")
        print(f"  Mean activation: {np.mean(mean_secondary_activations):.6f}")
        
        return mean_secondary_activations

    def analyze_condition_differences_all_features(self, output_dir):
        """
        Analysis 3 Visualization: Create comprehensive visualizations for condition differences across ALL secondary features.
        """
        print(f"\n=== ANALYSIS 3 VISUALIZATION: CONDITION DIFFERENCES FOR ALL SECONDARY FEATURES ===")
        
        # Process each condition separately
        condition_11_texts = STIMULUS_SETS['F1_high_F2_high']  # [1,1]
        condition_00_texts = STIMULUS_SETS['F1_low_F2_low']    # [0,0]
        condition_10_texts = STIMULUS_SETS['F1_high_F2_low']   # [1,0]
        condition_01_texts = STIMULUS_SETS['F1_low_F2_high']   # [0,1]
        
        (_, _, _, _, activations_11) = self.get_complete_pipeline_activations(condition_11_texts)
        (_, _, _, _, activations_00) = self.get_complete_pipeline_activations(condition_00_texts)
        (_, _, _, _, activations_10) = self.get_complete_pipeline_activations(condition_10_texts)
        (_, _, _, _, activations_01) = self.get_complete_pipeline_activations(condition_01_texts)
        
        # Calculate mean activations for each condition
        mean_11 = np.mean(activations_11, axis=0)
        mean_00 = np.mean(activations_00, axis=0)
        mean_10 = np.mean(activations_10, axis=0)
        mean_01 = np.mean(activations_01, axis=0)
        
        # Calculate differences
        diff_11_00 = mean_11 - mean_00  # High-High vs Low-Low
        diff_10_01 = mean_10 - mean_01  # Formality effect
        diff_11_10 = mean_11 - mean_10  # Emotion effect (High formality)
        diff_01_00 = mean_01 - mean_00  # Emotion effect (Low formality)
        
        # Define threshold for non-zero
        threshold = 1e-6
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # Row 1: Condition means
        axes[0, 0].hist(mean_11, bins=50, alpha=0.7, color='red', label='[1,1]')
        axes[0, 0].hist(mean_00, bins=50, alpha=0.7, color='blue', label='[0,0]')
        axes[0, 0].set_xlabel('Mean Activation')
        axes[0, 0].set_ylabel('Number of Features')
        axes[0, 0].set_title('Condition Means: [1,1] vs [0,0]')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].hist(mean_10, bins=50, alpha=0.7, color='orange', label='[1,0]')
        axes[0, 1].hist(mean_01, bins=50, alpha=0.7, color='green', label='[0,1]')
        axes[0, 1].set_xlabel('Mean Activation')
        axes[0, 1].set_ylabel('Number of Features')
        axes[0, 1].set_title('Condition Means: [1,0] vs [0,1]')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # All conditions together
        axes[0, 2].hist([mean_11, mean_10, mean_01, mean_00], bins=30, alpha=0.6, 
                       color=['red', 'orange', 'green', 'blue'], 
                       label=['[1,1]', '[1,0]', '[0,1]', '[0,0]'])
        axes[0, 2].set_xlabel('Mean Activation')
        axes[0, 2].set_ylabel('Number of Features')
        axes[0, 2].set_title('All Condition Means')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Row 2: Difference distributions
        axes[1, 0].hist(diff_11_00, bins=50, alpha=0.7, color='purple')
        axes[1, 0].axvline(x=0, color='black', linestyle='--', alpha=0.7)
        axes[1, 0].set_xlabel('Activation Difference')
        axes[1, 0].set_ylabel('Number of Features')
        axes[1, 0].set_title('[1,1] - [0,0] Differences')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].hist(diff_10_01, bins=50, alpha=0.7, color='brown')
        axes[1, 1].axvline(x=0, color='black', linestyle='--', alpha=0.7)
        axes[1, 1].set_xlabel('Activation Difference')
        axes[1, 1].set_ylabel('Number of Features')
        axes[1, 1].set_title('Formality Effect: [1,0] - [0,1]')
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].hist(diff_11_10, bins=50, alpha=0.7, color='pink')
        axes[1, 2].axvline(x=0, color='black', linestyle='--', alpha=0.7)
        axes[1, 2].set_xlabel('Activation Difference')
        axes[1, 2].set_ylabel('Number of Features')
        axes[1, 2].set_title('Emotion Effect (High Form): [1,1] - [1,0]')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Row 3: Top difference features
        # Top positive [1,1] - [0,0] differences
        top_pos_indices = np.argsort(diff_11_00)[-15:][::-1]
        top_pos_values = diff_11_00[top_pos_indices]
        bars1 = axes[2, 0].bar(range(len(top_pos_values)), top_pos_values, color='darkred')
        axes[2, 0].set_xlabel('Top Features (Rank)')
        axes[2, 0].set_ylabel('Difference ([1,1] - [0,0])')
        axes[2, 0].set_title('Top 15 Positive Differences')
        axes[2, 0].tick_params(axis='x', labelsize=8)
        axes[2, 0].grid(True, alpha=0.3)
        
        # Top negative [1,1] - [0,0] differences
        top_neg_indices = np.argsort(diff_11_00)[:15]
        top_neg_values = diff_11_00[top_neg_indices]
        bars2 = axes[2, 1].bar(range(len(top_neg_values)), top_neg_values, color='darkblue')
        axes[2, 1].set_xlabel('Top Features (Rank)')
        axes[2, 1].set_ylabel('Difference ([1,1] - [0,0])')
        axes[2, 1].set_title('Top 15 Negative Differences')
        axes[2, 1].tick_params(axis='x', labelsize=8)
        axes[2, 1].grid(True, alpha=0.3)
        
        # Scatter plot: [1,1] vs [0,0] activations
        axes[2, 2].scatter(mean_00, mean_11, alpha=0.6, s=20)
        axes[2, 2].plot([0, max(np.max(mean_00), np.max(mean_11))], 
                       [0, max(np.max(mean_00), np.max(mean_11))], 
                       'r--', alpha=0.7, label='y=x')
        axes[2, 2].set_xlabel('[0,0] Mean Activation')
        axes[2, 2].set_ylabel('[1,1] Mean Activation')
        axes[2, 2].set_title('Condition Correlation')
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / 'analysis3_condition_differences_all_features.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # EXPORT NON-ZERO ONLY VERSION FOR CONDITION MEANS
        # Filter non-zero activations for each condition
        mean_11_nonzero = mean_11[mean_11 > threshold]
        mean_00_nonzero = mean_00[mean_00 > threshold]
        mean_10_nonzero = mean_10[mean_10 > threshold]
        mean_01_nonzero = mean_01[mean_01 > threshold]
        
        if len(mean_11_nonzero) > 0 or len(mean_00_nonzero) > 0:
            fig_nonzero, axes_nonzero = plt.subplots(2, 3, figsize=(18, 12))
            
            # Non-zero condition means histograms
            if len(mean_11_nonzero) > 0 and len(mean_00_nonzero) > 0:
                axes_nonzero[0, 0].hist(mean_11_nonzero, bins=40, alpha=0.7, color='red', label=f'[1,1] (n={len(mean_11_nonzero)})')
                axes_nonzero[0, 0].hist(mean_00_nonzero, bins=40, alpha=0.7, color='blue', label=f'[0,0] (n={len(mean_00_nonzero)})')
                axes_nonzero[0, 0].set_xlabel('Mean Activation (Non-zero only)')
                axes_nonzero[0, 0].set_ylabel('Number of Features')
                axes_nonzero[0, 0].set_title('Non-zero Condition Means: [1,1] vs [0,0]')
                axes_nonzero[0, 0].legend()
                axes_nonzero[0, 0].grid(True, alpha=0.3)
            
            if len(mean_10_nonzero) > 0 and len(mean_01_nonzero) > 0:
                axes_nonzero[0, 1].hist(mean_10_nonzero, bins=40, alpha=0.7, color='orange', label=f'[1,0] (n={len(mean_10_nonzero)})')
                axes_nonzero[0, 1].hist(mean_01_nonzero, bins=40, alpha=0.7, color='green', label=f'[0,1] (n={len(mean_01_nonzero)})')
                axes_nonzero[0, 1].set_xlabel('Mean Activation (Non-zero only)')
                axes_nonzero[0, 1].set_ylabel('Number of Features')
                axes_nonzero[0, 1].set_title('Non-zero Condition Means: [1,0] vs [0,1]')
                axes_nonzero[0, 1].legend()
                axes_nonzero[0, 1].grid(True, alpha=0.3)
            
            # All non-zero conditions together
            all_nonzero_data = []
            all_nonzero_labels = []
            all_nonzero_colors = []
            
            if len(mean_11_nonzero) > 0:
                all_nonzero_data.append(mean_11_nonzero)
                all_nonzero_labels.append(f'[1,1] (n={len(mean_11_nonzero)})')
                all_nonzero_colors.append('red')
            if len(mean_10_nonzero) > 0:
                all_nonzero_data.append(mean_10_nonzero)
                all_nonzero_labels.append(f'[1,0] (n={len(mean_10_nonzero)})')
                all_nonzero_colors.append('orange')
            if len(mean_01_nonzero) > 0:
                all_nonzero_data.append(mean_01_nonzero)
                all_nonzero_labels.append(f'[0,1] (n={len(mean_01_nonzero)})')
                all_nonzero_colors.append('green')
            if len(mean_00_nonzero) > 0:
                all_nonzero_data.append(mean_00_nonzero)
                all_nonzero_labels.append(f'[0,0] (n={len(mean_00_nonzero)})')
                all_nonzero_colors.append('blue')
            
            if all_nonzero_data:
                axes_nonzero[0, 2].hist(all_nonzero_data, bins=30, alpha=0.6, 
                                       color=all_nonzero_colors, label=all_nonzero_labels)
                axes_nonzero[0, 2].set_xlabel('Mean Activation (Non-zero only)')
                axes_nonzero[0, 2].set_ylabel('Number of Features')
                axes_nonzero[0, 2].set_title('All Non-zero Condition Means')
                axes_nonzero[0, 2].legend(fontsize=8)
                axes_nonzero[0, 2].grid(True, alpha=0.3)
            
            # Statistics comparison table
            stats_text = "Non-zero Activation Statistics by Condition:\n\n"
            conditions = [('11', mean_11_nonzero), ('10', mean_10_nonzero), 
                         ('01', mean_01_nonzero), ('00', mean_00_nonzero)]
            
            for cond_name, data in conditions:
                if len(data) > 0:
                    stats_text += f"[{cond_name[0]},{cond_name[1]}]: n={len(data):4d}, "
                    stats_text += f"mean={np.mean(data):.5f}, std={np.std(data):.5f}\n"
                else:
                    stats_text += f"[{cond_name[0]},{cond_name[1]}]: n=   0, no non-zero activations\n"
            
            total_features = len(mean_11)
            stats_text += f"\nTotal features: {total_features}\n"
            stats_text += f"Threshold: {threshold}"
            
            axes_nonzero[1, 0].text(0.1, 0.5, stats_text, transform=axes_nonzero[1, 0].transAxes, 
                                   fontsize=10, verticalalignment='center', fontfamily='monospace')
            axes_nonzero[1, 0].axis('off')
            axes_nonzero[1, 0].set_title('Non-zero Statistics Summary')
            
            # Box plots for non-zero distributions
            if all_nonzero_data:
                box_data = [data for data in all_nonzero_data if len(data) > 0]
                box_labels = [label.split(' (')[0] for label, data in zip(all_nonzero_labels, all_nonzero_data) if len(data) > 0]
                
                if box_data:
                    axes_nonzero[1, 1].boxplot(box_data, labels=box_labels)
                    axes_nonzero[1, 1].set_ylabel('Mean Activation (Non-zero only)')
                    axes_nonzero[1, 1].set_title('Non-zero Activation Distributions')
                    axes_nonzero[1, 1].grid(True, alpha=0.3)
            
            # Log-scale version
            if all_nonzero_data:
                axes_nonzero[1, 2].hist(all_nonzero_data, bins=30, alpha=0.6, 
                                       color=all_nonzero_colors, label=all_nonzero_labels)
                axes_nonzero[1, 2].set_xlabel('Mean Activation (Non-zero only)')
                axes_nonzero[1, 2].set_ylabel('Number of Features (Log Scale)')
                axes_nonzero[1, 2].set_title('Non-zero Means - Log Y Scale')
                axes_nonzero[1, 2].set_yscale('log')
                axes_nonzero[1, 2].legend(fontsize=8)
                axes_nonzero[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_path_nonzero = output_dir / 'analysis3_condition_differences_nonzero_only.png'
            plt.savefig(output_path_nonzero, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved non-zero only visualization: {output_path_nonzero}")
        
        print(f"  Saved visualization: {output_path}")
        print(f"  Total features analyzed: {len(mean_11)}")
        print(f"  Non-zero features by condition (> {threshold}):")
        print(f"    [1,1]: {len(mean_11_nonzero)}/{len(mean_11)}")
        print(f"    [1,0]: {len(mean_10_nonzero)}/{len(mean_10)}")
        print(f"    [0,1]: {len(mean_01_nonzero)}/{len(mean_01)}")
        print(f"    [0,0]: {len(mean_00_nonzero)}/{len(mean_00)}")
        print(f"  Top positive difference: {np.max(diff_11_00):.6f}")
        print(f"  Top negative difference: {np.min(diff_11_00):.6f}")
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].hist(diff_10_01, bins=50, alpha=0.7, color='brown')
        axes[1, 1].axvline(x=0, color='black', linestyle='--', alpha=0.7)
        axes[1, 1].set_xlabel('Activation Difference')
        axes[1, 1].set_ylabel('Number of Features')
        axes[1, 1].set_title('Formality Effect: [1,0] - [0,1]')
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].hist(diff_11_10, bins=50, alpha=0.7, color='pink')
        axes[1, 2].axvline(x=0, color='black', linestyle='--', alpha=0.7)
        axes[1, 2].set_xlabel('Activation Difference')
        axes[1, 2].set_ylabel('Number of Features')
        axes[1, 2].set_title('Emotion Effect (High Form): [1,1] - [1,0]')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Row 3: Top difference features
        # Top positive [1,1] - [0,0] differences
        top_pos_indices = np.argsort(diff_11_00)[-15:][::-1]
        top_pos_values = diff_11_00[top_pos_indices]
        bars1 = axes[2, 0].bar(range(len(top_pos_values)), top_pos_values, color='darkred')
        axes[2, 0].set_xlabel('Top Features (Rank)')
        axes[2, 0].set_ylabel('Difference ([1,1] - [0,0])')
        axes[2, 0].set_title('Top 15 Positive Differences')
        axes[2, 0].tick_params(axis='x', labelsize=8)
        axes[2, 0].grid(True, alpha=0.3)
        
        # Top negative [1,1] - [0,0] differences
        top_neg_indices = np.argsort(diff_11_00)[:15]
        top_neg_values = diff_11_00[top_neg_indices]
        bars2 = axes[2, 1].bar(range(len(top_neg_values)), top_neg_values, color='darkblue')
        axes[2, 1].set_xlabel('Top Features (Rank)')
        axes[2, 1].set_ylabel('Difference ([1,1] - [0,0])')
        axes[2, 1].set_title('Top 15 Negative Differences')
        axes[2, 1].tick_params(axis='x', labelsize=8)
        axes[2, 1].grid(True, alpha=0.3)
        
        # Scatter plot: [1,1] vs [0,0] activations
        axes[2, 2].scatter(mean_00, mean_11, alpha=0.6, s=20)
        axes[2, 2].plot([0, max(np.max(mean_00), np.max(mean_11))], 
                       [0, max(np.max(mean_00), np.max(mean_11))], 
                       'r--', alpha=0.7, label='y=x')
        axes[2, 2].set_xlabel('[0,0] Mean Activation')
        axes[2, 2].set_ylabel('[1,1] Mean Activation')
        axes[2, 2].set_title('Condition Correlation')
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / 'analysis3_condition_differences_all_features.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved visualization: {output_path}")
        print(f"  Total features analyzed: {len(mean_11)}")
        print(f"  Top positive difference: {np.max(diff_11_00):.6f}")
        print(f"  Top negative difference: {np.min(diff_11_00):.6f}")
        
        return {
            'mean_activations': {'11': mean_11, '00': mean_00, '10': mean_10, '01': mean_01},
            'differences': {'11_00': diff_11_00, '10_01': diff_10_01, '11_10': diff_11_10, '01_00': diff_01_00}
        }

    def analyze_anova_sensitivity_all_features(self, output_dir):
        """
        Analysis 4 Visualization: ANOVA sensitivity across ALL secondary features with p<0.05 significance line.
        NaN/Inf values are treated as 0 for visualization.
        """
        print(f"\n=== ANALYSIS 4 VISUALIZATION: ANOVA SENSITIVITY FOR ALL SECONDARY FEATURES ===")
        
        # Process each condition separately
        condition_data = {}
        conditions = ['F1_high_F2_high', 'F1_high_F2_low', 'F1_low_F2_high', 'F1_low_F2_low']
        
        for condition in conditions:
            texts = STIMULUS_SETS[condition]
            (_, _, _, _, secondary_activations) = self.get_complete_pipeline_activations(texts)
            condition_data[condition] = secondary_activations
        
        # Compute ANOVA for each secondary feature
        num_secondary_features = condition_data[conditions[0]].shape[1]
        f_statistics = np.zeros(num_secondary_features)
        p_values = np.ones(num_secondary_features)  # Default to p=1 (not significant)
        
        for sec_idx in range(num_secondary_features):
            # Collect data for this secondary feature across all conditions
            groups = []
            for condition in conditions:
                groups.append(condition_data[condition][:, sec_idx])
            
            # Perform one-way ANOVA
            try:
                f_stat, p_val = stats.f_oneway(*groups)
                # Handle nan/inf by setting to 0 for F-statistic and 1 for p-value
                if np.isfinite(f_stat) and np.isfinite(p_val):
                    f_statistics[sec_idx] = f_stat
                    p_values[sec_idx] = p_val
                else:
                    f_statistics[sec_idx] = 0.0
                    p_values[sec_idx] = 1.0
            except:
                # Set to 0/1 if ANOVA fails
                f_statistics[sec_idx] = 0.0
                p_values[sec_idx] = 1.0
        
        # Create comprehensive ANOVA visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # F-statistics histogram
        axes[0, 0].hist(f_statistics, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('F-statistic')
        axes[0, 0].set_ylabel('Number of Features')
        axes[0, 0].set_title('Distribution of F-statistics')
        axes[0, 0].grid(True, alpha=0.3)
        
        # P-values histogram
        axes[0, 1].hist(p_values, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='p < 0.05')
        axes[0, 1].set_xlabel('p-value')
        axes[0, 1].set_ylabel('Number of Features')
        axes[0, 1].set_title('Distribution of p-values')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # -log10(p-values) to better show significance
        neg_log_p = -np.log10(p_values)
        # Handle inf values (p=0) by setting to a large value
        neg_log_p[np.isinf(neg_log_p)] = 50
        
        axes[0, 2].hist(neg_log_p, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 2].axvline(x=-np.log10(0.05), color='red', linestyle='--', linewidth=2, 
                          label=f'-log10(0.05) = {-np.log10(0.05):.2f}')
        axes[0, 2].set_xlabel('-log10(p-value)')
        axes[0, 2].set_ylabel('Number of Features')
        axes[0, 2].set_title('Significance: -log10(p-values)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Top F-statistics bar chart
        top_f_indices = np.argsort(f_statistics)[-20:][::-1]
        top_f_values = f_statistics[top_f_indices]
        bars1 = axes[1, 0].bar(range(len(top_f_values)), top_f_values, color='darkblue')
        axes[1, 0].set_xlabel('Top 20 Features (Rank)')
        axes[1, 0].set_ylabel('F-statistic')
        axes[1, 0].set_title('Top 20 F-statistics')
        axes[1, 0].tick_params(axis='x', labelsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add feature indices as labels
        for i, (bar, idx) in enumerate(zip(bars1, top_f_indices)):
            if i % 2 == 0:  # Only label every other bar to avoid crowding
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                               f'{idx}', ha='center', va='bottom', fontsize=7, rotation=45)
        
        # Significant features (p < 0.05)
        significant_mask = p_values < 0.05
        significant_indices = np.where(significant_mask)[0]
        significant_f_stats = f_statistics[significant_mask]
        
        if len(significant_indices) > 0:
            # Show top significant features
            top_sig_indices = significant_indices[np.argsort(significant_f_stats)[::-1]][:15]
            top_sig_f_values = f_statistics[top_sig_indices]
            bars2 = axes[1, 1].bar(range(len(top_sig_f_values)), top_sig_f_values, color='darkred')
            axes[1, 1].set_xlabel('Top Significant Features')
            axes[1, 1].set_ylabel('F-statistic')
            axes[1, 1].set_title(f'Top Significant Features (p < 0.05)\n{len(significant_indices)} total significant')
            axes[1, 1].tick_params(axis='x', labelsize=8)
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add feature indices as labels
            for i, (bar, idx) in enumerate(zip(bars2, top_sig_indices)):
                if i % 2 == 0:
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                                   f'{idx}', ha='center', va='bottom', fontsize=7, rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, 'No significant features\n(p < 0.05)', 
                           ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].set_title('No Significant Features')
        
        # Volcano plot: F-statistic vs -log10(p-value)
        axes[1, 2].scatter(f_statistics, neg_log_p, alpha=0.6, s=20, c='gray')
        if len(significant_indices) > 0:
            axes[1, 2].scatter(f_statistics[significant_mask], neg_log_p[significant_mask], 
                              alpha=0.8, s=30, c='red', label=f'{len(significant_indices)} significant')
        axes[1, 2].axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p = 0.05')
        axes[1, 2].set_xlabel('F-statistic')
        axes[1, 2].set_ylabel('-log10(p-value)')
        axes[1, 2].set_title('Volcano Plot: Effect Size vs Significance')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / 'analysis4_anova_sensitivity_all_features.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved visualization: {output_path}")
        print(f"  Total features analyzed: {num_secondary_features}")
        print(f"  Significant features (p < 0.05): {len(significant_indices)}")
        print(f"  Max F-statistic: {np.max(f_statistics):.6f}")
        print(f"  Min p-value: {np.min(p_values):.2e}")
        
        return {
            'f_statistics': f_statistics,
            'p_values': p_values,
            'significant_indices': significant_indices
        }

    def create_summary_visualization(self, primary_feature1, primary_feature2, output_dir):
        """
        Create a comprehensive summary visualization combining key insights from all analyses.
        """
        print(f"\n=== CREATING SUMMARY VISUALIZATION ===")
        
        # Get data for both primary features
        mean_sec_1 = self.analyze_all_secondary_features_for_primary(primary_feature1, output_dir)
        mean_sec_2 = self.analyze_all_secondary_features_for_primary(primary_feature2, output_dir)
        
        # Get condition differences
        condition_results = self.analyze_condition_differences_all_features(output_dir)
        
        # Get ANOVA results
        anova_results = self.analyze_anova_sensitivity_all_features(output_dir)
        
        # Create summary plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Compare primary feature effects
        axes[0, 0].hist([mean_sec_1, mean_sec_2], bins=30, alpha=0.6, 
                       color=['blue', 'red'], label=[f'Primary {primary_feature1}', f'Primary {primary_feature2}'])
        axes[0, 0].set_xlabel('Mean Secondary Feature Activation')
        axes[0, 0].set_ylabel('Number of Features')
        axes[0, 0].set_title('Primary Feature Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Condition effects summary
        diff_11_00 = condition_results['differences']['11_00']
        axes[0, 1].hist(diff_11_00, bins=50, alpha=0.7, color='purple')
        axes[0, 1].axvline(x=0, color='black', linestyle='--', alpha=0.7)
        axes[0, 1].set_xlabel('Activation Difference ([1,1] - [0,0])')
        axes[0, 1].set_ylabel('Number of Features')
        axes[0, 1].set_title('Condition Effects Summary')
        axes[0, 1].grid(True, alpha=0.3)
        
        # ANOVA significance summary
        f_stats = anova_results['f_statistics']
        p_vals = anova_results['p_values']
        significant_mask = p_vals < 0.05
        
        axes[0, 2].hist(f_stats, bins=50, alpha=0.7, color='green', label='All features')
        if np.any(significant_mask):
            axes[0, 2].hist(f_stats[significant_mask], bins=30, alpha=0.8, color='darkgreen', 
                           label=f'Significant (n={np.sum(significant_mask)})')
        axes[0, 2].set_xlabel('F-statistic')
        axes[0, 2].set_ylabel('Number of Features')
        axes[0, 2].set_title('ANOVA Sensitivity Summary')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Cross-analysis correlations
        axes[1, 0].scatter(mean_sec_1, mean_sec_2, alpha=0.6, s=20)
        axes[1, 0].set_xlabel(f'Primary {primary_feature1} Effect')
        axes[1, 0].set_ylabel(f'Primary {primary_feature2} Effect')
        axes[1, 0].set_title('Primary Feature Effect Correlation')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Primary effects vs condition effects
        max_primary_effect = np.maximum(mean_sec_1, mean_sec_2)
        axes[1, 1].scatter(max_primary_effect, np.abs(diff_11_00), alpha=0.6, s=20)
        axes[1, 1].set_xlabel('Max Primary Feature Effect')
        axes[1, 1].set_ylabel('|Condition Difference|')
        axes[1, 1].set_title('Primary vs Condition Effects')
        axes[1, 1].grid(True, alpha=0.3)
        
        # ANOVA vs other effects
        axes[1, 2].scatter(f_stats, max_primary_effect, alpha=0.6, s=20, c='gray', label='All')
        if np.any(significant_mask):
            axes[1, 2].scatter(f_stats[significant_mask], max_primary_effect[significant_mask], 
                              alpha=0.8, s=30, c='red', label='Significant')
        axes[1, 2].set_xlabel('F-statistic (ANOVA)')
        axes[1, 2].set_ylabel('Max Primary Feature Effect')
        axes[1, 2].set_title('ANOVA vs Primary Effects')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / 'summary_all_analyses.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved summary visualization: {output_path}")
        
        return {
            'primary_effects': {'feature1': mean_sec_1, 'feature2': mean_sec_2},
            'condition_effects': condition_results,
            'anova_effects': anova_results
        }

def load_sae_model(checkpoint_path, device="cuda"):
    """
    Load SAE model, detecting whether it's TopK or regular SAE.
    Returns the appropriate model instance.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract the actual model state dict
    if 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
        print(f"Loaded checkpoint with keys: {list(checkpoint.keys())}")
    else:
        state_dict = checkpoint
    
    print(f"State dict keys: {list(state_dict.keys())}")
    
    # Determine dimensions
    if 'decoder.weight' in state_dict:
        input_dim = state_dict['decoder.weight'].shape[0]
        hidden_dim = state_dict['decoder.weight'].shape[1]
    elif 'encoder.0.weight' in state_dict:
        encoder_weight = state_dict['encoder.0.weight']
        hidden_dim, input_dim = encoder_weight.shape
    else:
        # Try to infer from any available linear layer
        possible_keys = [k for k in state_dict.keys() if 'weight' in k]
        print(f"Available weight keys: {possible_keys}")
        
        # Look for encoder-like layers
        encoder_keys = [k for k in possible_keys if 'encoder' in k.lower()]
        decoder_keys = [k for k in possible_keys if 'decoder' in k.lower()]
        
        if encoder_keys:
            encoder_weight = state_dict[encoder_keys[0]]
            if len(encoder_weight.shape) == 2:
                hidden_dim, input_dim = encoder_weight.shape
            else:
                raise ValueError(f"Unexpected encoder weight shape: {encoder_weight.shape}")
        elif decoder_keys:
            decoder_weight = state_dict[decoder_keys[0]]
            if len(decoder_weight.shape) == 2:
                input_dim, hidden_dim = decoder_weight.shape
            else:
                raise ValueError(f"Unexpected decoder weight shape: {decoder_weight.shape}")
        else:
            raise ValueError(f"Cannot determine SAE dimensions from state dict keys: {possible_keys}")
    
    print(f"Detected SAE dimensions: {input_dim} → {hidden_dim}")
    
    # Check if this is a TopK SAE by looking for TopK-specific parameters or filename
    is_topk = False
    k_value = None
    
    # Check filename for TopK indicators
    filename = str(checkpoint_path).lower()
    if 'topk' in filename or 'top_k' in filename:
        is_topk = True
        # Try to extract K value from filename
        import re
        k_matches = re.findall(r'topk(\d+)', filename)
        if k_matches:
            k_value = int(k_matches[0])
        else:
            # Look for other patterns like _k1024_
            k_matches = re.findall(r'_k(\d+)_', filename)
            if k_matches:
                k_value = int(k_matches[0])
    
    # Check checkpoint metadata for TopK info
    if 'metrics_history' in checkpoint:
        metrics = checkpoint['metrics_history']
        if 'percent_active' in metrics:
            # TopK SAEs typically have very consistent percent_active values
            if len(metrics['percent_active']) > 0:
                recent_active = metrics['percent_active'][-10:]  # Last 10 values
                if len(recent_active) > 1:
                    std_active = np.std(recent_active)
                    mean_active = np.mean(recent_active)
                    # TopK SAEs have very low std in percent_active
                    if std_active < 0.1 and mean_active < 10:  # Less than 10% active
                        is_topk = True
                        # Estimate K from percent active
                        if k_value is None:
                            k_value = int((mean_active / 100.0) * hidden_dim)
    
    # Default K value if not found
    if is_topk and k_value is None:
        # Use a reasonable default (2% of features)
        k_value = max(1, int(0.02 * hidden_dim))
        print(f"Warning: TopK SAE detected but K value not found. Using default K={k_value} (2% of {hidden_dim})")
    
    # Create appropriate model
    if is_topk:
        print(f"Loading as TopK SAE with K={k_value}")
        model = TopKSparseAutoencoder(input_dim, hidden_dim, k_value)
    else:
        print(f"Loading as regular SAE")
        # Need to import regular SAE class
        class SparseAutoencoder(torch.nn.Module):
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
        
        model = SparseAutoencoder(input_dim, hidden_dim)
    
    model.load_state_dict(state_dict)
    model.to(device)
    
    return model

def main():
    parser = argparse.ArgumentParser(description="NFM Secondary SAE Analysis with Comprehensive Visualizations - TopK VERSION")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base LLM model")
    parser.add_argument("--primary_sae_path", type=str, required=True, help="Path to Primary SAE model")
    parser.add_argument("--nfm_path", type=str, required=True, help="Path to NFM model")
    parser.add_argument("--secondary_sae_path", type=str, required=True, help="Path to Secondary SAE model")
    parser.add_argument("--primary_feature1", type=int, required=True, help="First Primary SAE feature index")
    parser.add_argument("--primary_feature2", type=int, required=True, help="Second Primary SAE feature index")
    parser.add_argument("--output_dir", type=str, default="./nfm_sae_analysis_viz", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Setup output directory and logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / "nfm_sae_analysis_viz.log"),
            logging.StreamHandler()
        ]
    )
    
    # Set style for better plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    print("Loading models...")
    
    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")
    
    # Load Primary SAE (auto-detect TopK vs regular)
    print(f"Loading Primary SAE from {args.primary_sae_path}...")
    primary_sae = load_sae_model(args.primary_sae_path, args.device)
    
    # Load NFM model
    print(f"Loading NFM from {args.nfm_path}...")
    nfm_state_dict = torch.load(args.nfm_path, map_location=args.device)
    num_features = nfm_state_dict['feature_embeddings.weight'].shape[0]
    k_dim = nfm_state_dict['feature_embeddings.weight'].shape[1]
    output_dim = nfm_state_dict['linear.weight'].shape[0]
    
    nfm_model = NeuralFactorizationModel(num_features, k_dim, output_dim)
    nfm_model.load_state_dict(nfm_state_dict)
    nfm_model.to(args.device)
    
    # Load Secondary SAE (auto-detect TopK vs regular)
    print(f"Loading Secondary SAE from {args.secondary_sae_path}...")
    secondary_sae = load_sae_model(args.secondary_sae_path, args.device)
    
    print(f"Models loaded successfully!")
    
    # Print model information
    if hasattr(primary_sae, 'k'):
        primary_total = primary_sae.encoder[0].out_features
        primary_sparsity = (primary_sae.k / primary_total) * 100
        print(f"Primary TopK SAE: {primary_sae.encoder[0].in_features} → {primary_total} (K={primary_sae.k}, {primary_sparsity:.2f}% active)")
    else:
        print(f"Primary Regular SAE: {primary_sae.encoder[0].in_features} → {primary_sae.encoder[0].out_features}")
    
    print(f"NFM: {num_features} features → {k_dim} embedding dim")
    
    if hasattr(secondary_sae, 'k'):
        secondary_total = secondary_sae.encoder[0].out_features
        secondary_sparsity = (secondary_sae.k / secondary_total) * 100
        print(f"Secondary TopK SAE: {secondary_sae.encoder[0].in_features} → {secondary_total} (K={secondary_sae.k}, {secondary_sparsity:.2f}% active)")
    else:
        print(f"Secondary Regular SAE: {secondary_sae.encoder[0].in_features} → {secondary_sae.encoder[0].out_features}")
    
    # Initialize analyzer
    analyzer = NFMSAEAnalyzer(primary_sae, nfm_model, secondary_sae, tokenizer, base_model, args.device)
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE VISUALIZATION ANALYSIS")
    print(f"{'='*80}")
    
    # Generate all visualizations and collect results
    all_results = {}
    
    print(f"\nGenerating visualizations for Primary Features {args.primary_feature1} and {args.primary_feature2}...")
    
    # Analysis 1: Individual primary feature effects (generates individual plots)
    print(f"\n--- Analysis 1: Individual Primary Feature Effects ---")
    mean_sec_1 = analyzer.analyze_all_secondary_features_for_primary(args.primary_feature1, output_dir)
    mean_sec_2 = analyzer.analyze_all_secondary_features_for_primary(args.primary_feature2, output_dir)
    
    all_results['primary_feature_effects'] = {
        'feature1': {
            'index': args.primary_feature1,
            'secondary_activations': mean_sec_1.tolist()
        },
        'feature2': {
            'index': args.primary_feature2,
            'secondary_activations': mean_sec_2.tolist()
        }
    }
    
    # Analysis 3: Condition differences (generates comprehensive condition plot)
    print(f"\n--- Analysis 3: Condition Differences ---")
    condition_results = analyzer.analyze_condition_differences_all_features(output_dir)
    all_results['condition_differences'] = {
        'mean_activations': {k: v.tolist() for k, v in condition_results['mean_activations'].items()},
        'differences': {k: v.tolist() for k, v in condition_results['differences'].items()}
    }
    
    # Analysis 4: ANOVA sensitivity (generates ANOVA significance plot)
    print(f"\n--- Analysis 4: ANOVA Sensitivity ---")
    anova_results = analyzer.analyze_anova_sensitivity_all_features(output_dir)
    all_results['anova_sensitivity'] = {
        'f_statistics': anova_results['f_statistics'].tolist(),
        'p_values': anova_results['p_values'].tolist(),
        'significant_indices': anova_results['significant_indices'].tolist()
    }
    
    # Summary visualization combining all analyses
    print(f"\n--- Summary: Combined Analysis ---")
    summary_results = analyzer.create_summary_visualization(args.primary_feature1, args.primary_feature2, output_dir)
    all_results['summary'] = {
        'correlations': {
            'primary_1_vs_2': float(np.corrcoef(mean_sec_1, mean_sec_2)[0, 1]),
            'primary_vs_condition': float(np.corrcoef(
                np.maximum(mean_sec_1, mean_sec_2), 
                np.abs(condition_results['differences']['11_00'])
            )[0, 1])
        }
    }
    
    # Save comprehensive results
    results_path = output_dir / "comprehensive_analysis_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate summary statistics
    print(f"\n{'='*80}")
    print("COMPREHENSIVE ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    # Primary feature statistics
    print(f"\nPrimary Feature Effects:")
    print(f"  Feature {args.primary_feature1}:")
    print(f"    Active secondary features: {np.sum(mean_sec_1 > 0)}/{len(mean_sec_1)}")
    print(f"    Max activation: {np.max(mean_sec_1):.6f}")
    print(f"    Mean activation: {np.mean(mean_sec_1):.6f}")
    
    print(f"  Feature {args.primary_feature2}:")
    print(f"    Active secondary features: {np.sum(mean_sec_2 > 0)}/{len(mean_sec_2)}")
    print(f"    Max activation: {np.max(mean_sec_2):.6f}")
    print(f"    Mean activation: {np.mean(mean_sec_2):.6f}")
    
    print(f"  Correlation between primary effects: {np.corrcoef(mean_sec_1, mean_sec_2)[0, 1]:.4f}")
    
    # Condition effect statistics
    diff_11_00 = condition_results['differences']['11_00']
    print(f"\nCondition Effects ([1,1] - [0,0]):")
    print(f"  Features with positive difference: {np.sum(diff_11_00 > 0)}/{len(diff_11_00)}")
    print(f"  Features with negative difference: {np.sum(diff_11_00 < 0)}/{len(diff_11_00)}")
    print(f"  Max positive difference: {np.max(diff_11_00):.6f}")
    print(f"  Max negative difference: {np.min(diff_11_00):.6f}")
    print(f"  Mean absolute difference: {np.mean(np.abs(diff_11_00)):.6f}")
    
    # ANOVA statistics
    f_stats = anova_results['f_statistics']
    p_vals = anova_results['p_values']
    significant_count = len(anova_results['significant_indices'])
    
    print(f"\nANOVA Sensitivity:")
    print(f"  Total features analyzed: {len(f_stats)}")
    print(f"  Significant features (p < 0.05): {significant_count}")
    print(f"  Percentage significant: {(significant_count/len(f_stats)*100):.2f}%")
    print(f"  Max F-statistic: {np.max(f_stats):.6f}")
    print(f"  Min p-value: {np.min(p_vals):.2e}")
    
    # Generated files summary
    print(f"\n{'='*80}")
    print("GENERATED FILES")
    print(f"{'='*80}")
    
    generated_files = [
        f"analysis1_primary_{args.primary_feature1}_all_secondary_features.png",
        f"analysis1_primary_{args.primary_feature1}_nonzero_only.png",
        f"analysis1_primary_{args.primary_feature2}_all_secondary_features.png",
        f"analysis1_primary_{args.primary_feature2}_nonzero_only.png",
        "analysis3_condition_differences_all_features.png",
        "analysis3_condition_differences_nonzero_only.png",
        "analysis4_anova_sensitivity_all_features.png",
        "summary_all_analyses.png",
        "comprehensive_analysis_results.json",
        "nfm_sae_analysis_viz.log"
    ]
    
    print(f"Output directory: {output_dir}")
    print(f"Generated files:")
    for file in generated_files:
        file_path = output_dir / file
        if file_path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (missing)")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"All visualizations and results saved to: {output_dir}")

if __name__ == "__main__":
    main()