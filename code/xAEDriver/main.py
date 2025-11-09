import numpy as np
import pandas as pd
import math
import random
import argparse
from pathway_hierarchy import *
from neural_network import *
from utils import *

# Set random seeds for reproducibility
random.seed(2)
np.random.seed(2)


def main():
    """
    Main function to train an explainable autoencoder. This script preprocesses and integrates gene 
    dependency, gene expression, and somatic mutation data. It trains the model to generate driver 
    variant representations (DVRs) and then calculates pathway importance based on relevance scores.
    """
    
    # =========================================================================
    # Argument Parsing
    # =========================================================================
    parser = argparse.ArgumentParser(
        description="Train an explainable autoencoder to generate DVRs and identify important pathways.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help message
    )
    parser.add_argument('--dvr_file_name', type=str, required=True,
                        help='Path to save the output CSV file for generated DVRs.')
    parser.add_argument('--pathway_file_name', type=str, required=True,
                        help='Path to save the output CSV file with pathway relevance scores.')
    parser.add_argument('--encoded_dim', type=int, default=1024,
                        help='Dimension of the encoded layer (DVR).')
    parser.add_argument('--n_hidden', type=int, default=3,
                        help='Number of hidden layers in the neural network.')
    parser.add_argument("--learning_rate", type=float, default=0.01,
                        help="Learning rate for the Adam optimizer.")
    parser.add_argument("--num_epochs", type=int, default=1000,
                        help="Maximum number of training epochs.")
    parser.add_argument("--alpha_binomial", type=float, default=0.001,
                        help="Weight for the binomial distribution loss component in the autoencoder.")
    parser.add_argument("--alpha_regularization", type=float, default=0.1,
                        help="Weight for the regularization term.")
    parser.add_argument("--regularization_type", type=str, default='L2',
                        help="Type of regularization to apply ('L1' or 'L2').")
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="A small constant for the relevance calculation to ensure numerical stability.")
    args = parser.parse_args()
    
    # =========================================================================
    # Data Loading and Preprocessing
    # =========================================================================
    print("Loading and preprocessing datasets...")
    # Load gene dependency and expression datasets
    dependency_df = pd.read_csv("../../dataset/CRISPRGeneEffect.csv", index_col=0)
    expression_df = pd.read_csv("../../dataset/OmicsExpressionProteinCodingGenesTPMLogp1.csv", index_col=0)
    
    # --- Preprocess Gene Dependency Data ---
    # Clean up column names by keeping only the gene symbol part
    gene_col = [col.split(' ')[0] for col in dependency_df.columns]
    dependency_df.columns = gene_col
    # Filter genes based on a predefined list of screened genes
    gene_mapping_df = pd.read_csv("../../dataset/InputGene/ScreenedGene.csv")
    gene_mapping_df = gene_mapping_df.drop_duplicates(subset=['From'], keep='first')
    dependency_df = dependency_df[gene_mapping_df.iloc[:, 0]]
    dependency_df.columns = gene_mapping_df.iloc[:, 1]
    # Remove columns with any null values
    null_counts = dependency_df.isnull().sum()
    dependency_df = dependency_df[null_counts[null_counts == 0].index]
    
    # --- Preprocess Gene Expression Data ---
    # Clean up column names similarly
    gene_col = [col.split(' ')[0] for col in expression_df.columns]
    expression_df.columns = gene_col
    # Filter genes based on a predefined list of expression genes
    gene_mapping_df = pd.read_csv("../../dataset/InputGene/ExpressionGene.csv")
    gene_mapping_df = gene_mapping_df.drop_duplicates(subset=['From'], keep='first')
    expression_df = expression_df[gene_mapping_df.iloc[:, 0]]
    expression_df.columns = gene_mapping_df.iloc[:, 1]
    # Remove columns with any null values
    null_counts = expression_df.isnull().sum()
    expression_df = expression_df[null_counts[null_counts == 0].index]
    
    # --- Align and Combine Datasets ---
    # Remove duplicated columns (genes) that may have appeared after name cleaning
    dependency_df = dependency_df.loc[:, ~dependency_df.columns.duplicated()]
    expression_df = expression_df.loc[:, ~expression_df.columns.duplicated()]
    # Find the intersection of cell lines between the two datasets
    common_indices = sorted(list(set(dependency_df.index) & set(expression_df.index)))
    dependency_df = dependency_df.loc[common_indices, :]
    expression_df = expression_df.loc[common_indices, :]
    
    # Normalize both datasets
    dependency_df = (dependency_df - dependency_df.mean()) / (dependency_df.max() - dependency_df.min())
    expression_df = (expression_df - expression_df.mean()) / (expression_df.max() - expression_df.min())
    
    # Concatenate dependency and expression data into a single feature matrix
    features_df = pd.concat([dependency_df, expression_df], axis=1)
    features_df = features_df.dropna(axis=1)
    n_dep = dependency_df.shape[1]
    n_expr = expression_df.shape[1]
    
    # =========================================================================
    # Mutation Data Loading and Functional Filtering
    # =========================================================================
    print("Filtering for functional mutations using COSMIC and ClinVar...")
    # Load somatic mutation data and filter for SNPs from likely driver genes
    mutation = pd.read_csv("../../dataset/OmicsSomaticMutations.csv")
    mutation = mutation[mutation['VariantType'] == 'SNP']
    mutation = mutation[mutation['LikelyDriver']]
    mutation = mutation[['Chrom', 'Pos', 'HugoSymbol', 'ModelID']]
    
    # --- Process COSMIC Database ---
    # Load COSMIC data and filter for pathogenic SNVs
    cosmic = pd.read_csv("../../dataset/Cosmic_MutantCensus_v102_GRCh38.tsv", delimiter='\t')
    snv_mask = cosmic['HGVSG'].str.contains(r'[ACGT]>[ACGT]$', na=False)
    cosmic_somatic = cosmic[snv_mask].copy()
    pathogenic_impact_types = [
        'transcript_ablation', 'splice_acceptor_variant', 'splice_donor_variant',
        'stop_gained', 'frameshift_variant', 'stop_lost', 'start_lost',
        'transcript_amplification', 'inframe_insertion', 'inframe_deletion',
        'missense_variant', 'protein_altering_variant', 'splice_region_variant',
        'start_retained_variant'
    ]
    impact_mask = cosmic_somatic['MUTATION_DESCRIPTION'].isin(pathogenic_impact_types)
    cosmic_patho = cosmic_somatic[impact_mask].copy()
    # Standardize and prepare the pathogenic COSMIC data
    cosmic_patho = cosmic_patho[['GENE_SYMBOL', 'CHROMOSOME', 'GENOME_START']]
    cosmic_patho.rename(columns={'GENE_SYMBOL': 'GeneSymbol', 'CHROMOSOME': 'Chromosome', 'GENOME_START': 'Start'}, inplace=True)
    cosmic_patho['Chromosome'] = cosmic_patho['Chromosome'].apply(lambda x: 'chr' + str(x))
    cosmic_patho['Start'] = cosmic_patho['Start'].astype(int)
    # Create a unique ID for each mutation based on its genomic location
    cosmic_patho['ID'] = cosmic_patho['Chromosome'] + '-' + cosmic_patho['Start'].astype(str)
    
    # --- Process ClinVar Database ---
    # Load ClinVar data and filter for pathogenic SNVs
    clinvar = pd.read_csv("../../dataset/ClinVar/ClinVar_variant_summary.txt", delimiter='\t', low_memory=False)
    clinvar = clinvar[(clinvar['Assembly'] == "GRCh38") & (clinvar['Type'] == "single nucleotide variant")]
    pathogenicity = pd.read_csv("../../dataset/ClinVar/pathogenicity.csv", index_col=0)
    pathogenetic_type = list(pathogenicity[pathogenicity['Pathogenicity'] == 'Y']['Category'])
    clinvar_patho = clinvar[clinvar['ClinicalSignificance'].isin(pathogenetic_type)]
    # Standardize and prepare the pathogenic ClinVar data
    clinvar_patho = clinvar_patho[["Chromosome", "Start", "GeneSymbol"]]
    clinvar_patho['Chromosome'] = clinvar_patho['Chromosome'].apply(lambda x: 'chr' + str(x))
    clinvar_patho = clinvar_patho[pd.to_numeric(clinvar_patho['Start'], errors='coerce').notna()]
    clinvar_patho['Start'] = clinvar_patho['Start'].astype(int)
    # Create the same unique ID format as COSMIC
    clinvar_patho['ID'] = clinvar_patho['Chromosome'] + '-' + clinvar_patho['Start'].astype(str)
    
    # --- Combine Databases and Generate Final Mutation Matrix ---
    cosmic_final = cosmic_patho[['ID', 'GeneSymbol']]
    clinvar_final = clinvar_patho[['ID', 'GeneSymbol']]
    functional_mutations_union = pd.concat([cosmic_final, clinvar_final], ignore_index=True)
    functional_mutations_union.drop_duplicates(subset='ID', keep='first', inplace=True)
    
    # Filter the main mutation data to keep only functionally relevant mutations
    mutation['ID'] = mutation['Chrom'] + '-' + mutation['Pos'].astype(str)
    mutation = pd.merge(mutation, functional_mutations_union[['ID', 'GeneSymbol']], on='ID', how='inner')
    
    # Create a binary matrix (SNPs x cell lines) where 1 indicates a mutation
    mutation = mutation[["Chrom", "Pos", 'ModelID']].drop_duplicates().sort_values(["Chrom", "Pos"])
    mutation.index = range(mutation.shape[0])
    mutation['SNP'] = mutation['Chrom'] + '-' + mutation['Pos'].map(str)
    snp_matrix_df = pd.DataFrame(data=0, index=mutation['SNP'].unique(), columns=dependency_df.index)
    for i in range(mutation.shape[0]):
        if mutation["ModelID"][i] in snp_matrix_df.columns:
            snp_matrix_df.loc[mutation["SNP"][i]][mutation["ModelID"][i]] = 1
            
    # =========================================================================
    # Pathway and Network Structure Preparation
    # =========================================================================
    print("Constructing pathway hierarchy and network structure...")
    # Load Reactome pathway data
    pathway_genes = get_gene_pathways("../../dataset/reactome/Ensembl2Reactome_All_Levels.txt", species='human')
    pathway_names = '../../dataset/reactome/ReactomePathways.txt'
    relations_file_name = '../../dataset/reactome/ReactomePathwaysRelation.txt'
    # Generate the mask matrix and nodes in each layer for the neural network
    masking, layers_node, features_df = get_masking(pathway_names, pathway_genes, relations_file_name, features_df.T, args.encoded_dim, n_dep, n_expr, n_hidden=args.n_hidden, species='human')
    features_np = np.array(features_df.T)
    layers_node[len(layers_node) - 1] = list(features_df.index) # Ensure the final layer nodes match the input features
    
    # Define the architecture of the encoder and decoder based on the pathway hierarchy
    batch_size = features_np.shape[0]
    encoder_layer_sizes = []
    decoder_layer_sizes = [] 
    for i in range(len(masking)-1, -1, -1):
        encoder_layer_sizes.append(masking[i].shape[1])
    encoder_layer_sizes.append(masking[0].shape[0])
    encoder_layer_sizes.append(args.encoded_dim)
    decoder_layer_sizes.append(args.encoded_dim)
    decoder_layer_sizes.append(masking[0].shape[0])
    for i in range(len(masking)):
        decoder_layer_sizes.append(masking[i].shape[1])
    
    # =========================================================================
    # Autoencoder Training
    # =========================================================================
    print("Training the autoencoder...")
    # Filter out SNPs that are not present in any cell line
    snp_matrix_df = snp_matrix_df[snp_matrix_df.sum(axis=1) > 0]
    # Estimate the prior probability of a SNP occurring
    estimated_p = estimate_p(snp_matrix_df.sum(axis=1), snp_matrix_df.shape[1], initial_guess=0.8)
    
    # Train the autoencoder model
    autoencoder = train_autoencoder(features_np, masking, encoder_layer_sizes, decoder_layer_sizes, snp_matrix_df.shape[1], estimated_p, batch_size=batch_size, reg_type=args.regularization_type, alpha_reg=args.alpha_regularization, alpha_binomial=args.alpha_binomial, num_epochs=args.num_epochs, learning_rate=args.learning_rate, print_loss=False)
    
    # Use the trained encoder to generate DVR
    dvr_df = np.array(autoencoder.encoder(features_np))
    # Binarize the output to get 0s and 1s
    dvr_df = np.where(dvr_df < 0.5, 0, 1) 
    dvr_df = pd.DataFrame(dvr_df)
    dvr_df.index = snp_matrix_df.columns
    dvr_df.to_csv(args.dvr_file_name)
    print(f"DVRs saved to {args.dvr_file_name}")
    
    # =========================================================================
    # Relevance Calculation and Pathway Importance Analysis
    # =========================================================================
    print("Calculating pathway importance...")
    # Calculate relevance scores for each layer using the Layer-wise Relevance Propagation (LRP) algorithm
    relevance_scores = calculate_relevance(features_np, autoencoder, epsilon=args.epsilon)
    # Reshape the raw relevance scores into indexed DataFrames
    for i in range(len(layers_node)):
        relevance_scores[i] = pd.DataFrame(relevance_scores[i], index=features_df.columns, columns=layers_node[len(layers_node) - 1 - i])
    for i in range(1, len(layers_node)):
        relevance_scores[i + len(layers_node) - 1] = pd.DataFrame(relevance_scores[i + len(layers_node) - 1], index=features_df.columns, columns=layers_node[i])
        
    # Calculate the mean relevance for each node (pathway/gene) in each layer
    layer_mean_relevance = []
    for i in range(args.n_hidden * 2 + 3):
        layer_mean_relevance.append(relevance_scores[i].mean(axis=0))
        
    # Aggregate and rank pathways by their mean relevance scores
    all_layers_relevance_df = pd.DataFrame()
    num_pathway = []
    for i in range(args.n_hidden + 2, args.n_hidden * 2 + 2):
        all_layers_relevance_df = pd.concat([all_layers_relevance_df, layer_mean_relevance[i]])
        num_pathway += [len(layer_mean_relevance[i])] * len(layer_mean_relevance[i])
        
    # Create the final pathway importance results DataFrame and save it
    pathway_importance_df = pd.DataFrame({'layer': num_pathway, 'R': all_layers_relevance_df[0]})
    pathway_importance_df = pathway_importance_df.sort_values(by='R', ascending=False)
    pathway_importance_df.to_csv(args.pathway_file_name)
    print(f"Pathway relevance results saved to {args.pathway_file_name}")
    
if __name__ == '__main__':
    main()