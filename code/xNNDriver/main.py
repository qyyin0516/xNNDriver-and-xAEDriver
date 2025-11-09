import numpy as np
import pandas as pd
import random
import argparse
import os
from sklearn.model_selection import train_test_split
from pathway_hierarchy import *
from neural_network import *
from utils import *

# Set random seeds for reproducibility
random.seed(1999)
np.random.seed(1999)


def main():
    """
    Main function to train and evaluate a pathway-based neural network. This script iterates 
    through a list of 3,008 frequently mutated genes, training a separate, specialized model 
    for each gene to predictits mutation status from the gene dependency score.
    """

    # =========================================================================
    # Argument Parsing
    # =========================================================================
    # Defines all command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a pathway-based neural network to predict gene mutation status from gene dependency score.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help message
    )
    parser.add_argument('--input_gene', type=str, required=True,
                        help='Path to the input CSV file containing the list of frequently mutated genes.')
    parser.add_argument('--output_performance', type=str, required=True,
                        help='Path to save the output CSV file with model performance.')
    parser.add_argument('--if_functional', action='store_true',
                        help='If this flag is set, filter mutations to include only functional ones based on COSMIC and ClinVar.')
    parser.add_argument('--n_hidden', type=int, default=3,
                        help='Number of hidden layers in the neural network.')
    parser.add_argument("--learning_rate", type=float, default=0.01,
                        help="Learning rate for the Adam optimizer.")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for mini-batch gradient descent.")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of training epochs.")
    parser.add_argument("--gamma", type=float, default=0.0001,
                        help="L2 regularization parameter.")
    parser.add_argument("--score_cutoff", type=float, default=0.55,
                        help="Driver potential score threshold for saving pathway importance results.")
    parser.add_argument("--Dp_cutoff", type=float, default=0.1,
                        help="Dp cutoff for identifying important pathways.")

    args = parser.parse_args()

    # =========================================================================
    # Data Loading and Preprocessing
    # =========================================================================

    print("Loading datasets...")
    # Load gene dependency score data
    gene_dependency_df = pd.read_csv("../../dataset/CRISPRGeneEffect.csv", index_col=0)

    # Load and perform initial filtering on somatic mutation data
    mutation_df = pd.read_csv("../../dataset/OmicsSomaticMutations.csv")
    mutation_df = mutation_df[mutation_df['VariantType'] == 'SNP']
    mutation_df = mutation_df[['Chrom', 'Pos', 'HugoSymbol', 'ModelID']]

    # Filter for functional mutations if the flag is set
    if args.if_functional:
        print("Filtering for functional mutations using COSMIC and ClinVar...")
        
        # --- Process COSMIC database ---
        # Load the full COSMIC database of mutations and filter for single nucleotide variants only
        cosmic_df = pd.read_csv("../../dataset/Cosmic_MutantCensus_v102_GRCh38.tsv", delimiter='\t')
        snv_mask = cosmic_df['HGVSG'].str.contains(r'[ACGT]>[ACGT]$', na=False)
        somatic_snv_cosmic_df = cosmic_df[snv_mask].copy()
        
        # Define pathogenic types and get categories considered pathogenic
        pathogenic_impact_types = [
            'transcript_ablation', 'splice_acceptor_variant', 'splice_donor_variant',
            'stop_gained', 'frameshift_variant', 'stop_lost', 'start_lost',
            'transcript_amplification', 'inframe_insertion', 'inframe_deletion',
            'missense_variant', 'protein_altering_variant', 'splice_region_variant',
            'start_retained_variant'
        ]
        impact_mask = somatic_snv_cosmic_df['MUTATION_DESCRIPTION'].isin(pathogenic_impact_types)
        pathogenic_cosmic_df = somatic_snv_cosmic_df[impact_mask].copy()
        
        # Standardize and prepare the pathogenic COSMIC data for merging
        pathogenic_cosmic_df = pathogenic_cosmic_df[['GENE_SYMBOL', 'CHROMOSOME', 'GENOME_START']]
        pathogenic_cosmic_df.rename(columns={
            'GENE_NAME': 'GeneSymbol',
            'CHROMOSOME': 'Chromosome',
            'GENOME_START': 'Start'
        }, inplace=True)
        pathogenic_cosmic_df['Chromosome'] = pathogenic_cosmic_df['Chromosome'].apply(lambda x: 'chr' + str(x))
        pathogenic_cosmic_df['Start'] = pathogenic_cosmic_df['Start'].astype(int)
        # Create a unique ID for each mutation based on its genomic location
        pathogenic_cosmic_df['ID'] = pathogenic_cosmic_df['Chromosome'] + '-' + pathogenic_cosmic_df['Start'].astype(str)
        
        # --- Process ClinVar database ---
        # Load the full ClinVar database of mutations and filter for single nucleotide variants only
        clinvar_df = pd.read_csv("../../dataset/ClinVar/ClinVar_variant_summary.txt", delimiter='\t', low_memory=False)
        clinvar_df = clinvar_df[clinvar_df['Assembly'] == "GRCh38"]
        clinvar_df = clinvar_df[clinvar_df['Type'] == "single nucleotide variant"]
        
        # Load pathogenicity mapping and get categories considered pathogenic
        pathogenicity_map_df = pd.read_csv("../../dataset/ClinVar/pathogenicity.csv", index_col=0)
        pathogenic_categories = list(pathogenicity_map_df[pathogenicity_map_df['Pathogenicity'] == 'Y']['Category'])
        pathogenic_clinvar_df = clinvar_df[clinvar_df['ClinicalSignificance'].isin(pathogenic_categories)]
        
        # Standardize and prepare the pathogenic ClinVar data for merging
        pathogenic_clinvar_df = pathogenic_clinvar_df[["Chromosome", "Start", "GeneSymbol"]]
        pathogenic_clinvar_df['Chromosome'] = pathogenic_clinvar_df['Chromosome'].apply(lambda x: 'chr' + str(x))
        pathogenic_clinvar_df = pathogenic_clinvar_df[pd.to_numeric(pathogenic_clinvar_df['Start'], errors='coerce').notna()]
        pathogenic_clinvar_df['Start'] = pathogenic_clinvar_df['Start'].astype(int)
        # Create the same unique ID format as COSMIC
        pathogenic_clinvar_df['ID'] = pathogenic_clinvar_df['Chromosome'] + '-' + pathogenic_clinvar_df['Start'].astype(str)
        
        # --- Combine databases and filter the main mutation DataFrame ---
        # Combine unique functional mutation IDs from COSMIC and ClinVar
        cosmic_ids_df = pathogenic_cosmic_df[['ID', 'GeneSymbol']]
        clinvar_ids_df = pathogenic_clinvar_df[['ID', 'GeneSymbol']]
        functional_mutations_df = pd.concat([cosmic_ids_df, clinvar_ids_df], ignore_index=True)
        functional_mutations_df.drop_duplicates(subset='ID', keep='first', inplace=True)
        
        # Filter the main mutation_df to keep only mutations present in our functional list
        mutation_df['ID'] = mutation_df['Chrom'] + '-' + mutation_df['Pos'].astype(str)
        mutation_df = pd.merge(mutation_df, functional_mutations_df[['ID']], on='ID', how='inner')

    # Clean up the mutation dataframe
    mutation_df = mutation_df[['ModelID', 'HugoSymbol']].drop_duplicates().sort_values(["HugoSymbol", "ModelID"])
    mutation_df.index = range(mutation_df.shape[0])

    # Clean up the dependency score dataframe
    gene_column_names = [col.split(' ')[0] for col in gene_dependency_df.columns]
    gene_dependency_df.columns = gene_column_names
    screened_gene_info = pd.read_csv("../../dataset/InputGene/ScreenedGene.csv")
    screened_gene_info = screened_gene_info.drop_duplicates(subset=['From'], keep='first')
    gene_dependency_df = gene_dependency_df[screened_gene_info.iloc[:, 0]]
    gene_dependency_df.columns = screened_gene_info.iloc[:, 1]
    null_counts = gene_dependency_df.isnull().sum()
    gene_dependency_df = gene_dependency_df[null_counts[null_counts == 0].index]

    # =========================================================================
    # Label Matrix Generation
    # =========================================================================
    
    print("Generating binary label matrix based on mutation status...")
    # Create a binary label matrix (genes x cell lines) where 1 indicates a mutation
    target_genes_df = pd.read_csv(args.input_gene)
    label_matrix_df = pd.DataFrame(data=0, index=target_genes_df.iloc[:, 0], columns=gene_dependency_df.index)
    for i in range(label_matrix_df.shape[0]):
        target_gene = label_matrix_df.index[i]
        gene_specific_mutations = mutation_df[mutation_df['HugoSymbol'] == target_gene]
        cell_lines_with_mutation = list(set(gene_specific_mutations['ModelID']) & set(label_matrix_df.columns))
        label_matrix_df.loc[target_gene, cell_lines_with_mutation] = 1
    
    # =========================================================================
    # Model Training and Evaluation Loop
    # =========================================================================
    
    results_df = pd.DataFrame(columns=['driver potential score'])
    # Iterate over each target gene to train a specific model
    for i in range(label_matrix_df.shape[0]):
        target_gene = label_matrix_df.index[i]
        print(f"\n--- Training model for gene: {target_gene} ({i+1}/{label_matrix_df.shape[0]}) ---")
    
        # --- Data splitting ---
        x_train, x_test, y_train, y_test = train_test_split(
            gene_dependency_df, label_matrix_df.iloc[i, :], test_size=0.25, random_state=1999
        )
        y_train = pd.DataFrame(y_train)
        y_test = pd.DataFrame(y_test)
    
        # --- Pathway and network structure preparation ---
        pathway_genes = get_gene_pathways("../../dataset/reactome/Ensembl2Reactome_All_Levels.txt", species='human')
        pathway_names = '../../dataset/reactome/ReactomePathways.txt'
        relations_file_name = '../../dataset/reactome/ReactomePathwaysRelation.txt'
        masking, layers_node, gene_out = get_masking(
            pathway_names, pathway_genes, relations_file_name, x_train.T.index.tolist(),
            root_name=[0, 1], n_hidden=args.n_hidden
        )
        # Filter features to only include genes present in the final pathway structure
        x_train = x_train.T.loc[gene_out, :]
        x_test = x_test.T.loc[gene_out, :]
    
        # --- Handle class imbalance using random oversampling ---
        if y_train.iloc[:, 0].sum() > 0:
            train_df_for_resample = x_train.T.copy()
            train_df_for_resample['label'] = y_train.iloc[:, 0]
    
            majority_class_df = train_df_for_resample[train_df_for_resample['label'] == 0]
            minority_class_df = train_df_for_resample[train_df_for_resample['label'] == 1]
    
            rng = np.random.RandomState(1999)
            resample_indices = rng.choice(minority_class_df.index, size=len(majority_class_df), replace=True)
            upsampled_minority_df = minority_class_df.loc[resample_indices]
    
            resampled_train_df = pd.concat([upsampled_minority_df, majority_class_df])
            
            y_train = pd.DataFrame(resampled_train_df['label'])
            x_train = resampled_train_df.drop('label', axis=1).T
    
        # --- Train an ensemble of networks with varying depths ---
        y_test_pred_df = pd.DataFrame(data=0, index=x_test.columns, columns=list(range(2, len(masking) + 2)))
        activation_output = {}
        for num_layers in range(2, len(masking) + 2):
            print(f"Training with {num_layers - 1} hidden layers...")
            output_train, output_test = model(
                np.array(x_train), one_hot_coding(y_train), np.array(x_test),
                layers_node, masking, num_layers,
                learning_rate=args.learning_rate,
                minibatch_size=args.batch_size,
                num_epochs=args.num_epochs,
                gamma=args.gamma,
                print_cost=False
            )
    
            # Reshape raw model outputs into indexed DataFrames for later analysis
            for j in range(len(output_train)):
                if (j != num_layers - 1):
                    output_train[j + 1] = pd.DataFrame(data=output_train[j + 1],
                                                       index=layers_node[len(layers_node) - 2 - j],
                                                       columns=x_train.columns)
                else:
                    output_train[j + 1] = pd.DataFrame(data=output_train[j + 1], index=[0, 1],
                                                       columns=x_train.columns)
            activation_output[num_layers] = output_train
    
            # Store test set predictions for this model depth
            _, y_test_pred = get_predictions(output_train, output_test, num_layers)
            y_test_pred_df.loc[:, num_layers] = pd.DataFrame(y_test_pred, index=x_test.columns, columns=[num_layers])
    
        # Ensemble predictions by taking the mode across all trained models
        y_test_pred_final = y_test_pred_df.T.mode().T.iloc[:, 0]
    
        # --- Calculate and save performance ---
        driver_potential_score = get_driver_potential_score(y_test, y_test_pred_final)
        results_df.loc[target_gene] = [driver_potential_score]
        print(f"Finished training for {target_gene}. Driver potential score: {driver_potential_score:.4f}")
        results_df.to_csv(args.output_performance)
    
        # --- Pathway importance analysis for high-performing models ---
        if results_df.loc[target_gene, 'driver potential score'] >= args.score_cutoff:
            print(f"High performance detected. Calculating pathway importance for {target_gene}...")
            pathways = get_pathway_importance(y_train, activation_output, thr=args.Dp_cutoff)
            pathway_output_filename = f"pathways_{target_gene}.csv"
            pathways.to_csv(pathway_output_filename)
            print(f"Pathway importance saved to {pathway_output_filename}")


if __name__ == '__main__':
    main()