# Explainable deep learning for identifying cancer driver genes based on the Cancer Dependency Map


In this project, we utilized the Cancer Dependency Map (DepMap) to identify potential cancer driver genes and generate artificial representative driver variants. Using DepMap dependency score data, we developed a biologically informed, supervised deep learning model for each frequently mutated gene, leveraging functional SNPs to predict its mutation status. The fitness of the model represents how likely the gene is a cancer driver gene. To generate driver variant representations (DVRs), we built an explainable autoencoder, guided by the distribution of real driver SNPs. This allowed us to identify important pathways for the dependence profile of a cell line. 

This project corresponds to the following paper: Yin, Q., Chen, L.. Explainable deep learning for identifying cancer driver genes based on the Cancer Dependency Map, bioRxiv. https://www.biorxiv.org/content/10.1101/2025.04.28.651122

## Dependencies
The models are built with Python 3 (>= 3.10.16) with the following packages:

* numpy >= 1.21.6
* pandas >= 1.5.3
* scipy >= 1.11.2
* keras >= 2.9.0
* tensorflow >= 2.9.0
* scikit-learn >= 1.0.2
* networkx >= 2.6.3

## Installation
Clone the github repository and enter xNNDriver-and-xAEDriver directory with

    $ git clone https://github.com/qyyin0516/xNNDriver-and-xAEDriver.git
    $ cd xNNDriver-and-xAEDriver
  
However, the folder `xNNDriver-and-xAEDriver/dataset` is stored in Google Drive because of the file size limitations of GitHub. Please download the folder via https://drive.google.com/drive/folders/1CWI-P40QcIpNmYxleX5y-6KvhfznYadg?usp=sharing. Thank you! 

## Usage
Executing `code/gene/main.py` evaluates the supervised models (xNNDriver), providing driver potential scores along with identifying important pathways. Users need to specify the input gene list and the output file name for the fitness results of all genes. Similarly, executing `code/mutation/main.py` (xAEDriver) generates DVRs and calculates the relevance scores of pathways, averaged across all cell lines. Users should specify the output file name for the DVRs and pathways.

The list below is the options for `code/gene/main.py`.


    --input_gene                    path to the input CSV file containing the list of genes (required)
    --output_performance            path to save the output CSV file with model performance (required)
    --if_functional                 if this flag is set, filter mutations to include only functional ones based on COSMIC and ClinVar (optional, default: True)
    --n_hidden                      number of hidden layers in the neural network (optional, default: 3)
    --learning_rate                 learning rate for the Adam optimizer (optional, default: 0.01)
    --batch_size                    batch size for mini-batch gradient descent (optional, default: 128)
    --num_epochs                    number of training epochs (optional, default: 100)
    --gamma                         L2 regularization parameter (optional, default: 0.0001)
    --score_cutoff                  driver potential score threshold for saving pathway importance results (optional, default: 0.55)
    --Dp_cutoff                     Dp cutoff for identifying important pathways (optional, default: 0.1)

Here is an example.

    $ python code/gene/main.py  --input_gene "../../dataset/InputGene/3008Gene.csv"\
                                --output_performance "output_performance.csv"\

The list below is the options for `code/mutation/main.py`.

    --dvr_file_name                 path to save the output CSV file for generated DVRs (required)
    --pathway_file_name             path to save the output CSV file with pathway relevance scores (required)
    --encoded_dim                   dimension of the encoded layer (DVR) (optional, default: 1024)
    --n_hidden                      number of hidden layers in the neural network (optional, default: 3)
    --learning_rate                 learning rate for the Adam optimizer (optional, default: 0.01)
    --num_epochs                    maximum number of training epochs (optional, default: 1000)
    --alpha_binomial                weight for the binomial distribution loss component in the autoencoder (optional, default: 0.001)
    --alpha_regularization          weight for the regularization term (optional, default: 0.1)
    --regularization_type           type of regularization to apply ('L1' or 'L2') (optional, default: 'L2')
    --epsilon                       a small constant for the relevance calculation to ensure numerical stability (optional, default: 0.01)

Here is an example.

    $ python code/mutation/main.py  --fake_SNP_file_name "output_fake_SNP.csv"\
                                    --pathway_file_name "output_pathway.csv"\
