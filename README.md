# Bachelor's Thesis: Exploiting Temporal Coherence in Video


## Installation
Clone the repository and run:
    
    pip install .

## Download Datasets
    python datasets/download_data.py

## Download pretrained models
    python model_zoo/download_models.py

## Training
To train a tempo feature extractor 

    python tools/train_net.py \
        --lr 0.001 \ # SGD learning rate 
        --l 0.001 \ # BarlowTwins loss lambda parameter
        --epochs 10 \ # Number of epochs (complete video)
        --pdf uniform \ # Sampling pdf
        --proximity 30 \ # Proximity (tau or sigma depending on pdf)
        --save_model tempo.pth \ # Checkpoint name (not saved if none)
        --eval True # Perform linear eval at end

## Linear Evaluation
    python tools/linear_eval.py \
        --path model_zoo/tempo.pth \
        --runs 10 \
        --name tempo_linear_eval \
        --samples_pc 20

## Semi-Supervised Evaluation
    python tools/semi_sup_eval.py \
        --path model_zoo/tempo.pth \
        --runs 10 \
        --name tempo_semi_sup_eval