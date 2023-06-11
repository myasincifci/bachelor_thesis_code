# This is the repository for my bachelor thesis

## Installation
Clone the repository and run:
    
    pip install .

## Training
    python tools/train_net.py \
        --epochs 15 \
        --proximity 15 \
        --save_model tempo.pth \
        --eval True

## Linear Evaluation
    python tools/linear_eval.py \
        --path model_zoo/tempo.pth \
        --runs 10 \
        --name tempo_linear_eval

## Semi-Supervised Evaluation
    python tools/semi_sup_eval.py \
        --path model_zoo/tempo.pth \
        --runs 10 \
        --name tempo_semi_sup_eval