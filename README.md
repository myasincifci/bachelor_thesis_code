# This is the repository for my bachelor thesis

## Installation
Clone the repository and run:
    
    pip install .

## Training
    python tools/train_net.py \
        --lr 0.001 \
        --l 0.001 \
        --epochs 10 \
        --proximity 30 \
        --save_model tempo.pth \
        --eval True

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