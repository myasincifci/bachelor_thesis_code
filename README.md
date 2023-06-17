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
        --lr 0.001 \  
        --l 0.001 \ 
        --epochs 10 \ 
        --pdf uniform \ 
        --proximity 30 \ 
        --save_model tempo.pth \ 
        --eval True 

SGD learning rate
BarlowTwins loss lambda parameter
Number of epochs (complete video)
Sampling pdf
Proximity (tau or sigma depending on pdf)
Checkpoint name (not saved if none)
Perform linear eval at end

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

## Segmentation 
### Installation
Before running the segmentation example, please install the
detectron2 library following the [installation instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).
### Running
To run ...

## Explanation
