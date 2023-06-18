# Bachelor's Thesis: Exploiting Temporal Coherence in Video
<b>Abstract</b>: 
Pre-training convolutional neural networks on large-scale datasets like ImageNet is a well-established paradigm for transferring knowledge to a target task through features that have been learned in the previous training step. However, if there is video footage of the target tasks’ data-generating process, we are arguing that more useful features can be learned by pre-training on that video data, which will ultimately lead to a better downstream performance on the target-task. In this thesis, weare evaluating the effectiveness of this approach by developing a self-supervised method for learn-ing image features from video by exploiting its underlying temporal coherence and evaluating its effectiveness on various computer-vision tasks for American Sign Language hand signs. We are reporting a significant improvement in image-classification accuracy, which becomes especially apparent for small training sets, and a reduction in training time for object detection and instance segmentation.

[Thesis](main.pdf)

# Installation
Clone the repository and run:
    
    pip install .

## Download Datasets
    python datasets/download_data.py

## Download pretrained models
    python model_zoo/download_models.py

# Training a Tempo backbone
A script for training a Tempo backbone is provided in [train_net.py](tools/train_net.py). Run with the following command:

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

# Evaluation
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

# Explanation
