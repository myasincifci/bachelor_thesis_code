# Bachelor's Thesis: Exploiting Temporal Coherence in Video to Learn Useful Features
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
        --epochs 15 \
        --lr 0.001 \
        --l 0.001 \
        --pdf uniform \
        --proximity 30 \
        --save_model custom.pth \
        --eval True

<ol>
  <li><b>epochs (int)</b>: Number of epochs that the backbone is trained for.</li>
  <li><b>lr (float)</b>: SGD learning rate.</li>
  <li><b>l (float)</b>: Lambda parameter for Barlow Twins loss.</li>
  <li><b>eval (bool)</b>: Perform linear evaluation after training.</li>
  <li><b>baseline (bool)</b>: Create baseline model (no training).</li>  
  <li><b>pdf (str)</b>: Specify pdf for sampling (uniform or normal).</li>
  <li><b>proximity (int)</b>: Proximity parameter for pdf. Tau or Sigma depending on used pdf.</li>
  <li><b>save_model (str)</b>: Name of model checkpoint. Not saved if not specified.</li>

</ol>

# Evaluation
Linear and Semi-Supervised evaluation can be performed with the following scripts (results are written to tensorboard at ./runs):
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
        --name tempo_semi_sup_eval \
        --samples_pc 20

<ol>
    <li><b>path (str)</b>: Path to model checkpointy to be evaluated.</li>
    <li><b>runs (int)</b>: Number of eval. runs to be performed. Mean is reported.</li>
    <li><b>name (str)</b>: Name of run used in tensorboard.</li>
    <li><b>samples_pc (int)</b>: Number of samples per class in evaluation set. 20, 15, 10, 5, 1 possible, runs on complete set if not specified.</li>
</ol>

## Object Detection/ Instance Segmentation 
### Installation
Before running the segmentation example, please install the
detectron2 library following the [installation instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

## Training backone
An example for training a ResNet50 Tempo Backbone is provided in the following [jupyter-notebook](segmentation/resnet50.ipynb).

### Running

    python segmentation/segmentation.py \
        --path ./model_zoo/tempo50.pkl

Results will be written to tensorboard. 

# Explanation
Code for the explanation section is provided in the [xai subdirectory](xai).
## LRP
[lrp.ipynb](xai/lrp/lrp.ipynb) will create heatmaps for the complete test set.
## Adversarial Attack
[adversarial.ipynb](xai/adversarial_attack/adversarial.ipynb) shows additional examples for the adversarial attack experiment.
## Relative Relevance
[relative_relevance.ipynb](xai/relative_relevance/relative_relevance.ipynb) contains the code for computing the relative relevance over the complete test set.

Resulting heatmaps are written to the [supplementary_material](supplementary_material) directory.