# RGCNPPIS
We propose a novel residual graph convolutional network for structure-based PPI site prediction (RGCNPPIS). Specifically, we use a GCN module to extract the global structural features from all spatial neighborhoods, and utilize the GraphSage module to extract local structural features from local spatial neighborhoods. To the best of our knowledge, this is the first work utilizing local structural features for PPI site prediction. Besides, we propose an enhanced residual graph connection enables information transfer between layers and alleviates the over-smoothing problem.

## System requirement
### To run the code, you need the following dependencies:
python 3.7.7
numpy 1.19.1
pandas 1.1.0
torch 1.6.0
scikit-learn 0.23.2

## Dataset
Before run the code, you need to unzip the compressed files in the Feature folder.

# Usage
### for training the model you need to run:
    python train.py

### for evaluating the performance of our model, you need to run:
    python test.py

