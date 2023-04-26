# GLSPPIS
We developed a deep residual graph framework, called GLSPPIS, that combines global and local spatial features for PPI site prediction. GLSPPIS utilizes a novel residual graph connection to address the gradient vanishing problem in GCN. The residual graph connection includes node feature representation, the output of the previous layer, and local spatial features.

# System requirement
## To run the code, you need the following dependencies:
python 3.7.7
numpy 1.19.1
pandas 1.1.0
torch 1.6.0
scikit-learn 0.23.2

# Dataset
Before run the code, you need to unzip the compressed files in the Feature folder.

# Usage
## for training the model you need to run:
python train.py

## for evaluating the performance of our model, you need to run:
python test.py

