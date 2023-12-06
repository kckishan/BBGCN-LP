# Predicting Biomedical Interactions with Adaptive Graph Neural Network Structures

### Requirements
The codebase is implemented in Python 3.6.9 and the packages used for development are mentioned below.

```
argparse                1.1
numpy                   1.19.1
torch                   1.5.0
torch_sparse            0.6.4
pandas                  1.0.1
scikit-learn            0.22.1
matplotlib              3.2.2
scipy                   1.5.0
texttable               1.6.2
```

### Datasets
The details about dataset used in the experiments are provided in [README](https://github.com/kexinhuang12345/SkipGNN#dataset).

Train BBGCN-LP on DTI network:

```train
python3 main.py --network_type 'DTI'  
```
