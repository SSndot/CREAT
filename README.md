# Potent but Stealthy: A Constrained Reinforcement Driven Attack against Sequential Recommendation
### Introduction
We propose a constrained reinforcement driven attack CREAT that synergizes a bi-level optimization framework with multi-reward reinforcement learning to balance adversarial efficacy and stealthiness. We first develop a Pattern Balanced Rewarding Policy, which integrates pattern inversion rewards to invert critical patterns and distribution consistency rewards to minimize detectable shifts via unbalanced co-optimal transport. Then we employ a Constrained Group Relative Reinforcement Learning paradigm, enabling step-wise perturbations through dynamic barrier constraints and group-shared experience replay, achieving targeted pollution with minimal detectability.
### Datasets
The interaction dataset employed in this research was sourced from publicly available repositories *[Google Drive](https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI)*, and we use the "5-core" datasets as described in our paper.
### Backbone Models
We provide two backbone models, NARM and Bert4Rec, as described in the paper. You can modify the settings in `Rec_models`.
### Attack Implementation
You can modify the experiment configuration in `config.py` and run the following program to generate poisoned profiles.
```
python train.py
```
