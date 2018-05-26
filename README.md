# Overview

This repository contains the source code of the models submitted 
by NTUA-SLP team in SemEval 2018 tasks 1, 2 and 3.
- Task 1: Affect in Tweets https://arxiv.org/abs/1804.06658
- Task 2: Multilingual Emoji Prediction https://arxiv.org/abs/1804.06657
- Task 3: Irony Detection in English Tweets https://arxiv.org/abs/1804.06659


### Project Structure
- `datasets`: contains the datasets for the pretrainig (SemEval 2017 - Task4A) 
- `dataloaders` - contains scripts for loading the datasets
              and for tasks 1, 2 and 3 
- `embeddings`: in this folder you should put the word embedding files.
- `logger`: contains the source code for the `Trainer` class 
            and the accompanying helper functions for experiment management,
            including experiment logging, checkpoint and early-stoping mechanism
            and visualization of the training process.
- `model`: experiment runner scripts (dataset loading, training pipeline etc).
    - `pretraining`: the scripts for training the TL models
    - `task1`: the scripts for running the models for Task 1
    - `task2`: the scripts for running the models for Task 2
    - `task3`: the scripts for running the models for Task 3
- `modules`: the source code of the PyTorch deep-learning models 
             and the baseline models.
    - `nn`: the source code of the PyTorch modules
    - `sklearn`: scikit-learn Transformers for implementing the baseline 
                bag-of-word and neural bag-of-words models
- `out`: this directory contains the generated model predictions and 
            their corresponding attention files
- `predict`: scripts for generating predictions from saved models.
- `trained`: this is where all the model checkpoints are saved.
- `utils`: contains helper functions

Full documentation of the source code will be posted soon.
