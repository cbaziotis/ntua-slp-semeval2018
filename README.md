# Overview

This repository contains the source code of the models submitted 
by NTUA-SLP team in SemEval 2018 tasks 1, 2 and 3.
- Task 1: Affect in Tweets https://arxiv.org/abs/1804.06658
- Task 2: Multilingual Emoji Prediction https://arxiv.org/abs/1804.06657
- Task 3: Irony Detection in English Tweets https://arxiv.org/abs/1804.06659


## Prerequisites
Please follow the steps below in order to be able to train our models:

#### 1 - Install Requirements
```
pip install -r ./requirements.txt
```

#### 2 - Download our pre-trained word embeddings
The models were trained on top of word2vec embeddings pre-trained 
on a big collection of Twitter messages. We collected a big dataset of 
550M English Twitter messages posted from 12/2014 to 06/2017. 
For training the word embeddings we used 
[Gensim's implementation](https://radimrehurek.com/gensim/) 
of word2vec.
For preprocessing the tweets we used [ekphrasis](https://github.com/cbaziotis/ekphrasis).
Finally, used the following parameteres for training the word2vec embeddings: 
`window_size = 6`, `negative_sampling = 5` and `min_count = 20`.

You can download one of the following word embeddings:
- [ntua_twitter_300.txt](https://drive.google.com/open?id=1b-w7xf0d4zFmVoe9kipBHUwfoefFvU2t): 
300 dimensional embeddings.
- [ntua_twitter_affect_310.txt](https://drive.google.com/open?id=11zrXc1h_saJsMT6eo0VARKeZuzvK2bU0): 
310 dimensional embeddings, consisting of 300d word2vec embeddings + 10 affective dimensions.

**Important**: Finally, put the embeddings file in `/embeddings` folder.

#### 3 - Update mode configs
Our model definitions are stored in a python configuration file. 
Each config contains the model parameters and things like the batch size, 
number of epochs and embeddings file. You should update the 
`embeddings_file` parameter in the model's configuration in `model/params.py`.


### Example - Sentiment Analysis on SemEval 2017 Task 4A
You can test that you have a working setup by training 
a sentiment analysis model on [SemEval 2017 Task 4A](http://alt.qcri.org/semeval2017/task4/), 
which is used for pretraining for Task 1.  
```bash
python model/pretraining/sentiment2017.py
```

# Documentation 

### Project Structure
In order to make our codebase more accessible and easier to extend, 
we provide an overview of the structure of our project. 
The most important parts will be covered in greater detail.

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


**Note**: Full documentation of the source code will be posted soon.
