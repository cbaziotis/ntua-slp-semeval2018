"""
This file contains functions with logic that is used in almost all models,
with the goal of avoiding boilerplate code (and bugs due to copy-paste),
such as training pipelines.
"""
import glob
import math
import os
import pickle

import numpy
import torch
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, recall_score, accuracy_score, \
    precision_score, jaccard_similarity_score
from torch.nn import ModuleList
from torch.utils.data import DataLoader

from config import TRAINED_PATH, BASE_PATH, DEVICE
from logger.training import class_weigths, Checkpoint, EarlyStop, Trainer
from modules.nn.dataloading import WordDataset, CharDataset
from modules.nn.models import ModelWrapper
from utils.load_embeddings import load_word_vectors
from utils.nlp import twitter_preprocess


def load_pretrained_model(name):
    model_path = os.path.join(TRAINED_PATH, "{}.model".format(name))
    model_conf_path = os.path.join(TRAINED_PATH, "{}.conf".format(name))
    model = torch.load(model_path)
    model_conf = pickle.load(open(model_conf_path, 'rb'))

    return model, model_conf


def load_pretrained_models(name):
    models_path = os.path.join(TRAINED_PATH)
    fmodel_confs = sorted(glob.glob(os.path.join(models_path,
                                                 "{}*.conf".format(name))))
    fmodels = sorted(glob.glob(os.path.join(models_path,
                                            "{}*.model".format(name))))
    for model, model_conf in zip(fmodels, fmodel_confs):
        print("loading model {}".format(model))
        yield torch.load(model), pickle.load(open(model_conf, 'rb'))


def get_pretrained(pretrained):
    if isinstance(pretrained, list):
        pretrained_models = []
        pretrained_config = None
        for pt in pretrained:
            pretrained_model, pretrained_config = load_pretrained_model(pt)
            pretrained_models.append(pretrained_model)
        return pretrained_models, pretrained_config
    else:
        pretrained_model, pretrained_config = load_pretrained_model(pretrained)
        return pretrained_model, pretrained_config


def load_datasets(datasets, train_batch_size, eval_batch_size, token_type,
                  preprocessor=None,
                  params=None, word2idx=None, label_transformer=None):
    if params is not None:
        name = "_".join(params) if isinstance(params, list) else params
    else:
        name = None

    loaders = {}
    if token_type == "word":
        if word2idx is None:
            raise ValueError

        if preprocessor is None:
            preprocessor = twitter_preprocess()

        print("Building word-level datasets...")
        for k, v in datasets.items():
            _name = "{}_{}".format(name, k)
            dataset = WordDataset(v[0], v[1], word2idx, name=_name,
                                  preprocess=preprocessor,
                                  label_transformer=label_transformer)
            batch_size = train_batch_size if k == "train" else eval_batch_size
            loaders[k] = DataLoader(dataset, batch_size, shuffle=True,
                                    drop_last=True)

    elif token_type == "char":
        print("Building char-level datasets...")
        for k, v in datasets.items():
            _name = "{}_{}".format(name, k)
            dataset = CharDataset(v[0], v[1], name=_name,
                                  label_transformer=label_transformer)
            batch_size = train_batch_size if k == "train" else eval_batch_size
            loaders[k] = DataLoader(dataset, batch_size, shuffle=True,
                                    drop_last=True)

    else:
        raise ValueError("Invalid token_type.")

    return loaders


def load_embeddings(
        model_conf,
        absolute_path=False, embedding_size_auto_detect=None):
    if not absolute_path:
        word_vectors = os.path.join(
            BASE_PATH, "embeddings",
            "{}.txt".format(model_conf["embeddings_file"]))
    else:
        '''Absolute Path.'''
        word_vectors = model_conf["embeddings_file"]

    if embedding_size_auto_detect is not None:
        word_vectors_size = detect_embedding_dim(word_vectors)
    else:
        word_vectors_size = model_conf["embed_dim"]

    # load word embeddings
    print("loading word embeddings...")
    return load_word_vectors(word_vectors, word_vectors_size)


def detect_embedding_dim(embedding_file):
    """
    Auto detecting the dimensionality of embedding file.
    :param embedding_file:
    :return:
    """
    with open(embedding_file, 'r', encoding="utf8") as file:
        for line in file:
            return len(line.strip().split()) - 1


def get_pipeline(task, criterion=None, eval=False):
    """
    Generic classification pipeline
    Args:
        task (): available tasks
                - "clf": multiclass classification
                - "bclf": binary classification
                - "mclf": multilabel classification
                - "reg": regression
        criterion (): the loss function
        eval (): set to True if the pipeline will be used
            for evaluation and not for training.
            Note: this has nothing to do with the mode
            of the model (eval or train). If the pipeline will be used
            for making predictions, then set to True.

    Returns:

    """

    def pipeline(nn_model, curr_batch):
        # get the inputs (batch)
        inputs, labels, lengths, indices = curr_batch

        if task in ["reg", "mclf"]:
            labels = labels.float()

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        lengths = lengths.to(DEVICE)

        outputs, attentions = nn_model(inputs, lengths)

        if eval:
            return outputs, labels, attentions, None

        if task == "bclf":
            loss = criterion(outputs.view(-1), labels.float())
        else:
            loss = criterion(outputs.squeeze(), labels)

        return outputs, labels, attentions, loss

    return pipeline


def calc_pearson(y, y_hat):
    score = pearsonr(y, y_hat)[0]
    if math.isnan(score):
        return 0
    else:
        return score


def get_metrics(task, ordinal):
    _metrics = {
        "reg": {
            "pearson": calc_pearson,
        },
        "bclf": {
            "acc": lambda y, y_hat: accuracy_score(y, y_hat),
            "precision": lambda y, y_hat: precision_score(y, y_hat,
                                                          average='macro'),
            "recall": lambda y, y_hat: recall_score(y, y_hat,
                                                    average='macro'),
            "f1": lambda y, y_hat: f1_score(y, y_hat,
                                            average='macro'),
        },
        "clf": {
            "acc": lambda y, y_hat: accuracy_score(y, y_hat),
            "precision": lambda y, y_hat: precision_score(y, y_hat,
                                                          average='macro'),
            "recall": lambda y, y_hat: recall_score(y, y_hat,
                                                    average='macro'),
            "f1": lambda y, y_hat: f1_score(y, y_hat,
                                            average='macro'),
        },
        "mclf": {
            "jaccard": lambda y, y_hat: jaccard_similarity_score(
                numpy.array(y), numpy.array(y_hat)),
            "f1-macro": lambda y, y_hat: f1_score(numpy.array(y),
                                                  numpy.array(y_hat),
                                                  average='macro'),
            "f1-micro": lambda y, y_hat: f1_score(numpy.array(y),
                                                  numpy.array(y_hat),
                                                  average='micro'),
        },
    }
    _monitor = {
        "reg": "pearson",
        "bclf": "f1",
        "clf": "f1",
        "mclf": "jaccard",
    }
    _mode = {
        "reg": "max",
        "bclf": "max",
        "clf": "max",
        "mclf": "max",
    }

    if ordinal:
        task = "reg"

    metrics = _metrics[task]
    monitor = _monitor[task]
    mode = _mode[task]

    return metrics, monitor, mode


def define_trainer(task,
                   config,
                   name,
                   datasets,
                   monitor,
                   ordinal=False,
                   pretrained=None,
                   finetune=None,
                   label_transformer=None,
                   disable_cache=False,
                   absolute_path=False,
                   embedding_size_auto_detect=None):
    """

    Args:
        task (): available tasks
                - "clf": multiclass classification
                - "bclf": binary classification
                - "mclf": multilabel classification
                - "reg": regression
        config ():
        name ():
        datasets ():
        monitor ():
        ordinal ():
        pretrained ():
        finetune ():
        label_transformer ():
        disable_cache ():

    Returns:

    """
    ########################################################################
    # Load pre:trained models
    ########################################################################

    if task == "bclf":
        task = "clf"

    pretrained_models = None
    pretrained_config = None
    if pretrained is not None:
        pretrained_models, pretrained_config = get_pretrained(pretrained)

    if pretrained_config is not None:
        _config = pretrained_config
    else:
        _config = config

    ########################################################################
    # Load embeddings
    ########################################################################
    word2idx = None
    if _config["token_type"] == "word":
        word2idx, idx2word, embeddings = load_embeddings(_config,
                                                         absolute_path,
                                                         embedding_size_auto_detect)

    ########################################################################
    # DATASET
    # construct the pytorch Datasets and Dataloaders
    ########################################################################
    loaders = load_datasets(datasets,
                            train_batch_size=_config["batch_train"],
                            eval_batch_size=_config["batch_eval"],
                            token_type=_config["token_type"],
                            params=None if disable_cache else name,
                            word2idx=word2idx,
                            label_transformer=label_transformer)

    ########################################################################
    # MODEL
    # Define the model that will be trained and its parameters
    ########################################################################
    out_size = 1
    if task == "clf":
        classes = len(set(loaders["train"].dataset.labels))
        out_size = 1 if classes == 2 else classes
    elif task == "mclf":
        out_size = len(loaders["train"].dataset.labels[0])

    num_embeddings = None

    if _config["token_type"] == "char":
        num_embeddings = len(loaders["train"].dataset.char2idx) + 1
        embeddings = None

    model = ModelWrapper(embeddings=embeddings,
                         out_size=out_size,
                         num_embeddings=num_embeddings,
                         pretrained=pretrained_models,
                         finetune=finetune,
                         **_config)
    model.to(DEVICE)
    print(model)

    if task == "clf":
        weights = class_weigths(loaders["train"].dataset.labels,
                                to_pytorch=True)
    if task == "clf":
        weights = weights.to(DEVICE)

    ########################################################################
    # Loss function and optimizer
    ########################################################################
    if task == "clf":
        if out_size > 2:
            criterion = torch.nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = torch.nn.BCEWithLogitsLoss()
    elif task == "reg":
        criterion = torch.nn.MSELoss()
    elif task == "mclf":
        criterion = torch.nn.MultiLabelSoftMarginLoss()
    else:
        raise ValueError("Invalid task!")

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters,
                                 weight_decay=config["weight_decay"])

    ########################################################################
    # Trainer
    ########################################################################
    if task == "clf":
        pipeline = get_pipeline("bclf" if out_size == 1 else "clf", criterion)
    else:
        pipeline = get_pipeline("reg", criterion)

    metrics, monitor_metric, mode = get_metrics(task, ordinal)

    checkpoint = Checkpoint(name=name, model=model, model_conf=config,
                            monitor=monitor, keep_best=True, scorestamp=True,
                            metric=monitor_metric, mode=mode,
                            base=config["base"])
    early_stopping = EarlyStop(metric=monitor_metric, mode=mode,
                               monitor=monitor,
                               patience=config["patience"])

    trainer = Trainer(model=model,
                      loaders=loaders,
                      task=task,
                      config=config,
                      optimizer=optimizer,
                      pipeline=pipeline,
                      metrics=metrics,
                      use_exp=True,
                      inspect_weights=False,
                      checkpoint=checkpoint,
                      early_stopping=early_stopping)

    return trainer


def unfreeze_module(module, optimizer):
    for param in module.parameters():
        param.requires_grad = True

    optimizer.add_param_group(
        {'params': list(
            module.parameters())}
    )


def model_training(trainer, epochs, unfreeze=0, checkpoint=False):
    print("Training...")
    for epoch in range(epochs):
        trainer.train()
        trainer.eval()

        if unfreeze > 0:
            if epoch == unfreeze:
                print("Unfreeze transfer-learning model...")
                subnetwork = trainer.model.feature_extractor
                if isinstance(subnetwork, ModuleList):
                    for fe in subnetwork:
                        unfreeze_module(fe.encoder, trainer.optimizer)
                        unfreeze_module(fe.attention, trainer.optimizer)
                else:
                    unfreeze_module(subnetwork.encoder, trainer.optimizer)
                    unfreeze_module(subnetwork.attention, trainer.optimizer)

        print()

        if checkpoint:
            trainer.checkpoint.check()

        if trainer.early_stopping.stop():
            print("Early stopping...")
            break
