import json
import os

import numpy
from torch.utils.data import DataLoader

from config import DEVICE, ATT_PATH
from logger.training import predict
from modules.nn.dataloading import WordDataset, CharDataset
from utils.nlp import twitter_preprocess
from utils.train import load_embeddings, get_pipeline


def dump_attentions(X, y, name, model, conf, task):
    pred, posteriors, attentions, tokens = predictions(task, model, conf, X,
                                                       name=name)

    data = []
    for tweet, label, prediction, posterior, attention in zip(tokens, y,
                                                              pred, posteriors,
                                                              attentions):
        if task == "mclf":
            label = numpy.array(label)
            prediction = numpy.array(prediction).astype(label.dtype)

            item = {
                "text": tweet,
                "label": label.tolist(),
                "prediction": prediction.tolist(),
                "posterior": numpy.array(posterior).tolist(),
                "attention": numpy.array(attention).tolist(),
            }
        elif task in ["clf", "bclf", "reg"]:
            item = {
                "text": tweet,
                "label": label,
                "prediction": type(label)(prediction),
                "posterior": posterior,
                "attention": attention,
            }
        else:
            raise ValueError("Task not implemented!")

        data.append(item)
    with open(os.path.join(ATT_PATH, "{}.json".format(name)), 'w') as f:
        json.dump(data, f, indent=4, separators=(',', ': '))


def predictions(task, model, config, data, label_transformer=None,
                batch_size=128, preprocessor=None, name=None):
    """

    Args:
        task (): available tasks
                - "clf": multiclass classification
                - "bclf": binary classification
                - "mclf": multilabel classification
                - "reg": regression
        model ():
        config ():
        data ():
        label_transformer ():
        batch_size ():
        num_workers ():

    Returns:

    """
    word2idx = None
    if config["op_mode"] == "word":
        word2idx, idx2word, embeddings = load_embeddings(config)

    # dummy scores if order to utilize Dataset classes as they are
    dummy_y = [0] * len(data)

    if config["op_mode"] == "word":

        if preprocessor is None:
            preprocessor = twitter_preprocess()

        dataset = WordDataset(data, dummy_y, word2idx,
                              name=name,
                              preprocess=preprocessor,
                              label_transformer=label_transformer)
        loader = DataLoader(dataset, batch_size)

    elif config["op_mode"] == "char":
        print("Building char-level datasets...")
        dataset = CharDataset(data, dummy_y, name=name,
                              label_transformer=label_transformer)
        loader = DataLoader(dataset, batch_size)
    else:
        raise ValueError("Invalid op_mode")

    model.to(DEVICE)

    pipeline = get_pipeline(task=task, eval=True)
    avg_loss, (dummy_y, pred), posteriors, attentions = predict(model,
                                                                pipeline,
                                                                loader,
                                                                task,
                                                                "eval")

    return pred, posteriors, attentions, loader.dataset.data
