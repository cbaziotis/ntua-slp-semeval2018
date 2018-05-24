import json
import os

from config import EXPS_PATH
from dataloaders.task3 import parse
from model.params import TASK3_A, TASK3_B
from modules.sklearn.models import bow_model, nbow_model, eval_clf
from utils.train import load_embeddings, load_datasets


def train_task(task):
    if task == "a":
        model_config = TASK3_A
    else:
        model_config = TASK3_B

    X_train, y_train = parse(task=task, dataset="train")
    X_test, y_test = parse(task=task, dataset="gold")

    datasets = {
        "train": (X_train, y_train),
        "gold": (X_test, y_test),
    }

    word2idx, idx2word, embeddings = load_embeddings(model_config)

    loaders = load_datasets(datasets,
                            train_batch_size=model_config["batch_train"],
                            eval_batch_size=model_config["batch_eval"],
                            token_type=model_config["token_type"],
                            params=model_config["name"],
                            word2idx=word2idx)

    X_train = loaders['train'].dataset.data
    y_train = loaders['train'].dataset.labels
    X_test = loaders['gold'].dataset.data
    y_test = loaders['gold'].dataset.labels

    results = {}

    bow = bow_model("clf", max_features=10000)
    nbow = nbow_model("clf", embeddings, word2idx)

    print("Fitting BOW...")
    bow.fit(X_train, y_train)
    bow_test = eval_clf(bow.predict(X_test), y_test)
    results["bow"] = bow_test

    print("Fitting N-BOW...")
    nbow.fit(X_train, y_train)
    nbow_test = eval_clf(nbow.predict(X_test), y_test)
    results["nbow"] = nbow_test

    return results


results_a = train_task("a")
results_b = train_task("b")

results = {
    "A": results_a,
    "B": results_b,
}

with open(os.path.join(EXPS_PATH, "TASK3_baselines.json"), 'w') as f:
    json.dump(results, f)
