import json
import os

from config import EXPS_PATH
from dataloaders.task2 import load_task2
from model.params import TASK2_A
from modules.sklearn.models import bow_model, nbow_model, eval_clf
from utils.train import load_embeddings, load_datasets

config = TASK2_A

X_train, y_train = load_task2("train")
X_trial, y_trial = load_task2("trial")
X_test, y_test = load_task2("test")

datasets = {
    "train": (X_train, y_train),
    "trial": (X_trial, y_trial),
    "gold": (X_test, y_test)
}

word2idx, idx2word, embeddings = load_embeddings(config)

loaders = load_datasets(datasets,
                        train_batch_size=config["batch_train"],
                        eval_batch_size=config["batch_eval"],
                        token_type=config["token_type"],
                        params=config["name"],
                        word2idx=word2idx)

#########################################################################
#########################################################################
#########################################################################
X_train = loaders['train'].dataset.data
y_train = loaders['train'].dataset.labels
X_trial = loaders['trial'].dataset.data
y_trial = loaders['trial'].dataset.labels
X_test = loaders['gold'].dataset.data
y_test = loaders['gold'].dataset.labels

results = {}

bow = bow_model("clf", max_features=30000)
nbow = nbow_model("clf", embeddings, word2idx)

print("Fitting BOW...")
bow.fit(X_train, y_train)
bow_trial = eval_clf(bow.predict(X_trial), y_trial)
bow_test = eval_clf(bow.predict(X_test), y_test)
results["bow"] = {"trial": bow_trial, "test": bow_test}

print("Fitting N-BOW...")
nbow.fit(X_train, y_train)
nbow_trial = eval_clf(nbow.predict(X_trial), y_trial)
nbow_test = eval_clf(nbow.predict(X_test), y_test)
results["nbow"] = {"trial": nbow_trial, "test": nbow_test}

with open(os.path.join(EXPS_PATH, "TASK2_baselines.json"), 'w') as f:
    json.dump(results, f)
