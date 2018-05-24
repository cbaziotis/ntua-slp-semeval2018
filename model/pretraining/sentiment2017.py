"""
Model for sentiment classification (positive,negative,neutral)
for Semeval2017 TaskA
"""
import os

from sklearn.model_selection import train_test_split

from config import DATA_DIR
from dataloaders.rest import load_data_from_dir
from logger.training import LabelTransformer
from model.params import SEMEVAL_2017
from utils.train import define_trainer, model_training

config = SEMEVAL_2017
train = load_data_from_dir(os.path.join(DATA_DIR, 'semeval_2017_4A'))
X = [obs[1] for obs in train]
y = [obs[0] for obs in train]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.05,
                                                    stratify=y,
                                                    random_state=42)

# pass a transformer function, for preparing tha labels for training
label_map = {label: idx for idx, label in
             enumerate(sorted(list(set(y_train))))}
inv_label_map = {v: k for k, v in label_map.items()}
transformer = LabelTransformer(label_map, inv_label_map)

datasets = {
    "train": (X_train, y_train),
    "val": (X_test, y_test),
}

name = "_".join([config["name"], config["token_type"]])

trainer = define_trainer("clf", config=config, name=name, datasets=datasets,
                         monitor="val", label_transformer=transformer)

model_training(trainer, config["epochs"], checkpoint=True)
