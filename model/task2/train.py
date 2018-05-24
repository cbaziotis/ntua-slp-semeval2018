from dataloaders.task2 import load_task2
from model.params import TASK2_A
from utils.train import define_trainer, model_training

model_config = TASK2_A

X_train, y_train = load_task2("train")
X_trial, y_trial = load_task2("trial")
X_test, y_test = load_task2("test")

datasets = {
    "train": (X_train, y_train),
    "trial": (X_trial, y_trial),
    "gold": (X_test, y_test),
}

name = "_".join([model_config["name"]])

trainer = define_trainer("clf", config=model_config, name=name,
                         datasets=datasets, monitor="gold")

model_training(trainer, model_config["epochs"], checkpoint=True)
