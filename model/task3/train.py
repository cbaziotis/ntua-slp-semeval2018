from semeval2018.data.task3 import parse
from semeval2018.model.configs import TASK3_A, TASK3_B
from semeval2018.util.boiler import define_trainer, model_training

TASK = 'a'
# select config by args.task
if TASK == "a":
    model_config = TASK3_A
else:
    model_config = TASK3_B

X_train, y_train = parse(task=TASK, dataset="train")
X_test, y_test = parse(task=TASK, dataset="gold")

datasets = {
    "train": (X_train, y_train),
    "gold": (X_test, y_test),
}

name = "_".join([model_config["name"], model_config["token_type"]])

trainer = define_trainer("clf", config=model_config, name=name,
                         datasets=datasets,
                         monitor="gold")

model_training(trainer, model_config["epochs"])

trainer.log_training(name, model_config["token_type"])
