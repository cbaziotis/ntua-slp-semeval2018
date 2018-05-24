import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import argparse

from semeval2018.data.task3 import parse
from semeval2018.model.configs import TASK3_A, TASK3_B
from semeval2018.util.boiler import define_trainer, model_training

##############################################################################
# Command line Arguments
##############################################################################

parser = argparse.ArgumentParser(description='test',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--task', help="choose subtask a or b", default="b")
parser.add_argument('--mode', help='choose "char" or "word"', default="char")
args = parser.parse_args()
print(args)

if args.task not in ["a", "b"]:
    raise ValueError("Invalid task!")
if args.mode not in ["char", "word"]:
    raise ValueError("Invalid mode!")

# select config by args.task
if args.task == "a":
    model_config = TASK3_A
else:
    model_config = TASK3_B

if args.mode == "char":
    model_config["embed_dim"] = 25
    model_config["embed_finetune"] = True
    model_config["attention"] = True
    model_config["embed_noise"] = 0.0
    model_config["embed_dropout"] = 0.0
else:
    model_config["embed_dim"] = 300
    model_config["embed_finetune"] = False
    model_config["attention"] = True
    model_config["embed_noise"] = 0.05
    model_config["embed_dropout"] = 0.1

# set the operation mode
model_config["token_type"] = args.mode

##############################################################

# load the dataset and split it in train and val sets
X_train, y_train = parse(task=args.task, dataset="train")
X_test, y_test = parse(task=args.task, dataset="gold")

datasets = {
    "train": (X_train, y_train),
    "gold": (X_test, y_test),
}

name = "_".join([model_config["name"], model_config["token_type"]])

trainer = define_trainer("clf", config=model_config, name=name,
                         datasets=datasets, monitor="gold")

model_training(trainer, model_config["epochs"], checkpoint=True)

trainer.log_training(name, model_config["token_type"])
