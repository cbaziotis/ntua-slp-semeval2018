import sys

from dataloaders.task2 import load_task2
from model.params import TASK2_A
from utils.train import define_trainer, model_training

sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import argparse

##############################################################################
# Command line Arguments
##############################################################################

parser = argparse.ArgumentParser(description='test',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--context', help='options are ["none", "mean", "last"]',
                    default="none")

parser.add_argument('--finetune', dest='finetune', action='store_true')
parser.add_argument('--no-finetune', dest='finetune', action='store_false')
parser.set_defaults(finetune=True)

args = parser.parse_args()
print(args)

if args.context not in ["none", "mean", "last"]:
    raise ValueError("Invalid attention type!")

model_config = TASK2_A
model_config["attention_context"] = args.context
model_config["embed_finetune"] = args.finetune

X_train, y_train = load_task2("train")
X_trial, y_trial = load_task2("trial")
X_test, y_test = load_task2("test")

datasets = {
    "train": (X_train, y_train),
    "trial": (X_trial, y_trial),
    "gold": (X_test, y_test)
}

name = "_".join([model_config["name"], model_config["token_type"]])

trainer = define_trainer("clf", config=model_config, name=name,
                         datasets=datasets, monitor="gold")

model_training(trainer, model_config["epochs"])

desc = "context:{}, finetuning:{}".format(args.context, str(args.finetune))
trainer.log_training(name, desc)
