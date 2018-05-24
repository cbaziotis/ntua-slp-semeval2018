from dataloaders.task1 import parse as parse1
from dataloaders.task2 import load_task2
from dataloaders.task3 import parse as parse3
from predict.predictions import dump_attentions
from utils.train import load_pretrained_model

################################################################
# Task EI-reg: Detecting Emotion Intensity (regression)
################################################################
for emotion in ["anger", "fear", "joy", "sadness"]:
    task = "EI-oc"
    model, conf = load_pretrained_model("TASK1_{}_{}".format(task, emotion))
    for dataset in ["train", "dev", "gold"]:
        X, y = parse1(task=task, emotion=emotion, dataset=dataset)
        y = [label[1] for label in y]
        dump_attentions(X, y, "TASK1_{}_{}_{}".format(task, emotion, dataset),
                        model, conf, "reg")

################################################################
# Task E-c: Detecting Emotions (multi-label classification)
################################################################
task = "E-c"
model, conf = load_pretrained_model("TASK1_{}".format(task))
for dataset in ["train", "dev", "gold"]:
    X, y = parse1(task=task, dataset=dataset)
    dump_attentions(X, y, "TASK1_{}_{}".format(task, dataset), model, conf,
                    "mclf")

########################################################################
# TASK 2
########################################################################
model, conf = load_pretrained_model("TASK2_A_0.3711")
for task in ["train", "trial", "test"]:
    X, y = load_task2(task)
    dump_attentions(X, y, "TASK2_A_{}".format(task), model, conf, "clf")

########################################################################
# TASK 3
########################################################################

# TASK3-A
model, conf = load_pretrained_model("TASK3_A_word_0.7773")
for task in ["train", "gold"]:
    X, y = parse3(task="a", dataset=task)
    dump_attentions(X, y, "TASK3A_{}".format(task), model, conf, "bclf")

# TASK3-B
model, conf = load_pretrained_model("TASK3_B_word_0.5549")
for task in ["train", "gold"]:
    X, y = parse3(task="b", dataset=task)
    dump_attentions(X, y, "TASK3B_{}".format(task), model, conf, "clf")
