import argparse

from model.task1.neural_models import train_ei_reg, train_ei_oc, \
    train_v_reg, train_v_oc, train_e_c

tasks = """
Choose one from the following tasks
1. Task EI-reg: Detecting Emotion Intensity (regression)
2. Task EI-oc: Detecting Emotion Intensity (ordinal classification)
3. Task V-reg: Detecting Valence or Sentiment Intensity (regression)
4. Task V-oc: Detecting Valence (ordinal classification)
5. Task E-c: Detecting Emotions (multi-label classification)
"""
parser = argparse.ArgumentParser(description='test',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--task', type=int, help=tasks, required=True)
parser.add_argument('--pretrained', help='name of a pretrained model.')
args = parser.parse_args()

pretrained = args.pretrained

if args.task == 1:

    for i in range(10):
        for emotion in ["joy", "sadness", "fear", "anger"]:
            train_ei_reg(emotion=emotion)
            train_ei_reg(emotion=emotion, pretrained=pretrained, finetune=True)
            train_ei_reg(emotion=emotion, pretrained=pretrained,
                         finetune=False)
elif args.task == 2:

    for i in range(10):
        for emotion in ["joy", "sadness", "fear", "anger"]:
            train_ei_oc(emotion=emotion)
            train_ei_oc(emotion=emotion, pretrained=pretrained, finetune=True)
            train_ei_oc(emotion=emotion, pretrained=pretrained, finetune=False)

elif args.task == 3:

    for i in range(10):
        train_v_reg()
        train_v_reg(pretrained=pretrained, finetune=True)
        train_v_reg(pretrained=pretrained, finetune=False)

elif args.task == 4:

    for i in range(10):
        train_v_oc()
        train_v_oc(pretrained=pretrained, finetune=True)
        train_v_oc(pretrained=pretrained, finetune=False)

elif args.task == 5:

    for i in range(10):
        train_e_c()
        train_e_c(pretrained=pretrained, finetune=True)
        train_e_c(pretrained=pretrained, finetune=False)

else:
    raise ValueError("Invalid task!")
