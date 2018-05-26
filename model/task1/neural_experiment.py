import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import argparse

from model.task1.neural_models import train_ei_reg, train_ei_oc, \
    train_v_reg, train_v_oc, train_e_c

##############################################################################
# Command line Arguments
##############################################################################
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
parser.add_argument('--emotion', help='valid for tasks 1,2')
parser.add_argument('--pretrained', help='name of a pretrained model.')

parser.add_argument('--finetune', dest='finetune', action='store_true')
parser.add_argument('--no-finetune', dest='finetune', action='store_false')
parser.set_defaults(finetune=True)

parser.add_argument('--unfreeze', type=int, default=0,
                    help='epoch after which to unfreeze the pretrained model.')
args = parser.parse_args()
print(args)

if args.pretrained is not None:
    print("Using pretrained model: {}".format(args.pretrained))
    print("Finetuning: {}".format(args.finetune))
    if args.unfreeze > 0:
        print("Unfreezing after '{}' epochs.".format(args.unfreeze))

if args.task in [1, 2] and args.emotion is None:
    raise ValueError("Tasks 1,2 require the emotion argument!")

if args.task == 1:
    train_ei_reg(args.emotion,
                 pretrained=args.pretrained,
                 finetune=args.finetune,
                 unfreeze=args.unfreeze)
elif args.task == 2:
    train_ei_oc(args.emotion,
                pretrained=args.pretrained,
                finetune=args.finetune,
                unfreeze=args.unfreeze)
elif args.task == 3:
    train_v_reg(pretrained=args.pretrained,
                finetune=args.finetune,
                unfreeze=args.unfreeze)
elif args.task == 4:
    train_v_oc(pretrained=args.pretrained,
               finetune=args.finetune,
               unfreeze=args.unfreeze)
elif args.task == 5:
    train_e_c(pretrained=args.pretrained,
              finetune=args.finetune,
              unfreeze=args.unfreeze)
else:
    raise ValueError("Invalid task!")
