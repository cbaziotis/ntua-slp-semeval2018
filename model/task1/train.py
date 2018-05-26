"""
1. Task EI-reg: Detecting Emotion Intensity (regression)
2. Task EI-oc: Detecting Emotion Intensity (ordinal classification)
3. Task V-reg: Detecting Valence or Sentiment Intensity (regression)
4. Task V-oc: Detecting Valence (ordinal classification)
5. Task E-c: Detecting Emotions (multi-label classification)
"""
from model.task1.neural_models import train_ei_reg, train_ei_oc, train_v_reg, \
    train_v_oc, train_e_c

pretrained = None
# pretrained = "SEMEVAL_2017_att-rnn_word_0.6899"
finetune = True
unfreeze = 0
task = 5
emotion = "joy"

if pretrained is not None:
    print("Using pretrained model: {}".format(pretrained))
    print("Finetuning: {}".format(finetune))
    if unfreeze > 0:
        print("Unfreezing after '{}' epochs.".format(unfreeze))

if task in [1, 2] and emotion is None:
    raise ValueError("Tasks 1,2 require the emotion argument!")

if task == 1:
    train_ei_reg(emotion,
                 pretrained=pretrained,
                 finetune=finetune,
                 unfreeze=unfreeze)
elif task == 2:
    train_ei_oc(emotion,
                pretrained=pretrained,
                finetune=finetune,
                unfreeze=unfreeze)
elif task == 3:
    train_v_reg(pretrained=pretrained,
                finetune=finetune,
                unfreeze=unfreeze)
elif task == 4:
    train_v_oc(pretrained=pretrained,
               finetune=finetune,
               unfreeze=unfreeze)
elif task == 5:
    train_e_c(pretrained=pretrained,
              finetune=finetune,
              unfreeze=unfreeze)
else:
    raise ValueError("Invalid task!")
