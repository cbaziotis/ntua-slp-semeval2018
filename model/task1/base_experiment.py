import json
import os
from collections import defaultdict

from config import BASE_PATH, EXPS_PATH
from model.task1.baseline_models import train_ei_reg, train_ei_oc, train_v_reg, \
    train_v_oc, train_e_c
from modules.sklearn.models import nbow_model, bow_model, eval_reg, eval_mclf
from utils.load_embeddings import load_word_vectors
from utils.nlp import twitter_preprocess

emb_files = [
    ("word2vec_300_6_20_neg.txt", 300),
    ("word2vec_300_6_concatened.txt", 310),
    ("word2vec_500_6_20_neg.txt", 500),
    ("word2vec_500_6_concatened.txt", 510),
]
embeddings = {}
for e, d in emb_files:
    file = os.path.join(BASE_PATH, "embeddings", e)
    word2idx, idx2word, weights = load_word_vectors(file, d)
    embeddings[e.split(".")[0]] = (weights, word2idx)

bow_clf = bow_model("clf")
bow_reg = bow_model("reg")
nbow_clf = {"nbow_{}".format(name): nbow_model("clf", e, w2i)
            for name, (e, w2i) in embeddings.items()}
nbow_reg = {"nbow_{}".format(name): nbow_model("reg", e, w2i)
            for name, (e, w2i) in embeddings.items()}

preprocessor = twitter_preprocess()

# ###########################################################################
# # 1. Task EI-reg: Detecting Emotion Intensity (regression)
# ###########################################################################

results = defaultdict(dict)
print()

for emotion in ["joy", "sadness", "fear", "anger"]:
    task = "EI-reg:{}".format(emotion)

    dev, gold = train_ei_reg(emotion=emotion, model=bow_reg,
                             evaluation=eval_reg, preprocessor=preprocessor)
    results[task]["bow"] = {"dev": dev, "gold": gold}

    for name, model in nbow_reg.items():
        dev, gold = train_ei_reg(emotion=emotion,
                                 model=model,
                                 evaluation=eval_reg,
                                 preprocessor=preprocessor)
        results[task][name] = {"dev": dev, "gold": gold}

###########################################################################
# 2. Task EI-oc: Detecting Emotion Intensity (ordinal classification)
###########################################################################

for emotion in ["joy", "sadness", "fear", "anger"]:
    task = "EI-oc:{}".format(emotion)

    dev, gold = train_ei_oc(emotion=emotion,
                            model=bow_clf,
                            evaluation=eval_reg,
                            preprocessor=preprocessor)
    results[task]["bow"] = {"dev": dev, "gold": gold}

    for name, model in nbow_clf.items():
        dev, gold = train_ei_oc(emotion=emotion,
                                model=model,
                                evaluation=eval_reg,
                                preprocessor=preprocessor)
        results[task][name] = {"dev": dev, "gold": gold}

###########################################################################
# 3. Task V-reg: Detecting Valence or Sentiment Intensity (regression)
###########################################################################
task = "V-reg"
dev, gold = train_v_reg(model=bow_reg,
                        evaluation=eval_reg,
                        preprocessor=preprocessor)
results[task]["bow"] = {"dev": dev, "gold": gold}

for name, model in nbow_reg.items():
    dev, gold = train_v_reg(model=model,
                            evaluation=eval_reg,
                            preprocessor=preprocessor)
    results[task][name] = {"dev": dev, "gold": gold}

###########################################################################
# 4. Task V-oc: Detecting Valence (ordinal classification)
###########################################################################
task = "V-oc"
dev, gold = train_v_oc(model=bow_clf,
                       evaluation=eval_reg,
                       preprocessor=preprocessor)
results[task]["bow"] = {"dev": dev, "gold": gold}

for name, model in nbow_clf.items():
    dev, gold = train_v_oc(model=model,
                           evaluation=eval_reg,
                           preprocessor=preprocessor)
    results[task][name] = {"dev": dev, "gold": gold}

###########################################################################
# 5. Task E-c: Detecting Emotions (multi-label classification)
###########################################################################
task = "E-c"
dev, gold = train_e_c(model=bow_clf, evaluation=eval_mclf,
                      preprocessor=preprocessor)
results[task]["bow"] = {"dev": dev, "gold": gold}

for name, model in nbow_clf.items():
    dev, gold = train_e_c(model=model,
                          evaluation=eval_mclf,
                          preprocessor=preprocessor)
    results[task][name] = {"dev": dev, "gold": gold}

with open(os.path.join(EXPS_PATH, "TASK1_baselines.json"), 'w') as f:
    json.dump(results, f)
