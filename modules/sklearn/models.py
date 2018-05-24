import numpy
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, \
    accuracy_score, jaccard_similarity_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVR

from modules.sklearn.NBOWVectorizer import NBOWVectorizer


def eval_reg(y_hat, y):
    results = {
        "pearson": pearsonr([float(x) for x in y_hat],
                            [float(x) for x in y])[0]
    }

    return results


def eval_clf(y_test, y_p):
    results = {
        "f1": f1_score(y_test, y_p, average='macro'),
        "recall": recall_score(y_test, y_p, average='macro'),
        "precision": precision_score(y_test, y_p, average='macro'),
        "accuracy": accuracy_score(y_test, y_p)
    }

    return results


def eval_mclf(y, y_hat):
    results = {
        "jaccard": jaccard_similarity_score(numpy.array(y),
                                            numpy.array(y_hat)),
        "f1-macro": f1_score(numpy.array(y), numpy.array(y_hat),
                             average='macro'),
        "f1-micro": f1_score(numpy.array(y), numpy.array(y_hat),
                             average='micro')
    }

    return results


def bow_model(task, max_features=10000):
    if task == "clf":
        algo = LogisticRegression(C=0.6, random_state=0,
                                  class_weight='balanced')
    elif task == "reg":
        algo = SVR(kernel='linear', C=0.6)
    else:
        raise ValueError("invalid task!")

    word_features = TfidfVectorizer(ngram_range=(1, 1),
                                    tokenizer=lambda x: x,
                                    analyzer='word',
                                    min_df=5,
                                    # max_df=0.9,
                                    lowercase=False,
                                    use_idf=True,
                                    smooth_idf=True,
                                    max_features=max_features,
                                    sublinear_tf=True)

    model = Pipeline([
        ('bow-feats', word_features),
        ('normalizer', Normalizer(norm='l2')),
        ('clf', algo)
    ])

    return model


def nbow_model(task, embeddings, word2idx):
    if task == "clf":
        algo = LogisticRegression(C=0.6, random_state=0,
                                  class_weight='balanced')
    elif task == "reg":
        algo = SVR(kernel='linear', C=0.6)
    else:
        raise ValueError("invalid task!")

    embeddings_features = NBOWVectorizer(aggregation=["mean"],
                                         embeddings=embeddings,
                                         word2idx=word2idx,
                                         stopwords=False)

    model = Pipeline([
        ('embeddings-feats', embeddings_features),
        ('normalizer', Normalizer(norm='l2')),
        ('clf', algo)
    ])

    return model
