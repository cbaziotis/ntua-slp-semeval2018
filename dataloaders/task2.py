import os

from config import DATA_DIR


def parse_csv(data_file):
    """

    Returns:
        X: a list of tweets
        y: a list of labels corresponding to the tweets

    """
    with open(data_file, 'r') as fd:
        data = [l.strip().split('\t') for l in fd.readlines()]
    X = [d[0] for d in data]
    y = [int(d[2]) for d in data]
    return X, y


def fix_text(text):
    try:
        return text.encode().decode('unicode-escape')
    except:
        return text


def load_task2(dataset):
    data_file = os.path.join(DATA_DIR, "task2/us_{}.text".format(dataset))
    label_file = os.path.join(DATA_DIR, "task2/us_{}.labels".format(dataset))

    X = []
    y = []
    with open(data_file, 'r', encoding="utf-8") as dfile, \
            open(label_file, 'r', encoding="utf-8") as lfile:
        for tweet, label in zip(dfile, lfile):
            X.append(tweet.rstrip())
            y.append(int(label.rstrip()))

    return X, y
