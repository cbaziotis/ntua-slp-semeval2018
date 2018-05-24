import config


def parse_e_c(data_file):
    """

    Returns:
        X: a list of tweets
        y: a list of lists corresponding to the emotion labels of the tweets

    """
    with open(data_file, 'r') as fd:
        data = [l.strip().split('\t') for l in fd.readlines()][1:]
    X = [d[1] for d in data]
    # dict.values() does not guarantee the order of the elements
    # so we should avoid using a dict for the labels
    y = [[int(l) for l in d[2:]] for d in data]

    return X, y


def parse_oc(data_file, label_format='tuple'):
    """

    Returns:
        X: a list of tweets
        y: a list of (affect dimension, v) tuples corresponding to
         the ordinal classification targets of the tweets
    """
    with open(data_file, 'r') as fd:
        data = [l.strip().split('\t') for l in fd.readlines()][1:]
    X = [d[1] for d in data]
    y = [(d[2], int(d[3].split(':')[0])) for d in data]
    if label_format == 'list':
        y = [l[1] for l in y]
    return X, y


def parse_reg(data_file, label_format='tuple'):
    """
    The test datasets for the EI-reg and V-reg English tasks have two parts:
    1. The Tweet Test Set: tweets annotated for emotion/valence intensity;
    2. The Mystery Test Set: automatically generated sentences to test for
    unethical biases in NLP systems (with no emotion/valence annotations).

    Mystery Test Set: the last 16,937 lines with 'mystery' in the ID

    Returns:
        X: a list of tweets
        y: a list of (affect dimension, v) tuples corresponding to
         the regression targets of the tweets
    """
    with open(data_file, 'r') as fd:
        data = [l.strip().split('\t') for l in fd.readlines()][1:]
        data = [d for d in data if "mystery" not in d[0]]
    X = [d[1] for d in data]
    y = [(d[2], float(d[3])) for d in data]
    if label_format == 'list':
        y = [l[1] for l in y]
    return X, y


def parse(task, dataset, emotion=None):
    if task == 'E-c':
        data_train = config.TASK1.E_C[dataset]
        X, y = parse_e_c(data_train)
        return X, y
    elif task == 'EI-oc':
        data_train = config.TASK1.EI_oc[emotion][dataset]
        X, y = parse_oc(data_train)
        return X, y
    elif task == 'EI-reg':
        data_train = config.TASK1.EI_reg[emotion][dataset]
        X, y = parse_reg(data_train)
        return X, y
    elif task == 'V-oc':
        data_train = config.TASK1.V_oc[dataset]
        X, y = parse_oc(data_train)
        return X, y
    elif task == 'V-reg':
        data_train = config.TASK1.V_reg[dataset]
        X, y = parse_reg(data_train)
        return X, y
    else:
        return None, None
