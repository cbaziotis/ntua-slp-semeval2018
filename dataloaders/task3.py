import config


def parse_train_csv(data_file):
    """

    Returns:
        X: a list of tweets
        y: a list of labels corresponding to the tweets

    """
    with open(data_file, 'r') as fd:
        data = [l.strip().split('\t') for l in fd.readlines()][1:]
    X = [d[2] for d in data]
    y = [d[1] for d in data]
    return X, y


def parse_test_csv(data_file):
    """

    Returns:
        X: a list of tweets
        y: a list of labels corresponding to the tweets

    """
    with open(data_file, 'r') as fd:
        data = [l.strip().split('\t') for l in fd.readlines()][1:]
    X = [d[1] for d in data]
    return X


def parse(task, dataset):
    if task == 'a' and dataset == "train":
        data_file = config.TASK3.TASK_A
    elif task == 'a' and dataset == "gold":
        data_file = config.TASK3.TASK_A_GOLD
    elif task == 'b' and dataset == "train":
        data_file = config.TASK3.TASK_B
    elif task == 'b' and dataset == "gold":
        data_file = config.TASK3.TASK_B_GOLD
    else:
        raise ValueError("Invalid dataset.")

    X, y = parse_train_csv(data_file)

    y = [int(label) for label in y]
    return X, y


def parse_test(task='a'):
    if task == 'a':
        data_file = config.TASK3.TASK_A_TEST
    else:
        data_file = config.TASK3.TASK_B_TEST
    X = parse_test_csv(data_file)

    return X

# X, y = parse(task='a')
# X = parse_test(task='a')
# X, y = parse(task='b')
