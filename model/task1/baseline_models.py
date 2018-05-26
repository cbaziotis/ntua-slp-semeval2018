import numpy

from dataloaders.task1 import parse
from model.params import TASK1_VREG, TASK1_EIOC, TASK1_EC, \
    TASK1_EIREG, TASK1_VOC
from utils.nlp import twitter_preprocess


def train_ei_reg(emotion, model, evaluation, preprocessor=None):
    """
    1. Task EI-reg: Detecting Emotion Intensity (regression)

    Given:

        - a tweet
        - an emotion E (anger, fear, joy, or sadness)

    Task: determine the  intensity of E that best represents the mental state of
    the tweeter—a real-valued score between 0 and 1:

        - a score of 1: highest amount of E can be inferred
        - a score of 0: lowest amount of E can be inferred

    For each language: 4 training sets and 4 test sets: one for each emotion E.

    (Note that the absolute scores have no inherent meaning --
    they are used only as a means to convey that the instances
    with higher scores correspond to a greater degree of E
    than instances with lower scores.)

    :param emotion: emotions = ["anger", "fear", "joy", "sadness"]
    :param pretrained:
    :param finetune:
    :param unfreeze:
    :return:
    """

    if preprocessor is None:
        preprocessor = twitter_preprocess()

    model_config = TASK1_EIREG

    X_train, y_train = parse(task='EI-reg', emotion=emotion, dataset="train")
    X_dev, y_dev = parse(task='EI-reg', emotion=emotion, dataset="dev")
    X_test, y_test = parse(task='EI-reg', emotion=emotion, dataset="gold")

    # keep only scores
    y_train = [y[1] for y in y_train]
    y_dev = [y[1] for y in y_dev]
    y_test = [y[1] for y in y_test]

    name = model_config["name"] + "_" + emotion

    X_train = preprocessor("{}_{}".format(name, "train"), X_train)
    X_dev = preprocessor("{}_{}".format(name, "dev"), X_dev)
    X_test = preprocessor("{}_{}".format(name, "test"), X_test)

    model.fit(X_train, y_train)

    res_dev = evaluation(model.predict(X_dev), y_dev)
    res_test = evaluation(model.predict(X_test), y_test)

    return res_dev, res_test


def train_ei_oc(emotion, model, evaluation, preprocessor=None):
    """
    2. Task EI-oc: Detecting Emotion Intensity (ordinal classification)

    Given:

    a tweet
    an emotion E (anger, fear, joy, or sadness)

    Task: classify the tweet into one of four ordinal classes of intensity of E
    that best represents the mental state of the tweeter:

        0: no E can be inferred
        1: low amount of E can be inferred
        2: moderate amount of E can be inferred
        3: high amount of E can be inferred

    For each language: 4 training sets and 4 test sets: one for each emotion E.

    :param emotion: emotions = ["anger", "fear", "joy", "sadness"]
    :param pretrained:
    :param finetune:
    :param unfreeze:
    :return:
    """

    if preprocessor is None:
        preprocessor = twitter_preprocess()

    model_config = TASK1_EIOC

    X_train, y_train = parse(task='EI-oc', emotion=emotion, dataset="train")
    X_dev, y_dev = parse(task='EI-oc', emotion=emotion, dataset="dev")
    X_test, y_test = parse(task='EI-oc', emotion=emotion, dataset="gold")

    # keep only scores
    y_train = [y[1] for y in y_train]
    y_dev = [y[1] for y in y_dev]
    y_test = [y[1] for y in y_test]

    name = model_config["name"] + "_" + emotion

    X_train = preprocessor("{}_{}".format(name, "train"), X_train)
    X_dev = preprocessor("{}_{}".format(name, "dev"), X_dev)
    X_test = preprocessor("{}_{}".format(name, "test"), X_test)

    model.fit(X_train, y_train)

    res_dev = evaluation(model.predict(X_dev), y_dev)
    res_test = evaluation(model.predict(X_test), y_test)

    return res_dev, res_test


def train_v_reg(model, evaluation, preprocessor=None):
    """
    3. Task V-reg: Detecting Valence or Sentiment Intensity (regression)

    Given:
     - a tweet

    Task: determine the intensity of sentiment or valence (V)
    that best represents the mental state of the tweeter—a real-valued score
    between 0 and 1:

        a score of 1: most positive mental state can be inferred
        a score of 0: most negative mental state can be inferred

    For each language: 1 training set, 1 test set.

    (Note that the absolute scores have no inherent meaning --
    they are used only as a means to convey that the instances
    with higher scores correspond to a greater degree of positive sentiment
    than instances with lower scores.)

    :param pretrained:
    :param finetune:
    :param unfreeze:
    :return:
    """
    model_config = TASK1_VREG
    # load the dataset and split it in train and val sets
    X_train, y_train = parse(task='V-reg', dataset="train")
    X_dev, y_dev = parse(task='V-reg', dataset="dev")
    X_test, y_test = parse(task='V-reg', dataset="gold")

    # keep only scores
    y_train = [y[1] for y in y_train]
    y_dev = [y[1] for y in y_dev]
    y_test = [y[1] for y in y_test]

    name = model_config["name"]

    X_train = preprocessor("{}_{}".format(name, "train"), X_train)
    X_dev = preprocessor("{}_{}".format(name, "dev"), X_dev)
    X_test = preprocessor("{}_{}".format(name, "test"), X_test)

    model.fit(X_train, y_train)

    res_dev = evaluation(model.predict(X_dev), y_dev)
    res_test = evaluation(model.predict(X_test), y_test)

    return res_dev, res_test


def train_v_oc(model, evaluation, preprocessor=None):
    """
    4. Task V-oc: Detecting Valence (ordinal classification)
    -- This is the traditional Sentiment Analysis Task

    Given:
     - a tweet

    Task: classify the tweet into one of seven ordinal classes,
    corresponding to various levels of positive and negative sentiment
    intensity, that best represents the mental state of the tweeter:

        3: very positive mental state can be inferred
        2: moderately positive mental state can be inferred
        1: slightly positive mental state can be inferred
        0: neutral or mixed mental state can be inferred
        -1: slightly negative mental state can be inferred
        -2: moderately negative mental state can be inferred
        -3: very negative mental state can be inferred

    For each language: 1 training set, 1 test set.

    :param pretrained:
    :param finetune:
    :param unfreeze:
    :return:
    """
    model_config = TASK1_VOC

    # load the dataset and split it in train and val sets
    X_train, y_train = parse(task='V-oc', dataset="train")
    X_dev, y_dev = parse(task='V-oc', dataset="dev")
    X_test, y_test = parse(task='V-oc', dataset="gold")

    # keep only scores
    y_train = [str(y[1]) for y in y_train]
    y_dev = [str(y[1]) for y in y_dev]
    y_test = [str(y[1]) for y in y_test]

    name = model_config["name"]

    X_train = preprocessor("{}_{}".format(name, "train"), X_train)
    X_dev = preprocessor("{}_{}".format(name, "dev"), X_dev)
    X_test = preprocessor("{}_{}".format(name, "test"), X_test)

    model.fit(X_train, y_train)

    res_dev = evaluation(model.predict(X_dev), y_dev)
    res_test = evaluation(model.predict(X_test), y_test)

    return res_dev, res_test


def fit_multilabel(model, X_train, X_val, y_train, y_val):
    y_val = numpy.array(y_val)
    y_train = numpy.array(y_train)

    predictions = numpy.zeros(y_val.shape)

    for i in range(y_val.shape[1]):
        model.fit(X_train, y_train[:, i])
        y_p = model.predict(X_val)
        predictions[:, i] = y_p

    return predictions


def train_e_c(model, evaluation, preprocessor=None):
    """
    5. Task E-c: Detecting Emotions (multi-label classification)
    -- This is a traditional Emotion Classification Task

    Given:
     - a tweet

    Task: classify the tweet as 'neutral or no emotion' or as one, or more,
    of eleven given emotions that best represent the mental state
    of the tweeter:

        - anger (also includes annoyance and rage) can be inferred
        - anticipation (also includes interest and vigilance) can be inferred
        - disgust (also includes disinterest, dislike and loathing) can be
            inferred
        - fear (also includes apprehension, anxiety, concern, and terror)
            can be inferred
        - joy (also includes serenity and ecstasy) can be inferred
        - love (also includes affection) can be inferred
        - optimism (also includes hopefulness and confidence) can be inferred
        - pessimism (also includes cynicism and lack of confidence) can be
            inferred
        - sadness (also includes pensiveness and grief) can be inferred
        - surprise (also includes distraction and amazement) can be inferred
        - trust (also includes acceptance, liking, and admiration) can be
            inferred

    For each language: 1 training set, 1 test set.

    (Note that the set of emotions includes the eight basic emotions
    as per Plutchik (1980), as well as a few other emotions that are common
    in tweets (love, optimism, and pessimism).)

    :param pretrained:
    :param finetune:
    :param unfreeze:
    :return:
    """
    model_config = TASK1_EC
    # load the dataset and split it in train and val sets
    X_train, y_train = parse(task='E-c', dataset="train")
    X_dev, y_dev = parse(task='E-c', dataset="dev")
    X_test, y_test = parse(task='E-c', dataset="gold")

    name = model_config["name"]

    X_train = preprocessor("{}_{}".format(name, "train"), X_train)
    X_dev = preprocessor("{}_{}".format(name, "dev"), X_dev)
    X_test = preprocessor("{}_{}".format(name, "test"), X_test)

    res_dev = fit_multilabel(model, X_train, X_dev, y_train, y_dev)
    res_test = fit_multilabel(model, X_train, X_test, y_train, y_test)

    res_dev = evaluation(res_dev, y_dev)
    res_test = evaluation(res_test, y_test)

    return res_dev, res_test
