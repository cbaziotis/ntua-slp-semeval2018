import os

from config import DATA_DIR


class TASK1(object):
    E_C = {
        'train': os.path.join(DATA_DIR, 'task1/E-c/E-c-En-train.txt'),
        'dev': os.path.join(DATA_DIR, 'task1/E-c/E-c-En-dev.txt'),
        'gold': os.path.join(DATA_DIR, 'task1/E-c/E-c-En-test-gold.txt')
    }

    EI_oc = {
        'anger': {
            'train': os.path.join(
                DATA_DIR, 'task1/EI-oc/EI-oc-En-anger-train.txt'),
            'dev': os.path.join(
                DATA_DIR, 'task1/EI-oc/EI-oc-En-anger-dev.txt'),
            'gold': os.path.join(
                DATA_DIR, 'task1/EI-oc/EI-oc-En-anger-test-gold.txt')
        },
        'fear': {
            'train': os.path.join(
                DATA_DIR, 'task1/EI-oc/EI-oc-En-fear-train.txt'),
            'dev': os.path.join(
                DATA_DIR, 'task1/EI-oc/EI-oc-En-fear-dev.txt'),
            'gold': os.path.join(
                DATA_DIR, 'task1/EI-oc/EI-oc-En-fear-test-gold.txt')
        },
        'sadness': {
            'train': os.path.join(
                DATA_DIR, 'task1/EI-oc/EI-oc-En-sadness-train.txt'),
            'dev': os.path.join(
                DATA_DIR, 'task1/EI-oc/EI-oc-En-sadness-dev.txt'),
            'gold': os.path.join(
                DATA_DIR, 'task1/EI-oc/EI-oc-En-sadness-test-gold.txt')
        },
        'joy': {
            'train': os.path.join(
                DATA_DIR, 'task1/EI-oc/EI-oc-En-joy-train.txt'),
            'dev': os.path.join(
                DATA_DIR, 'task1/EI-oc/EI-oc-En-joy-dev.txt'),
            'gold': os.path.join(
                DATA_DIR, 'task1/EI-oc/EI-oc-En-joy-test-gold.txt')
        }
    }

    EI_reg = {
        'anger': {
            'train': os.path.join(
                DATA_DIR, 'task1/EI-reg/EI-reg-En-anger-train.txt'),
            'dev': os.path.join(
                DATA_DIR, 'task1/EI-reg/EI-reg-En-anger-dev.txt'),
            'gold': os.path.join(
                DATA_DIR, 'task1/EI-reg/EI-reg-En-anger-test-gold.txt')
        },
        'fear': {
            'train': os.path.join(
                DATA_DIR, 'task1/EI-reg/EI-reg-En-fear-train.txt'),
            'dev': os.path.join(
                DATA_DIR, 'task1/EI-reg/EI-reg-En-fear-dev.txt'),
            'gold': os.path.join(
                DATA_DIR, 'task1/EI-reg/EI-reg-En-fear-test-gold.txt')
        },
        'sadness': {
            'train': os.path.join(
                DATA_DIR, 'task1/EI-reg/EI-reg-En-sadness-train.txt'),
            'dev': os.path.join(
                DATA_DIR, 'task1/EI-reg/EI-reg-En-sadness-dev.txt'),
            'gold': os.path.join(
                DATA_DIR, 'task1/EI-reg/EI-reg-En-sadness-test-gold.txt')
        },
        'joy': {
            'train': os.path.join(
                DATA_DIR, 'task1/EI-reg/EI-reg-En-joy-train.txt'),
            'dev': os.path.join(
                DATA_DIR, 'task1/EI-reg/EI-reg-En-joy-dev.txt'),
            'gold': os.path.join(
                DATA_DIR, 'task1/EI-reg/EI-reg-En-joy-test-gold.txt')
        }
    }

    V_oc = {
        'train': os.path.join(
            DATA_DIR, 'task1/V-oc/Valence-oc-En-train.txt'),
        'dev': os.path.join(
            DATA_DIR, 'task1/V-oc/Valence-oc-En-dev.txt'),
        'gold': os.path.join(
            DATA_DIR, 'task1/V-oc/Valence-oc-En-test-gold.txt'),
    }

    V_reg = {
        'train': os.path.join(
            DATA_DIR, 'task1/V-reg/Valence-reg-En-train.txt'),
        'dev': os.path.join(
            DATA_DIR, 'task1/V-reg/Valence-reg-En-dev.txt'),
        'gold': os.path.join(
            DATA_DIR, 'task1/V-reg/Valence-reg-En-test-gold.txt'),
    }


class TASK2(object):
    EN = os.path.join(DATA_DIR, 'task2/tweet_by_ID_25_10_2017__10_29_45.txt')


class TASK3(object):
    TASK_A = os.path.join(
        DATA_DIR,
        'task3/train/SemEval2018-T3-train-taskA_emoji.txt')
    TASK_A_TEST = os.path.join(
        DATA_DIR,
        'task3/test/SemEval2018-T3_input_test_taskA_emoji.txt')
    TASK_A_GOLD = os.path.join(
        DATA_DIR,
        'task3/gold/SemEval2018-T3_gold_test_taskA_emoji.txt')
    TASK_B = os.path.join(
        DATA_DIR,
        'task3/train/SemEval2018-T3-train-taskB_emoji.txt')
    TASK_B_TEST = os.path.join(
        DATA_DIR,
        'task3/test/SemEval2018-T3_input_test_taskB_emoji.txt')
    TASK_B_GOLD = os.path.join(
        DATA_DIR,
        'task3/gold/SemEval2018-T3_gold_test_taskB_emoji.txt')
