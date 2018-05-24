import os
import pickle
import string
from collections import Counter

import numpy
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from torch.utils.data import Dataset
from tqdm import tqdm

from config import BASE_PATH
from utils.nlp import vectorize


class BaseDataset(Dataset):
    """
    This is a Base class which extends pytorch's Dataset, in order to avoid
    boilerplate code and equip our datasets with functionality such as
    caching.
    """

    def __init__(self, X, y,
                 max_length=0,
                 name=None,
                 label_transformer=None,
                 verbose=True,
                 preprocess=None):
        """

        Args:
            X (): List of training samples
            y (): List of training labels
            name (str): the name of the dataset. It is needed for caching.
                if None then caching is disabled.
            max_length (int): the max length for each sentence.
                if 0 then use the maximum length in the dataset
            label_transformer (LabelTransformer):
        """
        self.data = X
        self.labels = y
        self.name = name
        self.label_transformer = label_transformer

        if preprocess is not None:
            self.preprocess = preprocess

        self.data = self.load_preprocessed_data()

        self.set_max_length(max_length)

        if verbose:
            self.dataset_statistics()

    def set_max_length(self, max_length):
        # if max_length == 0, then set max_length
        # to the maximum sentence length in the dataset
        if max_length == 0:
            self.max_length = max([len(x) for x in self.data])
        else:
            self.max_length = max_length

    def dataset_statistics(self):
        raise NotImplementedError

    def preprocess(self, name, X):
        """
        Preprocessing pipeline
        Args:
            X (list): list of training examples

        Returns: list of processed examples

        """
        raise NotImplementedError

    @staticmethod
    def _check_cache():
        cache_dir = os.path.join(BASE_PATH, "_cache")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _get_cache_filename(self):
        return os.path.join(BASE_PATH, "_cache",
                            "preprocessed_{}.p".format(self.name))

    def _write_cache(self, data):
        self._check_cache()

        cache_file = self._get_cache_filename()

        with open(cache_file, 'wb') as pickle_file:
            pickle.dump(data, pickle_file)

    def load_preprocessed_data(self):

        # NOT using cache
        if self.name is None:
            print("cache deactivated!")
            return self.preprocess(self.name, self.data)

        # using cache
        cache_file = self._get_cache_filename()

        if os.path.exists(cache_file):
            print("Loading {} from cache!".format(self.name))
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        else:
            print("No cache file for {} ...".format(self.name))
            data = self.preprocess(self.name, self.data)
            self._write_cache(data)
            return data


class WordDataset(BaseDataset):

    def __init__(self, X, y, word2idx,
                 max_length=0,
                 name=None,
                 label_transformer=None,
                 verbose=True,
                 preprocess=None):
        """
        A PyTorch Dataset
        What we have to do is to implement the 2 abstract methods:

            - __len__(self): in order to let the DataLoader know the size
                of our dataset and to perform batching, shuffling and so on...
            - __getitem__(self, index): we have to return the properly
                processed data-item from our dataset with a given index

        Args:
            X (): list of training samples
            y (): list of training labels
            max_length (int): the max length for each sentence.
                if 0 then use the maximum length in the dataset
            word2idx (dict): a dictionary which maps words to indexes
            label_transformer (LabelTransformer):
        """
        self.word2idx = word2idx

        BaseDataset.__init__(self, X, y, max_length, name, label_transformer,
                             verbose, preprocess)

    def dataset_statistics(self):
        words = Counter()
        for x in self.data:
            words.update(x)
        unks = {w: v for w, v in words.items() if w not in self.word2idx}
        # unks = sorted(unks.items(), key=lambda x: x[1], reverse=True)
        total_words = sum(words.values())
        total_unks = sum(unks.values())

        print("Total words: {}, Total unks:{} ({:.2f}%)".format(
            total_words, total_unks, total_unks * 100 / total_words))

        print("Unique words: {}, Unique unks:{} ({:.2f}%)".format(
            len(words), len(unks), len(unks) * 100 / len(words)))

        # label statistics
        print("Labels statistics:")
        if isinstance(self.labels[0], float):
            print("Mean:{:.4f}, Std:{:.4f}".format(numpy.mean(self.labels),
                                                   numpy.std(self.labels)))
        else:
            try:
                counts = Counter(self.labels)
                stats = {k: "{:.2f}%".format(v * 100 / len(self.labels))
                         for k, v in sorted(counts.items())}
                print(stats)
            except:
                print("Not implemented for mclf")
        print()

    def preprocess(self, name, dataset):
        preprocessor = TextPreProcessor(
            normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                       'time',
                       'date', 'number'],
            annotate={"hashtag", "elongated", "allcaps", "repeated",
                      'emphasis',
                      'censored'},
            all_caps_tag="wrap",
            fix_text=True,
            segmenter="twitter_2018",
            corrector="twitter_2018",
            unpack_hashtags=True,
            unpack_contractions=True,
            spell_correct_elong=False,
            tokenizer=SocialTokenizer(lowercase=True).tokenize,
            dicts=[emoticons]
        ).pre_process_doc

        desc = "PreProcessing dataset {}...".format(name)

        data = [preprocessor(x) for x in tqdm(dataset, desc=desc)]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns the _transformed_ item from the dataset

        Args:
            index (int):

        Returns:
            (tuple):
                * example (ndarray): vector representation of a training sample
                * label (string): the class label
                * length (int): the length (tokens) of the sentence
                * index (int): the index of the dataitem in the dataset.
                  It is useful for getting the raw input for visualizations.
        """
        sample, label = self.data[index], self.labels[index]

        # transform the sample and the label,
        # in order to feed them to the model
        sample = vectorize(sample, self.word2idx, self.max_length)

        if self.label_transformer is not None:
            label = self.label_transformer.transform(label)

        if isinstance(label, (list, tuple)):
            label = numpy.array(label)

        return sample, label, len(self.data[index]), index


class CharDataset(BaseDataset):

    def __init__(self, X, y,
                 max_length=0,
                 lower=False,
                 name=None,
                 label_transformer=None,
                 verbose=True,
                 preprocess=None):
        """
        A PyTorch Dataset
        What we have to do is to implement the 2 abstract methods:

            - __len__(self): in order to let the DataLoader know the size
                of our dataset and to perform batching, shuffling and so on...
            - __getitem__(self, index): we have to return the properly
                processed data-item from our dataset with a given index

        Args:
            X (): list of training samples
            y (): list of training labels
            max_length (int): the max length for each sentence.
                if 0 then use the maximum length in the dataset
            word2idx (dict): a dictionary which maps words to indexes
            label_transformer (LabelTransformer):
        """
        self.char2idx, self.idx2char = self.build_chars(lower)

        BaseDataset.__init__(self, X, y, max_length, name, label_transformer,
                             verbose, preprocess)

    def build_chars(self, lower):
        chars = list(string.printable)

        if lower:
            chars = [x for x in chars if not x.isupper()]

        char2idx = dict()
        idx2char = dict()
        for i, char in enumerate(chars, 1):
            char2idx[char] = i
            idx2char[i] = char

        char2idx["<unk>"] = i + 1
        idx2char[i + 1] = "<unk>"

        return char2idx, idx2char

    def dataset_statistics(self):
        # print("--Dataset Statistics--")
        print("Labels statistics:")
        if isinstance(self.labels[0], float):
            print("Mean:{:.4f}, Std:{:.4f}".format(numpy.mean(self.labels),
                                                   numpy.std(self.labels)))
        else:
            try:
                counts = Counter(self.labels)
                stats = {k: "{:.2f}%".format(v * 100 / len(self.labels))
                         for k, v in sorted(counts.items())}
                print(stats)
            except:
                pass
        print()

    def preprocess(self, name, dataset):
        desc = "PreProcessing dataset {}...".format(name)
        data = [[c.lower() for c in x] for x in tqdm(dataset, desc=desc)]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns the _transformed_ item from the dataset

        Args:
            index (int):

        Returns:
            (tuple):
                * example (ndarray): vector representation of a training sample
                * label (string): the class label
                * length (int): the length (tokens) of the sentence
                * index (int): the index of the dataitem in the dataset.
                  It is useful for getting the raw input for visualizations.
        """
        sample, label = self.data[index], self.labels[index]

        # transform the sample and the label,
        # in order to feed them to the model
        sample = vectorize(sample, self.char2idx, self.max_length)

        if self.label_transformer is not None:
            label = self.label_transformer.transform(label)

        if isinstance(label, (list, tuple)):
            label = numpy.array(label)

        return sample, label, len(self.data[index]), index
