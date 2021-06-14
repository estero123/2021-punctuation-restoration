import string
import pandas as pd
from nltk import word_tokenize
import numpy as np
import spacy

nlp = spacy.load("pl_core_news_sm")

TRAIN_IN = 'train/in.tsv'
TRAIN_EXPECTED = 'train/expected.tsv'
TRAIN_OUT = 'train/out.tsv'
FILE_I = 'FileI'
ASR_OUT = 'ASROutput'
FIX_OUT = 'FixedOutput'
CTX_SIZE = 9
THRESH = 0.1
EMPTY = ''
NUM_ITEMS_PICK = 5


class Vocab:
    def __init__(self):
        self.words = {}

    def to_number(self, word):
        if word.lower() in self.words:
            return self.words[word]
        return -1

    def make_vocabulary_from_data(self, data):
        counter = 0
        words = {}
        for row in data:
            for tokenized_word in word_tokenize(row):
                e = tokenized_word.lower()
                if e not in words:
                    words[e] = counter
                    counter += 1
        self.words = words


def word_to_ctx(word, context, vector):
    context_str = str([x for x in context])
    if context_str in vector:
        if word in vector[context_str]:
            vector[context_str][word] += 1
        else:
            vector[context_str][word] = 0
    else:
        vector[context_str] = {}


class Model:
    def __init__(self, data, vocab):
        self.vec = {}
        self.data = data
        self.vocab = vocab

    def fit(self):
        vector = {}
        for row in self.data:
            tokenized_row = word_tokenize(row)
            for i in range(0, len(tokenized_row) - CTX_SIZE - 1):
                word = tokenized_row[i + CTX_SIZE + 1]
                ctx = tokenized_row[i: i + CTX_SIZE]
                word_to_ctx(word, ctx, vector)

    def predict(self, data):
        data = str([x for x in data])
        if data in self.vec:
            values_as_arr = np.asarray(list(self.vec[data].values()))
            nonzero = np.count_nonzero(values_as_arr)
            if values_as_arr.size == 0 or nonzero == 0:
                return EMPTY
            dr = self.get_dr(values_as_arr, data)
            return dr
        else:
            return EMPTY

    def get_probs(self, values_as_arr):
        return values_as_arr / np.sum(values_as_arr)

    def get_filter(self, thresh_value, values_as_arr):
        return np.asarray(values_as_arr >= thresh_value)

    def get_thresh_value(self, values_as_arr):
        return THRESH * np.max(values_as_arr)

    def get_keys(self, data):
        return np.asarray(list(self.vec[data].keys()))

    def get_dr(self, values_as_arr, data):
        thresh_value = self.get_thresh_value(values_as_arr)
        fltr = self.get_filter(thresh_value, values_as_arr)
        values_as_arr = values_as_arr[fltr]
        probs = self.get_probs(values_as_arr)
        keys = self.get_keys(data)[fltr]
        return np.random.choice(keys, size=NUM_ITEMS_PICK, p=probs)


def slv(row):
    an = row[0]
    se = row[0]
    for a in row[1:]:
        if a != se:
            se = a
            an += a
    return an


def mapper(element):
    return element[0] if type(element) == list else element


def get_ctx(i, w_tokenized):
    return w_tokenized[i: i + CTX_SIZE]


def transform(data, model):
    transformed = []
    for row in data:
        w_tokenized = word_tokenize(row)
        n_r = []
        for i in range(0, len(w_tokenized) - CTX_SIZE - 1):
            context = get_ctx(i, w_tokenized)
            dr = model.predict(context)
            n_r.append(context)
            for word in dr:
                if word in string.punctuation:
                    n_r.append(word)
                    break
        list_as_string = ' '.join(map(mapper, n_r))
        transformed.append(slv(list_as_string))
    return transformed


def load_data(path, cols):
    d = pd.read_csv(path, sep="\t")
    d.columns = cols
    return d


def create_vocabulary(exp):
    vocabulary = Vocab()
    vocabulary.make_vocabulary_from_data(exp[FIX_OUT])
    return vocabulary


def setup_model(exp, vocab):
    model = Model(exp[FIX_OUT], vocab)
    model.fit()
    return model


def solve_task():
    data = load_data(TRAIN_IN, [FILE_I, ASR_OUT])
    expected = load_data(TRAIN_EXPECTED, [FIX_OUT])

    vocabulary = create_vocabulary(expected)

    model = setup_model(expected, vocabulary)

    transformed_data = transform(data[ASR_OUT], model)

    data_frame = pd.DataFrame(transformed_data)
    data_frame.to_csv(TRAIN_OUT, sep='\t')


solve_task()
