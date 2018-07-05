from collections import namedtuple
import os

import nltk
import numpy as np
from tqdm import tqdm

from project.data.preprocessed import DataTuple

PAD_TOKEN = '<PAD>'
UNKNOWN_TOKEN = '<UNK>'
START_OF_TEXT_TOKEN = '<START>'
END_OF_TEXT_TOKEN = '<END>'

EmbedTuple = namedtuple(
    "EmbedTuple", ['word_weights', 'word2idx', 'char_weights', 'char2idx'])


def get_weights_char2idx(char_embed):
    # Weights are random, 300d
    dim = char_embed
    arg_alphabet = 'abcdefghijklmnopqrstuvwyxzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_*:'

    # ':' is a stop token
    char2idx = {a: i+1 for i, a in enumerate(arg_alphabet)}
    char2idx[END_OF_TEXT_TOKEN] = len(char2idx.keys())

    char_weights = np.random.uniform(
        low=-0.1, high=0.1, size=[len(arg_alphabet)+1, dim])
    return (char_weights, char2idx)


def get_weights_word2idx(desc_embed, vocab_size=100000):
    # Currently get the 300d embeddings from GloVe
    DIR = os.path.dirname(os.path.abspath(__file__))

    word2idx = {PAD_TOKEN: 0}
    weights = []

    embed_files = {
        50: "{}/glove/glove.6B.50d.txt".format(DIR),
        100: "{}/glove/glove.6B.100d.txt".format(DIR),
        200: "{}/glove/glove.6B.200d.txt".format(DIR),
        300: "{}/glove/glove.6B.300d.txt".format(DIR),
    }

    with open(embed_files[desc_embed], "r", encoding='utf-8') as f:
        for i, line in tqdm(enumerate(f)):
            values = line.split()

            word = values[0]
            word_weights = np.array(values[1:]).astype(np.float32)

            word2idx[word] = i + 1
            weights.append(word_weights)

            if i > vocab_size:
                break

    embed_dim = len(weights[0])
    weights.insert(0, np.random.randn(embed_dim))

    word2idx[UNKNOWN_TOKEN] = len(weights)
    weights.append(np.random.randn(embed_dim))

    word2idx[START_OF_TEXT_TOKEN] = len(weights)
    weights.append(np.random.randn(embed_dim))

    word2idx[END_OF_TEXT_TOKEN] = len(weights)
    weights.append(np.random.randn(embed_dim))

    weights = np.asarray(weights, dtype=np.float32)
    return (weights, word2idx)


def tokenize_descriptions(data, word2idx, char2idx):
    unk_token = word2idx[UNKNOWN_TOKEN]
    for i, d in enumerate(data):
        desc = d['arg_desc'].replace('\\n', " ").lower()
        desc_tok = nltk.word_tokenize(desc)

        d['arg_desc_tokens'] = [START_OF_TEXT_TOKEN]
        d['arg_desc_idx'] = [word2idx[START_OF_TEXT_TOKEN]]
        d['arg_desc_tokens'].extend(
            [w if w in word2idx else UNKNOWN_TOKEN for w in desc_tok])
        d['arg_desc_idx'].extend([word2idx.get(t, unk_token)
                                  for t in desc_tok])
        d['arg_desc_tokens'].append(END_OF_TEXT_TOKEN)
        d['arg_desc_idx'].append(word2idx[END_OF_TEXT_TOKEN])
        d['arg_name_tokens'] = [c for c in d['arg_name']]
        d['arg_name_idx'] = [char2idx[c] for c in d['arg_name']]
        d['arg_name_tokens'].append(END_OF_TEXT_TOKEN)
        d['arg_name_idx'].append(char2idx[END_OF_TEXT_TOKEN])

    return data


def extract_char_and_desc_idx_tensors(data, char_dim, desc_dim):
    chars = []
    descs = []
    for d in data:
        char_pad = [d['arg_name_idx'][i] if i < len(
            d['arg_name_idx']) else 0 for i in range(char_dim)]
        chars.append(np.array(char_pad))

        desc_pad = [d['arg_desc_idx'][i] if i < len(
            d['arg_desc_idx']) else 0 for i in range(desc_dim)]
        descs.append(np.array(desc_pad))
    return np.stack(chars), np.stack(descs)


def get_data_tuple(use_full_dataset, use_split_dataset):
    if use_full_dataset:
        if use_split_dataset:
            from project.data.preprocessed.split import split_data as data
        else:
            from project.data.preprocessed.unsplit import unsplit_data as data
    else:
        from project.data.preprocessed.overfit import overfit_data as data
    return data


def get_embed_tuple_and_data_tuple(vocab_size, char_seq, desc_seq, char_embed, desc_embed,
                                   use_full_dataset, use_split_dataset):
    data_tuple = get_data_tuple(use_full_dataset, use_split_dataset)

    print("Loading GloVe weights and word to index lookup table")
    word_weights, word2idx = get_weights_word2idx(desc_embed, vocab_size)
    print("Creating char to index look up table")
    char_weights, char2idx = get_weights_char2idx(char_embed)

    print("Tokenizing the word desctiptions and characters")
    train_data = tokenize_descriptions(data_tuple.train, word2idx, char2idx)
    valid_data = tokenize_descriptions(data_tuple.valid, word2idx, char2idx)
    test_data = tokenize_descriptions(data_tuple.test, word2idx, char2idx)

    print("Extracting tensors train and test")
    train_data = extract_char_and_desc_idx_tensors(
        train_data, char_seq, desc_seq)
    valid_data = extract_char_and_desc_idx_tensors(
        valid_data, char_seq, desc_seq)
    test_data = extract_char_and_desc_idx_tensors(
        test_data, char_seq, desc_seq)

    return EmbedTuple(word_weights, word2idx, char_weights, char2idx), DataTuple(train_data, valid_data, test_data, "Tensors")


if __name__ == '__main__':
    from project.data.preprocessed.overfit import overfit_data as DATA

    weights, word2idx = get_weights_word2idx()
    char_weights, char2idx = get_weights_char2idx(200)
    data = tokenize_descriptions(DATA.test, word2idx, char2idx)
    print(data[0])
