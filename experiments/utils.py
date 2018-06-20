import nltk 
import numpy as np
from tqdm import tqdm

import os

PAD_TOKEN = '<PAD>'
UNKNOWN_TOKEN = '<UNK>'
START_OF_TEXT_TOKEN = '<START>'
END_OF_TEXT_TOKEN = '<END>'

def get_char2idx():
    arg_alphabet = 'abcdefghijklmnopqrstuvwyxzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_*:'

    char2idx = {a:i+1 for i,a in enumerate(arg_alphabet)} # ':' is a stop token
    char2idx[END_OF_TEXT_TOKEN] = len(char2idx.keys()) 
    return char2idx 

def get_weights_word2idx(vocab_size=100000):
    DIR = os.path.dirname(os.path.abspath(__file__))

    word2idx = { PAD_TOKEN: 0 }
    weights = []
    with open("{}/glove/glove.42B.300d.txt".format(DIR), "r") as f:
        for i, line in tqdm(enumerate(f)):
            values = line.split()

            word = values[0]
            word_weights = np.asarray(values[1:], dtype=np.float32)  
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
        d['arg_desc_idx'] = [word2idx[END_OF_TEXT_TOKEN]]
        d['arg_desc_tokens'].extend([ w if w in word2idx else UNKNOWN_TOKEN for w in desc_tok ])
        d['arg_desc_idx'].extend([word2idx.get(t, unk_token) for t in desc_tok])
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
        char_pad = [d['arg_name_idx'][i] if i < len(d['arg_name_idx']) else 0 for i in range(char_dim)]
        chars.append(np.array(char_pad))

        desc_pad = [d['arg_desc_idx'][i] if i < len(d['arg_desc_idx']) else 0 for i in range(desc_dim)]
        descs.append(np.array(desc_pad))
    return np.stack(chars), np.stack(descs)

if __name__ == '__main__':
    from data.preprocessed.overfit import data as DATA

    weights, word2idx = get_weights_word2idx()
    char2idx = get_char2idx()
    data = tokenize_descriptions(DATA.test, word2idx, char2idx)
    print(data[0])
    
