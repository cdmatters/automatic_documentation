from collections import Counter
import os

import numpy as np
from project.utils.tokenize import get_data_tuple, choose_tokenizer, \
                     CHAR_VOCAB, nltk_tok,\
                     get_embed_filenames, get_weights_word2idx, \
                     get_weights_char2idx

NMT_PATH = "./nmt/nmt_data/{}"

def gen_embed_file(dim):
    DIR = os.path.dirname(os.path.abspath(__file__))

    with open(get_embed_filenames()[dim], 'r') as f:
        lines = f.readlines()
    start_toks = ["<unk>", "<s>", "</s>"]
    start_lines = ["{} {}\n".format(t, " ".join(["{:5f}".format(i) for i in np.random.randn(dim)])) for t in start_toks]
    start_lines.extend(lines)
    with open('{}/glove_inserted.{}.txt.de'.format(DIR, dim), 'w') as f:
        for s in start_lines:
            f.write(s)

def write_data_to_file(data, name):
    char_seq = []
    desc_seq = []
    for d in data:
        char_seq.append(d['arg_name_tokens'])
        desc_seq.append(nltk_tok(d['arg_desc']))

    with open(NMT_PATH.format(name+'.ch'), 'w') as f:
        for c in char_seq:
            f.write(" ".join(c))
            f.write("\n")

    with open(NMT_PATH.format(name+'.de'), 'w') as f:
        for d in desc_seq:
            f.write(" ".join(d))
            f.write("\n")

def split_by_char(word):
    return " ".join(word.split())

def gen_char_vocab_file():
    with open(NMT_PATH.format('vocab.ch'), 'w') as f:
        for c in CHAR_VOCAB:
            f.write("{}\n".format(c))

def gen_desc_vocab_file(train_data, vocab_size, dim):
    all_toks = []
    for d in train_data:
        all_toks.extend(nltk_tok(d['arg_desc']))
    common_tokens = Counter(all_toks).most_common()


    filename =  get_embed_filenames()[dim]
    vocab = ["<unk>", "<s>", "</s>"]

    with open(filename, 'r', encoding='utf-8') as f:
        file_voc = [ line.split()[0] for line in f ]
        for tok, count in common_tokens:

            if tok in file_voc and count > 2:
                vocab.append(tok)
                file_voc.remove(tok)

            if len(vocab) > vocab_size:
                break

        if len(vocab) < vocab_size:
            vocab.extend(file_voc[:vocab_size - len(vocab)])

    # vocab.extend(get_special_tokens())

    with open(NMT_PATH.format('vocab.de'), 'w', encoding='utf-8') as f:
        for v in vocab:
            f.write("{}\n".format(v))



if __name__ == "__main__":
    # gen_embed_file(200)

    desc_embed = 200
    char_embed = 200
    vocab_size=35000

    data_tuple = get_data_tuple(use_full_dataset=True, use_split_dataset=False)
    tokenizer = choose_tokenizer('var_only')



    gen_char_vocab_file()
    gen_desc_vocab_file(data_tuple.train, vocab_size, desc_embed)

    print("Loading GloVe weights and word to index lookup table")
    word_weights, word2idx = get_weights_word2idx(desc_embed, vocab_size, data_tuple.train)
    print("Creating char to index look up table")
    char_weights, char2idx = get_weights_char2idx(char_embed)

    train_data = tokenizer(data_tuple.train, word2idx, char2idx)
    valid_data = tokenizer(data_tuple.valid, word2idx, char2idx)
    test_data = tokenizer(data_tuple.test, word2idx, char2idx)

    write_data_to_file(train_data, 'train')
    write_data_to_file(valid_data, 'valid')
    write_data_to_file(valid_data, 'test')

