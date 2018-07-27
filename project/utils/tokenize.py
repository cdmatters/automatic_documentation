from collections import namedtuple, Counter
import os

from nltk import word_tokenize
import numpy as np
from tqdm import tqdm

from project.data.preprocessed import DataTuple

PAD_TOKEN = '<PAD>'
UNKNOWN_TOKEN = '<UNK>'
START_OF_TEXT_TOKEN = '<START>'
END_OF_TEXT_TOKEN = '<END>'
SEPARATOR_1 = '<SEP-1>'
SEPARATOR_2 = '<SEP-2>'
SEPARATOR_3 = '<SEP-3>'

CHAR_VOCAB = 'abcdefghijklmnopqrstuvwyxzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_*:'


EmbedTuple = namedtuple(
    "EmbedTuple", ['word_weights', 'word2idx', 'char_weights', 'char2idx'])

def get_special_tokens():
    return []
    return [PAD_TOKEN, UNKNOWN_TOKEN, START_OF_TEXT_TOKEN,
            END_OF_TEXT_TOKEN, SEPARATOR_1, SEPARATOR_2, SEPARATOR_3]

def get_hash_string(d):
    hash_string = []
    for l in d["arg_name_tokens"]:
        if l == SEPARATOR_1:
            hash_string.append("|")
        elif l == SEPARATOR_2:
            hash_string.append("-")        
        elif l == SEPARATOR_3:
            hash_string.append("+")
        elif l == END_OF_TEXT_TOKEN:
            hash_string.append("")    
        else:
            hash_string.append(l)
    return "".join(hash_string)


def get_weights_char2idx(char_embed):
    # Weights are random, 300d
    dim = char_embed
    arg_alphabet = CHAR_VOCAB

    # ':' is a stop token
    char2idx = {a: i+1 for i, a in enumerate(arg_alphabet)}
    char2idx[SEPARATOR_1] = len(char2idx.keys())
    char2idx[SEPARATOR_2] = len(char2idx.keys())
    char2idx[SEPARATOR_3] = len(char2idx.keys())
    char2idx[END_OF_TEXT_TOKEN] = len(char2idx.keys())

    char_weights = np.random.uniform(
        low=-0.1, high=0.1, size=[len(char2idx.keys())+1, dim])
    return (char_weights, char2idx)


def get_embed_filenames():
    DIR = os.path.dirname(os.path.abspath(__file__))

    return {
        50: "{}/glove/glove.6B.50d.txt".format(DIR),
        100: "{}/glove/glove.6B.100d.txt".format(DIR),
        200: "{}/glove/glove.6B.200d.txt".format(DIR),
        300: "{}/glove/glove.6B.300d.txt".format(DIR),
    }

def gen_train_vocab(train_data, embed_file, vocab_size):
    all_toks = []
    for d in train_data:
        all_toks.extend(nltk_tok(d['arg_desc']))
    most_common = Counter(all_toks).most_common()
    
    vocab = []
    if os.path.isfile(embed_file + ".vocab"):
        with open(embed_file + ".vocab", 'r', encoding='utf-8') as f:
            file_voc = [line.strip() for line in f]
        
    else:
        with open(embed_file, 'r', encoding='utf-8') as f:
            file_voc = [ line.split()[0] for line in f ]
        with open(embed_file + ".vocab", 'w', encoding='utf-8') as f:
            f.write("\n".join(file_voc))
    
    for tok, count in most_common:

        if tok in file_voc and count > 4:
            vocab.append(tok)
            file_voc.remove(tok)

        if len(vocab) > vocab_size:
            break

    if len(vocab) < vocab_size:
        vocab.extend(file_voc[:vocab_size - len(vocab)])
    return set(vocab)


def get_weights_word2idx(desc_embed, vocab_size=100000, train_data=None):
    # Currently get the 300d embeddings from GloVe
    embed_files = get_embed_filenames()
    embed_file = embed_files[desc_embed]

    if train_data is not None:
        desired_vocab = gen_train_vocab(train_data, embed_file, vocab_size)
        

    word2idx = {PAD_TOKEN: 0}
    weights = [np.random.randn(desc_embed)]


    with open(embed_file, "r", encoding='utf-8') as f:
        i = 0 
        for line in tqdm(f):
            values = line.split()

            word = values[0]
            if word in desired_vocab:
                word_weights = np.array(values[1:]).astype(np.float32)
                
                word2idx[word] = i + 1
                weights.append(word_weights)
                
                i += 1

            if i > vocab_size:
                break

    word2idx[UNKNOWN_TOKEN] = len(weights)
    weights.append(np.random.randn(desc_embed))

    word2idx[START_OF_TEXT_TOKEN] = len(weights)
    weights.append(np.random.randn(desc_embed))

    word2idx[END_OF_TEXT_TOKEN] = len(weights)
    weights.append(np.random.randn(desc_embed))

    weights = np.asarray(weights, dtype=np.float32)
    return (weights, word2idx)

def nltk_tok(desc):
    return word_tokenize(desc.replace('\\n', " ").lower())


def fill_descriptions_tok(d, word2idx):
    unk_token = word2idx[UNKNOWN_TOKEN]
    desc_tok = nltk_tok(d['arg_desc'])
    d['arg_desc_translate'] = desc_tok

    d['arg_desc_tokens'] = [START_OF_TEXT_TOKEN]
    d['arg_desc_idx'] = [word2idx[START_OF_TEXT_TOKEN]]

    d['arg_desc_tokens'].extend(
        [w if w in word2idx else UNKNOWN_TOKEN for w in desc_tok])
    d['arg_desc_idx'].extend([word2idx.get(t, unk_token)
                              for t in desc_tok])

    d['arg_desc_tokens'].append(END_OF_TEXT_TOKEN)
    d['arg_desc_idx'].append(word2idx[END_OF_TEXT_TOKEN])


def fill_name_tok(d, char2idx):
    d['arg_name_tokens'] = [c for c in d['arg_name']]
    d['arg_name_idx'] = [char2idx[c] for c in d['arg_name']]

    d['arg_name_tokens'].append(END_OF_TEXT_TOKEN)
    d['arg_name_idx'].append(char2idx[END_OF_TEXT_TOKEN])


def fill_name_funcname_tok(d, char2idx):
    d['arg_name_tokens'] = [c for c in d['arg_name']]
    d['arg_name_idx'] = [char2idx[c] for c in d['arg_name']]

    d['arg_name_tokens'].append(SEPARATOR_1)
    d['arg_name_idx'].append(char2idx[SEPARATOR_1])

    d['arg_name_tokens'].extend([c for c in d['name']])
    d['arg_name_idx'].extend([char2idx[c] for c in d['name']])

    d['arg_name_tokens'].append(END_OF_TEXT_TOKEN)
    d['arg_name_idx'].append(char2idx[END_OF_TEXT_TOKEN])


def fill_name_other_args_tok(d, char2idx):
    d['arg_name_tokens'] = [c for c in d['arg_name']]
    d['arg_name_idx'] = [char2idx[c] for c in d['arg_name']]

    d['arg_name_tokens'].append(SEPARATOR_1)
    d['arg_name_idx'].append(char2idx[SEPARATOR_1])

    for a in d["args"]:
        if a == d['arg_name']:
            continue
        else:
            d['arg_name_tokens'].extend([c for c in a])
            d['arg_name_idx'].extend([char2idx[c] for c in a])

            d['arg_name_tokens'].append(SEPARATOR_2)
            d['arg_name_idx'].append(char2idx[SEPARATOR_2])

    d['arg_name_tokens'].append(END_OF_TEXT_TOKEN)
    d['arg_name_idx'].append(char2idx[END_OF_TEXT_TOKEN])


def fill_name_funcname_other_args_tok(d, char2idx):
    d['arg_name_tokens'] = [c for c in d['arg_name']]
    d['arg_name_idx'] = [char2idx[c] for c in d['arg_name']]

    d['arg_name_tokens'].append(SEPARATOR_1)
    d['arg_name_idx'].append(char2idx[SEPARATOR_1])

    d['arg_name_tokens'].extend([c for c in d['name']])
    d['arg_name_idx'].extend([char2idx[c] for c in d['name']])

    d['arg_name_tokens'].append(SEPARATOR_2)
    d['arg_name_idx'].append(char2idx[SEPARATOR_2])

    for a in d["args"]:
        if a == d['arg_name']:
            continue
        else:
            d['arg_name_tokens'].extend([c for c in a])
            d['arg_name_idx'].extend([char2idx[c] for c in a])

            d['arg_name_tokens'].append(SEPARATOR_3)
            d['arg_name_idx'].append(char2idx[SEPARATOR_3])

    d['arg_name_tokens'].append(END_OF_TEXT_TOKEN)
    d['arg_name_idx'].append(char2idx[END_OF_TEXT_TOKEN])


def tokenize_vars_funcname_and_descriptions(data, word2idx, char2idx):
    for i, d in enumerate(data):
        fill_descriptions_tok(d, word2idx)
        fill_name_funcname_tok(d, char2idx)
    return data


def tokenize_vars_and_descriptions(data, word2idx, char2idx):
    for i, d in enumerate(data):
        fill_descriptions_tok(d, word2idx)
        fill_name_tok(d, char2idx)
    return data


def tokenize_vars_other_args_and_descriptions(data, word2idx, char2idx):
    for i, d in enumerate(data):
        fill_descriptions_tok(d, word2idx)
        fill_name_other_args_tok(d, char2idx)
    return data


def tokenize_vars_funcname_other_args_and_descriptions(data, word2idx, char2idx):
    for i, d in enumerate(data):
        fill_descriptions_tok(d, word2idx)
        fill_name_funcname_other_args_tok(d, char2idx)
    return data

def tokenize_code2vec(data, word2idx, vocab_size=20000):
    for d in data:
        d["path_idx"] = np.fromstring(d["path_idx"], dtype=int, sep=" ")
        d["target_var_idx"] = np.fromstring(d["target_var_idx"], dtype=int, sep=" ")
    
        d["path_idx"][d["path_idx"] > vocab_size] = 0
        d["target_var_idx"][d["target_var_idx"] > vocab_size] = 0

    return data 


def tokenize_src_all_basic_tokens(data, word2idx):
    for i, d in enumerate(data):
        unk_token = word2idx[UNKNOWN_TOKEN]
        src_tok = nltk_tok(d['src'])
        
        d['src_tokens'] = [w if w in word2idx else UNKNOWN_TOKEN for w in src_tok]
        d['src_idx'] = [word2idx.get(t, unk_token) for t in src_tok]
    
    return data

def extract_tensors(data, fields, seq_lengths):
    tensors = [[] for _ in fields]
    for d in data:
        for t, f, s in zip(tensors, fields, seq_lengths):
            pad = [d[f][i] if i < len(d[f]) else 0 for i in range(s)]
            t.append(np.array(pad))
    return [np.stack(t) for t in tensors]

def extract_transations(data):
    return [d['arg_desc_translate'] for d in data]

def extract_model_data(data, fields, seq_lengths):
    tensors = extract_tensors(data, fields, seq_lengths)
    translations = extract_transations(data)
    return tuple(tensors + [translations])


def get_data_tuple(use_full_dataset, use_split_dataset, no_dups, use_quickload=False):
    if use_full_dataset:
        if use_split_dataset:
            from project.data.preprocessed.split import split_quickload_data as q_data
            from project.data.preprocessed.split import split_data as data
        else:
            if no_dups == 0:
                from project.data.preprocessed.unsplit import unsplit_quickload_data as q_data
                from project.data.preprocessed.unsplit import unsplit_data as data
            elif no_dups == 1:
                from project.data.preprocessed.no_dups_1 import no_dups_1_quickload_data as q_data
                from project.data.preprocessed.no_dups_1 import no_dups_1_data as data
            elif no_dups == 2:
                from project.data.preprocessed.no_dups_2 import no_dups_2_quickload_data as q_data
                from project.data.preprocessed.no_dups_2 import no_dups_2_data as data
            elif no_dups == 3:
                from project.data.preprocessed.no_dups_3 import no_dups_3_quickload_data as q_data
                from project.data.preprocessed.no_dups_3 import no_dups_3_data as data
            elif no_dups == 4:
                from project.data.preprocessed.no_dups_4 import no_dups_4_quickload_data as q_data
                from project.data.preprocessed.no_dups_4 import no_dups_4_data as data
            elif no_dups == 5:
                from project.data.preprocessed.no_dups_5 import no_dups_5_quickload_data as q_data
                from project.data.preprocessed.no_dups_5 import no_dups_5_data as data
            elif no_dups == 10:
                from project.data.preprocessed.no_dups_X import no_dups_X_quickload_data as q_data
                from project.data.preprocessed.no_dups_X import no_dups_X_data as data
    else:
        from project.data.preprocessed.overfit import overfit_quickload_data as q_data
        from project.data.preprocessed.overfit import overfit_data as data
    if use_quickload:
        return q_data()
    else:
        return data()

def choose_code_tokenizer(tokenizer):
    if tokenizer == 'full':
        tokenize = tokenize_src_all_basic_tokens
    if tokenizer == 'code2vec':
        tokenize = tokenize_code2vec
    return tokenize

def choose_tokenizer(tokenizer):
    if tokenizer == 'var_only':
        tokenize = tokenize_vars_and_descriptions
    elif tokenizer == 'var_funcname':
        tokenize = tokenize_vars_funcname_and_descriptions
    elif tokenizer == 'var_otherargs':
        tokenize = tokenize_vars_other_args_and_descriptions
    elif tokenizer == 'var_funcname_otherargs':
        tokenize = tokenize_vars_funcname_other_args_and_descriptions
    return tokenize

def get_embed_tuple_and_data_tuple(vocab_size, char_seq, desc_seq, char_embed, desc_embed,
                                   use_full_dataset, use_split_dataset, tokenizer='var_only', 
                                   no_dups=0, code_tokenizer="full"):
    if code_tokenizer == "code2vec":
        data_tuple = get_data_tuple(use_full_dataset, use_split_dataset, no_dups, use_quickload=True)
    else:
        data_tuple = get_data_tuple(use_full_dataset, use_split_dataset, no_dups, use_quickload=False)

    print("Loading GloVe weights and word to index lookup table")
    word_weights, word2idx = get_weights_word2idx(desc_embed, vocab_size, data_tuple.train)
    print("Creating char to index look up table")
    char_weights, char2idx = get_weights_char2idx(char_embed)

    input_tokenize = choose_tokenizer(tokenizer)
    print("Tokenizing the word descriptions and characters")
    train_data = input_tokenize(data_tuple.train, word2idx, char2idx)
    valid_data = input_tokenize(data_tuple.valid, word2idx, char2idx)
    test_data = input_tokenize(data_tuple.test, word2idx, char2idx)

    code_tokenize = choose_code_tokenizer(code_tokenizer)
    print("Tokenizing the src code")
    train_data = code_tokenize(data_tuple.train, word2idx)
    valid_data = code_tokenize(data_tuple.valid, word2idx)
    test_data = code_tokenize(data_tuple.test, word2idx)
    
    print("Extracting tensors train and test")
    
    fields = ["arg_name_idx", "arg_desc_idx"]
    seq_lengths = [char_seq, desc_seq]

    if code_tokenizer == 'code2vec':
        fields.extend(["path_idx", "target_var_idx"])
        seq_lengths.extend([1000, 1000])
    elif code_tokenizer == "full":
        fields.extend(["src_idx"])
        seq_lengths.extend([200])
    
    train_data = extract_model_data(train_data, fields, seq_lengths)
    valid_data = extract_model_data(valid_data, fields, seq_lengths)
    test_data = extract_model_data(test_data, fields, seq_lengths)

    return EmbedTuple(word_weights, word2idx, char_weights, char2idx), DataTuple(train_data, valid_data, test_data, "Tensors")

if __name__ == '__main__':
    # from project.data.preprocessed.overfit import overfit_data as DATA

    # weights, word2idx = get_weights_word2idx()
    # char_weights, char2idx = get_weights_char2idx(200)
    # data = tokenize_vars_and_descriptions(DATA.test, word2idx, char2idx)

    data = get_embed_tuple_and_data_tuple(vocab_size=5000, char_seq=550, desc_seq=300,
                                   char_embed=50, desc_embed=50,
                                   use_full_dataset=True, use_split_dataset=False, tokenizer='var_funcname_otherargs')

    char_tensor = data[1].train[0]
    print(np.max(char_tensor, axis=0))
    print(data[1].train[0].shape)
