from project.utils.tokenize import get_data_tuple, choose_tokenizer, \
                     CHAR_VOCAB, get_special_tokens, \
                     get_embed_filenames, get_weights_word2idx, \
                    get_weights_char2idx

NMT_PATH = "./nmt/nmt_data/{}"

def write_data_to_file(data, name):
    char_seq = []
    desc_seq = []
    for d in data:
        char_seq.append(d['arg_name_tokens'])
        desc_seq.append(d['arg_desc_tokens'])
    
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

def gen_desc_vocab_file(vocab_size, dim):
    filename =  get_embed_filenames()[dim]
    vocab = []
    with open(filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            values = line.split()
            vocab.append(values[0])
            
            if i > vocab_size:
                break

    vocab.extend(get_special_tokens())

    with open(NMT_PATH.format('vocab.de'), 'w') as f:
        for v in vocab:
            f.write("{}\n".format(v))



if __name__ == "__main__":
    desc_embed=200
    char_embed = 200
    vocab_size=50000

    data_tuple = get_data_tuple(use_full_dataset=False, use_split_dataset=False)
    tokenizer = choose_tokenizer('var_only')

    gen_char_vocab_file()
    gen_desc_vocab_file(vocab_size, desc_embed)

    print("Loading GloVe weights and word to index lookup table")
    word_weights, word2idx = get_weights_word2idx(desc_embed, vocab_size)
    print("Creating char to index look up table")
    char_weights, char2idx = get_weights_char2idx(char_embed)

    train_data = tokenizer(data_tuple.train, word2idx, char2idx)
    valid_data = tokenizer(data_tuple.valid, word2idx, char2idx)
    test_data = tokenizer(data_tuple.test, word2idx, char2idx)
    
    write_data_to_file(train_data, 'train')
    write_data_to_file(valid_data, 'valid')
    
