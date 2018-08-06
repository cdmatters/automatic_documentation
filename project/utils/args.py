from functools import wraps


def data_args(parse_fn):
    @wraps(parse_fn)
    def wrapper(*args, **kwds):
        p = parse_fn(*args, **kwds)
        p.add_argument('--vocab-size', '-v', dest='vocab_size', action='store',
                       type=int, default=30000,
                       help='size of embedding vocab')
        p.add_argument('--char-seq', '-c', dest='char_seq', action='store',
                       type=int, default=24,
                       help='max char sequence length')
        p.add_argument('--desc-seq', '-d', dest='desc_seq', action='store',
                       type=int, default=120,
                       help='max desecription sequence length')
        #p.add_argument('--char-embed', '-f', dest='char_embed', action='store',
        #               type=int, default=200,
        #               help='size of char embedding')
        p.add_argument('--desc-embed', '-g', dest='desc_embed', action='store',
                       type=int, default=200,
                       help='size of glove embedding: 50, 100, 200 or 300')
        p.add_argument('--use-full-dataset', '-F', dest='use_full_dataset', action='store_true',
                       default=False,
                       help='use the complete data set (slow)')
        p.add_argument('--use-split-dataset', '-S', dest='use_split_dataset', action='store_true',
                       default=False,
                       help='use the dataset where train and test args are split by codebase. must be used with full dataset')
        p.add_argument('--no-dups', '-X', dest='no_dups', action='store',
                       type=int, default=0,
                       help='use the complete data set (slow)')
        p.add_argument('--tokenizer', '-to', dest='tokenizer', action='store',
                       type=str, default='var_only',
                       help='the type of tokenizer to build the char_sequence: var_only, var_funcname')
        p.add_argument('--code-tokenizer', '-ct', dest='code_tokenizer', action='store',
                       type=str, default='code2vec',
                       help='type of code tokenization "code2vec" or "full"')

        return p
    return wrapper


def log_args(parse_fn):
    @wraps(parse_fn)
    def wrapper(*args, **kwds):
        p = parse_fn(*args, **kwds)
        p.add_argument('--name', '-N', dest='name', action='store',
                       type=str, default='model',
                       help='name of the model')
        p.add_argument('--logdir', '-L', dest='logdir', action='store',
                       type=str, default='logs',
                       help='directory for storing logs and raw experiment runs')
        p.add_argument('--test-freq', '-T', dest='test_freq', action='store',
                       type=int, default=200,
                       help='how often to run a test and dump output')
        p.add_argument('--dump-translation', '-D', dest='test_translate', action='store',
                       type=int, default=5,
                       help='dump extensive test information on each test batch')
        p.add_argument('--save-every', '-E', dest='save_every', action='store',
                       type=int, default=5,
                       help='how often to save every run')
        p.add_argument('--mode', '-M', dest='mode', action='store',
                       type=str, default="TRAIN",
                       help='TRAIN, LOAD, RETURN')
        return p
    return wrapper

def code2vec_args(parse_fn):
    @wraps(parse_fn)
    def wrapper(*args, **kwds):
        p = parse_fn(*args, **kwds)
        p.add_argument('--path-seq', '-ps', dest='path_seq', action='store',
                        type=int, default=5000,
                        help='max number of paths to include')
        p.add_argument('--path-vocab', '-pv', dest='path_vocab', action='store',
                       type=int, default=10000,
                       help='vocab of paths recognised')
        p.add_argument('--path-embed', '-pe', dest='path_embed', action='store',
                       type=int, default=200,
                       help='size of initial path and var embeddings')
        p.add_argument('--code2vec-size', '-vs', dest='vec_size', action='store',
                       type=int, default=200,
                       help='size of code2vec vector')
        return p
    return wrapper

def encoder_args(parse_fn):
    @wraps(parse_fn)
    def wrapper(*args, **kwds):
        p = parse_fn(*args, **kwds)
        p.add_argument('--lstm-size', '-l', dest='lstm_size', action='store',
                        type=int, default=200,
                        help='size of LSTM size')
        p.add_argument('--bidirectional', '-bi', dest='bidirectional', action='store',
                       type=int, default=1,
                       help='use bidirectional lstm')
        return p
    return wrapper

def train_args(parse_fn):
    @wraps(parse_fn)
    def wrapper(*args, **kwds):
        p = parse_fn(*args, **kwds)
        p.add_argument('--epochs', '-e', dest='epochs', action='store',
                       type=int, default=200,
                       help='minibatch size for model')
        p.add_argument('--learning-rate', '-r', dest='learning_rate', action='store',
                       type=float, default=0.001,
                       help='learning rate for model')
        p.add_argument('--batch-size', '-b', dest='batch_size', action='store',
                       type=int, default=128,
                       help='minibatch size for model')
        p.add_argument('--dropout', '-dd', dest='dropout', action='store',
                       type=float, default=0.3,
                       help='minibatch size for model')
        return p
    return wrapper
