from functools import wraps


def data_args(parse_fn):
    @wraps(parse_fn)
    def wrapper(*args, **kwds):
        p = parse_fn(*args, **kwds)
        p.add_argument('--vocab-size', '-v', dest='vocab_size', action='store',
                       type=int, default=50000,
                       help='size of embedding vocab')
        p.add_argument('--char-seq', '-c', dest='char_seq', action='store',
                       type=int, default=24,
                       help='max char sequence length')
        p.add_argument('--desc-seq', '-d', dest='desc_seq', action='store',
                       type=int, default=120,
                       help='max desecription sequence length')
        p.add_argument('--char-embed', '-f', dest='char_embed', action='store',
                       type=int, default=100,
                       help='size of char embedding')
        p.add_argument('--desc-embed', '-g', dest='desc_embed', action='store',
                       type=int, default=100,
                       help='size of glove embedding: 50, 100, 200 or 300')
        p.add_argument('--use-full-dataset', '-F', dest='use_full_dataset', action='store_true',
                       default=False,
                       help='use the complete data set (slow)')
        p.add_argument('--use-split-dataset', '-S', dest='use_split_dataset', action='store_true',
                       default=False,
                       help='use the dataset where train and test args are split by codebase. must be used with full dataset')
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
                       type=int, default=100,
                       help='how often to run a test and dump output')
        p.add_argument('--dump-translation', '-D', dest='test_translate', action='store',
                       type=int, default=5,
                       help='dump extensive test information on each test batch')
        p.add_argument('--save-every', '-E', dest='save_every', action='store',
                       type=int, default=5,
                       help='how often to save every run')
        return p
    return wrapper

def train_args(parse_fn):
    @wraps(parse_fn)
    def wrapper(*args, **kwds):
        p = parse_fn(*args, **kwds)
        p.add_argument('--epochs', '-e', dest='epochs', action='store',
                       type=int, default=5000,
                       help='minibatch size for model')
        p.add_argument('--learning-rate', '-r', dest='lr', action='store',
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
