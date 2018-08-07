import argparse
from collections import defaultdict, Counter
import gc
import random

random.seed(100)
# from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import nltk
import numpy as np
from tqdm import tqdm

from project.external.nmt import bleu
from project.models.base_model import ArgumentSummary, SingleTranslation
from project.utils import args, tokenize


def default_dict_factory(): return defaultdict(list)


class HashtableBaseline(object):
    def __init__(self, code_mode, idx2path=None, idx2tv=None, model_name="Hashtable Model"):
        self.name = model_name

        self.code_only = ("code_only" in code_mode)
        self.code_mode = code_mode.replace("code_only_", "")

        self.lookup_list = defaultdict(default_dict_factory)
        self.codepath_lookup_list_soft = defaultdict(default_dict_factory)
        self.codepath_lookup_list_hard = defaultdict(list)

        self.idx2path = idx2path
        self.idx2tv = idx2tv
        if self.code_mode == 'soft':
            assert self.idx2path is not None and self.idx2tv is not None

        self.descriptions = []


    def __str__(self):
        summary_string = 'MODEL: {classname}\nName: {name}\n\n{summary}'
        args = "code_mode: {}, code_only: {}".format(self.code_mode, self.code_only)
        return summary_string.format(
            name=self.name, classname=self.__class__.__name__, summary="Args: {}".format(args))

    def get_n_grams(self, n, name):
        return [name[i:i + n] for i in range(len(name) + 1 - n)]


    def lookup_hard_codepaths(self, d, hardest=False):
        matches = []
        for p, tv in zip(d["path_idx"], d["target_var_idx"]):
            matches.extend(self.codepath_lookup_list_hard[(p,tv)])
        if matches:
            if hardest:
                mc = Counter(matches).most_common()
                top = mc[0][1]
                return [m for m,c in mc if c == top]
            else:
                return matches
        else:
            all_d_indices = range(len(self.descriptions))
            return [random.choice(all_d_indices)]


    def lookup_soft_codepaths(self, d, softest=False):
        matches = []
        for p, tv in zip(d["path_idx"], d["target_var_idx"]):
            path = tuple(self.idx2path[p] + self.idx2tv[tv])
            for i in reversed(range(len(path))):
                ngrams = self.get_n_grams(i + 1, path)
                path_match = []
                for n in ngrams:
                    path_match.extend(self.codepath_lookup_list_soft[i][n])
                if path_match:
                    matches.append((path_match, len(n)))
                    break

        if matches:
            if softest: # simply flatten
                return [i for j, _ in matches for i in j]
            else:
                highest = max(matches, key= lambda x:x[1])[1]
                return [i for j, c in matches for i in j if c == highest] # flatten and filter
        else:  # not even 1-gram!
            all_d_indices = range(len(self.descriptions))
            return [random.choice(all_d_indices)]

    def lookup_description_indices(self, name):
        for i in reversed(range(len(name))):
            ngrams = self.get_n_grams(i + 1, name)
            descriptions = []
            for n in ngrams:
                descriptions.extend(self.lookup_list[i][n])
            if descriptions:
                break

        if not descriptions:  # not even 1-gram!
            all_d_indices = range(len(self.descriptions))
            descriptions.append(random.choice(all_d_indices))
        return descriptions

    def tok(self, word):
        desc = word.replace('\\n', " ").lower()
        return nltk.word_tokenize(desc)

    def test(self, test_data):
        translations = []
        for d in tqdm(test_data, leave=False):

            hash_string = tokenize.get_hash_string(d)
            if self.code_only:
                indices = []
            else:
                indices = self.lookup_description_indices(hash_string)

            if self.code_mode == "hard":
                indices += self.lookup_hard_codepaths(d)
            if self.code_mode == "hardest":
                indices += self.lookup_hard_codepaths(d, hardest=True)

            if self.code_mode == "soft":
                indices += self.lookup_soft_codepaths(d)
            if self.code_mode == "softest":
                indices += self.lookup_soft_codepaths(d, softest=True)

            descriptions = [self.descriptions[i] for i in indices]
            translation = random.choice(descriptions)

            translations.append(SingleTranslation(
                hash_string, d["arg_desc_translate"], d["arg_desc_tokens"], translation))
        return translations

    def train(self, train_data):
        for i, d in enumerate(tqdm(train_data, leave=False)):
            hash_string = tokenize.get_hash_string(d)
            l = len(hash_string)
            self.descriptions.append(d["arg_desc_tokens"])

            for j in range(l):
                ngrams = self.get_n_grams(j + 1, hash_string)
                for n in ngrams:
                    self.lookup_list[j][n].append(i)

            if self.code_mode in ["hard", "hardest"]:
                for p, tv in zip(d["path_idx"], d["target_var_idx"]):
                    if (p, tv) != (0, 0) and p != 1 and tv != 1:
                        self.codepath_lookup_list_hard[(p,tv)].append(i)

            if self.code_mode in ["soft"]:
                for p, tv in zip(d["path_idx"], d["target_var_idx"]):
                    path = tuple(self.idx2path[p] + self.idx2tv[tv])
                    l = len(path)
                    for j in range(l):
                        ngrams = self.get_n_grams(j + 1, path)
                        for n in ngrams:
                            self.codepath_lookup_list_soft[j][n].append(i)


    def evaluate(self, all_translations):
        references = [[t.description] for t in all_translations]
        translations = [t.translation for t in all_translations]

        return bleu.compute_bleu(references, translations, max_order=4, smooth=False)

    def main(self, train_data, test_data):
        self.train(train_data)
        translations = self.test(test_data)
        bleu_tuple = self.evaluate(translations)
        bleu = bleu_tuple[0]*100
        print(bleu)
        return bleu

def setup_log(**kwargs):
    logfile = kwargs["logfile"]
    if logfile is None:
        return print
    else:
        def _log(*args):
            str_args = [str(a) for a in args]
            with open(logfile, 'a') as f:
                f.write(*str_args)
                f.write("\n")
            print(*args)
        return _log


def _run_model(**kwargs):
    LOG = setup_log(**kwargs)
    data_tuple = tokenize.get_data_tuple(
        kwargs['use_full_dataset'], kwargs['use_split_dataset'],
        kwargs['no_dups'], use_code2vec_cache=True)


    idx2path, idx2tv = tokenize.get_idx2code2vec(
            kwargs['use_full_dataset'], kwargs['use_split_dataset'],
            kwargs['no_dups'])
    idx2path = {i: path.split(" ") for i, path in idx2path.items()}
    idx2tv = {i: [tv]for i, tv in idx2tv.items()}


    LOG("Loading GloVe weights and word to index lookup table")
    _, word2idx = tokenize.get_weights_word2idx(
        kwargs['desc_embed'], kwargs['vocab_size'], data_tuple.train)
    _ = defaultdict(int)

    this_tokenizer = tokenize.choose_tokenizer(kwargs['tokenizer'])
    train_data = this_tokenizer(data_tuple.train, word2idx, _)
    valid_data = this_tokenizer(data_tuple.valid, word2idx, _)
    test_data = this_tokenizer(data_tuple.test, word2idx, _)

    this_code_tokenizer = tokenize.choose_code_tokenizer(kwargs["code_tokenizer"])
    train_data = this_code_tokenizer(data_tuple.train, word2idx=word2idx, path_vocab=kwargs["path_vocab"])
    valid_data = this_code_tokenizer(data_tuple.valid, word2idx=word2idx, path_vocab=kwargs["path_vocab"])
    test_data = this_code_tokenizer(data_tuple.test, word2idx=word2idx, path_vocab=kwargs["path_vocab"])

    train_data = tokenize.trim_paths(data_tuple.train, kwargs["path_seq"])
    valid_data = tokenize.trim_paths(data_tuple.valid, kwargs["path_seq"])
    test_data = tokenize.trim_paths(data_tuple.test, kwargs["path_seq"])



    all_results = []
    for mode in kwargs['code_mode']:
        model = HashtableBaseline(mode, idx2path, idx2tv)
        summary = ArgumentSummary(model, kwargs)
        LOG(summary)

        results = []
        test_results = []
        model.train(train_data)
        for i in range(kwargs['n_times']):
            random.seed(i)

            bleu = model.evaluate(model.test(valid_data))[0]*100
            bleu_test = model.evaluate(model.test(test_data))[0]*100
            LOG(bleu)
            results.append(bleu)
            test_results.append(bleu_test)

        r = (np.mean(results), np.std(results))
        r_test = (np.mean(test_results), np.std(test_results))
        LOG("----- {} -----".format(len(results)))
        LOG("VALID  {:.5f} +/-  {:.5f}   TEST  {:.5f} +/-  {:.5f} ".format(
            r[0], r[1], r_test[0], r_test[1]))
        LOG("      & $ {:.5f} \pm  {:.5f} $ & $ {:.5f} \pm {:.5f} $ &".format(
            r[0], r[1], r_test[0], r_test[1]))

        all_results.append((mode, r))
        gc.collect()

    for m, r in all_results:
        LOG("Mode: {},  Score {:.5f} +/- {:.5f}".format(m, r[0], r[1]))


@args.data_args
@args.code2vec_args
def _build_argparser():
    parser = argparse.ArgumentParser(
        description='Run the non-neural hashtable baseline')
    parser.add_argument('--no-times', '-N', dest='n_times', action='store',
                        type=int, default=10,
                        help='no of times (print std dev & mean')
    parser.add_argument('--modes', '-m', dest='code_mode', action='store',
                        nargs='*', type=str, default=["none"],
                        help='possible modes of hashtable lookup:  '
                             '(code_only_+) [softest, soft, hard, hardest, none]')
    parser.add_argument('--logfile', '-lf', dest='logfile', action='store',
                        type=str, default=None,
                        help='logfile to write to.')
    return parser

if __name__ == "__main__":
    parser = _build_argparser()
    args = parser.parse_args()
    _run_model(**vars(args))

