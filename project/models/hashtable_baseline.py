import argparse
from collections import defaultdict
import random

random.seed(100)
# from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import nltk

from project.external.nmt import bleu
from project.models.base_model import ExperimentSummary, SingleTranslation
from project.utils import args, tokenize


def default_dict_factory(): return defaultdict(list)


class HashtableBaseline(object):
    def __init__(self, name="Hashtable Model"):
        self.name = name
        self.lookup_list = defaultdict(default_dict_factory)
        self.descriptions = []

    def __str__(self):
        summary_string = 'MODEL: {classname}\nName: {name}\n\n{summary}'
        return summary_string.format(
            name=self.name, classname=self.__class__.__name__, summary="Args: no args")

    def get_n_grams(self, n, name):
        return [name[i:i + n] for i in range(len(name) + 1 - n)]

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
        for d in test_data:
            hash_string = tokenize.get_hash_string(d)
            indices = self.lookup_description_indices(hash_string)

            descriptions = [self.descriptions[i] for i in indices]
            translation = random.choice(descriptions)

            translations.append(SingleTranslation(
                hash_string, d["arg_desc_tokens"], translation))
        return translations

    def train(self, train_data):
        for i, d in enumerate(train_data):
            hash_string = tokenize.get_hash_string(d)
            l = len(hash_string)
            self.descriptions.append(d["arg_desc_tokens"])

            for j in range(l):
                ngrams = self.get_n_grams(j + 1, hash_string)
                for n in ngrams:
                    self.lookup_list[j][n].append(i)

    def evaluate(self, all_translations):
        references = [[t.description] for t in all_translations]
        translations = [t.translation for t in all_translations]

        return bleu.compute_bleu(references, translations, max_order=4, smooth=False)

    def main(self, train_data, test_data):
        self.train(train_data)
        translations = self.test(test_data)
        bleu_tuple = self.evaluate(translations)
        print(bleu_tuple[0]*100)


def _run_model(vocab_size, char_seq, desc_seq, use_full_dataset, use_split_dataset, tokenizer, 
               no_dups, desc_embed, char_embed, **kwargs):
    data_tuple = tokenize.get_data_tuple(use_full_dataset, use_split_dataset, no_dups)
    print("Loading GloVe weights and word to index lookup table")

    _, word2idx = tokenize.get_weights_word2idx(desc_embed, vocab_size, data_tuple.train)
    _ = defaultdict(int)

    this_tokenizer = tokenize.choose_tokenizer(tokenizer)
    train_data = this_tokenizer(data_tuple.train, word2idx, _)
    valid_data = this_tokenizer(data_tuple.valid, word2idx, _)

    model = HashtableBaseline()
    summary = ExperimentSummary(
        model, vocab_size, char_seq, desc_seq, None, None, use_full_dataset, use_split_dataset)
    print(summary)

    model.main(train_data, valid_data)


@args.data_args
def _build_argparser():
    parser = argparse.ArgumentParser(
        description='Run the non-neural hashtable baseline')
    return parser


if __name__ == "__main__":
    parser = _build_argparser()
    args = parser.parse_args()

    _run_model(**vars(args))
