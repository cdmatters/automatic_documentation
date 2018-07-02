import argparse
from collections import defaultdict
import random

# from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction 
import nltk

from project.external.nmt import bleu
from project.models.base_model import ExperimentSummary, SingleTranslation
import project.utils.args as args

default_dict_factory = lambda : defaultdict(list) 

class HashtableBaseline(object):
    def __init__(self, name="Hashtable Model"):
        self.name  = name
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

        if not descriptions: # not even 1-gram!
            all_d_indices = range(len(self.descriptions))
            descriptions.append(random.choice(all_d_indices))
        return descriptions

    def tok(self, word):
        desc = word.replace('\\n', " ").lower()
        return nltk.word_tokenize(desc)

    def test(self, test_data):
        translations = []
        for d in test_data:
            indices = self.lookup_description_indices(d["arg_name"])

            descriptions = [self.descriptions[i] for i in indices]
            translation = random.choice(descriptions)
            
            translations.append(SingleTranslation(d["arg_name"], self.tok(d["arg_desc"]), self.tok(translation)))
        return translations

    def train(self, train_data):
        for i, d in enumerate(train_data):
            name = d["arg_name"]
            l = len(d["arg_name"])
            self.descriptions.append(d["arg_desc"])
            
            for j in range(l):
                ngrams = self.get_n_grams(j + 1, name)
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



def _run_model( vocab_size, char_seq, desc_seq, use_full_dataset, use_split_dataset):
    if use_full_dataset:
        if use_split_dataset:
            from project.data.preprocessed import main_data_split as DATA
        else:
            from project.data.preprocessed import main_data as DATA
    else:
        from project.data.preprocessed.overfit import overfit_data as DATA

 
    model = HashtableBaseline()
    
    summary = ExperimentSummary(model, vocab_size, char_seq, desc_seq, use_full_dataset)
    print(summary)

    model.main(DATA.train, DATA.test) 

    # filewriters = {
    #     'train_continuous':  tf.summary.FileWriter('logs/{}/train_continuous'.format(log_str), sess.graph),
    #     'train': tf.summary.FileWriter('logs/{}/train'.format(log_str), sess.graph),
    #     'test': tf.summary.FileWriter('logs/{}/test'.format(log_str))
    # }

    # plogging.load(sess, "logdir_0618_204400", "BasicModel.ckpt-1" )
    # nn.main(sess, epochs, data_tuple.train, log_str, filewriters, data_tuple.test, 
    #         test_check=test_freq, test_translate=test_translate)


@args.data_args
def _build_argparser():
    parser = argparse.ArgumentParser(description='Run the non-neural hashtable baseline')
    return parser




if __name__ == "__main__":
    parser = _build_argparser()
    args = parser.parse_args()

    _run_model(**vars(args))