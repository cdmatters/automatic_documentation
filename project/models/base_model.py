
import abc
from collections import namedtuple
import logging

# from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import numpy as np
import tensorflow as tf

from project.external.nmt import bleu
from project.utils.tokenize import START_OF_TEXT_TOKEN


EXPERIMENT_SUMMARY_STRING = '''
------------------------------------------------------
------------------------------------------------------
DATA: vocab_size: {vocab}, char_seq: {char}, desc_seq: {desc},
       char_embed: {c_embed}, desc_embed: {d_embed},
       full_dataset: {full}, split_dataset: {split}
------------------------------------------------------
{nn}
------------------------------------------------------
------------------------------------------------------
'''

SingleTranslation = namedtuple(
    "Translation", ['name', 'description', 'translation','translation2', 'translation3' ])
SingleTranslation.__str__ = lambda s: "ARGN: {}\nDESC: {}\nINFR1: {}\nINFR2: {}\nINFR3: {}".format(
    s.name, " ".join(s.description), " ".join(s.translation)," ".join(s.translation2)," ".join(s.translation3) )

exp_sum = ['nn', 'vocab', 'char_seq', 'desc_seq', 'char_embed',
           'desc_embed', 'full_dataset', 'split_dataset']
ExperimentSummary = namedtuple("ExperimentSummary", exp_sum)
ExperimentSummary.__str__ = lambda s: EXPERIMENT_SUMMARY_STRING.format(
    vocab=s.vocab, char=s.char_seq, desc=s.desc_seq,
    full=s.full_dataset, nn=s.nn, split=s.split_dataset,
    c_embed=s.char_embed, d_embed=s.desc_embed)


LOGGER = logging.getLogger('')


class BasicRNNModel(abc.ABC):

    summary_string = 'MODEL: {classname}\nName: {name}\n\n{summary}'

    def __init__(self, embed_tuple, name="BasicModel"):
        # To Do; all these args from config, to make saving model easier.
        self.name = name

        self.word_weights = embed_tuple.word_weights
        self.char_weights = embed_tuple.char_weights

        self.word2idx = embed_tuple.word2idx
        self.idx2word = dict((v, k) for k, v in embed_tuple.word2idx.items())
        self.char2idx = embed_tuple.char2idx
        self.idx2char = dict((v, k) for k, v in embed_tuple.char2idx.items())

        self.input_data_sequence = None
        self.input_label_sequence = None
        self.dropout_tensor = None
        self.update = None

        self.train_loss = None
        self.train_id = None

        self.inference_loss = None
        self.inference_id = None

    @abc.abstractmethod
    def _build_train_graph(self):
        '''Build the tensorflow graph'''
        pass

    @abc.abstractmethod
    def main(self, session, epochs, train_data, filewriters, test_data=None, test_check=20, test_translate=0):
        '''Run an experiment'''
        pass

    @abc.abstractmethod
    def arg_summary(self):
        '''Describe the models parameters in a string, for logging'''
        string = "Args: arg1: {}, arg2: {} ".format(1, 2)
        return string

    def __str__(self):
        return self.__class__.summary_string.format(
            name=self.name, classname=self.__class__.__name__, summary=self.arg_summary())

    @staticmethod
    def _log_in_tensorboard(scalar_vars):
        for name, var in scalar_vars:
            tf.summary.scalar(name, var)
        return tf.summary.merge_all()

    @staticmethod
    def _build_encode_decode_embeddings(input_data_sequence, char_weights,
                                        input_label_sequence, word_weights):
        with tf.name_scope("embed_vars"):
            # 1. Embed Our "arg_names" char by char
            char_vocab_size, char_embed_size = char_weights.shape
            char_initializer = tf.constant_initializer(char_weights)
            char_embedding = tf.get_variable("char_embed", [char_vocab_size, char_embed_size],
                                             initializer=char_initializer, trainable=True)
            encode_embedded = tf.nn.embedding_lookup(
                char_embedding, input_data_sequence)

            # 2. Embed Our "arg_desc" word by word
            desc_vocab_size, word_embed_size = word_weights.shape
            word_initializer = tf.constant_initializer(word_weights)
            word_embedding = tf.get_variable("desc_embed", [desc_vocab_size, word_embed_size],
                                             initializer=word_initializer, trainable=False)
            decode_embedded = tf.nn.embedding_lookup(
                word_embedding, input_label_sequence)

            return encode_embedded, decode_embedded, char_embedding, word_embedding

    @staticmethod
    def _build_bi_rnn_encoder(input_data_seq_length, rnn_size, encode_embedded, dropout_keep_prob):
        with tf.name_scope("encoder"):
            batch_size = tf.shape(input_data_seq_length)
            encoder_rnn_cell_fw = tf.contrib.rnn.BasicLSTMCell(
                rnn_size, name="RNNencoder")
            initial_state_fw = encoder_rnn_cell_fw.zero_state(
                batch_size, dtype=tf.float32)

            encoder_rnn_cell_fw = tf.contrib.rnn.DropoutWrapper(encoder_rnn_cell_fw,
                                                                input_keep_prob=dropout_keep_prob,
                                                                output_keep_prob=dropout_keep_prob,
                                                                state_keep_prob=dropout_keep_prob)

            encoder_rnn_cell_bk = tf.contrib.rnn.BasicLSTMCell(
                rnn_size, name="RNNencoder")
            initial_state_bk = encoder_rnn_cell_bk.zero_state(
                batch_size, dtype=tf.float32)

            encoder_rnn_cell_bk = tf.contrib.rnn.DropoutWrapper(encoder_rnn_cell_bk,
                                                                input_keep_prob=dropout_keep_prob,
                                                                output_keep_prob=dropout_keep_prob,
                                                                state_keep_prob=dropout_keep_prob)

            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(encoder_rnn_cell_fw, encoder_rnn_cell_bk,
                                                                     encode_embedded,
                                                                     sequence_length=input_data_seq_length,
                                                                     initial_state_fw=initial_state_fw,
                                                                     initial_state_bk=initial_state_bk,
                                                                     time_major=False)

            return tf.concat(outputs, 2), tf.concat(output_states, 2)

    @staticmethod
    def _build_rnn_encoder(input_data_seq_length, rnn_size, encode_embedded, dropout_keep_prob):
        with tf.name_scope("encoder"):
            batch_size = tf.shape(input_data_seq_length)
            encoder_rnn_cell = tf.contrib.rnn.BasicLSTMCell(
                rnn_size, name="RNNencoder")
            initial_state = encoder_rnn_cell.zero_state(
                batch_size, dtype=tf.float32)

            encoder_rnn_cell = tf.contrib.rnn.DropoutWrapper(encoder_rnn_cell,
                                                             input_keep_prob=dropout_keep_prob,
                                                             output_keep_prob=dropout_keep_prob,
                                                             state_keep_prob=dropout_keep_prob)

            return tf.nn.dynamic_rnn(encoder_rnn_cell, encode_embedded,
                                     sequence_length=input_data_seq_length,
                                     initial_state=initial_state, time_major=False)

    @staticmethod
    def _build_rnn_training_decoder(decoder_rnn_cell, state, projection_layer, decoder_weights,
                                    input_label_seq_length, decode_embedded):
        with tf.name_scope("training"):
            # batch_size = tf.shape(state[0])[0]

            helper = tf.contrib.seq2seq.TrainingHelper(
                decode_embedded, input_label_seq_length, time_major=False)

            # decoder_initial_state = decoder_rnn_cell.zero_state(batch_size, dtype=tf.float32).clone(
            #     cell_state=state)

            decoder = tf.contrib.seq2seq.BasicDecoder(
                decoder_rnn_cell, helper, state, #decoder_initial_state,
                output_layer=projection_layer)

            return tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True)

    @staticmethod
    def _build_rnn_greedy_inference_decoder(decoder_rnn_cell, state, projection_layer, decoder_weights,
                                            start_tok, end_tok):
        with tf.name_scope("inference"):
            batch_size = tf.shape(state[0])[0]

            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_weights,
                                                              tf.fill([batch_size], start_tok), end_tok)

            decoder_initial_state = decoder_rnn_cell.zero_state(batch_size, dtype=tf.float32).clone(
                cell_state=state)

            decoder = tf.contrib.seq2seq.BasicDecoder(
                decoder_rnn_cell, helper, decoder_initial_state,
                output_layer=projection_layer)

            maximum_iterations = 300
            return tf.contrib.seq2seq.dynamic_decode(
                decoder, impute_finished=True, maximum_iterations=maximum_iterations)

    @staticmethod
    def _build_rnn_beam_inference_decoder(decoder_rnn_cell, state, projection_layer, decoder_weights,
                                          start_tok, end_tok, beam_width=10):
        with tf.name_scope("beam_inference"):
            batch_size = tf.shape(state[0])[0]  #* beam_width

            tiled_decoder_initial_state = tf.contrib.seq2seq.tile_batch(
                state, multiplier=beam_width)

            # decoder_initial_state = decoder_rnn_cell.zero_state(batch_size, dtype=tf.float32).clone(
            #     cell_state=tiled_decoder_initial_state)
  

            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=decoder_rnn_cell,
                embedding=decoder_weights,
                start_tokens=tf.fill([batch_size], start_tok),
                end_token=end_tok,
                initial_state=tiled_decoder_initial_state, #decoder_initial_state,
                beam_width=beam_width,
                output_layer=projection_layer,
                length_penalty_weight=0.5)

            maximum_iterations = 300
            return tf.contrib.seq2seq.dynamic_decode(
                decoder, impute_finished=False, maximum_iterations=maximum_iterations)

    @staticmethod
    def _get_loss(logits, input_label_sequence, input_label_seq_length):
        with tf.name_scope("loss"):
            batch_size = tf.shape(input_label_sequence)[0]
            zero_col = tf.zeros([batch_size, 1], dtype=tf.int32)

            # Shift the decoder to be the next word, and then clip it
            decoder_outputs = tf.concat(
                [input_label_sequence[:, 1:], zero_col], 1)  # TODO transform this
            maximum_length = tf.reduce_max(input_label_seq_length)
            decoder_outputs = decoder_outputs[:, :maximum_length]

            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=decoder_outputs, logits=logits)

            target_weights = tf.logical_not(
                tf.equal(decoder_outputs, tf.zeros_like(decoder_outputs)))
            target_weights = tf.cast(target_weights, tf.float32)
            train_loss = (tf.reduce_sum(crossent * target_weights) /
                          tf.cast(batch_size, tf.float32))
        return train_loss

    @staticmethod
    def _do_updates(train_loss, learning_rate):
        with tf.name_scope("opt"):
            # Clip the gradients
            max_gradient_norm = 1
            params = tf.trainable_variables()
            gradients = tf.gradients(train_loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(
                gradients, max_gradient_norm)

            # Create Optimiser and Apply Update
            optimizer = tf.train.AdamOptimizer(learning_rate)
            update = optimizer.apply_gradients(zip(clipped_gradients, params))
        return update

    def translate(self, translate_id, filter_pad=True, lookup=None, do_join=True, prepend_tok=None):
        if lookup is None:
            lookup = self.idx2word
        if filter_pad:
            translate_id = np.trim_zeros(translate_id, 'b')
        if prepend_tok is not None:
            translate_id = np.insert(translate_id, 0, prepend_tok)

        if do_join:
            return " ".join([lookup[i] for i in translate_id])
        else:
            return [lookup[i] for i in translate_id]

    def _feed_fwd(self, session, input_data, input_labels, operation, mode=None):
        """
        Evaluates a node in the graph
        Args
            session: session that is being run
            input_data, array: batch of comments
            input_labels, array: batch of labels
            operation: node in graph to be evaluated
        Returns
            output of the operation
        """
        run_ouputs = operation
        feed_dict = {self.input_data_sequence: input_data,
                     self.input_label_sequence: input_labels}
        if mode == 'TRAIN':
            feed_dict[self.dropout_keep_prob] = 1 - self.dropout

        return session.run(run_ouputs, feed_dict=feed_dict)

    def _to_batch(self, arg_name, arg_desc, epochs=1, do_prog_bar=False):
        assert arg_name.shape[0] == arg_desc.shape[0]
        size = arg_name.shape[0]

        batch_per_epoch = (size // self.batch_size) + 1

        for e in range(epochs):
            zipped = list(zip(arg_name, arg_desc))
            np.random.shuffle(zipped)
            arg_name, arg_desc = zip(*zipped)

            for i in range(batch_per_epoch):
                idx_start = i * self.batch_size
                idx_end = (i + 1) * self.batch_size

                arg_name_batch = arg_name[idx_start: idx_end]
                arg_desc_batch = arg_desc[idx_start: idx_end]
                yield arg_name_batch, arg_desc_batch

    def evaluate_bleu(self, session, data, max_points=10000, max_translations=200):
        all_names = []
        all_references = []
        all_translations = []
        all_training_loss = []
        all_backups1 = []
        all_backups2 = []

        ops = [self.merged_metrics, self.train_loss, self.inference_id]
        for arg_name, arg_desc in self._to_batch(data[0][:max_points], data[1][:max_points]):

            metrics, train_loss, inference_ids = self._feed_fwd(
                session, arg_name, arg_desc, ops)

            # Translating quirks:
            #    names: RETURN: 'axis<END>' NOT 'a x i s <END>'
            #    references: RETURN: [['<START>', 'this', 'reference', '<END>']]
            #                NOT: ['<START>', 'this', 'reference','<END>'],
            #                     because compute_bleu takes multiple references
            #    translations: RETURN: ['<START>', 'this', 'translation', '<END>']
            #                  NOT: ['this', 'translation', '<END>']
            names = [self.translate(i, lookup=self.idx2char).replace(
                " ", "") for i in arg_name]
            references = [[self.translate(i, do_join=False)] for i in arg_desc]
            translations = [self.translate(
                i, do_join=False, prepend_tok=self.word2idx[START_OF_TEXT_TOKEN]) for i in inference_ids[:,:,0]]

            backups1 = [self.translate(
                i, do_join=False, prepend_tok=self.word2idx[START_OF_TEXT_TOKEN]) for i in inference_ids[:,:,1]]
            
            backups2 = [self.translate(
                i, do_join=False, prepend_tok=self.word2idx[START_OF_TEXT_TOKEN]) for i in inference_ids[:,:,2]]

            all_training_loss.append(train_loss)
            all_names.extend(names)
            all_references.extend(references)
            all_translations.extend(translations)
            all_backups1.extend(backups1)
            all_backups2.extend(backups2)

        # BLEU TUPLE = (bleu_score, precisions, bp, ratio, translation_length, reference_length)
        # To Do: Replace with NLTK:
        #         smoother = SmoothingFunction()
        #         bleu_score = corpus_bleu(all_references, all_translations,
        #                                  smoothing_function=smoother.method2)
        bleu_tuple = bleu.compute_bleu(
            all_references, all_translations, max_order=4, smooth=False)
        av_loss = np.mean(all_training_loss)

        translations = [SingleTranslation(n, d[0], t, b1, b2) for n, d, t, b1, b2 in zip(
            all_names, all_references, all_translations, all_backups1, all_backups2 )]

        return bleu_tuple, av_loss, translations[:max_translations]


def argparse_basic_wrap(parser):
    parser.add_argument('--epochs', '-e', dest='epochs', action='store',
                        type=int, default=5000,
                        help='minibatch size for model')
    parser.add_argument('--vocab-size', '-v', dest='vocab_size', action='store',
                        type=int, default=50000,
                        help='size of embedding vocab')
    parser.add_argument('--char-seq', '-c', dest='char_seq', action='store',
                        type=int, default=24,
                        help='max char sequence length')
    parser.add_argument('--desc-seq', '-d', dest='desc_seq', action='store',
                        type=int, default=120,
                        help='max desecription sequence length')
    parser.add_argument('--test-freq', '-t', dest='test_freq', action='store',
                        type=int, default=100,
                        help='how often to run a test and dump output')
    parser.add_argument('--dump-translation', '-D', dest='test_translate', action='store',
                        type=int, default=5,
                        help='dump extensive test information on each test batch')
    parser.add_argument('--use-full-dataset', '-F', dest='use_full_dataset', action='store_true',
                        default=False,
                        help='dump extensive test information on each test batch')
    parser.add_argument('--logdir', '-L', dest='logdir', action='store',
                        type=str, default='logdir',
                        help='directory for storing logs and raw experiment runs')
    return parser


if __name__ == "__main__":
    print("ABSTRACT BASE CLASS")
