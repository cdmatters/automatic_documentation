# -*- coding: utf-8 -*-
import argparse
import logging
from collections import namedtuple

import numpy as np
import tensorflow as tf
from tensorflow.python.layers import core as layers_core

from project.models.base_model import BasicRNNModel, _run_model
from project.models.code2vec_encoder import Code2VecEncoder
import project.utils.args as args
from project.utils.tokenize import START_OF_TEXT_TOKEN, END_OF_TEXT_TOKEN


LOGGER = logging.getLogger('')

SingleTranslationWithPaths = namedtuple(
    "Translation", ['name', 'description', 'tokenized', 'translation', 'code'])
SingleTranslationWithPaths.__str__ = lambda s: "ARGN: {}\nCODE: {}\nDESC: {}\nTOKN: {}\nINFR: {}\n".format(
    s.name, s.code," ".join(s.description)," ".join(s.tokenized), " ".join(s.translation))


class Code2VecSolo(Code2VecEncoder):

    def __init__(self, embed_tuple, batch_size, learning_rate,
                dropout, vec_size, path_vocab, path_embed, path_seq, model_name="BasicModel", **_):
        BasicRNNModel.__init__(self, embed_tuple, model_name)
        # To Do; all these args from config, to make saving model easier.

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.code2vec_size = vec_size
        self.dropout = dropout

        # Graph Variables (built later)
        self.input_data_sequence = None
        self.input_label_sequence = None
        self.update = None

        self.train_loss = None
        self.train_id = None

        self.inference_loss = None
        self.inference_id = None

        self.path_seq = path_seq
        self.path_embed = path_embed
        self.path_vocab = path_vocab + 50 # ARGS
        self.path_weights = np.random.uniform(
             low=-0.1, high=0.1, size=[self.path_vocab, self.path_embed])
        self.input_target_var_weights = np.random.uniform(
             low=-0.1, high=0.1, size=[self.path_vocab, self.path_embed])

        self._build_train_graph()
        self.merged_metrics = self._log_in_tensorboard()

        LOGGER.debug("Init loaded")

    def arg_summary(self):
        mod_args = "ModArgs: code2vec_size: {}, path_embed: {}, lr: {}, batch_size: {}, ".format(
            self.code2vec_size, self.path_embed, self.learning_rate, self.batch_size)

        data_args = "DataArgs: path_seq {}, vocab_size: {}, path_vocab: {}, word_embed: {}, dropout: {} ".format(
            self.path_seq, len(self.word2idx), self.path_vocab, self.word_weights.shape[1], self.dropout)
        return "\n".join([mod_args, data_args])

    def _log_in_tensorboard(self):
        tf.summary.scalar('loss', self.train_loss)
        return tf.summary.merge_all()

    def build_translations(self, all_names, all_references, all_references_tok, all_translations, all_data):
        get_path_stats = lambda x: "Paths: {}/{},  Of Which <UNK> {}".format(np.count_nonzero(x), len(x), (x==1).sum())
        return [SingleTranslationWithPaths(n, r[0], t[0], tr, get_path_stats(s)) for n, r, t, tr, s in zip(
            all_names, all_references, all_references_tok, all_translations, all_data[2])]

    def _build_train_graph(self):
        with tf.name_scope("Model_{}".format(self.name)):
            # 0. Define our placeholders and derived vars
            # # input_data_sequence : [batch_size x max_variable_length]
            input_data_sequence = tf.placeholder(tf.int32, [None, None], "arg_name")
            input_data_seq_length = tf.argmin(
                input_data_sequence, axis=1, output_type=tf.int32) + 1

            # # input_label_sequence  : [batch_size x max_docstring_length]
            input_label_sequence = tf.placeholder(tf.int32, [None, None], "arg_desc")
            input_label_seq_length = tf.argmin(
                input_label_sequence, axis=1, output_type=tf.int32) + 1
            dropout_keep_prob = tf.placeholder_with_default(1.0, shape=())

            # CODE 2 VEC
            # # input_codepaths : [batch_size x max_codepaths]
            # # input_target_vars : [batch_size x max_codepaths]
            input_codepaths = tf.placeholder(tf.int32, [None, None], "paths")
            input_target_vars = tf.placeholder(tf.int32, [None, None], "paths")
            input_codepaths_seq_length = tf.argmin(
                input_codepaths, axis=1, output_type=tf.int32)

            # 1. Get Embeddings
            _, decode_embedded, _, decoder_weights = self._build_encode_decode_embeddings(
                input_data_sequence, self.char_weights,
                input_label_sequence, self.word_weights)

            # 1.1 Get the Embeddings for out Code2Vec embedder
            encode_path_embedded, encode_tv_embedded, _, _ = self._build_code2vec_encoder(
                input_codepaths, self.path_weights,
                input_target_vars, self.input_target_var_weights)

            # 1.2 Build the Code2Vec Vector
            code2vec_embedding = self._build_code2vec_vector(
                encode_path_embedded,
                encode_tv_embedded,
                self.path_embed,
                self.code2vec_size,
                dropout_keep_prob,
                input_codepaths_seq_length
                )

            c = code2vec_embedding
            h = code2vec_embedding
            state = tf.contrib.rnn.LSTMStateTuple(c, h)

            # 3. Build out Cell ith attention
            decoder_rnn_size = self.code2vec_size

            decoder_rnn_cell = tf.contrib.rnn.BasicLSTMCell(
                    decoder_rnn_size, name="RNNdecoder")

            desc_vocab_size, _ = self.word_weights.shape
            projection_layer = layers_core.Dense(
                desc_vocab_size, use_bias=False)

            decoder_rnn_cell = tf.contrib.rnn.DropoutWrapper(
                decoder_rnn_cell,
                input_keep_prob=dropout_keep_prob,
                output_keep_prob=dropout_keep_prob,
                state_keep_prob=dropout_keep_prob)



            # 4. Build out helpers
            train_outputs, _, _ = self._build_rnn_training_decoder(decoder_rnn_cell,
                                                                   state, projection_layer, decoder_weights,
                                                                   input_label_seq_length,
                                                                   decode_embedded,
                                                                   use_attention=False)

            inf_outputs, _, _ = self._build_rnn_greedy_inference_decoder(decoder_rnn_cell,
                                                                         state, projection_layer, decoder_weights,
                                                                         self.word2idx[START_OF_TEXT_TOKEN],
                                                                         self.word2idx[END_OF_TEXT_TOKEN],
                                                                         use_attention=False)

            # 5. Define Train Loss
            train_logits = train_outputs.rnn_output
            train_loss = self._get_loss(
                train_logits, input_label_sequence, input_label_seq_length)
            train_translate = train_outputs.sample_id

            # 6. Define Translation
            inf_logits = inf_outputs.rnn_output
            inf_translate = inf_outputs.sample_id
            inf_loss = self._get_loss(
                inf_logits, input_label_sequence, input_label_seq_length)

            # 7. Do Updates
            update = self._do_updates(train_loss, self.learning_rate)

            # 8. Save Variables to Model
            self.input_data_sequence = input_data_sequence
            self.input_codepaths = input_codepaths
            self.input_target_vars = input_target_vars
            self.input_label_sequence = input_label_sequence
            self.dropout_keep_prob = dropout_keep_prob
            self.update = update
            self.train_loss = train_loss
            self.train_id = train_translate

            self.inference_loss = inf_loss
            self.inference_id = inf_translate


    def _feed_fwd(self, session, minibatch, operation, mode=None):
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
        input_data, input_labels, input_paths, input_target_vars  = minibatch[0], minibatch[1], minibatch[2], minibatch[3]
        run_ouputs = operation
        feed_dict = {self.input_data_sequence: input_data,
                     self.input_label_sequence: input_labels,
                     self.input_codepaths: input_paths,
                     self.input_target_vars: input_target_vars,
                      }
        if mode == 'TRAIN':
            feed_dict[self.dropout_keep_prob] = 1 - self.dropout

        return session.run(run_ouputs, feed_dict=feed_dict)

def run_model(**kwargs):
    _run_model(Code2VecSolo, **kwargs)

@args.code2vec_args
@args.encoder_args
@args.log_args
@args.train_args
@args.data_args
def _build_argparser():
    parser = argparse.ArgumentParser(
        description='Run the code2vec model')
    return parser

if __name__ == "__main__":
    parser = _build_argparser()
    args = parser.parse_args()

    run_model(**vars(args))
