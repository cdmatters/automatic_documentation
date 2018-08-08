# -*- coding: utf-8 -*-
import argparse
import logging
from collections import namedtuple

import numpy as np
import tensorflow as tf
from tensorflow.python.layers import core as layers_core

from project.models.base_model import BasicRNNModel, _run_model
import project.utils.args as args
from project.utils.tokenize import START_OF_TEXT_TOKEN, END_OF_TEXT_TOKEN


LOGGER = logging.getLogger('')

SingleTranslationWithCode = namedtuple(
    "Translation", ['name', 'description', 'tokenized', 'translation', 'code'])
SingleTranslationWithCode.__str__ = lambda s: "ARGN: {}\nCODE: {}\nDESC: {}\nTOKN: {}\nINFR: {}\n".format(
    s.name, " ".join(s.code)," ".join(s.description)," ".join(s.tokenized), " ".join(s.translation))


class DoubleEncoderBaseline(BasicRNNModel):

    def __init__(self, embed_tuple, rnn_size=300, batch_size=128, learning_rate=0.001,
                dropout=0.3, bidirectional=False, model_name="BasicModel", **_):
        super().__init__(embed_tuple, model_name)
        # To Do; all these args from config, to make saving model easier.

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.rnn_size = rnn_size
        self.second_rnn_size = 128
        self.dropout = dropout
        self.bidirectional = bidirectional

        # Graph Variables (built later)
        self.input_data_sequence = None
        self.input_label_sequence = None
        self.update = None

        self.train_loss = None
        self.train_id = None

        self.inference_loss = None
        self.inference_id = None

        self._build_train_graph()
        self.merged_metrics = self._log_in_tensorboard()

        self.idx2voc = Non
        self.code_weights = np.random.uniform(
                low=-0.1, high=0.1, size=[40003, 200])

        LOGGER.debug("Init loaded")

    def translate_code(self, code, do_join=False):
        if self.idx2voc is None:
            from project.utils.tokenize import SRC_VOCAB
            self.idx2voc = {i:k for i,k in enumerate(SRC_VOCAB)}
        print(len(self.idx2voc))
        return self.translate(code, lookup=self.idx2voc)

    def build_translations(self, all_names, all_references, all_references_tok, all_translations, all_data):
        return [SingleTranslationWithCode(n, r[0], t[0], tr, self.translate_code(s, do_join=False)) for n, r, t, tr, s in zip(
            all_names, all_references, all_references_tok, all_translations, all_data[2])]

    def arg_summary(self):
        mod_args = "ModArgs: rnn_size: {}, lr: {}, batch_size: {}, ".format(
            self.rnn_size, self.learning_rate, self.batch_size)

        data_args = "DataArgs: vocab_size: {}, char_embed: {}, word_embed: {}, dropout: {} ".format(
            len(self.word2idx), self.char_weights.shape[1], self.word_weights.shape[1], self.dropout)
        return "\n".join([mod_args, data_args])

    @staticmethod
    def _concat_vectors(first_state, second_state, concat_size, rnn_size):
        with tf.variable_scope("double_enc_concat", reuse=tf.AUTO_REUSE):
            c = tf.concat([first_state.c, second_state.c], axis = 1)
            h = tf.concat([first_state.h, second_state.h], axis = 1)


            W = tf.get_variable("ConcatMLP_W",
                [concat_size, rnn_size],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())

            B = tf.get_variable("ConcatMLP_B",
                [rnn_size],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())

            Zc = tf.add(tf.matmul(c, W), B)
            Zh = tf.add(tf.matmul(h, W), B)
            # Ac = tf.nn.tanh(Zc)
            # Ah = tf.nn.tanh(Zh)

            return  tf.contrib.rnn.LSTMStateTuple(Zc, Zh), (W, B)

    def _log_in_tensorboard(self):
        tf.summary.scalar('loss', self.train_loss)
        return tf.summary.merge_all()

    def _build_train_graph(self):
        with tf.name_scope("Model_{}".format(self.name)):
            # 0. Define our placeholders and derived vars
            # # input_data_sequence : [batch_size x max_variable_length]
            input_data_sequence = tf.placeholder(tf.int32, [None, None], "arg_name")
            input_data_seq_length = tf.argmin(
                input_data_sequence, axis=1, output_type=tf.int32) + 1
            # # second_data_sequence : [batch_size x max_variable_length]
            second_data_sequence = tf.placeholder(tf.int32, [None, None], "code_seq")
            second_data_seq_length = tf.argmin(
                second_data_sequence, axis=1, output_type=tf.int32) + 1
            # # input_label_sequence  : [batch_size x max_docstring_length]
            input_label_sequence = tf.placeholder(tf.int32, [None, None], "arg_desc")
            input_label_seq_length = tf.argmin(
                input_label_sequence, axis=1, output_type=tf.int32) + 1
            dropout_keep_prob = tf.placeholder_with_default(1.0, shape=())


            # 1. Get Embeddings
            encode_embedded, decode_embedded, _, decoder_weights = self._build_encode_decode_embeddings(
                input_data_sequence, self.char_weights,
                input_label_sequence, self.word_weights)

            # 1.5 Get Second Embeddings
            second_encode_embedded, _build_encode_random_embeddings(second_data_seq_length, self.code_weights)

            # 2. Build out Encoder
            if self.bidirectional:
                first_encoder_outputs, first_state = self._build_bi_rnn_encoder(
                    input_data_seq_length, self.rnn_size, encode_embedded, dropout_keep_prob, name="FirstRNN")

                second_encoder_outputs, second_state = self._build_bi_rnn_encoder(
                    second_data_seq_length, self.second_rnn_size, second_encode_embedded, dropout_keep_prob, name="SecondRNN")

            else:
                first_encoder_outputs, first_state = self._build_rnn_encoder(
                    input_data_seq_length, self.rnn_size, encode_embedded, dropout_keep_prob,  name="FirstRNN")
                second_encoder_outputs, second_state = self._build_rnn_encoder(
                    second_data_seq_length, self.second_rnn_size, second_encode_embedded, dropout_keep_prob, name="SecondRNN")

            # 3. Concatenate the Vectors and Resize With Linear Layer
            concat_size = self.rnn_size + self.second_rnn_size
            if self.bidirectional:
                concat_size = concat_size * 2
            state, _ = self._concat_vectors(first_state, second_state, concat_size, self.rnn_size)

            # 4. Build out Cell ith attention
            decoder_rnn_size = self.rnn_size
            decoder_rnn_cell = tf.contrib.rnn.BasicLSTMCell(
                    decoder_rnn_size, name="RNNdecoder")

            desc_vocab_size, _ = self.word_weights.shape
            projection_layer = layers_core.Dense(
                desc_vocab_size, use_bias=False)


            attention_mechanism1 = tf.contrib.seq2seq.LuongAttention(
                decoder_rnn_size, first_encoder_outputs,
                memory_sequence_length=input_data_seq_length,
                name="LuongAttention1")

            decoder_rnn_cell = tf.contrib.rnn.DropoutWrapper(
                decoder_rnn_cell,
                input_keep_prob=dropout_keep_prob,
                output_keep_prob=dropout_keep_prob,
                state_keep_prob=dropout_keep_prob)

            decoder_rnn_cell = tf.contrib.seq2seq.AttentionWrapper(
                decoder_rnn_cell, attention_mechanism1,
                attention_layer_size=self.rnn_size)


            # 5. Build out helpers
            train_outputs, _, _ = self._build_rnn_training_decoder(decoder_rnn_cell,
                                                                   state, projection_layer, decoder_weights,
                                                                   input_label_seq_length,
                                                                   decode_embedded)

            inf_outputs, _, _ = self._build_rnn_greedy_inference_decoder(decoder_rnn_cell,
                                                                         state, projection_layer, decoder_weights,
                                                                         self.word2idx[START_OF_TEXT_TOKEN],
                                                                         self.word2idx[END_OF_TEXT_TOKEN])

            # 6. Define Train Loss
            train_logits = train_outputs.rnn_output
            train_loss = self._get_loss(
                train_logits, input_label_sequence, input_label_seq_length)
            train_translate = train_outputs.sample_id

            # 7. Define Translation
            inf_logits = inf_outputs.rnn_output
            inf_translate = inf_outputs.sample_id
            inf_loss = self._get_loss(
                inf_logits, input_label_sequence, input_label_seq_length)

            # 8. Do Updates
            update = self._do_updates(train_loss, self.learning_rate)

            # 9. Save Variables to Model
            self.input_data_sequence = input_data_sequence
            self.second_data_sequence = second_data_sequence
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
        input_data, input_labels, input_code = minibatch[0], minibatch[1], minibatch[2]
        run_ouputs = operation
        feed_dict = {self.input_data_sequence: input_data,
                     self.input_label_sequence: input_labels,
                     self.second_data_sequence: input_code }
        if mode == 'TRAIN':
            feed_dict[self.dropout_keep_prob] = 1 - self.dropout

        return session.run(run_ouputs, feed_dict=feed_dict)

def run_model(**kwargs):
    kwargs['code_tokenizer'] = 'full'
    _run_model(DoubleEncoderBaseline, **kwargs)

@args.encoder_args
@args.log_args
@args.train_args
@args.data_args
def _build_argparser():
    parser = argparse.ArgumentParser(
        description='Run the basic LSTM model on the overfit dataset')
    return parser

if __name__ == "__main__":
    parser = _build_argparser()
    args = parser.parse_args()

    run_model(**vars(args))
