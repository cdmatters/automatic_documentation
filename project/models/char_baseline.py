# -*- coding: utf-8 -*-
import argparse
import logging

import tensorflow as tf
from tensorflow.python.layers import core as layers_core

from project.models.base_model import BasicRNNModel, SingleTranslation, _run_model
import project.utils.args as args
from project.utils.tokenize import START_OF_TEXT_TOKEN, END_OF_TEXT_TOKEN


LOGGER = logging.getLogger('')



class CharSeqBaseline(BasicRNNModel):

    def __init__(self, embed_tuple, lstm_size, batch_size, learning_rate, 
                dropout, bidirectional, model_name="BasicModel", **_):
        super().__init__(embed_tuple, model_name)
        # To Do; all these args from config, to make saving model easier.

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.rnn_size = lstm_size
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

        LOGGER.debug("Init loaded")

    def build_translations(self, all_names, all_references, all_tokenized, all_translations, all_data):
        return [SingleTranslation(n, r[0], t[0], tr) for n, r, t, tr in zip(
            all_names, all_references, all_tokenized, all_translations)]


    def arg_summary(self):
        mod_args = "ModArgs: rnn_size: {}, lr: {}, batch_size: {}, ".format(
            self.rnn_size, self.learning_rate, self.batch_size)

        data_args = "DataArgs: vocab_size: {}, char_embed: {}, word_embed: {}, dropout: {} ".format(
            len(self.word2idx), self.char_weights.shape[1], self.word_weights.shape[1], self.dropout)
        return "\n".join([mod_args, data_args])

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
            # # input_label_sequence  : [batch_size x max_docstring_length]
            input_label_sequence = tf.placeholder(tf.int32, [None, None], "arg_desc")
            input_label_seq_length = tf.argmin(
                input_label_sequence, axis=1, output_type=tf.int32) + 1
            dropout_keep_prob = tf.placeholder_with_default(1.0, shape=())

            # 1. Get Embeddings
            encode_embedded, decode_embedded, _, decoder_weights = self._build_encode_decode_embeddings(
                input_data_sequence, self.char_weights,
                input_label_sequence, self.word_weights)

            # 2. Build out Encoder
            if self.bidirectional:
                encoder_outputs, state = self._build_bi_rnn_encoder(
                    input_data_seq_length, self.rnn_size, encode_embedded, dropout_keep_prob)

            else:
                encoder_outputs, state = self._build_rnn_encoder(
                    input_data_seq_length, self.rnn_size, encode_embedded, dropout_keep_prob)

            # 3. Build out Cell ith attention
            decode_rnn_size = self.rnn_size
            if self.bidirectional:
                decode_rnn_size = self.rnn_size * 2
                decoder_rnn_cell = tf.contrib.rnn.BasicLSTMCell(
                    decode_rnn_size, name="RNNdecoder")
            else:
                decoder_rnn_cell = tf.contrib.rnn.BasicLSTMCell(
                    decode_rnn_size, name="RNNdecoder")

            desc_vocab_size, _ = self.word_weights.shape
            projection_layer = layers_core.Dense(
                desc_vocab_size, use_bias=False)

            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                decode_rnn_size, encoder_outputs,
                memory_sequence_length=input_data_seq_length)

            decoder_rnn_cell = tf.contrib.rnn.DropoutWrapper(
                decoder_rnn_cell,
                input_keep_prob=dropout_keep_prob,
                output_keep_prob=dropout_keep_prob,
                state_keep_prob=dropout_keep_prob)

            decoder_rnn_cell = tf.contrib.seq2seq.AttentionWrapper(
                decoder_rnn_cell, attention_mechanism,
                attention_layer_size=decode_rnn_size)

            # 4. Build out helpers
            train_outputs, _, _ = self._build_rnn_training_decoder(decoder_rnn_cell,
                                                                   state, projection_layer, decoder_weights, 
                                                                   input_label_seq_length,
                                                                   decode_embedded)

            inf_outputs, _, _ = self._build_rnn_greedy_inference_decoder(decoder_rnn_cell,
                                                                         state, projection_layer, decoder_weights,
                                                                         self.word2idx[START_OF_TEXT_TOKEN],
                                                                         self.word2idx[END_OF_TEXT_TOKEN])

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
            self.input_label_sequence = input_label_sequence
            self.dropout_keep_prob = dropout_keep_prob
            self.update = update
            self.train_loss = train_loss
            self.train_id = train_translate

            self.inference_loss = inf_loss
            self.inference_id = inf_translate

def run_model(**kwargs):
    return _run_model(CharSeqBaseline, **kwargs)

@args.log_args
@args.train_args
@args.data_args
@args.encoder_args
def _build_argparser():
    parser = argparse.ArgumentParser(
        description='Run the basic LSTM model on the overfit dataset')
    return parser

if __name__ == "__main__":
    parser = _build_argparser()
    args = parser.parse_args()

    run_model(**vars(args))
