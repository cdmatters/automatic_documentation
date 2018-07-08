# -*- coding: utf-8 -*-
import argparse
from datetime import datetime
import logging
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.layers import core as layers_core

from project.models.base_model import BasicRNNModel, ExperimentSummary, \
    argparse_basic_wrap
import project.utils.args as args
import project.utils.logging as log_util
import project.utils.saveload as saveload
import project.utils.tokenize as tokenize
from project.utils.tokenize import PAD_TOKEN, UNKNOWN_TOKEN, \
    START_OF_TEXT_TOKEN, END_OF_TEXT_TOKEN


LOGGER = logging.getLogger('')


class CharSeqBaseline(BasicRNNModel):

    def __init__(self, embed_tuple, rnn_size=300, batch_size=128, learning_rate=0.001, 
                dropout=0.3, bidirectional=False, name="BasicModel"):
        super().__init__(embed_tuple, name)
        # To Do; all these args from config, to make saving model easier.

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.rnn_size = rnn_size
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
            if self.bidirectional:
                decoder_rnn_cell = tf.contrib.rnn.BasicLSTMCell(
                    self.rnn_size * 2, name="RNNencoder")
            else:
                decoder_rnn_cell = tf.contrib.rnn.BasicLSTMCell(
                    self.rnn_size, name="RNNencoder")

            desc_vocab_size, _ = self.word_weights.shape
            projection_layer = layers_core.Dense(
                desc_vocab_size, use_bias=False)

            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                self.rnn_size, encoder_outputs,
                memory_sequence_length=input_data_seq_length)

            decoder_rnn_cell = tf.contrib.rnn.DropoutWrapper(
                decoder_rnn_cell,
                input_keep_prob=dropout_keep_prob,
                output_keep_prob=dropout_keep_prob,
                state_keep_prob=dropout_keep_prob)

            # decoder_rnn_cell = tf.contrib.seq2seq.AttentionWrapper(
            #     decoder_rnn_cell, attention_mechanism,
            #     attention_layer_size=self.rnn_size)

            # 4. Build out helpers
            train_outputs, _, _ = self._build_rnn_training_decoder(decoder_rnn_cell,
                                                                   state, projection_layer, decoder_weights, 
                                                                   input_label_seq_length,
                                                                   decode_embedded)

            inf_outputs, _, _ = self._build_rnn_beam_inference_decoder(decoder_rnn_cell,
                                                                         state, projection_layer, decoder_weights,
                                                                         self.word2idx[START_OF_TEXT_TOKEN],
                                                                         self.word2idx[END_OF_TEXT_TOKEN])            
            # inf_outputs, _, _ = self._build_rnn_greedy_inference_decoder(decoder_rnn_cell,
            #                                                              state, projection_layer, decoder_weights,
            #                                                              self.word2idx[START_OF_TEXT_TOKEN],
            #                                                              self.word2idx[END_OF_TEXT_TOKEN])

            # 5. Define Train Loss
            train_logits = train_outputs.rnn_output
            train_loss = self._get_loss(
                train_logits, input_label_sequence, input_label_seq_length)
            train_translate = train_outputs.sample_id

            # 6. Define Translation
            # inf_logits = inf_outputs.rnn_output
            # inf_loss = self._get_loss(
                # inf_logits, input_label_sequence, input_label_seq_length)
            # inf_translate = inf_outputs.sample_id
            inf_translate = inf_outputs.predicted_ids

            # 7. Do Updates
            update = self._do_updates(train_loss, self.learning_rate)

            # 8. Save Variables to Model
            self.input_data_sequence = input_data_sequence
            self.input_label_sequence = input_label_sequence
            self.dropout_keep_prob = dropout_keep_prob
            self.update = update
            self.train_loss = train_loss
            self.train_id = train_translate

            # self.inference_loss = inf_loss
            self.inference_id = inf_translate

    def main(self, session, epochs, data_tuple,  log_dir, filewriters, test_check=20, test_translate=0):
        try:
            recent_losses = [1e8] * 10  # should use a queue
            for i, (arg_name, arg_desc) in enumerate(self._to_batch(*data_tuple.train, epochs)):

                ops = [self.update, self.train_loss,
                       self.train_id, self.merged_metrics]
                _,  _, train_id, train_summary = self._feed_fwd(
                    session, arg_name, arg_desc, ops, 'TRAIN')
                filewriters["train_continuous"].add_summary(train_summary, i)

                if i % test_check == 0:
                    evaluation_tuple = self.evaluate_bleu(
                        session, data_tuple.train, max_points=5000)
                    log_util.log_tensorboard(
                        filewriters['train'], i, *evaluation_tuple)

                    valid_evaluation_tuple = self.evaluate_bleu(
                        session, data_tuple.valid, max_points=5000)
                    log_util.log_tensorboard(
                        filewriters['valid'], i, *valid_evaluation_tuple)

                    test_evaluation_tuple = ((-1,), -1, "--") 
                    # test_evaluation_tuple = self.evaluate_bleu(
                    #     session, data_tuple.test, max_points=10000)
                    # log_util.log_tensorboard(
                    #     filewriters['test'], i, *test_evaluation_tuple)

                    log_util.log_std_out(
                        i, evaluation_tuple, valid_evaluation_tuple, test_evaluation_tuple)

                    if i > 0:
                        saveload.save(session, log_dir, self.name, i)

                    recent_losses.append(valid_evaluation_tuple[-2])
                    if np.argmin(recent_losses) == 0:
                        return
                    else:
                        recent_losses.pop(0)

        except KeyboardInterrupt as e:
            saveload.save(session, log_dir, self.name, i)


@args.log_args
@args.train_args
@args.data_args
def _build_argparser():
    parser = argparse.ArgumentParser(
        description='Run the basic LSTM model on the overfit dataset')
    parser.add_argument('--lstm-size', '-l', dest='lstm_size', action='store',
                        type=int, default=300,
                        help='size of LSTM size')
    parser.add_argument('--tokenizer', '-to', dest='tokenizer', action='store',
                       type=str, default='var_only',
                       help='the type of tokenizer to build the char_sequence: var_only, var_funcname')
    parser.add_argument('--bidirectional', '-bi', dest='bidirectional', action='store',
                       type=bool, default=True,
                       help='use bidirectional lstm')

    return parser


def _run_model(name, logdir, test_freq, test_translate, save_every,
               lstm_size, dropout, lr, batch_size, epochs,
               vocab_size, char_seq, desc_seq, char_embed, desc_embed,
               use_full_dataset, use_split_dataset, tokenizer, bidirectional, **kwargs):
    log_path = log_util.to_log_path(logdir, name)
    log_util.setup_logger(log_path)

    embed_tuple, data_tuple = tokenize.get_embed_tuple_and_data_tuple(
        vocab_size, char_seq, desc_seq, char_embed, desc_embed,
        use_full_dataset, use_split_dataset, tokenizer)
    nn = CharSeqBaseline(embed_tuple, lstm_size, batch_size, lr, dropout)

    summary = ExperimentSummary(nn, vocab_size, char_seq, desc_seq, char_embed, desc_embed,
                                use_full_dataset, use_split_dataset)

    LOGGER.warning("Follow logs with \n\n ./log_summary.sh -f {}/main.log \n".format(log_path))
    LOGGER.multiline_info(summary)

    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    
    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=4,
        inter_op_parallelism_threads=4)
    sess = tf.Session(config=session_conf)

    saveload.setup_saver(save_every)
    filewriters = log_util.get_filewriters(log_path, sess)

    sess.run(init)
    # log_util.load(sess, "logdir_0618_204400", "BasicModel.ckpt-1" )
    nn.main(sess, epochs, data_tuple, log_path, filewriters,
            test_check=test_freq, test_translate=test_translate)


if __name__ == "__main__":
    parser = _build_argparser()
    args = parser.parse_args()

    _run_model(**vars(args))
