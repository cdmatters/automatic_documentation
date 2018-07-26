# -*- coding: utf-8 -*-
import argparse
import logging
from collections import namedtuple

import numpy as np
import tensorflow as tf
from tensorflow.python.layers import core as layers_core

from project.external.nmt import bleu
from project.models.base_model import BasicRNNModel, ExperimentSummary
import project.utils.args as args
import project.utils.logging as log_util
import project.utils.saveload as saveload
import project.utils.tokenize as tokenize
from project.utils.tokenize import PAD_TOKEN, UNKNOWN_TOKEN, \
    START_OF_TEXT_TOKEN, END_OF_TEXT_TOKEN


LOGGER = logging.getLogger('')

SingleTranslationWithCode = namedtuple(
    "Translation", ['name', 'description', 'translation', 'code'])
SingleTranslationWithCode.__str__ = lambda s: "ARGN: {}\nCODE: {}\nDESC: {}\nINFR: {}\n".format(
    s.name, " ".join(s.code)," ".join(s.description), " ".join(s.translation))


class DoubleEncoderBaseline(BasicRNNModel):

    def __init__(self, embed_tuple, rnn_size=300, batch_size=128, learning_rate=0.001, 
                dropout=0.3, bidirectional=False, name="BasicModel"):
        super().__init__(embed_tuple, name)
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

            # 2. Build out Encoder
            if self.bidirectional:
                first_encoder_outputs, first_state = self._build_bi_rnn_encoder(
                    input_data_seq_length, self.rnn_size, encode_embedded, dropout_keep_prob, name="FirstRNN")

                second_encoder_outputs, second_state = self._build_bi_rnn_encoder(
                    second_data_seq_length, self.second_rnn_size, encode_embedded, dropout_keep_prob, name="SecondRNN")
                
            else:
                first_encoder_outputs, first_state = self._build_rnn_encoder(
                    input_data_seq_length, self.rnn_size, encode_embedded, dropout_keep_prob,  name="FirstRNN")
                second_encoder_outputs, second_state = self._build_rnn_encoder(
                    second_data_seq_length, self.second_rnn_size, encode_embedded, dropout_keep_prob, name="SecondRNN")

            c = tf.concat([first_state.c, second_state.c], axis = 1)
            h = tf.concat([first_state.h, second_state.h], axis = 1)
            state = tf.contrib.rnn.LSTMStateTuple(c, h)

         
            
            # 3. Build out Cell ith attention
            decoder_rnn_size = self.rnn_size + self.second_rnn_size
            if self.bidirectional:
                decoder_rnn_size = decoder_rnn_size * 2
                decoder_rnn_cell = tf.contrib.rnn.BasicLSTMCell(
                    (self.rnn_size + self.second_rnn_size) * 2, name="RNNencoder")
            else:
                decoder_rnn_cell = tf.contrib.rnn.BasicLSTMCell(
                    decoder_rnn_size, name="RNNencoder")

            desc_vocab_size, _ = self.word_weights.shape
            projection_layer = layers_core.Dense(
                desc_vocab_size, use_bias=False)


            attention_mechanism1 = tf.contrib.seq2seq.LuongAttention(
                decoder_rnn_size, first_encoder_outputs,
                memory_sequence_length=input_data_seq_length,
                name="LuongAttention1")
            

            # attention_mechanism2 = tf.contrib.seq2seq.LuongAttention(
            #     decoder_rnn_size, second_encoder_outputs,
            #     memory_sequence_length=second_data_seq_length,
            #     name="LuongAttention2")

            decoder_rnn_cell = tf.contrib.rnn.DropoutWrapper(
                decoder_rnn_cell,
                input_keep_prob=dropout_keep_prob,
                output_keep_prob=dropout_keep_prob,
                state_keep_prob=dropout_keep_prob)

            decoder_rnn_cell = tf.contrib.seq2seq.AttentionWrapper(
                decoder_rnn_cell, attention_mechanism1,
                attention_layer_size=self.rnn_size)

            # decoder_rnn_cell = tf.contrib.seq2seq.AttentionWrapper(
            #     decoder_rnn_cell, attention_mechanism2,
            #     attention_layer_size=self.second_rnn_size)

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
            self.second_data_sequence = second_data_sequence
            self.input_label_sequence = input_label_sequence
            self.dropout_keep_prob = dropout_keep_prob
            self.update = update
            self.train_loss = train_loss
            self.train_id = train_translate

            self.inference_loss = inf_loss
            self.inference_id = inf_translate
    
    def evaluate_bleu(self, session, data, max_points=10000, max_translations=200):
        all_names = []
        all_references = []
        all_translations = []
        all_training_loss = []
        all_src_code = []

        ops = [self.merged_metrics, self.train_loss, self.inference_id]
        restricted_data = tuple([d[:max_points] for d in data])
        for _, minibatch in self._to_batch(restricted_data):

            metrics, train_loss, inference_ids = self._feed_fwd(
                session, minibatch, ops)

            # Translating quirks:
            #    names: RETURN: 'axis<END>' NOT 'a x i s <END>'
            #    references: RETURN: [['<START>', 'this', 'reference', '<END>']] 
            #                NOT: ['<START>', 'this', 'reference','<END>'],
            #                     because compute_bleu takes multiple references
            #    translations: RETURN: ['<START>', 'this', 'translation', '<END>'] 
            #                  NOT: ['this', 'translation', '<END>']
            arg_name, arg_desc, src = minibatch[0], minibatch[1], minibatch[2]
            names = [self.translate(i, lookup=self.idx2char).replace(
                " ", "") for i in arg_name]
            src_code = [self.translate(i, do_join=False) for i in src]
            references = [[self.translate(i, do_join=False)[1:-1]] for i in arg_desc]
            translations = [self.translate(
                i, do_join=False, prepend_tok=self.word2idx[START_OF_TEXT_TOKEN])[1:-1] for i in inference_ids]

            all_training_loss.append(train_loss)
            all_names.extend(names)
            all_references.extend(references)
            all_translations.extend(translations)
            all_src_code.extend(src_code)

        # BLEU TUPLE = (bleu_score, precisions, bp, ratio, translation_length, reference_length)
        # To Do: Replace with NLTK:
        #         smoother = SmoothingFunction()
        #         bleu_score = corpus_bleu(all_references, all_translations, 
        #                                  smoothing_function=smoother.method2)
        bleu_tuple = bleu.compute_bleu(
            all_references, all_translations, max_order=4, smooth=False)
        av_loss = np.mean(all_training_loss)

        translations = [SingleTranslationWithCode(n, d[0], t, s) for n, d, t, s in zip(
            all_names, all_references, all_translations, all_src_code)]

        return bleu_tuple, av_loss, translations[:max_translations]
    
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
        input_data, input_labels, input_code = minibatch
        run_ouputs = operation
        feed_dict = {self.input_data_sequence: input_data,
                     self.input_label_sequence: input_labels,
                     self.second_data_sequence: input_code }
        if mode == 'TRAIN':
            feed_dict[self.dropout_keep_prob] = 1 - self.dropout

        return session.run(run_ouputs, feed_dict=feed_dict)


    def main(self, session, epochs, data_tuple,  log_dir, filewriters, test_check=20, test_translate=0):
        epoch = 0
        try:
            recent_losses = [1e8] * 50  # should use a queue
            for i, (e, minibatch) in enumerate(self._to_batch(data_tuple.train, epochs)):
                ops = [self.update, self.train_loss,
                       self.train_id, self.merged_metrics]
                _,  _, train_id, train_summary = self._feed_fwd(
                    session, minibatch, ops, 'TRAIN')
                filewriters["train_continuous"].add_summary(train_summary, i)

                if epoch != e:
                    epoch = e
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
                        e, i, evaluation_tuple, valid_evaluation_tuple, test_evaluation_tuple)

                    if i > 0:
                        saveload.save(session, log_dir, self.name, i)

                    recent_losses.append(valid_evaluation_tuple[-2])
                    # if np.argmin(recent_losses) == 0:
                    #     return
                    # else:
                    #     recent_losses.pop(0)
            saveload.save(session, log_dir, self.name, i)
            
        except KeyboardInterrupt as e:
            saveload.save(session, log_dir, self.name, i)

@args.log_args
@args.train_args
@args.data_args
def _build_argparser():
    parser = argparse.ArgumentParser(
        description='Run the basic LSTM model on the overfit dataset')
    parser.add_argument('--lstm-size', '-l', dest='lstm_size', action='store',
                        type=int, default=128,
                        help='size of LSTM size')
    parser.add_argument('--bidirectional', '-bi', dest='bidirectional', action='store',
                       type=int, default=1,
                       help='use bidirectional lstm')

    return parser


def _run_model(name, logdir, test_freq, test_translate, save_every,
               lstm_size, dropout, lr, batch_size, epochs,
               vocab_size, char_seq, desc_seq, char_embed, desc_embed,
               use_full_dataset, use_split_dataset, tokenizer, bidirectional, no_dups, **kwargs):
    log_path = log_util.to_log_path(logdir, name)
    log_util.setup_logger(log_path)

    code_tokenizer = 'full'
    # bidirectional = False #DELETE
    bidirectional = bidirectional > 0
    embed_tuple, data_tuple = tokenize.get_embed_tuple_and_data_tuple(
        vocab_size, char_seq, desc_seq, char_embed, desc_embed,
        use_full_dataset, use_split_dataset, tokenizer, no_dups, code_tokenizer)
    nn = DoubleEncoderBaseline(embed_tuple, lstm_size, batch_size, lr, dropout, bidirectional)

    summary = ExperimentSummary(nn, vocab_size, char_seq, desc_seq, char_embed, desc_embed,
                                use_full_dataset, use_split_dataset)

    LOGGER.warning("\n./log_summary.sh -f {}/main.log # to follow\n".format(log_path))
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
