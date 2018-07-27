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

SingleTranslationWithPaths = namedtuple(
    "Translation", ['name', 'description', 'tokenized', 'translation', 'code'])
SingleTranslationWithPaths.__str__ = lambda s: "ARGN: {}\nCODE: {}\nDESC: {}\nTOKN: {}\nINFR: {}\n".format(
    s.name, s.code," ".join(s.description)," ".join(s.tokenized), " ".join(s.translation))


class Code2VecEncoder(BasicRNNModel):

    def __init__(self, embed_tuple, rnn_size=300, batch_size=128, learning_rate=0.001, 
                dropout=0.3, bidirectional=False, vec_size=128, name="BasicModel"):
        super().__init__(embed_tuple, name)
        # To Do; all these args from config, to make saving model easier.

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.rnn_size = rnn_size
        self.code2vec_final_size = vec_size
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
        

        self.dim = 300
        vocab = 20000
        self.path_weights = np.random.uniform(
             low=-0.1, high=0.1, size=[vocab, self.dim])
        self.input_target_var_weights = np.random.uniform(
             low=-0.1, high=0.1, size=[vocab, self.dim])

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

    def build_translations(self, all_names, all_references, all_references_tok, all_translations, all_data):
        get_path_stats = lambda x: "Paths: {}/{},  Of Which <UNK> {}".format(np.count_nonzero(x), len(x), (x==1).sum())
        return [SingleTranslationWithPaths(n, r[0], t[0], tr, get_path_stats(s)) for n, r, t, tr, s in zip(
            all_names, all_references, all_references_tok, all_translations, all_data[2])]

    @staticmethod
    def _build_code2vec_encoder(input_codepaths, path_weights,
                                target_var_variable, target_var_weights):
        with tf.name_scope("code2vec_embeddings"):
            # 1. Embed Our "codepath" as a whole sequence
            codepath_vocab_size, codepath_size = path_weights.shape
            path_initializer = tf.constant_initializer(path_weights)
            path_embedding = tf.get_variable("path_embed", [codepath_vocab_size, codepath_size],
                                             initializer=path_initializer, trainable=True)
            encode_path_embedded = tf.nn.embedding_lookup(
                path_embedding, input_codepaths)

            # 2. Embed Our "target_var" as a word
            target_var_vocab_size, target_var_size = target_var_weights.shape
            target_var_initializer = tf.constant_initializer(target_var_weights)
            target_var_embedding = tf.get_variable("target_embed", [target_var_vocab_size, target_var_size],
                                             initializer=target_var_initializer, trainable=True)
            encode_target_var_embedded = tf.nn.embedding_lookup(
                target_var_embedding, target_var_variable)

            return encode_path_embedded, encode_target_var_embedded, path_embedding, target_var_embedding

    @staticmethod
    def _build_code2vec_vector(encode_path_embedded, encode_target_var_embedded, dim, code2vec_final_size):
        with tf.name_scope("code2vec_vector"):
            # 1. Concat Our Vector
            path_context = tf.concat([encode_path_embedded, encode_target_var_embedded], axis=2)
            
            path_context_size = 2 * dim
            # 2. Feed it through an MLP
            W = tf.get_variable("MLP_W", 
                [path_context_size, code2vec_final_size],  
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
            B = tf.get_variable("MLP_B", 
                [code2vec_final_size],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())

            Z = tf.add(tf.tensordot(path_context, W, axes=[[2], [0]]), B,  )
            A = tf.nn.tanh(Z)
            # 3. Add attention & Return the Vector
            attention_param = tf.get_variable("attention", 
                [code2vec_final_size],  
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())

            attention_vector = tf.tensordot(A, attention_param, axes=[[2], [0]])
            attention_vector = tf.nn.softmax(attention_vector)


            multiplied = tf.transpose(tf.multiply(tf.transpose(A, [2,0,1]), attention_vector), [1,2,0])
            code_vec = tf.reduce_sum(multiplied, axis=1)

            return code_vec

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
            # input_codepaths_seq_length = tf.argmin(
            #     input_codepaths, axis=1, output_type=tf.int32) + 1
            
            # 1. Get Embeddings
            encode_embedded, decode_embedded, _, decoder_weights = self._build_encode_decode_embeddings(
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
                self.dim,
                self.code2vec_final_size
                )

            # 2. Build out Encoder
            if self.bidirectional:
                first_encoder_outputs, first_state = self._build_bi_rnn_encoder(
                    input_data_seq_length, self.rnn_size, encode_embedded, dropout_keep_prob, name="FirstRNN")
            else:
                first_encoder_outputs, first_state = self._build_rnn_encoder(
                    input_data_seq_length, self.rnn_size, encode_embedded, dropout_keep_prob,  name="FirstRNN")

            c = tf.concat([first_state.c, code2vec_embedding], axis = 1)
            h = tf.concat([first_state.h, code2vec_embedding], axis = 1)
            state = tf.contrib.rnn.LSTMStateTuple(c, h)

            # 3. Build out Cell ith attention
            decoder_rnn_size = self.rnn_size + self.code2vec_final_size
            if self.bidirectional:
                decoder_rnn_size = decoder_rnn_size + self.rnn_size
                decoder_rnn_cell = tf.contrib.rnn.BasicLSTMCell(
                    decoder_rnn_size, name="RNNencoder")
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
            
            decoder_rnn_cell = tf.contrib.rnn.DropoutWrapper(
                decoder_rnn_cell,
                input_keep_prob=dropout_keep_prob,
                output_keep_prob=dropout_keep_prob,
                state_keep_prob=dropout_keep_prob)

            decoder_rnn_cell = tf.contrib.seq2seq.AttentionWrapper(
                decoder_rnn_cell, attention_mechanism1,
                attention_layer_size=self.rnn_size)

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


    def main(self, session, epochs, data_tuple,  log_dir, filewriters, test_check=20, test_translate=0):
        epoch = 0
        try:
            recent_losses = [1e8] * 50  # should use a queue
            for i, (e, minibatch) in enumerate(self._to_batch(data_tuple.train, epochs)):
                ops = [self.update, self.train_loss,
                       self.train_id, self.merged_metrics]
                _,  loss, train_id, train_summary = self._feed_fwd(
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

@args.encoder_args
@args.log_args
@args.train_args
@args.data_args
def _build_argparser():
    parser = argparse.ArgumentParser(
        description='Run the basic LSTM model on the overfit dataset')
    parser.add_argument('--vec-size', '-vs', dest='vec_size', action='store',
                       type=int, default=200,
                       help='size of code2vec vector')
    return parser


def _run_model(name, logdir, test_freq, test_translate, save_every,
               lstm_size, dropout, lr, batch_size, epochs,
               vocab_size, char_seq, desc_seq, char_embed, desc_embed,
               use_full_dataset, use_split_dataset, tokenizer, bidirectional, no_dups, vec_size, **kwargs):
    log_path = log_util.to_log_path(logdir, name)
    log_util.setup_logger(log_path)

    bidirectional = bidirectional > 0
    embed_tuple, data_tuple = tokenize.get_embed_tuple_and_data_tuple(
        vocab_size, char_seq, desc_seq, char_embed, desc_embed,
        use_full_dataset, use_split_dataset, tokenizer, no_dups, "code2vec")
    print(len(data_tuple))



    nn = Code2VecEncoder(embed_tuple, lstm_size, batch_size, lr, dropout, bidirectional, vec_size)
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