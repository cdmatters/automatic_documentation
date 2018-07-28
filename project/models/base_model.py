import abc
from collections import namedtuple
import logging


# from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import numpy as np
import tensorflow as tf

from project.external.nmt import bleu
from project.utils.tokenize import START_OF_TEXT_TOKEN
import project.utils.logging as log_util
import project.utils.saveload as saveload

LOGGER = logging.getLogger('')


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
    "Translation", ['name', 'description', 'tokenized', 'translation'])
SingleTranslation.__str__ = lambda s: "ARGN: {}\nDESC: {}\nTOKN: {}\nINFR: {}\n".format(
    s.name," ".join(s.description), " ".join(s.tokenized), " ".join(s.translation))


exp_sum = ['nn', 'vocab', 'char_seq', 'desc_seq', 'char_embed', 'desc_embed', 'full_dataset', 'split_dataset']
ExperimentSummary = namedtuple("ExperimentSummary", exp_sum)
ExperimentSummary.__str__ = lambda s: EXPERIMENT_SUMMARY_STRING.format(
    vocab=s.vocab, char=s.char_seq, desc=s.desc_seq,
    full=s.full_dataset, nn=s.nn, split=s.split_dataset,
    c_embed=s.char_embed, d_embed=s.desc_embed)

np.random.seed(100)

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
    def build_translations(self, all_names, all_references, all_tokenized, all_translations, all_data):
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
    def _get_scope_variable(scope, var, shape=None):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            v = tf.get_variable(var, shape)
        return v

    def get_scope_variable(self, sess, scope, var):
        return sess.run([self._get_scope_variable(scope, var)], feed_dict={})
        

    @staticmethod
    def _build_encode_decode_embeddings(input_data_sequence, char_weights,
                                        input_label_sequence, word_weights):
        with tf.variable_scope("embed_vars", reuse=tf.AUTO_REUSE):
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
    def _build_bi_rnn_encoder(input_data_seq_length, rnn_size, encode_embedded, dropout_keep_prob, name="RNNencoder"):
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            batch_size = tf.shape(input_data_seq_length)
            encoder_rnn_cell_fw = tf.contrib.rnn.BasicLSTMCell(
                rnn_size, name=name)
            initial_state_fw = encoder_rnn_cell_fw.zero_state(
                batch_size, dtype=tf.float32)

            encoder_rnn_cell_fw = tf.contrib.rnn.DropoutWrapper(encoder_rnn_cell_fw,
                                                             input_keep_prob=dropout_keep_prob,
                                                             output_keep_prob=dropout_keep_prob,
                                                             state_keep_prob=dropout_keep_prob)

            encoder_rnn_cell_bk = tf.contrib.rnn.BasicLSTMCell(
                rnn_size, name=name)
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
                                                  initial_state_bw=initial_state_bk,
                                                  time_major=False)

            final_states = tf.concat(output_states, 2)
            final_states = tf.contrib.rnn.LSTMStateTuple(final_states[0,:,:], final_states[1,:,:])
            return tf.concat(outputs, 2), final_states


    @staticmethod
    def _build_rnn_encoder(input_data_seq_length, rnn_size, encode_embedded, dropout_keep_prob, name="RNNencoder"):
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            batch_size = tf.shape(input_data_seq_length)
            encoder_rnn_cell = tf.contrib.rnn.BasicLSTMCell(
                rnn_size, name=name)
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
                                    input_label_seq_length, decode_embedded, use_attention=True):
        with tf.variable_scope("training", reuse=tf.AUTO_REUSE):
            batch_size = tf.shape(state[0])[0]

            helper = tf.contrib.seq2seq.TrainingHelper(
                decode_embedded, input_label_seq_length, time_major=False)

            if use_attention:
                decoder_initial_state = decoder_rnn_cell.zero_state(batch_size, dtype=tf.float32).clone(
                    cell_state=state)
            else:
                decoder_initial_state = state

            decoder = tf.contrib.seq2seq.BasicDecoder(
                decoder_rnn_cell, helper, decoder_initial_state,
                output_layer=projection_layer)

            return tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True)

    @staticmethod
    def _build_rnn_greedy_inference_decoder(decoder_rnn_cell, state, projection_layer, decoder_weights,
                                            start_tok, end_tok, use_attention=True):
        with tf.variable_scope("inference", reuse=tf.AUTO_REUSE):
            batch_size = tf.shape(state[0])[0]

            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_weights,
                                                              tf.fill([batch_size], start_tok), end_tok)
            if use_attention:
                decoder_initial_state = decoder_rnn_cell.zero_state(batch_size, dtype=tf.float32).clone(
                    cell_state=state)
            else:
                decoder_initial_state = state

            decoder = tf.contrib.seq2seq.BasicDecoder(
                decoder_rnn_cell, helper, decoder_initial_state,
                output_layer=projection_layer)

            maximum_iterations = 300
            return tf.contrib.seq2seq.dynamic_decode(
                decoder, impute_finished=True, maximum_iterations=maximum_iterations)

    @staticmethod
    def _get_loss(logits, input_label_sequence, input_label_seq_length):
        with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
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
        with tf.variable_scope("opt", reuse=tf.AUTO_REUSE):
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

    def _feed_fwd(self, session, minibatch_data, operation, mode=None):
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
        feed_dict = {self.input_data_sequence: minibatch_data[0],
                     self.input_label_sequence: minibatch_data[1]}
        if mode == 'TRAIN':
            feed_dict[self.dropout_keep_prob] = 1 - self.dropout

        return session.run(run_ouputs, feed_dict=feed_dict)

    def _to_batch(self, full_data, epochs=1, do_prog_bar=False):
        # arg_name, arg_desc, arg_code = full_data 
        # assert arg_name.shape[0] == arg_desc.shape[0]
        size = full_data[0].shape[0]

        batch_per_epoch = (size // self.batch_size) + 1

        for e in range(epochs):
            zipped = list(zip(*full_data))
            np.random.shuffle(zipped)
            shuffled = list(zip(*zipped))

            for i in range(batch_per_epoch):
                idx_start = i * self.batch_size
                idx_end = (i + 1) * self.batch_size

                mb_data = [d[idx_start: idx_end] for d in shuffled]
                # arg_name_batch = arg_name[idx_start: idx_end]
                # arg_desc_batch = arg_desc[idx_start: idx_end]
                yield e, tuple(mb_data)

    def evaluate_bleu(self, session, data, max_points=10000, max_translations=200):
        all_names = []
        all_references = []
        all_references_tok = []
        all_translations = []
        all_training_loss = []

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
            arg_name, arg_desc, arg_desc_translate = minibatch[0], minibatch[1], minibatch[-1]
            names = [self.translate(i, lookup=self.idx2char).replace(
                " ", "") for i in arg_name]

            references = [[t] for t in arg_desc_translate]
            references_tokenized = [[self.translate(i, do_join=False)[1:-1]] for i in arg_desc]
            translations = [self.translate(
                i, do_join=False, prepend_tok=self.word2idx[START_OF_TEXT_TOKEN])[1:-1] for i in inference_ids]


            for t in all_translations:
                if t == []:
                    LOGGER.warning("EMPTY TRANSLATION")
                    t.append("")

            all_training_loss.append(train_loss)
            all_names.extend(names)
            all_references.extend(references)
            all_references_tok.extend(references_tokenized)
            all_translations.extend(translations)

        # BLEU TUPLE = (bleu_score, precisions, bp, ratio, translation_length, reference_length)
        # To Do: Replace with NLTK:
        #         smoother = SmoothingFunction()
        #         bleu_score = corpus_bleu(all_references, all_translations, 
        #                                  smoothing_function=smoother.method2)
        bleu_tuple = bleu.compute_bleu(
            all_references, all_translations, max_order=4, smooth=False)
        av_loss = np.mean(all_training_loss)

        translations = self.build_translations(all_names, all_references, all_references_tok, all_translations, restricted_data)
        return bleu_tuple, av_loss, translations[:max_translations]
    
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




if __name__ == "__main__":
    print("ABSTRACT BASE CLASS")
