import tensorflow as tf
import numpy as np
from tqdm import tqdm

from tensorflow.python.layers import core as layers_core
from tensorflow.python import debug as tf_debug


from  experiments.utils import PAD_TOKEN, UNKNOWN_TOKEN, START_OF_TEXT_TOKEN, END_OF_TEXT_TOKEN


class BasicRNNModel(object):

    def __init__(self, word2idx, word_weights, char_weights, rnn_size=300, batch_size=128,
                 learning_rate=0.001):
        # To Do; all these args from config, to make saving model easier.
        self.name = "SimpleModel"

        self.word_weights = word_weights
        self.char_weights = char_weights
        self.word2idx = word2idx
        self.idx2word = dict((v,k) for k,v in word2idx.items())


        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.rnn_size = rnn_size

        # Graph Variables (built later)
        self.input_data_sequence = None
        self.input_label_sequence = None
        self.update = None
        self.loss = None
        
        self._build_train_graph()
        
        print("Init loaded")

    @staticmethod
    def _build_encode_decode_embeddings(input_data_sequence, char_weights, 
                                        input_label_sequence, word_weights):
        with tf.name_scope("embed_vars"): 
            # 1. Embed Our "arg_names" char by char
            char_vocab_size, char_embed_size = char_weights.shape
            char_initializer =  tf.constant_initializer(char_weights)        
            char_embedding = tf.get_variable("char_embed", [char_vocab_size, char_embed_size],
                                             initializer=char_initializer, trainable=True)
            encode_embedded = tf.nn.embedding_lookup(char_embedding, input_data_sequence)
                
            # 2. Embed Our "arg_desc" word by word
            desc_vocab_size, word_embed_size = word_weights.shape
            word_initializer = tf.constant_initializer(word_weights)
            word_embedding = tf.get_variable("desc_embed", [desc_vocab_size, word_embed_size],
                                             initializer=word_initializer, trainable=True)
            decode_embedded = tf.nn.embedding_lookup(word_embedding, input_label_sequence)
            
            return encode_embedded, decode_embedded, char_embedding, word_embedding

    @staticmethod
    def _build_rnn_encoder(input_data_seq_length, rnn_size, encode_embedded):  
        with tf.name_scope("encoder"):
            batch_size = tf.shape(input_data_seq_length)          
            encoder_rnn_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size, name="RNNencoder")
            initial_state = encoder_rnn_cell.zero_state(batch_size, dtype=tf.float32)
               
            return tf.nn.dynamic_rnn(encoder_rnn_cell, encode_embedded,
                                        sequence_length=input_data_seq_length,
                                        initial_state=initial_state, time_major=False)
    
    @staticmethod
    def _build_rnn_training_decoder(decoder_rnn_cell, state, projection_layer, decoder_weights,
                                    input_label_seq_length, decode_embedded):
        with tf.name_scope("training"):        
            helper = tf.contrib.seq2seq.TrainingHelper(
                     decode_embedded, input_label_seq_length, time_major=False)
            
            decoder = tf.contrib.seq2seq.BasicDecoder(
                              decoder_rnn_cell, helper, state,
                              output_layer=projection_layer)
            
            return tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True)

    @staticmethod
    def _build_rnn_greedy_inference_decoder(decoder_rnn_cell, state, projection_layer, decoder_weights,
                                            start_tok, end_tok):
        with tf.name_scope("inference"):    
            batch_size = tf.shape(state[0])[0]  

            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_weights,
                tf.fill([batch_size], start_tok), end_tok)

            decoder = tf.contrib.seq2seq.BasicDecoder(
                           decoder_rnn_cell, helper, state,
                           output_layer=projection_layer)
            
            maximum_iterations = 300
            return tf.contrib.seq2seq.dynamic_decode(
                        decoder, impute_finished=True, maximum_iterations=maximum_iterations)

    @staticmethod
    def _get_loss(logits, input_label_sequence, input_label_seq_length):
        with tf.name_scope("loss"):
            batch_size = tf.shape(input_label_sequence)[0]
            zero_col = tf.zeros([batch_size,1], dtype=tf.int32)

            # Shift the decoder to be the next word, and then clip it
            decoder_outputs = tf.concat([input_label_sequence[:, 1:], zero_col], 1)  # TODO transform this
            maximum_length = tf.reduce_max(input_label_seq_length)
            decoder_outputs = decoder_outputs[:,:maximum_length]
            
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                           labels=decoder_outputs, logits=logits)

            target_weights = tf.logical_not(tf.equal(decoder_outputs, tf.zeros_like(decoder_outputs)))
            target_weights = tf.cast(target_weights, tf.float32)
            train_loss = (tf.reduce_sum(crossent * target_weights)) # / batch_size)
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

    def _build_train_graph(self):
        with tf.name_scope("Model_{}".format(self.name)):
            # 0. Define our placeholders and derived vars
            # # input_data_sequence : [batch_size x max_variable_length]
            input_data_sequence = tf.placeholder(tf.int32, [None, None], "arg_name")
            input_data_seq_length = tf.argmin(input_data_sequence, axis=1, output_type=tf.int32)
            # # input_label_sequence  : [batch_size x max_docstring_length]
            input_label_sequence = tf.placeholder(tf.int32, [None, None], "arg_desc")
            input_label_seq_length = tf.argmin(input_label_sequence, axis=1, output_type=tf.int32)
            
            # 1. Get Embeddings
            encode_embedded, decode_embedded, _, decoder_weights = self._build_encode_decode_embeddings(
                                                    input_data_sequence, self.char_weights, 
                                                    input_label_sequence, self.word_weights)
            
            # 2. Build out Encoder
            _, state = self._build_rnn_encoder(input_data_seq_length, self.rnn_size, encode_embedded)

            # 3. Build out Training and Inference Decoder
            decoder_rnn_cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_size, name="RNNencoder")

            desc_vocab_size, _ = self.word_weights.shape 
            projection_layer = layers_core.Dense(desc_vocab_size, use_bias=False)
            
            train_outputs, _, _ = self._build_rnn_training_decoder(decoder_rnn_cell,
                                                    state,projection_layer, decoder_weights, input_label_seq_length,
                                                    decode_embedded)
            
            inf_outputs, _, _ = self._build_rnn_greedy_inference_decoder(decoder_rnn_cell,
                                                    state,projection_layer, decoder_weights,
                                                    self.word2idx[START_OF_TEXT_TOKEN],
                                                    self.word2idx[END_OF_TEXT_TOKEN])
            
            # 4. Define Train Loss 
            train_logits = train_outputs.rnn_output
            train_loss = self._get_loss(train_logits, input_label_sequence, input_label_seq_length)
            train_translate = train_outputs.sample_id

            # 5. Define Translation
            inf_logits = inf_outputs.rnn_output
            inf_translate = inf_outputs.sample_id
            inf_loss = self._get_loss(inf_logits, input_label_sequence, input_label_seq_length)


            # 6. Do Updates
            update = self._do_updates(train_loss, self.learning_rate)

            # 7. Save Variables to Model
            self.input_data_sequence = input_data_sequence
            self.input_label_sequence = input_label_sequence
            self.update = update
            self.train_loss = train_loss
            self.train_id = train_translate 

            self.inference_loss = inf_loss
            self.inference_id = inf_translate

    def translate(self, translate_id, filter_pad=True):
        if filter_pad:
            translate_id = np.trim_zeros(translate_id, 'b')
        return  " ".join([self.idx2word[i] for i in translate_id])
        
    
    def _feed_fwd(self, session, input_data, input_labels, operation):
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

        return session.run(run_ouputs, feed_dict=feed_dict)

    def _to_batch(self, arg_name, arg_desc, epochs=1e5, do_prog_bar=False):
        assert arg_name.shape[0] == arg_desc.shape[0]
        size = arg_name.shape[0]
        
        batch_per_epoch = (size // self.batch_size) + 1

        for i in tqdm(range(batch_per_epoch * epochs), disable=True):
            idx_start = (i % batch_per_epoch) * self.batch_size
            idx_end = ( (i % batch_per_epoch) + 1)  * self.batch_size

            arg_name_batch = arg_name[idx_start: idx_end]
            arg_desc_batch = arg_desc[idx_start: idx_end]
            yield arg_name_batch, arg_desc_batch

    # def train(self, session, train_data, test_data=None):

    #     i = 0

    #     epochs = 5_000
    #     arg_name, arg_desc = train_data[0], train_data[1]
    #     for e in range(epochs):

    #         # for arg_name, arg_desc in self._to_batch(*train_data):
    #             _,  current_loss, train_id = self._feed_fwd(sess, arg_name, arg_desc, [self.update, self.train_loss, self.train_id])
    #             if e % 100 == 0:
    #                 inf_id = self._feed_fwd(session, arg_name, arg_desc, self.inference_id)
    #                 for inf_t, train_t in zip(inf_id[:3], arg_desc[:3]):
    #                     print(train_t.shape)
    #                     print(self.translate(train_t))
    #                     print(i, current_loss)
    #                     print(self.translate(inf_t))


    #                     print()



    def evaluate_model(self, session, test_data, data_limit=None):
        for test_arg_name, test_arg_desc in self._to_batch(*test_data, 1):
            [inference_ids] = self._feed_fwd(session, test_arg_name, test_arg_desc, [self.inference_id])
            for test_t, true_t in zip(inference_ids[:3], test_arg_desc[:3]):
                        print(self.translate(true_t))
                        print("----")
                        print(START_OF_TEXT_TOKEN + " " + self.translate(test_t))
                        print()
            print("------------------------------------")


    def main(self, session, train_data, test_data=None, test_check=20):

        epochs = 5_000        
        for i, (arg_name, arg_desc) in enumerate(self._to_batch(*train_data, epochs)):

                ops = [self.update, self.train_loss, self.train_id]
                _,  train_loss, train_id = self._feed_fwd(sess, arg_name, arg_desc, ops)
                
                if i % test_check == 0:
                    print("EPOCH: {}, LOSS: {}".format(i, train_loss))
                    self.evaluate_model(sess, train_data)

                if i % test_check == 0 and test_data is not None:
                    self.evaluate_model(sess, test_data)





if __name__=="__main__":
    from data.preprocessed.overfit import data as DATA
    import experiments.utils as utils

    vocab_size = 10_000
    char_seq_len = 24
    desc_seq_len = 150
    
    print("Loading GloVe weights and word to index lookup table")
    word_weights, word2idx = utils.get_weights_word2idx(vocab_size)
    print("Creating char to index look up table")
    char_weights, char2idx = utils.get_weights_char2idx()
    
    print("Tokenizing the word desctiptions and characters") 
    data = utils.tokenize_descriptions(DATA.train, word2idx, char2idx)


    test_data = utils.extract_char_and_desc_idx_tensors(data[:50], char_seq_len, desc_seq_len)
        
    nn = BasicRNNModel(word2idx, word_weights, char_weights)

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    sess = tf.Session()
    sess.run(init)

    nn.main(sess, test_data)

