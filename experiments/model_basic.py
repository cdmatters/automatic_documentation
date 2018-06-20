import tensorflow as tf
from tensorflow.python.layers import core as layers_core

import numpy as np

from tensorflow.python import debug as tf_debug


class BasicRNNModel(object):

    def __init__(self, word2idx, word_weights, char_weights, rnn_size=300, batch_size=128,
                 learning_rate=0.001):
        # To Do; all these args from config, to make saving model easier.
        self.name = "SimpleModel"

        self.word_weights = word_weights
        self.char_weights = char_weights
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
            
            return encode_embedded, decode_embedded

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
    def _build_rnn_training_decoder(state, input_label_seq_length, rnn_size, decode_embedded, word_weights):
        with tf.name_scope("decoder"):
            decoder_rnn_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size, name="RNNencoder")
        
            helper = tf.contrib.seq2seq.TrainingHelper(
                     decode_embedded, input_label_seq_length, time_major=False)
            
            desc_vocab_size = word_weights.shape[0]
            projection_layer = layers_core.Dense(desc_vocab_size, use_bias=False)
            
            decoder = tf.contrib.seq2seq.BasicDecoder(
                              decoder_rnn_cell, helper, state,
                              output_layer=projection_layer)
            
            return tf.contrib.seq2seq.dynamic_decode(decoder)

    @staticmethod
    def _build_rnn_inference_decoder(state):
        pass

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
            encode_embedded, decode_embedded = self._build_encode_decode_embeddings(
                                                    input_data_sequence, self.char_weights, 
                                                    input_label_sequence, self.word_weights)
            
            # 2. Build out Encoder
            _, state = self._build_rnn_encoder(input_data_seq_length, self.rnn_size, encode_embedded)

            # 3. Build out Training Decoder
            outputs, _, _ = self._build_rnn_training_decoder(state, 
                                            input_label_seq_length, self.rnn_size,
                                            decode_embedded, self.word_weights)

            # 4. Define Loss 
            logits = outputs.rnn_output
            train_loss = self._get_loss(logits, input_label_sequence, input_label_seq_length)

            # 5. Do Updates
            update = self._do_updates(train_loss, self.learning_rate)

            # 6. Save Variables to Model
            self.input_data_sequence = input_data_sequence
            self.input_label_sequence = input_label_sequence
            self.update = update
            self.loss = train_loss
    
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

  
    def _to_batch(self, arg_name, arg_desc, do_prog_bar=False):
        assert arg_name.shape[0] == arg_desc.shape[0]
        size = arg_name.shape[0]

        for i in tqdm(range((size//self.batch_size)+1), leave=do_prog_bar):
            arg_name_batch = arg_name[i * self.batch_size: (i+1) * self.batch_size]
            arg_desc_batch = arg_desc[i * self.batch_size: (i+1) * self.batch_size]
            yield arg_name_batch, arg_desc_batch

    def train(self, session, data):
        for _ in range(5_000):
            # for arg_name, arg_desc in _to_batch(*data):
            arg_name, arg_desc = data[0], data[1]
            _, current_loss = self._feed_fwd(sess, arg_name[:2, :], arg_desc[:2, :], [self.update, self.loss])
            print(current_loss)
        pass


if __name__=="__main__":
    from data.preprocessed.overfit import data as DATA
    import experiments.utils as utils

    vocab_size = 20_000
    char_seq_len = 24
    desc_seq_len = 300
    
    print("Loading GloVe weights and word to index lookup table")
    word_weights, word2idx = utils.get_weights_word2idx(vocab_size)
    print("Creating char to index look up table")
    char_weights =np.random.uniform(low=-0.1, high=0.1, size=[70, 300])
    char2idx = utils.get_char2idx()
    
    print("Tokenizing the word desctiptions and characters") 
    data = utils.tokenize_descriptions(DATA.train, word2idx, char2idx)

    test = utils.extract_char_and_desc_idx_tensors(data, char_seq_len, desc_seq_len)
    
    nn = BasicRNNModel(word2idx, word_weights, char_weights)

    sess = tf.Session()

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    nn.train(sess, test)

