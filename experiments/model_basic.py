import tensorflow as tf
from tensorflow.python.layers import core as layers_core

class BasicRNNModel(object):

    def __init__(self, word2idx, word_weights, rnn_size=300, embed_size=300, batch_size=128,
                 char_vocab_size=70, output_labels=6):
        # all these args from config, to make saving model easier.
        self.word_weights = word_weights

        self.name = "SimpleModel"
        self.char_vocab_size = char_vocab_size
        self.desc_vocab_size = self.word_weights.shape[0]
        self.embed_size = self.word_weights.shape[1]

        self.rnn_size = rnn_size
        self.embed_size = embed_size
        self.batch_size = batch_size

        # Training Graph Variables

        self._build_train_graph()

    def _build_train_graph(self):
       
        with tf.name_scope("Model_{}".format(self.name)):

            # input_data_sequence : [batch_size x max_variable_length]
            input_data_sequence = tf.placeholder(tf.int32, [None, None], "arg_name")
            input_data_seq_length = tf.argmax(input_data_sequence, axis=1, output_type=tf.int32)
            # input_label_sequence  : [batch_size x max_docstring_length]
            input_label_sequence = tf.placeholder(tf.int32, [None, None], "arg_desc")
            input_label_seq_length = tf.argmax(input_label_sequence, axis=1, output_type=tf.int32)


            batch_size = tf.shape(input_data_sequence)[0]
            unroll_size = tf.shape(input_data_sequence)[1]

            # 1. Embed Our "arg_names" character by character
            initializer = tf.random_uniform_initializer(-0.1, 0.1)
            char_embedding = tf.get_variable("char_embed", [self.char_vocab_size, self.embed_size],
                                         initializer=initializer, trainable=True)
            encode_embedded = tf.nn.embedding_lookup(char_embedding, input_data_sequence)
            
            # 2. Embed Our "arg_desc" word by word
            glove_initializer = tf.constant_initializer(self.word_weights)
            word_embedding = tf.get_variable("desc_embed", [self.desc_vocab_size, self.embed_size],
                                         initializer=glove_initializer, trainable=False)
            decode_embedded = tf.nn.embedding_lookup(word_embedding, input_label_sequence)


            # 3. Build out encoder LSTM
            encoder_rnn_cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_size, name="LSTM_enc_"+self.name)
            initial_state = encoder_rnn_cell.zero_state(batch_size, dtype=tf.float32)
           
            _, state = tf.nn.dynamic_rnn(encoder_rnn_cell, encode_embedded,
                                        sequence_length=input_data_seq_length,
                                        initial_state=initial_state, time_major=False)
            
            # 4. Build out decoder LSTM
            decoder_rnn_cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_size, name="LSTM_dec_"+self.name)
            projection_layer = layers_core.Dense(self.desc_vocab_size, use_bias=False)
            
            helper = tf.contrib.seq2seq.TrainingHelper(
                     decode_embedded, input_label_seq_length , time_major=False)
            
            decoder = tf.contrib.seq2seq.BasicDecoder(
                              decoder_rnn_cell, helper, state,
                              output_layer=projection_layer)
            
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
            
            # 5. Define Loss 
            logits = outputs.rnn_output
            zero_col = tf.zeros([batch_size,1], dtype=tf.int32)
            decoder_outputs = tf.concat([input_label_sequence[:, 1:], zero_col], 1)  # TODO transform this
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                           labels=decoder_outputs, logits=logits)

            target_weights = tf.logical_not(tf.equal(decoder_outputs, tf.zeros_like(decoder_outputs)))
            target_weights = tf.cast(target_weights, tf.float32)
            train_loss = (tf.reduce_sum(crossent * target_weights))# / batch_size)

            # 6. Clip gradients

            max_gradient_norm = 1
            params = tf.trainable_variables()
            gradients = tf.gradients(train_loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(
                                    gradients, max_gradient_norm)

            # 7. Optimiser
            optimizer = tf.train.AdamOptimizer(0.001)
            update = optimizer.apply_gradients(zip(clipped_gradients, params))


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
        for _ in range(5):
            # for arg_name, arg_desc in _to_batch(*data):
            arg_name, arg_desc = data[0], data[1]
            print(arg_name[0])
            print(arg_desc[1])
            _, current_loss = self._feed_fwd(sess, arg_name[:1, :], arg_desc[:1, :], [self.update, self.loss])
            print(current_loss)
        pass


if __name__=="__main__":
    from data.preprocessed.overfit import data as DATA
    import experiments.utils as utils
    weights, word2idx = utils.get_weights_word2idx(40000)
    char2idx = utils.get_char2idx()
    print(weights.shape)
    data = utils.tokenize_descriptions(DATA.train, word2idx, char2idx)
    test = utils.extract_char_and_desc_idx_tensors(data, 24, 100)
    nn = BasicRNNModel(word2idx, weights)

    sess = tf.Session()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    nn.train(sess, test)

