import tensorflow as tf
from tensorflow.contrib import rnn,seq2seq
from hyperparameters import Hyperparameters
hp = Hyperparameters()

class Model():
    def __init__(self,mode):
        self.rnn_size = hp.rnn_size
        self.num_layers = hp.lstm_layers
        self.source_vocab_size = hp.source_vocab_size
        self.embedding_size = hp.embedding_size
        self.beam_size = hp.beam_search
        self.target_vocab_size = hp.target_vocab_size
        self.mode = mode
        self.learning_rate = hp.learning_rate
        self.max_gradient_norm = hp.max_gradient_norm
        self.create_placeholder()
        self.model()
    def create_placeholder(self):
        self.source_input = tf.placeholder(dtype=tf.int32,shape=[None,None])
        self.target_input = tf.placeholder(dtype=tf.int32,shape = [None,None])
        self.target_length = tf.placeholder(dtype=tf.int32,shape=[None])
        self.max_target_length = tf.placeholder(dtype=tf.int32,shape=[])
        self.is_training = tf.placeholder(dtype=tf.bool,shape=[])
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])

    def _create_rnn_cell(self):
        def single_rnn_cell():
            single_rnn_cell = rnn.LSTMCell(num_units=self.rnn_size)

            return single_rnn_cell

        cell = rnn.MultiRNNCell([single_rnn_cell() for _ in range(self.num_layers)])
        return cell
    def model(self):
        with tf.variable_scope("encoder"):
            encoder_cell = self._create_rnn_cell()
            source_embedding = tf.get_variable(name="source_embedding",
                                               shape=[self.source_vocab_size, self.embedding_size],
                                               initializer=tf.initializers.truncated_normal())
            encoder_embedding_inputs = tf.nn.embedding_lookup(source_embedding, self.source_input)
            encoder_outputs, encoder_states = tf.nn.dynamic_rnn(cell=encoder_cell,
                                                                inputs=encoder_embedding_inputs,
                                                                dtype=tf.float32)
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            if self.mode=="test":
                encoder_states = seq2seq.tile_batch(encoder_states, self.beam_size)
            decoder_cell = self._create_rnn_cell()
            decoder_cell = rnn.DropoutWrapper(decoder_cell,output_keep_prob=0.5)

            if self.mode=="test":
                batch_size = self.batch_size*self.beam_size
            else:
                batch_size = self.batch_size
            #decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size,dtype=tf.float32)

            output_layer = tf.layers.Dense(units=self.target_vocab_size,
                                           kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            target_embedding = tf.get_variable(name="target_embedding",
                                               shape=[self.target_vocab_size, self.embedding_size])
            if self.mode == "train":
                self.mask = tf.sequence_mask(self.target_length,self.max_target_length,dtype=tf.float32)
                del_end = tf.strided_slice(self.target_input,[0,0],[self.batch_size,-1],[1,1])
                decoder_input = tf.concat([tf.fill([self.batch_size, 1],2),del_end],axis=1)
                decoder_input_embedding = tf.nn.embedding_lookup(target_embedding,decoder_input)
                training_helper = seq2seq.TrainingHelper(inputs=decoder_input_embedding,
                                                         sequence_length=tf.fill([self.batch_size],self.max_target_length))
                training_decoder = seq2seq.BasicDecoder(cell=decoder_cell,
                                                        helper=training_helper,
                                                        initial_state=encoder_states,
                                                        output_layer=output_layer)
                decoder_outputs,_,_ = seq2seq.dynamic_decode(decoder=training_decoder,output_time_major=False,
                                                             impute_finished=True,
                                                             maximum_iterations=self.max_target_length)
                self.decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
                self.decoder_predict_train = tf.argmax(self.decoder_logits_train,axis=-1)
                self.loss_op = tf.reduce_mean(tf.losses.softmax_cross_entropy(
                    onehot_labels=tf.one_hot(self.target_input, depth=self.target_vocab_size),
                    logits=self.decoder_logits_train, weights=self.mask))
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                trainable_params = tf.trainable_variables()
                gradients = tf.gradients(self.loss_op, trainable_params)
                clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
                self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))
            elif self.mode =="test":
                start_tokens = tf.fill([self.batch_size], value=2)
                end_token = 3
                inference_decoder = seq2seq.BeamSearchDecoder(cell=decoder_cell, embedding=target_embedding,
                                                              start_tokens=start_tokens, end_token=end_token,
                                                              initial_state=encoder_states,
                                                              beam_width=self.beam_size, output_layer=output_layer)
                decoder_outputs, _, _ = seq2seq.dynamic_decode(decoder=inference_decoder, maximum_iterations=self.max_target_length)
                print(decoder_outputs.predicted_ids.get_shape().as_list())
                self.decoder_predict_decode = decoder_outputs.predicted_ids[:, :, 0]
    def train(self,sess,source_train_datas,target_train_datas,target_train_length,batch_size):
        index = 0
        max_target_length = target_train_datas.shape[1]
        n = int(len(source_train_datas)/batch_size)
        while index<len(source_train_datas):
            batch_source_train_datas = source_train_datas[index:index+batch_size]
            batch_target_train_datas = target_train_datas[index:index+batch_size]
            batch_target_train_length = target_train_length[index:index+batch_size]
            loss,_ = sess.run([self.loss_op,self.train_op],feed_dict={self.source_input:batch_source_train_datas,
                                                                      self.target_input:batch_target_train_datas,
                                                                      self.target_length:batch_target_train_length,
                                                                      self.batch_size:len(batch_target_train_length),
                                                                      self.is_training:True,
                                                                      self.max_target_length:max_target_length})
            if index%(100*batch_size)==0:
                print (int(index/(100*batch_size)*100),"/ "+str(n))
                print ("Trining loss is:",loss)
            index+=batch_size
    def infer(self,sess,source_test_datas,target_test_datas,batch_size):
        index = 0
        max_target_length = target_test_datas.shape[1]
        results = []
        while index<len(source_test_datas):
            batch_source_test_datas = source_test_datas[index:index+batch_size]
            batch_target_test_datas = target_test_datas[index:index+batch_size]
            predict = sess.run(self.decoder_predict_decode,feed_dict = {self.source_input:batch_source_test_datas,
                                                                      self.target_input:batch_target_test_datas,
                                                                      self.batch_size:len(batch_source_test_datas),
                                                                      self.is_training:False,
                                                                      self.max_target_length:max_target_length})
            results = results+list(predict)
            index+=batch_size
        return results


