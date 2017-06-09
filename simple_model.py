import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
import random
import numpy as np


class Model():
    def __init__(self, data_dir,input_encoding,log_dir,save_dir,rnn_size,num_layers,model,batch_size,seq_length,num_epochs,save_every,grad_clip,learning_rate,decay_rate,gpu_mem,init_from, vocab_size, infer=False):#

        #infers in True if we want to generate text.
        #-------------------------------------------
        #In this case, batch size and sequence lenghth must be equal to 1 (only one word !).
        if infer:
            batch_size = 1
            seq_length = 1

        #set up LSTM and cells definition
        #--------------------------------
        #if you want, you can test another cells:
        # 'gru': cell_fn = rnn.GRUCell
        # 'basic rnn': cell_fn = rnn.BasicRNNCell
        cell_fn = rnn.BasicLSTMCell

        cells = []
        #create the cells: for each layers, we create rnn_size cells
        for _ in range(num_layers):
            cell = cell_fn(rnn_size)
            cells.append(cell)
        #define the cell of the neural network.
        self.cell = cell = rnn.MultiRNNCell(cells)

        #placeholders, states and batch pointers
        #----------------------------------------
        #insert a placeholder for the input tensor. We define a shape [size of a batch, lenght of the sequence]
        #all elements of the tensor are integer.
        self.input_data = tf.placeholder(tf.int32, [batch_size, seq_length])
        #insert a placeholder for the target tensor.
        #it's shape is similar to the input. Elements are also intergers.
        self.targets = tf.placeholder(tf.int32, [batch_size, seq_length])
        #define the initial state: integer over the size of the batch.
        self.initial_state = cell.zero_state(batch_size, tf.float32)
        #we define as a variable to point the batch. it's a integer, set up to 0 at the biginning, and not trainable.
        self.batch_pointer = tf.Variable(0, name="batch_pointer", trainable=False, dtype=tf.int32)
        #the incremental batch pointer for the operation is autmatically set: batch_pointer+1
        self.inc_batch_pointer_op = tf.assign(self.batch_pointer, self.batch_pointer + 1)
        #we define as a variable to point the epoch. it's a integer, set up to 0 at the biginning, and not trainable.
        self.epoch_pointer = tf.Variable(0, name="epoch_pointer", trainable=False)
        #we set up a variable to calculate the time spent on a batch.
        #set up to 0 at the biginning, and not trainable.
        self.batch_time = tf.Variable(0.0, name="batch_time", trainable=False)
        tf.summary.scalar("time_batch", self.batch_time)

        #define a new python op, named variables_sumaries
        #-------------------------------------------------
        def variable_summaries(var):
            """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
            #Outputs a Summary protocol buffer containing a single scalar value.
            with tf.name_scope('summaries'):
                #calculate the mean
                mean = tf.reduce_mean(var)
                #create scalar values:
                tf.summary.scalar('mean', mean)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [rnn_size, vocab_size])
            variable_summaries(softmax_w)
            softmax_b = tf.get_variable("softmax_b", [vocab_size])
            variable_summaries(softmax_b)
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding", [vocab_size, rnn_size])
                inputs = tf.split(tf.nn.embedding_lookup(embedding, self.input_data), seq_length, 1)
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        #create the loop function
        #-------------------------
        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        ## RNN Decoder initilization
        #---------------------------
        #RNN decoder for the sequence-to-sequence model. It requires:
            # - inputs,
            # - initial_state,
            # - cell function and size,
            # - loop_function,
            # - scope scope for the created subgraph
        #this function returns:
            # - outputs: (the generated outputs, a list of the same length as inputs)
            # - last_state : the state of each cell at the final time-step.
        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop, scope='rnnlm')#if infer else None

        #Output tensor
        #-------------
        #we reshape the outputs tensor:
        #tf.concact: Concatenates tensors along one dimension. Exemple:
            # tensor t1 with shape [2, 3]
            # tensor t2 with shape [2, 3]
            #tf.shape(tf.concat([t1, t2], 1)) ==> t3 with shape [2, 6]
        #then, we flatten the tensor:
        output = tf.reshape(tf.concat(outputs, 1), [-1, rnn_size])

        #predictions
        #------------
        ## The LSTM output can be used to make next word predictions

        #multiply matrix output with softmax_w, then add softmax_b
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        #Computes softmax activations using logits calculated above.
        self.probs = tf.nn.softmax(self.logits)
        #Weighted cross-entropy loss for a sequence of logits (per example).
        #   logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
        #   targets: List of 1D batch-sized int32 Tensors of the same length as logits.
        #   weights: List of 1D batch-sized float-Tensors of the same length as logits.

        #calculate cost function
        #-----------------------
        #We want to minimize the average negative log probability of the target words:
        #first, define the loss:
        loss = legacy_seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([batch_size * seq_length])],
                vocab_size)

        #cost calculation
        #-----------------
        self.cost = tf.reduce_sum(loss) / batch_size / seq_length
        #output a summary protocol buffer containing the cost value:
        tf.summary.scalar("cost", self.cost)

        #final state
        #-----------
        #define the final state as the last one:
        self.final_state = last_state
        #reset learning rate:
        self.lr = tf.Variable(0.0, trainable=False)

        #Trainable variables
        #-------------------
        #Returns all variables created with trainable=True.
        tvars = tf.trainable_variables()
        #Clip values of multiple tensors by the ratio of the sum of their norms.
        #grad_clip is the clipping ratio
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                grad_clip)

        #Optimizer:
        #----------
        #Optimizer that implements the Adam algorithm.
        optimizer = tf.train.AdamOptimizer(self.lr)

        #Apply gradients to variables.
        #-----------------------------
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, words, vocab, num, prime='first', sampling_type=1):
        '''
        This function is used to generate text, based on a saved model, with
        a text as input.
        It returns a string, composed of words chosen one by one by the model.
        '''
        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))


        #set up the string for the result:
        ret = ''

        #define the state:
        state = sess.run(self.cell.zero_state(1, tf.float32))

        ret = prime

        #we took the last word of the string:
        word = prime.split()[-1]

        #we loop of the number of words we want to generate:
        for n in range(num):
            #set up input:
            x = np.zeros((1, 1))

            #we retrieve the index of the word
            x[0, 0] = vocab.get(word, 0)

            #we create the feeding string for the model
            feed = {self.input_data: x, self.initial_state:state}

            #we ask the model to return predictions:
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            #you can play with this variable sampling_type to modify the way text is generate.
            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if word == '\n':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else: # sampling_type == 1 default:
                sample = weighted_pick(p)

            #restrieve the words by its index
            pred = words[sample]
            #generate the string
            ret += ' ' + pred
            #take last word to loop
            word = pred
        return ret
