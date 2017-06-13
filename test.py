import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import os

from tensorflow.contrib import rnn
from EURNN import EURNNCell


os.environ["CUDA_VISIBLE_DEVICES"]=""

def get_pixel_rep(n):   
    
    number_dict = {
                    0 : np.array([[1,1,1], [1,0,1], [1,0,1], [1,0,1], [1,1,1]]),
                    1 : np.array([[0,1,0], [0,1,0], [0,1,0], [0,1,0], [0,1,0]]),
                    2 : np.array([[1,1,1], [0,0,1], [1,1,1], [1,0,0], [1,1,1]]),
                    3 : np.array([[1,1,1], [0,0,1], [1,1,1], [0,0,1], [1,1,1]]),
                    4 : np.array([[1,0,1], [1,0,1], [1,1,1], [0,0,1], [0,0,1]]),
                    5 : np.array([[1,1,1], [1,0,0], [1,1,1], [0,0,1], [1,1,1]]),
                    6 : np.array([[1,1,1], [1,0,0], [1,1,1], [1,0,1], [1,1,1]]),
                    7 : np.array([[1,1,1], [0,0,1], [0,0,1], [0,0,1], [0,0,1]]),
                    8 : np.array([[1,1,1], [1,0,1], [1,1,1], [1,0,1], [1,1,1]]),
                    9 : np.array([[1,1,1], [1,0,1], [1,1,1], [0,0,1], [0,0,1]]),
                   11 : np.array([[0,0,0], [0,1,0], [1,1,1], [0,1,0], [0,0,0]]), #plus sign
                   12 : np.array([[0,0,0], [1,1,1], [0,0,0], [1,1,1], [0,0,0]])  #equal sign
                  }
    
    def num_gen():
        spacing = np.array([ [0] for i in range(5) ])    
        symbols = {'+' : 11, '=' : 12}
        
        for char in str(n):
            if char in symbols:
                yield number_dict[symbols.get(char)]
            else:
                yield number_dict[int(char)]
            
            yield spacing
     
    return np.concatenate(list(num_gen()), axis = 1)


"""
    Produces a random n digit number capped at (10^n-1)//2
"""
def random_with_N_digits(n):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return random.randint(range_start, range_end//2)


def random_data(num_of_points):
 
    pluses = np.array([ get_pixel_rep("+") ]  * num_of_points)
    equals = np.array([ get_pixel_rep("=") ] * num_of_points)

    first_reps = []
    sec_reps   = []
    labels     = []

    for i in range(num_of_points):
        first_num = random_with_N_digits(3)
        first_rep = get_pixel_rep(first_num)
        sec_num   = random_with_N_digits(3)
        sec_rep   = get_pixel_rep(sec_num)

        labels.append(first_num + sec_num)
        first_reps.append(first_rep)
        sec_reps.append(sec_rep)

    labels     = np.array(labels)
    first_reps = np.array(first_reps)
    sec_reps   = np.array(sec_reps)

    data = np.concatenate((first_reps, pluses, sec_reps, equals), axis = 2).astype(np.float32)
    

    return data, labels

def data_gen(start, end):
    
    char_width = 4
    
    digit_length = len(str(end))
    im_height = 5
    im_width  = char_width * (digit_length * 2 + 2) #2 accounts for equal and plus signs
    
    length = (end - start)**2
    curr_loc = 0
    data = np.empty((length, im_height, im_width))
    labels = np.empty(length)
    
    new_index = np.random.permutation(length)
    
    for first_num in range(start, end):
        for sec_num in range(start, end):
            data[new_index[curr_loc]] = get_pixel_rep(str(first_num) + "+" + str(sec_num) + "=" )
            labels[new_index[curr_loc]] = first_num + sec_num          
            curr_loc += 1
    
    return data, labels

def split_into_train_and_test(data, labels, percent_train = 0.6):
    
    train_data_num = int(data.shape[0]*percent_train)
    
    train_data = data[:train_data_num]
    train_labels = labels[:train_data_num]
    
    test_data  = data[train_data_num:]
    test_labels = labels[train_data_num:]
    
    return train_data, train_labels, test_data, test_labels


def gen_batches(data, labels, batch_size):
    
    indices = random.sample(range(data.shape[0]), batch_size)
    
    data_batch = data[indices, :]
    label_batch = np.take(labels, indices)
    
    return data_batch, label_batch


#Set up hyper-params
learning_rate = 0.01
iters         = 6000
batch_size    = 512
 
#Set up neural-net parameters
n_input   = 5    #rows in image
n_steps   = 32   #columns to read in image
n_hidden  = 128  #hidden neurons
n_classes = 1998 

init_val = np.sqrt(6.)/np.sqrt(n_classes * 2)

#input for graph
x = tf.placeholder("float", [None, 5, 32])
y = tf.placeholder("int64", [None])


def RNN(x, model = "RNN", capacity = 2, FFT = False, comp = False):
    
    #Choose cell and assign output and state   
    if model == "LSTM":
        cell = rnn.BasicLSTMCell(n_hidden, state_is_tuple=True, forget_bias=1)
        outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    elif model == "RNN":
        cell = rnn.BasicRNNCell(n_hidden)
        outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    elif model == "EURNN":
        cell = EURNNCell(n_hidden, capacity, FFT, comp)
        if comp:
            comp_outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.complex64)
            outputs = tf.real(comp_outputs)
        else:
            outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    with tf.variable_scope("params"):
        weights = tf.get_variable("weights", shape = [n_hidden, n_classes], \
                    dtype=tf.float32, initializer=tf.random_uniform_initializer(-init_val, init_val))
        
        biases = tf.get_variable("biases", shape=[n_classes], \
                 dtype=tf.float32, initializer=tf.constant_initializer(1) )
        
    output_list = tf.unstack(outputs, axis=1)
    last_out = output_list[-1]
    weight_prod = tf.matmul(last_out, weights)
    return tf.nn.bias_add(weight_prod, biases)

rnn_out = RNN(x)

# --- evaluate process ----------------------
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rnn_out, labels=y))
correct_pred = tf.equal(tf.argmax(rnn_out, 1), y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# --- Initialization ----------------------
optimizer = tf.train.RMSPropOptimizer(learning_rate = learning_rate, decay = 0.9).minimize(cost)
init = tf.global_variables_initializer()

print("Preparing data...")
data, labels = data_gen(100, 999)
train_data, train_labels, test_data, test_labels = split_into_train_and_test(data, labels)
print("Data prepared.")

with tf.Session() as sess:
    sess.run(init)
    epochs = []
    errors = []
    accs = []
    
    for i in range(iters):     
        batch_X, batch_Y = gen_batches(train_data, train_labels, batch_size)
        sess.run(optimizer, feed_dict={x: batch_X, y: batch_Y})
        error = sess.run(cost, feed_dict={x: batch_X, y: batch_Y})
        acc = sess.run(accuracy, feed_dict={x: batch_X, y: batch_Y})
        accs.append(acc)
        
        
        print("Epoch number: " + str(i) + ", Error = " + "{:.6f}".format(error), \
              "Accuracy = " + "{:.6f}".format(acc))
        
        epochs.append(i)
        errors.append(error)
    
    print("done!")
    
    plt.plot(epochs, errors)
    plt.xlabel = "epochs"
    plt.ylabel = "error"
    plt.show()
    
    plt.plot(epochs, accs)
    plt.xlabel = "epochs"
    plt.ylabel = "accuracy"
    plt.show() 
    
    sess.run(optimizer, feed_dict={x: test_data, y: test_labels})
    test_acc = sess.run(accuracy, feed_dict={x: test_data, y: test_labels})
    test_loss = sess.run(cost, feed_dict={x: test_data, y: test_labels})
    print("Test result: Loss= " + "{:.6f}".format(test_loss) + \
          ", Accuracy= " + "{:.5f}".format(test_acc))




    
