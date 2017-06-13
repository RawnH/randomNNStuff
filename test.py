import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse
import os

from tensorflow.contrib import rnn
from EURNN import EURNNCell


os.environ["CUDA_VISIBLE_DEVICES"]="" #not enough gpu memory to test

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

def save_plot(filename, title, x_label, y_label, x_data, y_data):
    
    
    plt.plot(x_data, y_data)
    plt.xlabel = x_label
    plt.ylabel = y_label
    plt.title(title)   
    plt.tight_layout()
    plt.savefig(filename, format = "png")
    plt.clf()
    

def main(model_arg = "RNN", learning_rate = 0.01, iters = 3000, batch_size = 256, n_hidden = 128, cap_arg = 2, FFT_arg = False, comp_arg = False):
      
    #Set up neural-net parameters
    n_steps   = 5     #rows in image
    n_input   = 32    #columns to read in image
    n_classes = 2000  #possible outputs
    
    #init_val = np.sqrt(6.)/np.sqrt(n_classes * 2)
    
    #input for graph
    x = tf.placeholder("float", [None, n_steps, n_input])
    y = tf.placeholder("int64", [None])

    def RNN(x, model = model_arg, capacity = cap_arg, FFT = FFT_arg, comp = comp_arg):
        
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
                        dtype=tf.float32, initializer=tf.random_uniform_initializer(1, 2))
            
            biases = tf.get_variable("biases", shape=[n_classes], \
                     dtype=tf.float32, initializer=tf.constant_initializer(1) )
            
        output_list = tf.unstack(outputs, axis=1)
        last_out = output_list[-1]
        weight_prod = tf.matmul(last_out, weights)
        return tf.nn.bias_add(weight_prod, biases)

    model_description = "{} LR = {}, iters = {}, batch_size = {}, n_hidden = {}, capacity = {} FFT = {}, complex = {}"\
                        .format(model_arg, learning_rate, iters, batch_size, n_hidden, cap_arg, FFT_arg, comp_arg)

    title             = "{} LR = {}, iters = {}, batch_size = {}, n_hidden = {}"\
                        .format(model_arg, learning_rate, iters, batch_size, n_hidden)
    
    rnn_out = RNN(x)
    
    # --- evaluate process ----------------------
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rnn_out, labels=y))
    correct_pred = tf.equal(tf.argmax(rnn_out, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    
    # --- Initialization ----------------------
    optimizer = tf.train.RMSPropOptimizer(learning_rate = learning_rate, decay = 0.9).minimize(cost)
    init = tf.global_variables_initializer()
    
    print("Preparing data...")
    data, labels = data_gen(100, 200)
    train_data, train_labels, test_data, test_labels = split_into_train_and_test(data, labels)
    
    epochs = []
    errors = []
    accs = []
    
    with tf.Session() as sess:
        sess.run(init)
        
        for i in range(iters):     
            batch_X, batch_Y = gen_batches(train_data, train_labels, batch_size)
            sess.run(optimizer, feed_dict={x: batch_X, y: batch_Y})
            error = sess.run(cost, feed_dict={x: batch_X, y: batch_Y})
            acc = sess.run(accuracy, feed_dict={x: batch_X, y: batch_Y})
            accs.append(acc)
            
            
            #print("Epoch number: " + str(i) + ", Error = " + "{:.6f}".format(error), \
            #     "Accuracy = " + "{:.6f}".format(acc))
            
            epochs.append(i)
            errors.append(error)
        
        print("done!")
        
    
        save_plot("Error " + model_description, title, "Epoch Number", "Error", epochs, errors)
        save_plot("Accuracy" + model_description, title, "Epoch Number", "Accuracy", epochs, accs)

        
        sess.run(optimizer, feed_dict={x: test_data, y: test_labels})
        test_acc = sess.run(accuracy, feed_dict={x: test_data, y: test_labels})
        test_loss = sess.run(cost, feed_dict={x: test_data, y: test_labels})
        
        print(model_description)
        print("Test result: Loss= " + "{:.6f}".format(test_loss) + \
              ", Accuracy= " + "{:.5f}".format(test_acc))


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Addition task")
    parser.add_argument("model", default='LSTM', help="Model name: LSTM, EURNN")
    parser.add_argument('--n_iter', '-I', type=int, default=300, help='training iteration number')
    parser.add_argument('--learning_rate', '-N', type = float, default=0.01, help = 'learning rate of network')
    parser.add_argument('--n_batch', '-B', type=int, default=128, help='batch size')
    parser.add_argument('--n_hidden', '-H', type=int, default=128, help='hidden layer size')
    parser.add_argument('--capacity', '-L', type=int, default=2, help='Tunable style capacity, only for EURNN, default value is 2')
    parser.add_argument('--comp', '-C', type=str, default="False", help='Complex domain or Real domain. Default is False: real domain')
    parser.add_argument('--FFT', '-F', type=str, default="False", help='FFT style, only for EURNN, default is False')

    args = parser.parse_args()
    arg_dict = vars(args)

    for i in arg_dict:
        if (arg_dict[i]=="False"):
            arg_dict[i] = False
        elif arg_dict[i]=="True":
            arg_dict[i] = True
        
    kwargs = {    
                'model_arg': arg_dict['model'],
                'learning_rate': arg_dict['learning_rate'],
                'iters': arg_dict['n_iter'],
                  'batch_size': arg_dict['n_batch'],
                  'n_hidden': arg_dict['n_hidden'],
                  'cap_arg': arg_dict['capacity'],
                  'comp_arg': arg_dict['comp'],
                  'FFT_arg': arg_dict['FFT'],
            }

    main(**kwargs)



    
