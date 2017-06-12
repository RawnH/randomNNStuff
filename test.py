import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

from tensorflow.contrib import rnn
#from EURNN import EURNNCell

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

def generate_data(num_of_points):
 
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

        labels.append(get_pixel_rep(first_num + sec_num))
        first_reps.append(first_rep)
        sec_reps.append(sec_rep)

    labels     = np.array(labels)
    first_reps = np.array(first_reps)
    sec_reps   = np.array(sec_reps)

    data = np.concatenate((first_reps, pluses, sec_reps, equals), axis = 2)
    

    return data, labels


#Set up hyper-params
learning_rate = 0.01
epochs        = 10000
batch_size    = 128
 
#Set up neural-net parameters
n_input   = 5    #rows in image
n_steps   = 32   #columns to read in image
n_hidden  = 128  #hidden neurons
n_output  = 5    #numbered output of NN 
n_classes = 12

#input for graph
x = tf.placeholder("tf.int32", [None, n_steps,  n_input])
y = tf.placeholder("tf.int32", [None, n_output, n_classes])


def RNN(x, weights, biases, model = "RNN", capacity = 2, FFT = False, comp = False):
    
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
