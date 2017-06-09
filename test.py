import tensorflow as tf
import numpy as np
import random


def generate_data(num_of_points):

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

	num_digits = 3

	pluses = np.array([number_dict[11]] * num_of_points )
	equals = np.array([number_dict[12]] * num_of_points)


	first_reps = []
	sec_reps   = []
	labels     = []

	for i in range(num_of_points):
		first_num, first_rep = make_n_digit_number(num_digits, number_dict)
		sec_num,   sec_rep   = make_n_digit_number(num_digits, number_dict)

		labels.append(first_num + sec_num)
		first_reps.append(first_rep)
		sec_reps.append(sec_rep)

	labels     = np.array(labels)
	first_reps = np.array(first_reps)
	sec_reps   = np.array(sec_reps)

	print(first_reps)
	data = np.concatenate((first_reps, pluses, sec_reps), axis = 2)

	return data, labels


def make_n_digit_number(n, number_dict):
	power = n - 1
	number = 0
	seq = []

	while(power >= 0):
		if power == n - 1:
			digit = random.randint(1,9)
		else:
			digit = random.randint(0,9)

		number += digit * 10**power
		power -= 1
		seq.append(number_dict[digit])

	return number, np.concatenate(seq, axis = 1)


