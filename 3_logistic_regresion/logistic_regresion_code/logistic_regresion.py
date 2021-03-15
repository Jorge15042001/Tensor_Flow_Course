import tensorflow as tf
import numpy as np
import pandas as pd
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#load de dataset
iris_datset = load_iris()#iris is a dictionary, that have data and traget as keys

print ( type ( iris_datset ) )
iris_imput_data = iris_datset.data # take the input data

iris_output_data = iris_datset.target # take the labels 
#print(iris_imput_data)
iris_output_data = pd.get_dummies( iris_output_data ) # separates the data into 3 columns one for each class, puts 1 where the class was defined and 0 in the rest
# [1, 2, 1, 2, 3]-> becomes 
#[[1, 0, 1, 0, 0],
# [0, 1, 0, 1, 0],
# [0, 0, 0, 0, 1]]


train_x, test_x, train_y, test_y = train_test_split(
        iris_imput_data, 
        iris_output_data, 
        test_size= 0.33, 
        random_state= 1)

numbers_of_features = train_x.shape[1] # dimentionality of the input vector, aka: number of independent variables, aka: number of variables for each input
print("Iris_data_set constains ", numbers_of_features ," features")
numbers_of_labels = train_y.shape[1] # dimentionality of the output vector, aka: numbr of classes that can be predicted
print ("Iris_data_set have "+ numbers_of_labels + " valid classes")


# using tensorflow 

# make the data set a tensorflow constant
tf_train_x = tf.constant ( train_x, dtype = 'float32' )
tf_train_y = tf.constant ( train_y, dtype = 'float32' )
tf_test_x = tf.constant ( test_x, dtype = 'float32' )
tf_test_y = tf.constant ( test_y, dtype = 'float32' )

# defintion of trainable variables 

model_weights = tf.Variable (tf.zeros([4, 3]))
model_biases = tf.Variable (tf.zeros ([3])) 

# definition of logistic_regression function
@tf.function
def logistic_regression(input_vector):
    # first lutiply the input vector by the weights
    input_mult_weights = tf.matmul(input_vector, model_weights, name = "input_mult_weights")
    # add the biases 
    biases_added = tf.add(input_mult_weights, model_biases, name = "biases_added")
    # apply sigmoid function 
    sigmoid_activation = tf.nn.sigmoid(biases_added)

    return sigmoid_activation


# training variables 

number_of_epochs =  700

learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = 0.008,
        decay_steps = train_x.shape[0],
        decay_rate = 0.95,
        staircase = True)



# 
