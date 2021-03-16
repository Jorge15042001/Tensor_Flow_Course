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
print("Iris_data_set constains ", str(numbers_of_features) ," features")
numbers_of_labels = train_y.shape[1] # dimentionality of the output vector, aka: numbr of classes that can be predicted
print ("Iris_data_set have "+ str(numbers_of_labels) + " valid classes")


# using tensorflow 

# make the data set a tensorflow constant
tf_train_x = tf.constant ( train_x, dtype = 'float64' )
tf_train_y = tf.constant ( train_y, dtype = 'float64' )
tf_test_x = tf.constant ( test_x, dtype = 'float64' )
tf_test_y = tf.constant ( test_y, dtype = 'float64' )

# defintion of trainable variables 


model_weights = tf.Variable (tf.zeros([4, 3], dtype = 'float64'))
model_biases = tf.Variable (tf.zeros ([3], dtype = 'float64') ) 

#use 64 bits floats
#model_weights = tf.cast(model_weights, tf.float64)
#model_biases = tf.cast(model_biases, tf.float64)

# definition of logistic_regression function
@tf.function
def logistic_regression(input_vector):
    # first lutiply the input vector by the weights
    input_mult_weights = tf.matmul(input_vector, model_weights) 
    # add the biases 
    biases_added = tf.add(input_mult_weights, model_biases)
    # apply sigmoid function 
    sigmoid_activation = tf.nn.sigmoid(biases_added)

    return sigmoid_activation


# training variables 

number_of_epochs =  2000

learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = 0.008,
        decay_steps = tf_train_x.shape[0],
        decay_rate = 0.95,
        staircase = True)



#  Definiton of cost function
loss_object = tf.keras.losses.MeanSquaredLogarithmicError()

# Definition of thhe optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate)

# Defintiono of the function  for accuracy metrics 
@tf.function
def accuracy(predicted_result, expected_result ):
    # definiton of a boolean vector that determines which inputs are the same and which arent
    correct_prediction_vector = tf.equal(tf.argmax(predicted_result, 1), tf.argmax(expected_result, 1))

    # correct_prediction_vector is a boolean vector, so in order to compute the mean, it needs to be cast(tranform) to a nunmeric vector 
    return tf.reduce_mean(tf.cast(correct_prediction_vector, tf.float64))


@tf.function
def run_optimization(input_vector, expected_output):
    with tf.GradientTape() as tape :
        tape.watch(tf_train_x)
        tape.watch(tf_train_y)
        tape.watch(tf_test_x)
        tape.watch(tf_test_y)

        # evaluate the model with the input vector  to get a prediction
        prediction = logistic_regression(input_vector)
        
        # compute the error of the prediction
        loss = loss_object(expected_output, prediction)
        
    print ("LOSS:" ,loss)
    print ("MODEL WEIGHTS:" ,model_weights)
    print ("MODEL BIASES:" ,model_biases)

    # compute the gradient with respect to the weights and biases
    gradients = tape.gradient(loss, [model_weights, model_biases])
    print(gradients)
    #apply gradients
    optimizer.apply_gradients(zip(gradients, [model_weights, model_biases]))


    
# training the model

# reporting variables, usefull to vasualize how the model behaves

display_steps = 10 # after how many epochs, more data should be store

epoch_values = []
accuracy_values = [] # stores the accuracy of the model each display_steps
loss_values = [] # stores the loss measurements of the model each display_steps

loss = 0
diff = 1

# TRAINING LOOP
for i in range(number_of_epochs):
    if i > 1 and diff < 0.0001:
        print("TRAING TERMINADED DUE TO SMALL CHANGE IN DIFF")
        break
    # for each iteration
    run_optimization(tf_train_x,tf_train_y)
    if i % display_steps == 0: # if this epoch must present results
        epoch_values.append(i)# add the epoch number to the lsit

        # run the tests
        prediction = logistic_regression(tf_test_x) # compute the prediction with the test data

        new_loss = loss_object(tf_test_y, prediction) # compute the new values for the loss

        accuracy_value = accuracy(prediction, tf_test_y) # compute the accuracy of the model at its current state

        # append the values to the mettrics 
        loss_values.append(new_loss)
        accuracy_values.append(accuracy_value)

        diff = abs(loss - new_loss)
        loss = new_loss

        #present the data 
        print ("step %d, training accuracy %g, loss %g, change in loss %g"%(i, accuracy_value, new_loss, diff))


        

















