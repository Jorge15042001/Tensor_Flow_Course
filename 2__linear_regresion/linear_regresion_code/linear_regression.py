'''linear regresion'''
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import tensorflow as tf
import matplotlib.patches as mpatches



#Y=aX+b

#imput data
X=np.arange(0.0,5.0,0.1)

a=1
b=0
Y=a*X+b# a, b are the parameters that need to  be tuned

#unsing simple data set "FuelConsumption.csv"


df=pd.read_csv("FuelConsumption.csv")

print(df.head())

trainx=np.asanyarray(df[["ENGINESIZE"]])#input data for the linear regresion
trainy=np.asanyarray(df[["CO2EMISSIONS"]])# expected data

#variables

a=tf.Variable(20.0)#slope, set to a random initial value
b=tf.Variable(30.2)#intercept, set to a random initial value

function= lambda x:a*x+b#linear function

def loss_object(predicted,expected):
    '''computes the least square '''
    return tf.reduce_mean(tf.square(predicted-expected))

learning_rate=0.01
training_data=[]
loss_values=[]

epochs=200

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        predicted=function(trainx)
        loss_value=loss_object(predicted,trainy)
        loss_values.append(loss_value)

        gradients= tape.gradient(loss_value,[b,a])

        b.assign_sub(gradients[0]*learning_rate)
        a.assign_sub(gradients[1]*learning_rate)

        if epoch%5==0:
            training_data.append((a.numpy(), b.numpy()))
print(training_data)
plt.plot(loss_values,'ro')
plt.show()


