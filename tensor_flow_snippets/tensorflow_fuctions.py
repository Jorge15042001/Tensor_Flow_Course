'''using tensorflow function definition'''
import tensorflow as tf
one=tf.constant(1, name="one")
two= tf.constant(2,name="two")
fib0_5=tf.constant([0,1,1,2,3,5],name="Fib0_5")
square=tf.constant([0,1,4,9,16,25],name="Squeare0_5")


@tf.function
def add(a,b):
    """this function adds 2 tensors"""
    c=tf.add(a,b)
    print("result: ",c)
    return c
print("\n\n\nComputind add(one,two)\n")
result1=add(one,two)
print(result1.numpy(),"\nDone computing ")
print("\n\n\nComputind add(fib0_5,square)\n")
result2=add(fib0_5,square)
print(result2.numpy(),"\nDone computing ")
