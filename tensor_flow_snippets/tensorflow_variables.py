'''tensorflow variables'''
import tensorflow as tf

#definin variables

variable1= tf.Variable(1)
variable_vector=tf.Variable([0,1,0])

@tf.function
def add_one(value):
    return tf.add(value,1)
@tf.function
def dot_product(value):
    return tf.multiply(value,value)

variable1=add_one(variable1)
variable_vector=dot_product(variable_vector)

print(variable1.numpy())
print(variable_vector.numpy())
