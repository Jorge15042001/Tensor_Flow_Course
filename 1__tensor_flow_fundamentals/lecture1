There are two ways of manipulating tensor with in python

1 defining a tensorflow function using @tf.function after a normal function definition

@tf.function
def add(a,b): #normal function definition
    return tf.add(a,b)


2 using normal python sintax and letting tensorflow flow api to everithing required to run the code

a= tf.constant([0,1,2,3])#defining a tensor
b= tf.constant([0,1,2,3])#defining another tensor

#manipulating tensor using normal python sintax
c=a+b # tensorflow knows how to handle this 


