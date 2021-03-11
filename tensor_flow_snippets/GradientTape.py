import tensorflow as tf

a=tf.constant(1.0) #constants are not being what by default
b=tf.Variable(2.0) #vaiable are being whatch

@tf.function
def f_sin(x):
    return tf.sin(x)
@tf.function
def f_cos(x):
    return tf.cos(x)

with tf.GradientTape(persistent=False,watch_accessed_variables=True) as tape:
    #persistent allows tape to persist when resources are liberaded, so it can be call outsite this scope, allows tape.gradient to be call for mutlple  derivatives
    # watch_accessed_variables if set to False, no variables are being whatch
    
    tape.watch(a)#if a constant needs to be whatch it must be manully set for that, other wise it wont compute dervatives

    

    y=f_sin(a)
    y2=f_cos(f_sin(a))

    d_f_sin_dx=tape.gradient(y2,a)
    
    print(d_f_sin_dx)
    




