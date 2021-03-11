'''tenor multiplication '''
import tensorflow as tf

identity_matrix=tf.constant([[1,0,0],
                             [0,1,0],
                             [0,0,1]])
all_1= tf.constant([[1,1,1],
                    [1,1,1],
                    [1,1,1]])
result=tf.matmul(identity_matrix,all_1)
print(result.numpy())
