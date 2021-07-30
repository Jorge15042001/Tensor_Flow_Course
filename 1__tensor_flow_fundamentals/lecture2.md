# Constants in TensorFlow

Like in any other programming language constants are values that can be read but not changed through the execution of the code

```python
import tensorflow as tf

matrrix = [[1,2,3],
           [4,5,6],
           [7,8,9]]

const_1 = tf.constant( [2]        , name = "constant_0" )
const_2 = tf.constant( [2,2,3,5]  , name = "constant_1" )
const_3 = tf.constant( matrrix    , name = "constant_2" )
const_4 = tf.constant( matrrix[0] , name = "constant_3" )

```

# Variables in TensorFlow

A TensorFlow variable is a tensor whose values are allowed to change, just like in any programming language

```python 
import tensorflow as tf

matrrix = [[1,2,3],
           [4,5,6],
           [7,8,9]]

var_1 = tf.variable( [2]        , name = "constant_0" )
var_2 = tf.variable( [2,2,3,5]  , name = "constant_1" )
var_3 = tf.variable( matrrix    , name = "constant_2" )
var_4 = tf.variable( matrrix[0] , name = "constant_3" )
```

