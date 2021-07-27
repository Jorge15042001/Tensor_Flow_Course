# TensorFlow basics

![https://youtu.be/I5mCd8Yb7Ss](lecture)

# What is

* Tensor flow its a mathematical framework created by Google Brain to deal with heavy numerical computation
* It provides a c++ and a python interface
* The fastest compilation times
* supports CPU ans GPU acceleration

# How it works?

It uses a data FlowGraph that is composed by 2 units

* **nodes** operations
  * defines the operations to be executed
  * they get a tensor transform it and return another tensor that will be send through the output edge
* **edges** tensor
  * each tensor is a multi-dimensional array 
  * it travels through the nodes being transform at each one


A tensor could be a:

Name | dimensionality
--------------------
Value | 0 dimension
Vector | 1 dimension
Matrix | 2 dimensions
N-dimensional-array | N dimensions

# Computational Graph ingredients

Name | Purpose
---
tf.placeholder() | inputs to the graph, like parameters in a function
tf.Variable() | everything that needs to be referenced or persists

# Why deep learning with TensorFlow

* Tensor flow allows us to run applications in any type of device
* Built in support for neural networks and deep learning
* Deep learning will benefit from auto-differentiation and optimizer
