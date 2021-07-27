# TensorFlow 2.x i and Eager-Execution

![Lecture](https://youtu.be/oHL-Y4frOPs)

# TensorFlow 2.x main features

* It offers some key new capabilities
* Keras is the default high-level API for TensorFlow, which is known for it ease of use
* Includes performance optimization and GPU enhancements
* it includes  Eager-Execution mode and its active by default eve at the low Level API

# Eager-Execution mode

Without Eager-Execution | With Eager-Execution
------------------------|--------------------
The code is not run automatically | Code is executed line by line, like normal python code
Intermediate results aren't accessible | you can access every variable along the code, useful for debugging
The FlowGraph needs to be compile and then executed | The code is run line by line
