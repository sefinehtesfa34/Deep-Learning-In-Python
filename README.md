# Deep Learning with python
# The main focus here is TensorFlow and Keras API for Neural Network
# The Activation Function
It turns out, however, that two dense layers with nothing in between are no better than a single dense layer by itself. 
Dense layers by themselves can never move us out of the world of lines and planes. 
What we need is something nonlinear. What we need are activation functions.
ithout activation functions, neural networks can only learn linear relationships. 
In order to fit curves, we'll need to use activation functions.
An activation function is simply some function we apply to each of a layer's outputs (its activations). 
The most common is the rectifier function  max(0,x) .