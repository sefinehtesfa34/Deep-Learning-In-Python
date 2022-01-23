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
<br>
# Stochastic Gradient Descent
## Introduction
In the first two lessons, we learned how to build fully-connected networks out of stacks of dense layers. When first created, all of the network's weights are set randomly -- the network doesn't "know" anything yet. In this lesson we're going to see how to train a neural network; we're going to see how neural networks learn.

As with all machine learning tasks, we begin with a set of training data. Each example in the training data consists of some features (the inputs) together with an expected target (the output). Training the network means adjusting its weights in such a way that it can transform the features into the target. In the 80 Cereals dataset, for instance, we want a network that can take each cereal's 'sugar', 'fiber', and 'protein' content and produce a prediction for that cereal's 'calories'. If we can successfully train a network to do that, its weights must represent in some way the relationship between those features and that target as expressed in the training data.
<br>
<h1>In addition to the training data, we need two more things:</h1>
<br>
A <strong><em>"loss function"</em></strong> that measures how good the network's predictions are.<br>
An <strong><em>"optimizer"</em></strong> that can tell the network how to change its weights.<br>
The Loss Function
We've seen how to design an architecture for a network, but we haven't seen how to tell a network what problem to solve. This is the job of the loss function.
<br>
The loss function measures the disparity between the the target's true value and the value the model predicts.
<br>
Different problems call for different loss functions. We have been looking at regression problems, where the task is to predict some numerical value -- calories in 80 Cereals, rating in Red Wine Quality. Other regression tasks might be predicting the price of a house or the fuel efficiency of a car.
<br>
A common loss function for regression problems is the mean absolute error or MAE. For each prediction y_pred, MAE measures the disparity from the true target y_true by an absolute difference abs(y_true - y_pred).

##### The total MAE loss on a dataset is the mean of all these absolute differences.
# N.B 
<small>The</small> <em>loss function</em> <small>measures the disparity between the the target's true value and the value the model predicts.</small>