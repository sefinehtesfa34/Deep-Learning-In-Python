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
A common loss function for regression problems is <h1><em>the mean absolute error or MAE</em></h1>. For each prediction y_pred, MAE measures the disparity from the true target y_true by an absolute difference abs(y_true - y_pred).

##### The total MAE loss on a dataset is the mean of all these absolute differences.
# N.B 
<small>The</small> <h1><em>loss function</em> </h1><small>measures the disparity between the the target's true value and the value the model predicts.</small>


# <i>The Optimizer - Stochastic Gradient Descent</i><br>
We've described the problem we want the network to solve, but now we need to say how to solve it. This is the job of the optimizer. The optimizer is an algorithm that adjusts the weights to minimize the loss.<br>

## One step of training goes like this:

### 1. Sample some training data and run it through the network to make predictions.<br>
### 2. Measure the loss between the predictions and the true values.<br>
### 3. Finally, adjust the weights in a direction that makes the loss smaller.<br>


# Learning Rate and Batch Size
<br>
Notice that the line only makes a small shift in the direction of each batch (instead of moving all the way). The size of <br>these shifts is determined by <strong> the learning rate.</strong> A smaller learning rate means the network needs to see more minibatches<br> before its weights converge to their best values.<br>
<br>
The learning rate and the size of the minibatches are the two parameters that have the largest effect on how the SGD training proceeds.<br> Their interaction is often subtle and the right choice for these parameters isn't always obvious. <br>
(We'll explore these effects in the exercise.)
<br>
Fortunately, for most work it won't be necessary to do an extensive hyperparameter search to get satisfactory results. Adam is an SGD algorithm that has an adaptive learning rate that makes it suitable for most problems without any parameter tuning (it is "self tuning", in a sense). Adam is a great general-purpose optimizer.
<br>
<h1>Bear in mind!</h1>
<strong>The learning rate and the size of the minibatches are the two parameters that have the largest effect on how the SGD training proceeds.</strong>

# 2) Train Model

Once you've defined the model and compiled it with a loss and optimizer you're ready for training. Train the network for 200 epochs with a batch size of 128. The input data is `X` with target `y`.
<br>
With the learning rate and the batch size, you have some control over:
- How long it takes to train a model
- How noisy the learning curves are
- How small the loss becomes

To get a better understanding of these two parameters, we'll look at the linear model, our ppsimplest neural network. Having only a single weight and a bias, it's easier to see what effect a change of parameter has.

The next cell will generate an animation like the one in the tutorial. Change the values for `learning_rate`, `batch_size`, and `num_examples` (how many data points) and then run the cell. (It may take a moment or two.) Try the following combinations, or try some of your own:

| `learning_rate` | `batch_size` | `num_examples` |
|-----------------|--------------|----------------|
| 0.05            | 32           | 256            |
| 0.05            | 2            | 256            |
| 0.05            | 128          | 256            |
| 0.02            | 32           | 256            |
| 0.2             | 32           | 256            |
| 1.0             | 32           | 256            |
| 0.9             | 4096         | 8192           |
| 0.99            | 4096         | 8192           |








