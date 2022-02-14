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





# 1) Define Model #

The model we'll use this time will have both batch normalization and dropout layers. To ease reading we've broken the diagram into blocks, but you can define it layer by layer as usual.

Define a model with an architecture given by this diagram:

<figure style="padding: 1em;">
<img src="https://i.imgur.com/V04o59Z.png" width="400" alt="Diagram of network architecture: BatchNorm, Dense, BatchNorm, Dropout, Dense, BatchNorm, Dropout, Dense.">
<figcaption style="textalign: center; font-style: italic"><center>Diagram of a binary classifier.</center></figcaption>
</figure>


# The Receptive Field #

Trace back all the connections from some neuron and eventually you reach the input image. All of the input pixels a neuron is connected to is that neuron's **receptive field**. The receptive field just tells you which parts of the input image a neuron receives information from.

As we've seen, if your first layer is a convolution with $3 \times 3$ kernels, then each neuron in that layer gets input from a $3 \times 3$ patch of pixels (except maybe at the border).

What happens if you add another convolutional layer with $3 \times 3$ kernels? Consider this next illustration:

<figure>
<img src="https://i.imgur.com/HmwQm2S.png" alt="Illustration of the receptive field of two stacked convolutions." width=250>
</figure>

Now trace back the connections from the neuron at top and you can see that it's connected to a $5 \times 5$ patch of pixels in the input (the bottom layer): each neuron in the $3 \times 3$ patch in the middle layer is connected to a $3 \times 3$ input patch, but they overlap in a $5 \times 5$ patch. So that neuron at top has a $5 \times 5$ receptive field.
# Computer Vision

# Design a Convnet #

Let's design a convolutional network with a block architecture like we saw in the tutorial. The model from the example had three blocks, each with a single convolutional layer. Its performance on the "Car or Truck" problem was okay, but far from what the pretrained VGG16 could achieve. It might be that our simple network lacks the ability to extract sufficiently complex features. We could try improving the model either by adding more blocks or by adding convolutions to the blocks we have.

Let's go with the second approach. We'll keep the three block structure, but increase the number of `Conv2D` layer in the second block to two, and in the third block to three.

<figure>
<!-- <img src="./images/2-convmodel-2.png" width="250" alt="Diagram of a convolutional model."> -->
<img src="https://i.imgur.com/Vko6nCK.png" width="250" alt="Diagram of a convolutional model.">
</figure>

# 1) Define Model #

Given the diagram above, complete the model by defining the layers of the third block.

# 2) Compile #

To prepare for training, compile the model with an appropriate loss and accuracy metric for the "Car or Truck" dataset.
## Finally, let's test the performance of this new model. First run this cell to fit the model to the training set.
## And now run the cell below to plot the loss and metric curves for this training run.
# 3) Train the Model #

How would you interpret these training curves? Did this model improve upon the model from the tutorial?
### This would indicate that it was prone to overfitting and in need of some regularization. The additional layer in our new model would make it even more prone to overfitting. However, adding some regularization with the Dropout layer helped prevent this. These changes improved the validation accuracy of the model by several points.

# Conclusion
These exercises showed you how to design a custom convolutional network to solve a specific classification problem. Though most models these days will be built on top of a pretrained base, it certain circumstances a smaller custom convnet might still be preferable -- such as with a smaller or unusual dataset or when computing resources are very limited. As you saw here, for certain problems they can perform just as well as a pretrained model.


# Image Classifications in keras API
This short introduction uses [Keras](https://www.tensorflow.org/guide/keras/overview) to:

1. Load a prebuilt dataset.
1. Build a neural network machine learning model that classifies images.
2. Train this neural network.
3. Evaluate the accuracy of the model.



#!/usr/bin/env python
# coding: utf-8

# # Implementing the Gradient Descent Algorithm
# 
# In this notebook, you'll be implementing the functions that build the gradient descent algorithm, namely:
# 
# * `sigmoid`: The sigmoid activation function.
# * `output_formula`: The formula for the prediction.
# * `error_formula`: The formula for the error at a point.
# * `update_weights`: The function that updates the parameters with one gradient descent step.
# 
# Your goal is to find the boundary on a small dataset that has two classes:
# 
#
# 
# After you implement the gradient descent functions, be sure to run the `train` function. This will graph several of the lines that are drawn in successive gradient descent steps. It will also graph the error function, and you'll be able to see it decreasing as the number of epochs grows.
# 
# First, we'll start with some functions that will help us plot and visualize the data.

# In[2]:
# ## TODO: Implementing the basic functions
# Here is your turn to shine. Implement the following formulas, as explained in the text.
# - Sigmoid activation function
# 
# $$\sigma(x) = \frac{1}{1+e^{-x}}$$
# 
# - Output (prediction) formula
# 
# $$\hat{y} = \sigma(w_1 x_1 + w_2 x_2 + b)$$
# 
# - Error function
# 
# $$Error(y, \hat{y}) = - y \log(\hat{y}) - (1-y) \log(1-\hat{y})$$
# 
# - The function that updates the weights
# 
# $$ w_i \longrightarrow w_i + \alpha (y - \hat{y}) x_i$$
# 
# $$ b \longrightarrow b + \alpha (y - \hat{y})$$





















