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
# The dataset
Cement	BlastFurnaceSlag	FlyAsh	Water	Superplasticizer	CoarseAggregate	FineAggregate	Age	CompressiveStrength
0	540.0	0.0	0.0	162.0	2.5	1040.0	676.0	28	79.99
1	540.0	0.0	0.0	162.0	2.5	1055.0	676.0	28	61.89
2	332.5	142.5	0.0	228.0	0.0	932.0	594.0	270	40.27
3	332.5	142.5	0.0	228.0	0.0	932.0	594.0	365	41.05
4	198.6	132.4	0.0	192.0	0.0	978.4	825.5	360	44.30