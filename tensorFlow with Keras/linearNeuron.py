#Linear Neural Network
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

model = keras.Sequential(layers.Dense(units=1,input_shape=(11,)))
w, b = model.weights
print(w.shape,b.shape)
print(w[:])
#plot the untrained dataset
model = keras.Sequential([
    layers.Dense(1, input_shape=[1]),
])

x = tf.linspace(-1.0, 1.0, 100)
y = model.predict(x)

plt.figure(dpi=100)
plt.plot(x, y, 'k')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("Input: x")
plt.ylabel("Target y")
w, b = model.weights # you could also use model.get_weights() here
plt.title("Weight: {:0.2f}\nBias: {:0.2f}".format(w[0][0], b[0]))
plt.show()