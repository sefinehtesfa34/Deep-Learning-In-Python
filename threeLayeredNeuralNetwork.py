from tensorflow import keras
from tensorflow.keras import layers

# YOUR CODE HERE
model = keras.Sequential([layers.Dense(units=512,activation='relu',input_shape=(8,)),
                        layers.Dense(units=512,activation="relu"),
                        layers.Dense(units=512,activation="relu"),
                        layers.Dense(units=1)])