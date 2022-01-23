from tensorflow import keras
from tensorflow.keras import layers

# YOUR CODE HERE
model = keras.Sequential([layers.Dense(units=512,activation='relu',input_shape=(8,)),
                        layers.Dense(units=512,activation="relu"),
                        layers.Dense(units=512,activation="relu"),
                        layers.Dense(units=1)])

#After defining the model, we compile in the optimizer and loss function.
model.compile(
    optimizer='adam',
    loss='mae',
)
# Now we're ready to start the training! We've told Keras to feed the optimizer 256 rows of the training data at a time (the batch_size) 
# and to do that 10 times all the way through the dataset (the epochs).
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=10,
)
import pandas as pd

# convert the training history to a dataframe
history_df = pd.DataFrame(history.history)
# use Pandas native plot method
history_df['loss'].plot();


