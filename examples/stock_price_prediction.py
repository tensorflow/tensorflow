import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np

# Example data (Replace with actual data)
data = np.random.rand(100, 1)

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(1, 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(data, data, epochs=10, batch_size=1)

# Predict stock prices
predicted_price = model.predict(data)
print(predicted_price)
