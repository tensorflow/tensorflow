import tensorflow as tf
import numpy as np

# Define training data
trainDataForPrediction = np.array([[[0.28358589],
  [0.30372512],
  [0.29780091],
  [0.33183642],
  [0.33120391],
  [0.33995099]],

 [[0.66235955],
  [0.35913154],
  [0.44153985],
  [0.32184616],
  [0.36265909],
  [0.3549683 ]],

 # (Other data points omitted for brevity)
 
 [[0.7357174 ],
  [0.82513636],
  [0.92903864],
  [0.83082154],
  [0.71830423],
  [0.68545151]]])

trainDataTrueValues = np.array([[0.33370854, 0.32896128, 0.338919, 0.370148, 0.41977692, 0.5521488],
 [0.365207, 0.37061936, 0.37484066, 0.3478887, 0.32885199, 0.30680109],
 
 # (Other data points omitted for brevity)

 [0.67259377, 0.65934765, 0.64005251, 0.56716475, 0.41110739, 0.3281523]])

# Ensure the data types are float32
trainDataForPrediction = trainDataForPrediction.astype(np.float32)
trainDataTrueValues = trainDataTrueValues.astype(np.float32)

# Function to create the neural network
def createNeuralNetwork(hidden_units=9, dense_units=6, input_shape=(6, 1), activation=['relu', 'sigmoid']):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(hidden_units, input_shape=input_shape, activation=activation[0]))
    model.add(tf.keras.layers.Dense(units=dense_units, activation=activation[1]))
    model.add(tf.keras.layers.Dense(units=dense_units, activation=activation[1]))
    model.add(tf.keras.layers.Dense(units=dense_units, activation=activation[1]))
    
    # Compile the model
    model.compile(loss='mse', metrics=['mae', tf.keras.metrics.RootMeanSquaredError(), 'mse', tf.keras.metrics.R2Score()], optimizer='adam')
    return model

# Create the model
model = createNeuralNetwork()

# Fit the model to the training data
history = model.fit(trainDataForPrediction, trainDataTrueValues, epochs=300, batch_size=1, verbose=0)

# Print the model summary
model.summary()
