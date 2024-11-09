# How to Create a Keras-like Model with keras.layers.RNN Supporting Extension Types

## Introduction

In TensorFlow, the `keras.layers.RNN` layer allows us to create custom RNN models. In this example, we will demonstrate how to create a Keras-like model using `keras.layers.RNN` that supports extension types, both for inputs and states.

### What are Extension Types?

Extension types in TensorFlow are custom data types that extend the functionality of the standard TensorFlow types. These are often used for special cases, such as when dealing with sequences or higher-dimensional states.

## Creating the Model

To create a Keras-like model using `keras.layers.RNN` that supports extension types, you can follow these steps:

1. **Define the RNN Cell**: First, define a custom RNN cell that supports extension types.

    ```python
    import tensorflow as tf

    class CustomRNNCell(tf.keras.layers.Layer):
        def __init__(self, units):
            super(CustomRNNCell, self).__init__()
            self.units = units
            self.state_size = units

        def build(self, input_shape):
            self.kernel = self.add_weight("kernel", shape=(input_shape[-1], self.units))
            self.bias = self.add_weight("bias", shape=(self.units,))
        
        def call(self, inputs, states):
            output = tf.matmul(inputs, self.kernel) + self.bias
            return output, output
    ```

2. **Create the RNN Layer**: Use `keras.layers.RNN` to create the RNN layer with the custom cell.

    ```python
    # Create an RNN layer using the custom cell
    rnn_layer = tf.keras.layers.RNN(CustomRNNCell(units=64))
    ```

3. **Build the Model**: Create a Keras model using the `RNN` layer.

    ```python
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(None, 10)),  # Example input shape
        rnn_layer
    ])
    model.summary()
    ```

4. **Compile and Train the Model**: Now, compile and train the model as usual.

    ```python
    model.compile(optimizer='adam', loss='mse')
    # Example random data
    x_train = tf.random.normal([100, 10, 10])  # 100 samples, 10 timesteps, 10 features
    y_train = tf.random.normal([100, 64])      # 100 samples, 64 output units
    model.fit(x_train, y_train, epochs=10)
    ```

## Conclusion

In this guide, we have shown how to build a Keras-like model using `keras.layers.RNN` with a custom cell that supports extension types for both inputs and states. This provides greater flexibility for custom RNN designs and integration with TensorFlow's extended types.
