#!/usr/bin/env python3

"""
Minimal reproduction of the GitHub issue #105131.
This is the exact code from the issue report.
"""

import tensorflow as tf

class TestModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(32, activation='relu')
        self.d2 = tf.keras.layers.Dense(16, activation='tanh')
        self.d3 = tf.keras.layers.Dense(8)

    def call(self, x):
        filtered_features = list(filter(lambda z: tf.reduce_sum(z) > 0.5, [x, x * 2, x * 3]))
        mapped_features = list(map(lambda z: tf.nn.sigmoid(z), filtered_features))
        zipped_data = list(zip(mapped_features, [tf.ones_like(x) for _ in range(len(mapped_features))]))
        combined = tf.concat(zipped_data, axis=-1)
        return self.d3(combined)

def get_default_model():
    return TestModel()

def get_sample_inputs():
    x = tf.random.normal([4, 16])
    return (x,)

def main():
    model = get_default_model()
    inputs = get_sample_inputs()
    eager_out = model(*inputs)
    print('Eager Input shape:', inputs[0].shape)
    print('Eager Output shape:', eager_out.shape)
    
    @tf.function(jit_compile=True)
    def compiled_forward(*args):
        return model(*args)
    
    compiled_out = compiled_forward(*inputs)
    print('XLA Output shape:', compiled_out.shape)

if __name__ == '__main__':
    main()