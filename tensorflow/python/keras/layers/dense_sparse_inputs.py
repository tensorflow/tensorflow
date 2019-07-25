"""Dense layers do not support sparse inputs in version 1.13.1

This file propose an implementation of the Dense layer to ingest both sparse and dense tensors.

.. moduleauthor:: Sharone DAYAN <sharone-dayan@hotmail.com>
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class DenseLayerForSparse(tf.keras.layers.Layer):
    def __init__(self, vocabulary_size, num_units, activation, **kwargs):
        super(DenseLayerForSparse, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.num_units = num_units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_variable(
            "kernel", shape=[self.vocabulary_size, self.num_units]
        )
        self.bias = self.add_variable("bias", shape=[self.num_units])

    def call(self, inputs, **kwargs):
        if isinstance(inputs, tf.SparseTensor):
            outputs = tf.add(tf.sparse.matmul(inputs, self.kernel), self.bias)
        if not isinstance(inputs, tf.SparseTensor):
            outputs = tf.add(tf.matmul(inputs, self.kernel), self.bias)
        return self.activation(outputs)

    def compute_output_shape(self, input_shape):
        input_shape = input_shape.get_shape().as_list()
        return input_shape[0], self.num_units
