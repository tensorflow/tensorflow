from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import random
import functools
import math

from tensorflow.python.keras.layers import dense_sparse_inputs


class CategoryPrediction(tf.keras.Model):
    def __init__(self, vocab_size, num_units_1, num_units_2):
        super(CategoryPrediction, self).__init__(name="CategoryPrediction")
        self.vocab_size = vocab_size
        self.dense_1 = dense_sparse_inputs.DenseLayerForSparse(vocab_size, num_units_1, activation="relu")
        self.dense_2 = tf.keras.layers.Dense(num_units_2, activation="sigmoid")

    def call(self, inputs):
        out_1 = self.dense_1(inputs)
        out_2 = self.dense_2(out_1)
        return out_2


# parameters
batch_size = 128
hidden_layer_kernel_size = 500
number_of_labels = 10
nb_examples = 5000
vocabulary_size = 2000
learning_rate = 0.001
max_epochs = 2

features = tf.sparse.SparseTensor(indices=np.column_stack(([random.randrange(0, nb_examples) for _ in range(10)],
                                                           [random.randrange(10, vocabulary_size) for _ in range(10)])),
                                  values=np.random.rand(10),
                                  dense_shape=np.array([nb_examples, vocabulary_size]))

labels = np.zeros((nb_examples, number_of_labels))
ind = np.random.randint(number_of_labels, size=nb_examples)
for i, j in enumerate(ind):
    labels[i, j] = 1
labels = tf.convert_to_tensor(labels)

model = CategoryPrediction(vocabulary_size, hidden_layer_kernel_size, number_of_labels)
model.compile(optimizer=tf.train.AdamOptimizer(learning_rate), loss="binary_crossentropy", metrics=["accuracy"])
model.fit(features, labels, epochs=max_epochs, steps_per_epoch=math.ceil(nb_examples / batch_size))
