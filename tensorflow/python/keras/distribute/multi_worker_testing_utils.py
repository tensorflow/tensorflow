# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities for testing multi-worker distribution strategies with Keras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops


def mnist_synthetic_dataset(batch_size, steps_per_epoch):
  """Generate synthetic MNIST dataset for testing."""
  # train dataset
  x_train = array_ops.ones([batch_size * steps_per_epoch, 28, 28, 1],
                           dtype=dtypes.float32)
  y_train = array_ops.ones([batch_size * steps_per_epoch, 1],
                           dtype=dtypes.int32)
  train_ds = dataset_ops.Dataset.from_tensor_slices((x_train, y_train))
  train_ds = train_ds.repeat()
  # train_ds = train_ds.shuffle(100)
  train_ds = train_ds.batch(64, drop_remainder=True)

  # eval dataset
  x_test = random_ops.random_uniform([10000, 28, 28, 1], dtype=dtypes.float32)
  y_test = random_ops.random_uniform([10000, 1],
                                     minval=0,
                                     maxval=9,
                                     dtype=dtypes.int32)
  eval_ds = dataset_ops.Dataset.from_tensor_slices((x_test, y_test))
  eval_ds = eval_ds.batch(64, drop_remainder=True)

  return train_ds, eval_ds


def get_mnist_model(input_shape):
  """Define a deterministically-initialized CNN model for MNIST testing."""
  inputs = keras.Input(shape=input_shape)
  x = keras.layers.Conv2D(
      32,
      kernel_size=(3, 3),
      activation="relu",
      kernel_initializer=keras.initializers.TruncatedNormal(seed=99))(inputs)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Flatten()(x) + keras.layers.Flatten()(x)
  x = keras.layers.Dense(
      10,
      activation="softmax",
      kernel_initializer=keras.initializers.TruncatedNormal(seed=99))(x)
  model = keras.Model(inputs=inputs, outputs=x)

  # TODO(yuefengz): optimizer with slot variables doesn't work because of
  # optimizer's bug.
  # TODO(yuefengz): we should not allow non-v2 optimizer.
  model.compile(
      loss=keras.losses.sparse_categorical_crossentropy,
      optimizer=gradient_descent.SGD(learning_rate=0.001),
      metrics=["accuracy"])
  return model
