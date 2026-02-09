# coding=utf-8
# Copyright 2025 TF.Text Authors.
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

"""Tests for ToDense Keras layer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import test_util
from tensorflow_text.python.keras.layers.todense import ToDense


class Final(tf.keras.layers.Layer):
  """This is a helper layer that can be used as the last layer in a network for testing purposes."""

  def call(self, inputs):
    return tf.dtypes.cast(inputs, tf.dtypes.float32)

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    base_config = super(Final, self).get_config()
    return dict(list(base_config.items()))


def get_input_dataset(in_data, out_data=None):
  batch_size = in_data.shape[0]
  if out_data is None:
    return tf.data.Dataset.from_tensor_slices(in_data).batch(
        batch_size)

  return tf.data.Dataset.from_tensor_slices(
      (in_data, out_data)).batch(batch_size)


def get_model_from_layers(
    layers,
    input_shape,
    input_sparse=False,
    input_ragged=False,
    input_dtype=None):
  layers = [
      tf.keras.Input(
          shape=input_shape,
          dtype=input_dtype,
          sparse=input_sparse,
          ragged=input_ragged,
      )
  ] + layers
  return tf.keras.models.Sequential(layers)


class RaggedTensorsToDenseLayerTest(tf.test.TestCase, parameterized.TestCase):

  def SKIP_test_ragged_input_default_padding(self):
    input_data = get_input_dataset(
        tf.ragged.constant([[1, 2, 3, 4, 5], [2, 3]]))
    expected_output = np.array([[1, 2, 3, 4, 5], [2, 3, 0, 0, 0]])

    layers = [ToDense(), Final()]
    model = get_model_from_layers(
        layers,
        input_shape=(None,),
        input_ragged=True,
        input_dtype=tf.dtypes.int32)
    model.compile(
        optimizer="sgd",
        loss="mse",
        metrics=["accuracy"])
    output = model.predict(input_data)
    self.assertAllEqual(output, expected_output)

  def SKIP_test_ragged_input_with_padding(self):
    input_data = get_input_dataset(
        tf.ragged.constant([[[1, 2, 3, 4, 5]], [[2], [3]]]))
    expected_output = np.array([[[1., 2., 3., 4., 5.],
                                 [-1., -1., -1., -1., -1.]],
                                [[2., -1., -1., -1., -1.],
                                 [3., -1., -1., -1., -1.]]])

    layers = [ToDense(pad_value=-1), Final()]
    model = get_model_from_layers(
        layers,
        input_shape=(None, None),
        input_ragged=True,
        input_dtype=tf.dtypes.int32)
    model.compile(
        optimizer="sgd",
        loss="mse",
        metrics=["accuracy"])
    output = model.predict(input_data)
    self.assertAllEqual(output, expected_output)

  def test_ragged_input_pad_and_mask(self):
    input_data = tf.ragged.constant([[1, 2, 3, 4, 5], []])
    expected_mask = np.array([True, False])

    output = ToDense(pad_value=-1, mask=True)(input_data)
    self.assertTrue(hasattr(output, "_keras_mask"))
    self.assertIsNot(output._keras_mask, None)
    self.assertAllEqual(
        tf.keras.backend.get_value(output._keras_mask), expected_mask)

  def test_ragged_input_shape(self):
    input_data = get_input_dataset(
        tf.ragged.constant([[1, 2, 3, 4, 5], [2, 3]]))
    expected_output = np.array([[1, 2, 3, 4, 5, 0, 0], [2, 3, 0, 0, 0, 0, 0]])

    layers = [ToDense(shape=[2, 7]), Final()]
    model = get_model_from_layers(
        layers,
        input_shape=(None,),
        input_ragged=True,
        input_dtype=tf.dtypes.int32)
    model.compile(
        optimizer="sgd",
        loss="mse",
        metrics=["accuracy"])
    output = model.predict(input_data)
    self.assertAllEqual(output, expected_output)

  @parameterized.named_parameters(
      *test_util.generate_combinations_with_testcase_name(layer=[
          tf.keras.layers.SimpleRNN, tf.compat.v1.keras.layers.GRU,
          tf.compat.v1.keras.layers.LSTM, tf.keras.layers.GRU,
          tf.keras.layers.LSTM
      ]))
  def SKIP_test_ragged_input_RNN_layer(self, layer):  # pylint: disable=invalid-name
    input_data = get_input_dataset(
        tf.ragged.constant([[1, 2, 3, 4, 5], [5, 6]]))

    layers = [
        ToDense(pad_value=7, mask=True),
        tf.keras.layers.Embedding(8, 16),
        layer(16),
        tf.keras.layers.Dense(3, activation="softmax"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ]
    model = get_model_from_layers(
        layers,
        input_shape=(None,),
        input_ragged=True,
        input_dtype=tf.dtypes.int32)
    model.compile(
        optimizer="rmsprop",
        loss="binary_crossentropy",
        metrics=["accuracy"])

    output = model.predict(input_data)
    self.assertAllEqual(np.zeros((2, 1)).shape, output.shape)


class SparseTensorsToDenseLayerTest(tf.test.TestCase):

  def SKIP_test_sparse_input_default_padding(self):
    input_data = get_input_dataset(
        tf.sparse.SparseTensor(
            indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4]))

    expected_output = np.array([[1., 0., 0., 0.], [0., 0., 2., 0.],
                                [0., 0., 0., 0.]])

    layers = [ToDense(), Final()]
    model = get_model_from_layers(
        layers,
        input_shape=(None,),
        input_sparse=True,
        input_dtype=tf.dtypes.int32)
    model.compile(
        optimizer="sgd",
        loss="mse",
        metrics=["accuracy"])
    output = model.predict(input_data)
    self.assertAllEqual(output, expected_output)

  def SKIP_test_sparse_input_with_padding(self):
    input_data = get_input_dataset(
        tf.sparse.SparseTensor(
            indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4]))

    expected_output = np.array([[1., -1., -1., -1.], [-1., -1., 2., -1.],
                                [-1., -1., -1., -1.]])

    layers = [ToDense(pad_value=-1, trainable=False), Final()]
    model = get_model_from_layers(
        layers,
        input_shape=(None,),
        input_sparse=True,
        input_dtype=tf.dtypes.int32)
    model.compile(
        optimizer="sgd",
        loss="mse",
        metrics=["accuracy"])
    output = model.predict(input_data)
    self.assertAllEqual(output, expected_output)

  def test_sparse_input_pad_and_mask(self):
    input_data = tf.sparse.SparseTensor(
        indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])

    expected_mask = np.array([True, True, False])

    output = ToDense(pad_value=-1, mask=True)(input_data)
    self.assertTrue(hasattr(output, "_keras_mask"))
    self.assertIsNot(output._keras_mask, None)
    self.assertAllEqual(
        tf.keras.backend.get_value(output._keras_mask), expected_mask)

  def test_sparse_input_shape(self):
    input_data = get_input_dataset(
        tf.sparse.SparseTensor(
            indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4]))

    expected_output = np.array([[1., 0., 0., 0.], [0., 0., 2., 0.],
                                [0., 0., 0., 0.]])

    layers = [ToDense(shape=[3, 4]), Final()]
    model = get_model_from_layers(
        layers,
        input_shape=(None,),
        input_sparse=True,
        input_dtype=tf.dtypes.int32)
    model.compile(
        optimizer="sgd",
        loss="mse",
        metrics=["accuracy"])
    output = model.predict(input_data)
    self.assertAllEqual(output, expected_output)


if __name__ == "__main__":
  tf.test.main()
