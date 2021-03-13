# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Keras discretization preprocessing layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np

from tensorflow.python import keras

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.layers.preprocessing import discretization
from tensorflow.python.keras.layers.preprocessing import discretization_v1
from tensorflow.python.keras.layers.preprocessing import preprocessing_test_utils
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test


def get_layer_class():
  if context.executing_eagerly():
    return discretization.Discretization
  else:
    return discretization_v1.Discretization


@keras_parameterized.run_all_keras_modes
class DiscretizationTest(keras_parameterized.TestCase,
                         preprocessing_test_utils.PreprocessingLayerTest):

  def test_bucketize_with_explicit_buckets_integer(self):
    input_array = np.array([[-1.5, 1.0, 3.4, .5], [0.0, 3.0, 1.3, 0.0]])

    expected_output = [[0, 1, 3, 1], [0, 3, 2, 0]]
    expected_output_shape = [None, 4]

    input_data = keras.Input(shape=(4,))
    layer = get_layer_class()(bin_boundaries=[0., 1., 2.])
    bucket_data = layer(input_data)
    self.assertAllEqual(expected_output_shape, bucket_data.shape.as_list())

    model = keras.Model(inputs=input_data, outputs=bucket_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_bucketize_with_explicit_buckets_int_input(self):
    input_array = np.array([[-1, 1, 3, 0], [0, 3, 1, 0]], dtype=np.int64)

    expected_output = [[0, 2, 3, 1], [1, 3, 2, 1]]
    expected_output_shape = [None, 4]

    input_data = keras.Input(shape=(4,), dtype=dtypes.int64)
    layer = get_layer_class()(bin_boundaries=[-.5, 0.5, 1.5])
    bucket_data = layer(input_data)
    self.assertAllEqual(expected_output_shape, bucket_data.shape.as_list())

    model = keras.Model(inputs=input_data, outputs=bucket_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_bucketize_with_explicit_buckets_sparse_float_input(self):
    indices = [[0, 1], [0, 2], [1, 1]]
    input_array = sparse_tensor.SparseTensor(
        indices=indices, values=[-1.5, 1.0, 3.4], dense_shape=[2, 3])
    expected_output = [0, 2, 3]
    input_data = keras.Input(shape=(3,), dtype=dtypes.float32, sparse=True)
    layer = get_layer_class()(bin_boundaries=[-.5, 0.5, 1.5])
    bucket_data = layer(input_data)

    model = keras.Model(inputs=input_data, outputs=bucket_data)
    output_dataset = model.predict(input_array, steps=1)
    self.assertAllEqual(indices, output_dataset.indices)
    self.assertAllEqual(expected_output, output_dataset.values)

  def test_bucketize_with_explicit_buckets_ragged_float_input(self):
    input_array = ragged_factory_ops.constant([[-1.5, 1.0, 3.4, .5],
                                               [0.0, 3.0, 1.3]])

    expected_output = [[0, 1, 3, 1], [0, 3, 2]]
    expected_output_shape = [None, None]

    input_data = keras.Input(shape=(None,), ragged=True)
    layer = get_layer_class()(bin_boundaries=[0., 1., 2.])
    bucket_data = layer(input_data)
    self.assertAllEqual(expected_output_shape, bucket_data.shape.as_list())

    model = keras.Model(inputs=input_data, outputs=bucket_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_bucketize_with_explicit_buckets_ragged_int_input(self):
    input_array = ragged_factory_ops.constant([[-1, 1, 3, 0], [0, 3, 1]],
                                              dtype=dtypes.int64)

    expected_output = [[0, 2, 3, 1], [1, 3, 2]]
    expected_output_shape = [None, None]

    input_data = keras.Input(shape=(None,), ragged=True, dtype=dtypes.int64)
    layer = get_layer_class()(bin_boundaries=[-.5, 0.5, 1.5])
    bucket_data = layer(input_data)
    self.assertAllEqual(expected_output_shape, bucket_data.shape.as_list())
    model = keras.Model(inputs=input_data, outputs=bucket_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_bucketize_with_explicit_buckets_sparse_int_input(self):
    indices = [[0, 1], [0, 2], [1, 1]]
    input_array = sparse_tensor.SparseTensor(
        indices=indices, values=[-1, 1, 3], dense_shape=[2, 3])
    expected_output = [0, 2, 3]
    input_data = keras.Input(shape=(3,), dtype=dtypes.int32, sparse=True)
    layer = get_layer_class()(bin_boundaries=[-.5, 0.5, 1.5])
    bucket_data = layer(input_data)

    model = keras.Model(inputs=input_data, outputs=bucket_data)
    output_dataset = model.predict(input_array, steps=1)
    self.assertAllEqual(indices, output_dataset.indices)
    self.assertAllEqual(expected_output, output_dataset.values)

  def test_num_bins_negative_fails(self):
    with self.assertRaisesRegex(ValueError, "`num_bins` must be.*num_bins=-7"):
      _ = get_layer_class()(num_bins=-7)

  def test_num_bins_and_bins_set_fails(self):
    with self.assertRaisesRegex(
        ValueError,
        r"`num_bins` and `bin_boundaries` should not be set.*5.*\[1, 2\]"):
      _ = get_layer_class()(num_bins=5, bins=[1, 2])

  @parameterized.named_parameters([
      {
          "testcase_name": "2d_single_element",
          "adapt_data": np.array([[1.], [2.], [3.], [4.], [5.]]),
          "test_data": np.array([[1.], [2.], [3.]]),
          "use_dataset": True,
          "expected": np.array([[0], [1], [2]]),
          "num_bins": 5,
          "epsilon": 0.01
      }, {
          "testcase_name": "2d_multi_element",
          "adapt_data": np.array([[1., 6.], [2., 7.], [3., 8.], [4., 9.],
                                  [5., 10.]]),
          "test_data": np.array([[1., 10.], [2., 6.], [3., 8.]]),
          "use_dataset": True,
          "expected": np.array([[0, 4], [0, 2], [1, 3]]),
          "num_bins": 5,
          "epsilon": 0.01
      }, {
          "testcase_name": "1d_single_element",
          "adapt_data": np.array([3., 2., 1., 5., 4.]),
          "test_data": np.array([1., 2., 3.]),
          "use_dataset": True,
          "expected": np.array([0, 1, 2]),
          "num_bins": 5,
          "epsilon": 0.01
      }, {
          "testcase_name": "300_batch_1d_single_element_1",
          "adapt_data": np.arange(300),
          "test_data": np.arange(300),
          "use_dataset": True,
          "expected":
              np.concatenate([np.zeros(101), np.ones(99), 2 * np.ones(100)]),
          "num_bins": 3,
          "epsilon": 0.01
      }, {
          "testcase_name": "300_batch_1d_single_element_2",
          "adapt_data": np.arange(300) ** 2,
          "test_data": np.arange(300) ** 2,
          "use_dataset": True,
          "expected":
              np.concatenate([np.zeros(101), np.ones(99), 2 * np.ones(100)]),
          "num_bins": 3,
          "epsilon": 0.01
      }, {
          "testcase_name": "300_batch_1d_single_element_large_epsilon",
          "adapt_data": np.arange(300),
          "test_data": np.arange(300),
          "use_dataset": True,
          "expected": np.concatenate([np.zeros(137), np.ones(163)]),
          "num_bins": 2,
          "epsilon": 0.1
      }])
  def test_layer_computation(self, adapt_data, test_data, use_dataset,
                             expected, num_bins=5, epsilon=0.01):

    input_shape = tuple(list(test_data.shape)[1:])
    np.random.shuffle(adapt_data)
    if use_dataset:
      # Keras APIs expect batched datasets
      adapt_data = dataset_ops.Dataset.from_tensor_slices(adapt_data).batch(
          test_data.shape[0] // 2)
      test_data = dataset_ops.Dataset.from_tensor_slices(test_data).batch(
          test_data.shape[0] // 2)

    cls = get_layer_class()
    layer = cls(epsilon=epsilon, num_bins=num_bins)
    layer.adapt(adapt_data)

    input_data = keras.Input(shape=input_shape)
    output = layer(input_data)
    model = keras.Model(input_data, output)
    model._run_eagerly = testing_utils.should_run_eagerly()
    output_data = model.predict(test_data)
    self.assertAllClose(expected, output_data)

  @parameterized.named_parameters(
      {
          "num_bins": 5,
          "data": np.array([[1.], [2.], [3.], [4.], [5.]]),
          "expected": {
              "bins": np.array([1., 2., 3., 4., np.Inf])
          },
          "testcase_name": "2d_single_element_all_bins"
      }, {
          "num_bins": 5,
          "data": np.array([[1., 6.], [2., 7.], [3., 8.], [4., 9.], [5., 10.]]),
          "expected": {
              "bins": np.array([2., 4., 6., 8., np.Inf])
          },
          "testcase_name": "2d_multi_element_all_bins",
      }, {
          "num_bins": 3,
          "data": np.array([[0.], [1.], [2.], [3.], [4.], [5.]]),
          "expected": {
              "bins": np.array([1., 3., np.Inf])
          },
          "testcase_name": "2d_single_element_3_bins"
      })
  def test_combiner_computation(self, num_bins, data, expected):
    epsilon = 0.01
    combiner = discretization.Discretization.DiscretizingCombiner(epsilon,
                                                                  num_bins)
    self.validate_accumulator_extract(combiner, data, expected)

if __name__ == "__main__":
  test.main()
