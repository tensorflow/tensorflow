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
"""Tests for Keras lookup table utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.layers.preprocessing import preprocessing_test_utils
from tensorflow.python.keras.layers.preprocessing import table_utils
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test


def get_table(dtype=dtypes.string, oov_tokens=None):
  table = lookup_ops.MutableHashTable(
      key_dtype=dtype,
      value_dtype=dtypes.int64,
      default_value=-7,
      name="index_table")
  return table_utils.TableHandler(
      table, oov_tokens, use_v1_apis=(not context.executing_eagerly()))


@keras_parameterized.run_all_keras_modes
class CategoricalEncodingInputTest(
    keras_parameterized.TestCase,
    preprocessing_test_utils.PreprocessingLayerTest):

  def test_sparse_string_input(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 2]],
        values=["fire", "michigan"],
        dense_shape=[3, 4])

    expected_indices = [[0, 0], [1, 2]]
    expected_values = [5, 1]
    expected_dense_shape = [3, 4]

    table = get_table(oov_tokens=[1])
    table.insert(vocab_data, range(2, len(vocab_data) + 2))
    output_data = table.lookup(input_array)

    self.assertAllEqual(expected_indices, output_data.indices)
    self.assertAllEqual(expected_values, output_data.values)
    self.assertAllEqual(expected_dense_shape, output_data.dense_shape)

  def test_sparse_int_input(self):
    vocab_data = np.array([10, 11, 12, 13], dtype=np.int64)
    input_array = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 2]],
        values=np.array([13, 32], dtype=np.int64),
        dense_shape=[3, 4])

    expected_indices = [[0, 0], [1, 2]]
    expected_values = [5, 1]
    expected_dense_shape = [3, 4]

    table = get_table(dtype=dtypes.int64, oov_tokens=[1])
    table.insert(vocab_data, range(2, len(vocab_data) + 2))
    output_data = table.lookup(input_array)

    self.assertAllEqual(expected_indices, output_data.indices)
    self.assertAllEqual(expected_values, output_data.values)
    self.assertAllEqual(expected_dense_shape, output_data.dense_shape)

  def test_ragged_string_input(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = ragged_factory_ops.constant(
        [["earth", "wind", "fire"], ["fire", "and", "earth", "michigan"]])
    expected_output = [[2, 3, 5], [5, 4, 2, 1]]

    table = get_table(oov_tokens=[1])
    table.insert(vocab_data, range(2, len(vocab_data) + 2))
    output_data = table.lookup(input_array)

    self.assertAllEqual(expected_output, output_data)

  def test_ragged_int_input(self):
    vocab_data = np.array([10, 11, 12, 13], dtype=np.int64)
    input_array = ragged_factory_ops.constant([[10, 11, 13], [13, 12, 10, 42]],
                                              dtype=np.int64)
    expected_output = [[2, 3, 5], [5, 4, 2, 1]]

    table = get_table(dtype=dtypes.int64, oov_tokens=[1])
    table.insert(vocab_data, range(2, len(vocab_data) + 2))
    output_data = table.lookup(input_array)

    self.assertAllEqual(expected_output, output_data)


@keras_parameterized.run_all_keras_modes
class CategoricalEncodingMultiOOVTest(
    keras_parameterized.TestCase,
    preprocessing_test_utils.PreprocessingLayerTest):

  def test_sparse_string_input_multi_bucket(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 2]], values=["fire", "ohio"], dense_shape=[3, 4])

    expected_indices = [[0, 0], [1, 2]]
    expected_values = [6, 2]
    expected_dense_shape = [3, 4]

    table = get_table(oov_tokens=[1, 2])
    table.insert(vocab_data, range(3, len(vocab_data) + 3))
    output_data = table.lookup(input_array)

    self.assertAllEqual(expected_indices, output_data.indices)
    self.assertAllEqual(expected_values, output_data.values)
    self.assertAllEqual(expected_dense_shape, output_data.dense_shape)

  def test_sparse_int_input_multi_bucket(self):
    vocab_data = np.array([10, 11, 12, 13], dtype=np.int64)
    input_array = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 2]],
        values=np.array([13, 132], dtype=np.int64),
        dense_shape=[3, 4])

    expected_indices = [[0, 0], [1, 2]]
    expected_values = [6, 1]
    expected_dense_shape = [3, 4]

    table = get_table(dtype=dtypes.int64, oov_tokens=[1, 2])
    table.insert(vocab_data, range(3, len(vocab_data) + 3))
    output_data = table.lookup(input_array)

    self.assertAllEqual(expected_indices, output_data.indices)
    self.assertAllEqual(expected_values, output_data.values)
    self.assertAllEqual(expected_dense_shape, output_data.dense_shape)

  def test_ragged_string_input_multi_bucket(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = ragged_factory_ops.constant([["earth", "wind", "fire"],
                                               ["fire", "and", "earth",
                                                "ohio"]])
    expected_output = [[3, 4, 6], [6, 5, 3, 2]]

    table = get_table(oov_tokens=[1, 2])
    table.insert(vocab_data, range(3, len(vocab_data) + 3))
    output_data = table.lookup(input_array)

    self.assertAllEqual(expected_output, output_data)

  def test_ragged_int_input_multi_bucket(self):
    vocab_data = np.array([10, 11, 12, 13], dtype=np.int64)
    input_array = ragged_factory_ops.constant([[10, 11, 13], [13, 12, 10, 132]],
                                              dtype=np.int64)
    expected_output = [[3, 4, 6], [6, 5, 3, 1]]

    table = get_table(dtype=dtypes.int64, oov_tokens=[1, 2])
    table.insert(vocab_data, range(3, len(vocab_data) + 3))
    output_data = table.lookup(input_array)

    self.assertAllEqual(expected_output, output_data)

  def test_tensor_int_input_multi_bucket(self):
    vocab_data = np.array([10, 11, 12, 13], dtype=np.int64)
    input_array = np.array([[13, 132], [13, 133]], dtype=np.int64)
    expected_values = [[6, 1], [6, 2]]

    table = get_table(dtype=dtypes.int64, oov_tokens=[1, 2])
    table.insert(vocab_data, range(3, len(vocab_data) + 3))
    output_data = table.lookup(input_array)

    self.assertAllEqual(expected_values, output_data)

  def test_tensor_string_input_multi_bucket(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = [["earth", "wind", "fire", "michigan"],
                   ["fire", "and", "earth", "ohio"]]
    expected_output = [[3, 4, 6, 1], [6, 5, 3, 2]]

    table = get_table(oov_tokens=[1, 2])
    table.insert(vocab_data, range(3, len(vocab_data) + 3))
    output_data = table.lookup(input_array)

    self.assertAllEqual(expected_output, output_data)


@keras_parameterized.run_all_keras_modes
class IndexLookupOutputTest(keras_parameterized.TestCase,
                            preprocessing_test_utils.PreprocessingLayerTest):

  def test_int_output_default_lookup_value(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", "michigan"]])
    expected_output = [[1, 2, 3, 4], [4, 3, 1, -7]]

    table = get_table(oov_tokens=None)
    table.insert(vocab_data, range(1, len(vocab_data) + 1))
    output_data = table.lookup(input_array)

    self.assertAllEqual(expected_output, output_data)

  def test_output_shape(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", "michigan"]])

    table = get_table()
    table.insert(vocab_data, range(1, len(vocab_data) + 1))
    output_data = table.lookup(input_array)

    self.assertAllEqual(input_array.shape[1:], output_data.shape[1:])

  def test_int_output_no_reserved_zero_default_lookup_value(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", "michigan"]])
    expected_output = [[0, 1, 2, 3], [3, 2, 0, -7]]

    table = get_table(oov_tokens=None)
    table.insert(vocab_data, range(len(vocab_data)))
    output_data = table.lookup(input_array)

    self.assertAllEqual(expected_output, output_data)


if __name__ == "__main__":
  test.main()
