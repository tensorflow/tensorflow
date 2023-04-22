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

import os
import tempfile

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.layers.preprocessing import preprocessing_test_utils
from tensorflow.python.keras.layers.preprocessing import table_utils
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test


def get_table(dtype=dtypes.string, oov_tokens=None):
  table = lookup_ops.MutableHashTable(
      key_dtype=dtype,
      value_dtype=dtypes.int64,
      default_value=-7,
      name="index_table")
  return table_utils.TableHandler(table, oov_tokens)


def get_static_table(tmpdir,
                     vocab_list,
                     mask_token=None,
                     dtype=dtypes.string,
                     oov_tokens=None):
  vocabulary_file = os.path.join(tmpdir, "tmp_vocab.txt")

  if dtype == dtypes.string:
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(vocab_list) + "\n")
  else:
    with open(vocabulary_file, "w") as f:
      f.write("\n".join([str(v) for v in vocab_list]) + "\n")

  offset = ((0 if mask_token is None else 1) +
            (len(oov_tokens) if oov_tokens is not None else 0))
  init = lookup_ops.TextFileInitializer(
      vocabulary_file,
      dtype,
      lookup_ops.TextFileIndex.WHOLE_LINE,
      dtypes.int64,
      lookup_ops.TextFileIndex.LINE_NUMBER,
      value_index_offset=offset)
  table = lookup_ops.StaticHashTable(init, default_value=-7)
  return table_utils.TableHandler(
      table,
      oov_tokens,
      mask_token=mask_token)


@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
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

  def test_tensor_multi_dim_values_fails(self):
    key_data = np.array([0, 1], dtype=np.int64)
    value_data = np.array([[11, 12], [21, 22]])

    table = get_table(dtype=dtypes.int64, oov_tokens=[1, 2])

    with self.assertRaisesRegex(ValueError, "must be 1-dimensional"):
      table.insert(key_data, value_data)


@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
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


@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
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


@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class StaticIndexLookupOutputTest(
    keras_parameterized.TestCase,
    preprocessing_test_utils.PreprocessingLayerTest):

  def test_int_output_default_lookup_value(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", "michigan"]])
    expected_output = [[1, 2, 3, 4], [4, 3, 1, -7]]

    table = get_static_table(
        tmpdir=self.get_temp_dir(),
        vocab_list=vocab_data,
        mask_token="",
        oov_tokens=None)
    output_data = table.lookup(input_array)

    self.assertAllEqual(expected_output, output_data)

  def test_output_shape(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", "michigan"]])

    table = get_static_table(
        tmpdir=self.get_temp_dir(), vocab_list=vocab_data, oov_tokens=None)
    output_data = table.lookup(input_array)

    self.assertAllEqual(input_array.shape[1:], output_data.shape[1:])

  def test_int_output_no_reserved_zero_default_lookup_value(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", "michigan"]])
    expected_output = [[0, 1, 2, 3], [3, 2, 0, -7]]

    table = get_static_table(
        tmpdir=self.get_temp_dir(), vocab_list=vocab_data, oov_tokens=None)
    output_data = table.lookup(input_array)

    self.assertAllEqual(expected_output, output_data)


@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class CategoricalEncodingStaticInputTest(
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

    table = get_static_table(
        tmpdir=self.get_temp_dir(),
        vocab_list=vocab_data,
        mask_token="",
        oov_tokens=[1])
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

    table = get_static_table(
        tmpdir=self.get_temp_dir(),
        vocab_list=vocab_data,
        dtype=dtypes.int64,
        mask_token=0,
        oov_tokens=[1])
    output_data = table.lookup(input_array)

    self.assertAllEqual(expected_indices, output_data.indices)
    self.assertAllEqual(expected_values, output_data.values)
    self.assertAllEqual(expected_dense_shape, output_data.dense_shape)

  def test_ragged_string_input(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = ragged_factory_ops.constant(
        [["earth", "wind", "fire"], ["fire", "and", "earth", "michigan"]])
    expected_output = [[2, 3, 5], [5, 4, 2, 1]]

    table = get_static_table(
        tmpdir=self.get_temp_dir(),
        vocab_list=vocab_data,
        mask_token="",
        oov_tokens=[1])
    output_data = table.lookup(input_array)

    self.assertAllEqual(expected_output, output_data)

  def test_ragged_int_input(self):
    vocab_data = np.array([10, 11, 12, 13], dtype=np.int64)
    input_array = ragged_factory_ops.constant([[10, 11, 13], [13, 12, 10, 42]],
                                              dtype=np.int64)
    expected_output = [[2, 3, 5], [5, 4, 2, 1]]

    table = get_static_table(
        tmpdir=self.get_temp_dir(),
        vocab_list=vocab_data,
        dtype=dtypes.int64,
        mask_token=0,
        oov_tokens=[1])
    output_data = table.lookup(input_array)

    self.assertAllEqual(expected_output, output_data)


class GetVocabularyFromFileTest(test.TestCase):

  def setUp(self):
    super(GetVocabularyFromFileTest, self).setUp()
    dir_path = tempfile.mkdtemp(prefix=test.get_temp_dir())
    self._vocab_path = os.path.join(dir_path, "vocab")

  def test_only_line_separator_is_stripped(self):
    expected = ["foo", " foo", "foo ", " foo "]
    with gfile.GFile(self._vocab_path, "w") as writer:
      for word in expected:
        writer.write(word)
        writer.write(os.linesep)

    actual = actual = table_utils.get_vocabulary_from_file(self._vocab_path)
    self.assertAllEqual(expected, actual)

  def test_linux_file(self):
    content = b"line1\nline2\nline3"
    with gfile.GFile(self._vocab_path, "wb") as writer:
      writer.write(content)

    actual = table_utils.get_vocabulary_from_file(self._vocab_path)
    self.assertAllEqual(["line1", "line2", "line3"], actual)

  def test_windows_file(self):
    content = b"line1\r\nline2\r\nline3"
    with gfile.GFile(self._vocab_path, "wb") as writer:
      writer.write(content)

    actual = table_utils.get_vocabulary_from_file(self._vocab_path)
    self.assertAllEqual(["line1", "line2", "line3"], actual)

if __name__ == "__main__":
  test.main()
