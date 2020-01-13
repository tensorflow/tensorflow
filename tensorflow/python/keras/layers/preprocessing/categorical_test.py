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
"""Tests for image preprocessing layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.layers.preprocessing import categorical
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class CategoryLookupVocabListTest(keras_parameterized.TestCase):

  def test_vocab_list_basic(self):
    vocabulary_list = ['A', 'B', 'C', 'D', 'E']
    layer = categorical.CategoryLookup(
        vocabulary=vocabulary_list, num_oov_tokens=0)
    inp = np.asarray([['A', 'D'], ['E', 'C'], ['D', 'A']])
    output = layer(inp)
    self.assertAllClose(np.asarray([[0, 3], [4, 2], [3, 0]]), output)

  def test_vocab_list_unknown_input(self):
    vocabulary_list = ['A', 'B', 'C', 'D', 'E']
    layer = categorical.CategoryLookup(vocabulary=vocabulary_list)
    inp = np.asarray([['A', ''], ['E', 'C'], ['D', 'A']])
    output = layer(inp)
    self.assertAllClose(np.asarray([[0, 5], [4, 2], [3, 0]]), output)

  def test_vocab_list_invalid_input(self):
    vocabulary_list = ['A', 'B', 'C', 'D', 'E']
    layer = categorical.CategoryLookup(
        vocabulary=vocabulary_list, num_oov_tokens=0)
    inp = np.asarray([['A', ''], ['E', 'C'], ['D', 'A']])
    output = layer(inp)
    self.assertAllClose(np.asarray([[0, -1], [4, 2], [3, 0]]), output)

  def test_vocab_list_compute_output_signature(self):
    input_shape = tensor_shape.TensorShape([2, 3])
    input_spec = tensor_spec.TensorSpec(input_shape, dtypes.string)
    vocabulary_list = ['A', 'B', 'C', 'D', 'E']
    layer = categorical.CategoryLookup(
        vocabulary=vocabulary_list, num_oov_tokens=0)
    output_spec = layer.compute_output_signature(input_spec)
    self.assertEqual(output_spec.shape.dims, input_shape.dims)
    self.assertEqual(output_spec.dtype, dtypes.int64)

  def test_vocab_list_sparse_input(self):
    vocabulary_list = ['A', 'B', 'C', 'D', 'E']
    layer = categorical.CategoryLookup(
        vocabulary=vocabulary_list, num_oov_tokens=0)
    inp = np.asarray([['A', ''], ['E', 'C'], ['D', 'A']])
    indices = array_ops.where_v2(math_ops.not_equal(inp, ''))
    sp_inp = sparse_tensor.SparseTensor(
        indices,
        array_ops.gather_nd_v2(inp, indices),
        dense_shape=array_ops.shape_v2(inp, out_type=dtypes.int64))
    output = layer(sp_inp)
    self.assertIsInstance(output, sparse_tensor.SparseTensor)
    self.assertAllClose(
        np.asarray([[0, 0], [1, 0], [1, 1], [2, 0], [2, 1]]), output.indices)
    self.assertAllClose(np.asarray([0, 4, 2, 3, 0]), output.values)


@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class CategoryLookupVocabFileTest(keras_parameterized.TestCase):

  def setUp(self):
    super(CategoryLookupVocabFileTest, self).setUp()

    # Contains strings, character names from 'The Wire': omar, stringer, marlo
    self._wire_vocabulary_file_name = test.test_src_dir_path(
        'python/keras/layers/preprocessing/testdata/wire_vocabulary.txt')
    self._wire_vocabulary_size = 3

  def test_vocab_file_basic(self):
    layer = categorical.CategoryLookup(
        vocabulary=self._wire_vocabulary_file_name, num_oov_tokens=0)
    inp = np.asarray([['marlo', 'omar'], ['stringer', 'omar']])
    output = layer(inp)
    self.assertAllClose(np.asarray([[2, 0], [1, 0]]), output)

  def test_vocab_file_unknown_input(self):
    layer = categorical.CategoryLookup(
        vocabulary=self._wire_vocabulary_file_name)
    inp = np.asarray([['marlo', 'omar'], ['skywalker', 'omar']])
    output = layer(inp)
    self.assertAllClose(np.asarray([[2, 0], [3, 0]]), output)

  def test_vocab_file_invalid_input(self):
    layer = categorical.CategoryLookup(
        vocabulary=self._wire_vocabulary_file_name, num_oov_tokens=0)
    inp = np.asarray([['marlo', 'omar'], ['skywalker', 'omar']])
    output = layer(inp)
    self.assertAllClose(np.asarray([[2, 0], [-1, 0]]), output)

  def test_vocab_file_compute_output_signature(self):
    input_shape = tensor_shape.TensorShape([2, 3])
    input_spec = tensor_spec.TensorSpec(input_shape, dtypes.string)
    layer = categorical.CategoryLookup(
        vocabulary=self._wire_vocabulary_file_name, num_oov_tokens=0)
    output_spec = layer.compute_output_signature(input_spec)
    self.assertEqual(output_spec.shape.dims, input_shape.dims)
    self.assertEqual(output_spec.dtype, dtypes.int64)

  def test_vocab_list_sparse_input(self):
    layer = categorical.CategoryLookup(
        vocabulary=self._wire_vocabulary_file_name, num_oov_tokens=0)
    inp = np.asarray([['omar', ''], ['stringer', 'marlo'], ['marlo', 'omar']])
    indices = array_ops.where_v2(math_ops.not_equal(inp, ''))
    sp_inp = sparse_tensor.SparseTensor(
        indices,
        array_ops.gather_nd_v2(inp, indices),
        dense_shape=array_ops.shape_v2(inp, out_type=dtypes.int64))
    output = layer(sp_inp)
    self.assertIsInstance(output, sparse_tensor.SparseTensor)
    self.assertAllClose(
        np.asarray([[0, 0], [1, 0], [1, 1], [2, 0], [2, 1]]), output.indices)
    self.assertAllClose(np.asarray([0, 1, 2, 2, 0]), output.values)


if __name__ == '__main__':
  test.main()
