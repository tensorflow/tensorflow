
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
"""Tests deprecation warnings in a few special cases."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging


class DeprecationTest(test.TestCase):

  @test.mock.patch.object(logging, "warning", autospec=True)
  def testDeprecatedFunction(self, mock_warning):
    self.assertEqual(0, mock_warning.call_count)
    tf.compat.v1.initializers.tables_initializer()
    self.assertEqual(0, mock_warning.call_count)

    tf.tables_initializer()
    self.assertEqual(1, mock_warning.call_count)
    self.assertRegexpMatches(
        mock_warning.call_args[0][1],
        "deprecation_test.py:")
    self.assertRegexpMatches(
        mock_warning.call_args[0][2], r"tables_initializer")
    self.assertRegexpMatches(
        mock_warning.call_args[0][3],
        r"compat.v1.initializers.tables_initializer")
    tf.tables_initializer()
    self.assertEqual(1, mock_warning.call_count)

  @test.mock.patch.object(logging, "warning", autospec=True)
  def testDeprecatedClass(self, mock_warning):
    value = np.array([1, 2, 3])
    row_splits = np.array([1])

    self.assertEqual(0, mock_warning.call_count)
    tf.compat.v1.ragged.RaggedTensorValue(value, row_splits)
    self.assertEqual(0, mock_warning.call_count)

    tf.ragged.RaggedTensorValue(value, row_splits)
    self.assertEqual(1, mock_warning.call_count)
    self.assertRegexpMatches(
        mock_warning.call_args[0][1],
        "deprecation_test.py:")
    self.assertRegexpMatches(
        mock_warning.call_args[0][2], r"ragged.RaggedTensorValue")
    self.assertRegexpMatches(
        mock_warning.call_args[0][3],
        r"compat.v1.ragged.RaggedTensorValue")
    tf.ragged.RaggedTensorValue(value, row_splits)
    self.assertEqual(1, mock_warning.call_count)

  @test.mock.patch.object(logging, "warning", autospec=True)
  def testDeprecatedFunctionEndpoint(self, mock_warning):
    array = tf.IndexedSlices(
        tf.compat.v1.convert_to_tensor(np.array([1, 2])),
        tf.compat.v1.convert_to_tensor(np.array([0, 2])))
    mask_indices = tf.compat.v1.convert_to_tensor(np.array([2]))

    self.assertEqual(0, mock_warning.call_count)
    tf.sparse.mask(array, mask_indices)
    self.assertEqual(0, mock_warning.call_count)

    tf.sparse_mask(array, mask_indices)
    self.assertEqual(1, mock_warning.call_count)
    self.assertRegexpMatches(
        mock_warning.call_args[0][1],
        "deprecation_test.py:")
    self.assertRegexpMatches(
        mock_warning.call_args[0][2], r"sparse_mask")
    self.assertRegexpMatches(
        mock_warning.call_args[0][3],
        "sparse.mask")
    tf.sparse_mask(array, mask_indices)
    self.assertEqual(1, mock_warning.call_count)

  @test.mock.patch.object(logging, "warning", autospec=True)
  def testDeprecatedClassEndpoint(self, mock_warning):
    self.assertEqual(0, mock_warning.call_count)
    tf.io.VarLenFeature(tf.dtypes.int32)
    self.assertEqual(0, mock_warning.call_count)

    tf.VarLenFeature(tf.dtypes.int32)
    self.assertEqual(1, mock_warning.call_count)
    self.assertRegexpMatches(
        mock_warning.call_args[0][1],
        "deprecation_test.py:")
    self.assertRegexpMatches(
        mock_warning.call_args[0][2], r"VarLenFeature")
    self.assertRegexpMatches(
        mock_warning.call_args[0][3],
        r"io.VarLenFeature")
    tf.VarLenFeature(tf.dtypes.int32)
    self.assertEqual(1, mock_warning.call_count)


if __name__ == "__main__":
  test.main()
