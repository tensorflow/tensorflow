# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for ragged_placeholder op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedPlaceholderOpTest(test_util.TensorFlowTestCase,
                              parameterized.TestCase):

  @parameterized.parameters([
      # dtype, ragged_rank, value_shape, name -> expected
      (dtypes.int32, 0, [5], None,
       'Tensor("Placeholder:0", shape=(5,), dtype=int32)'),
      (dtypes.int32, 1, [], 'ph',
       'tf.RaggedTensor('
       'values=Tensor("ph/flat_values:0", shape=(None,), dtype=int32), '
       'row_splits=Tensor("ph/row_splits_0:0", shape=(None,), dtype=int64))'),
      (dtypes.string, 1, [5], 'ph',
       'tf.RaggedTensor('
       'values=Tensor("ph/flat_values:0", shape=(None, 5), dtype=string), '
       'row_splits=Tensor("ph/row_splits_0:0", shape=(None,), dtype=int64))'),
      (dtypes.float32, 2, [], 'ph',
       'tf.RaggedTensor(values=tf.RaggedTensor('
       'values=Tensor("ph/flat_values:0", shape=(None,), dtype=float32), '
       'row_splits=Tensor("ph/row_splits_1:0", shape=(None,), dtype=int64)), '
       'row_splits=Tensor("ph/row_splits_0:0", shape=(None,), dtype=int64))'),
      (dtypes.int32, 2, [3, 5], 'ph',
       'tf.RaggedTensor(values=tf.RaggedTensor('
       'values=Tensor("ph/flat_values:0", shape=(None, 3, 5), dtype=int32), '
       'row_splits=Tensor("ph/row_splits_1:0", shape=(None,), dtype=int64)), '
       'row_splits=Tensor("ph/row_splits_0:0", shape=(None,), dtype=int64))'),

  ])
  def testRaggedPlaceholder(self, dtype, ragged_rank, value_shape, name,
                            expected):
    if not context.executing_eagerly():
      placeholder = ragged_factory_ops.placeholder(
          dtype, ragged_rank, value_shape, name)
      result = str(placeholder).replace('?', 'None')
      self.assertEqual(result, expected)

  def testRaggedPlaceholderRaisesExceptionInEagerMode(self):
    if context.executing_eagerly():
      with self.assertRaises(RuntimeError):
        ragged_factory_ops.placeholder(dtypes.int32, 1, [])


if __name__ == '__main__':
  googletest.main()
