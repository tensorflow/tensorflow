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
"""Tests for tf.ragged in eager execution mode."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import googletest


class RaggedTensorTest(test_util.TensorFlowTestCase,
                       parameterized.TestCase):

  @parameterized.parameters([
      dict(pylist=[[b'a', b'b'], [b'c']]),
      dict(pylist=[[[1, 2], [3]], [[4, 5, 6], [], [7]]]),
      dict(pylist=[[[1, 2], [3, 4]], [[5, 6], [], [7, 8]]], ragged_rank=1),
  ])
  def testRaggedTensorToList(self, pylist, ragged_rank=None):
    rt = ragged_factory_ops.constant(pylist, ragged_rank)
    self.assertAllEqual(rt, pylist)

  @parameterized.parameters([
      dict(pylist=[[b'a', b'b'], [b'c']],
           expected_str="[[b'a', b'b'], [b'c']]"),
      dict(pylist=[[[1, 2], [3]], [[4, 5, 6], [], [7]]],
           expected_str='[[[1, 2], [3]], [[4, 5, 6], [], [7]]]'),
      dict(pylist=[[0, 1], np.arange(2, 2000)],
           expected_str='[[0, 1], [2, 3, 4, ..., 1997, 1998, 1999]]'),
      dict(pylist=[[[0, 1]], [np.arange(2, 2000)]],
           expected_str='[[[0, 1]],\n [[2, 3, 4, ..., 1997, 1998, 1999]]]'),
  ])
  def testRaggedTensorStr(self, pylist, expected_str):
    rt = ragged_factory_ops.constant(pylist)
    self.assertEqual(str(rt), f'<tf.RaggedTensor {expected_str}>')


if __name__ == '__main__':
  ops.enable_eager_execution()
  googletest.main()
