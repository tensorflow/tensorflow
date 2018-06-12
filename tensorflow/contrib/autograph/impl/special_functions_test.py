# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for special_functions module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.autograph.impl import special_functions
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import list_ops
from tensorflow.python.platform import test


class SpecialFunctionsTest(test.TestCase):

  def test_basic(self):
    self.assertEqual(special_functions.stack(1), 1)
    self.assertListEqual(special_functions.stack([1, 2, 3]), [1, 2, 3])
    # TODO(mdan): This should probably forward to tf.stack.
    self.assertTrue(
        isinstance(
            special_functions.stack(
                [constant_op.constant(1),
                 constant_op.constant(2)]), list))

    t = constant_op.constant([1.0, 2.0])
    l = list_ops.tensor_list_from_tensor(
        t, element_shape=constant_op.constant([], dtype=dtypes.int32))
    self.assertTrue(
        tensor_util.is_tensor(
            special_functions.stack(l, element_dtype=dtypes.float32)))


if __name__ == '__main__':
  test.main()
