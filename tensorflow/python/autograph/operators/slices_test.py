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
"""Tests for slices module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.operators import slices
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import list_ops
from tensorflow.python.platform import test


class SlicesTest(test.TestCase):

  def test_set_item_tensor_list(self):
    initial_list = constant_op.constant([[1, 2], [3, 4]])
    elem_shape = constant_op.constant([2])
    l = list_ops.tensor_list_from_tensor(initial_list, element_shape=elem_shape)
    l = slices.set_item(l, 0, [5, 6])

    with self.cached_session() as sess:
      t = list_ops.tensor_list_stack(l, element_dtype=initial_list.dtype)
      self.assertAllEqual(self.evaluate(t), [[5, 6], [3, 4]])

  def test_get_item_tensor_list(self):
    initial_list = constant_op.constant([[1, 2], [3, 4]])
    elem_shape = constant_op.constant([2])
    l = list_ops.tensor_list_from_tensor(initial_list, element_shape=elem_shape)
    t = slices.get_item(
        l, 1, slices.GetItemOpts(element_dtype=initial_list.dtype))

    with self.cached_session() as sess:
      self.assertAllEqual(self.evaluate(t), [3, 4])

  def test_get_item_tensor_string(self):
    initial_str = constant_op.constant('abcd')
    t = slices.get_item(initial_str, 1,
                        slices.GetItemOpts(element_dtype=initial_str.dtype))

    with self.cached_session() as sess:
      self.assertEqual(self.evaluate(t), b'b')

    initial_list_str = constant_op.constant(['abcd', 'bcde'])
    t = slices.get_item(initial_list_str, 1,
                        slices.GetItemOpts(element_dtype=initial_str.dtype))

    with self.cached_session() as sess:
      self.assertEqual(self.evaluate(t), b'bcde')


if __name__ == '__main__':
  test.main()
