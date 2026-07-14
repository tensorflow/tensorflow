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

import numpy as np

from tensorflow.python.autograph.lang import special_functions
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import list_ops
from tensorflow.python.platform import test


class SpecialFunctionsTest(test.TestCase):

  def test_match_staging_level(self):
    some_tensor = constant_op.constant(0)
    tensor_one = special_functions.match_staging_level(1, some_tensor)
    python_one = special_functions.match_staging_level(1, 1)
    with self.cached_session() as sess:
      self.assertTrue(tensor_util.is_tf_type(tensor_one))
      self.assertAllEqual(self.evaluate(tensor_one), 1)
      self.assertEqual(python_one, 1)

  def test_tensor_list_empty_list(self):
    l = special_functions.tensor_list([],
                                      element_dtype=dtypes.int32,
                                      element_shape=())
    sl = list_ops.tensor_list_stack(l, element_dtype=dtypes.int32)
    with self.cached_session() as sess:
      self.assertAllEqual(self.evaluate(sl), [])

    l = special_functions.tensor_list((),
                                      element_dtype=dtypes.int32,
                                      element_shape=())
    sl = list_ops.tensor_list_stack(l, element_dtype=dtypes.int32)
    with self.cached_session() as sess:
      self.assertAllEqual(self.evaluate(sl), [])

  def test_tensor_list_tensor(self):
    l = special_functions.tensor_list(
        constant_op.constant([], dtype=dtypes.int32))
    sl = list_ops.tensor_list_stack(l, element_dtype=dtypes.int32)
    with self.cached_session() as sess:
      self.assertAllEqual(self.evaluate(sl), [])

  def test_tensor_list_unsupported_initializer(self):
    with self.assertRaisesRegex(ValueError, 'unknown type'):
      special_functions.tensor_list(np.array([1, 2, 3]))

  def test_tensor_list_empty_list_no_type(self):
    with self.assertRaisesRegex(ValueError,
                                'element_dtype and element_shape are required'):
      special_functions.tensor_list([])

  def test_tensor_list_from_elements(self):
    elements = [constant_op.constant([1, 2]), constant_op.constant([3, 4])]

    l = special_functions.tensor_list(elements)
    sl = list_ops.tensor_list_stack(l, element_dtype=dtypes.int32)
    with self.cached_session() as sess:
      self.assertAllEqual(self.evaluate(sl), [[1, 2], [3, 4]])

  def test_tensor_list_array_from_elements(self):
    elements = [constant_op.constant([1, 2]), constant_op.constant([3, 4])]

    l = special_functions.tensor_list(elements, use_tensor_array=True)
    sl = l.stack()
    with self.cached_session() as sess:
      self.assertAllEqual(self.evaluate(sl), [[1, 2], [3, 4]])

  def test_stack(self):
    self.assertEqual(special_functions.stack(1, strict=False), 1)
    self.assertListEqual(
        special_functions.stack([1, 2, 3], strict=False), [1, 2, 3])
    # TODO(mdan): This should probably forward to tf.stack.
    self.assertTrue(
        isinstance(
            special_functions.stack(
                [constant_op.constant(1),
                 constant_op.constant(2)], strict=False), list))

    with self.assertRaises(ValueError):
      special_functions.stack([1, 2, 3])

    t = constant_op.constant([1.0, 2.0])
    l = list_ops.tensor_list_from_tensor(
        t, element_shape=constant_op.constant([], dtype=dtypes.int32))
    self.assertTrue(
        tensor_util.is_tf_type(
            special_functions.stack(l, element_dtype=dtypes.float32)))


if __name__ == '__main__':
  test.main()
