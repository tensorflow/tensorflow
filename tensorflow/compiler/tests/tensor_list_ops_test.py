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
"""Tests for ops which manipulate lists of tensors via bridge."""

# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.platform import test


def scalar_shape():
  return ops.convert_to_tensor([], dtype=dtypes.int32)


class ListOpsTest(xla_test.XLATestCase):

  def testElementShape(self):
    with self.cached_session() as sess, self.test_scope():
      dim = array_ops.placeholder(dtypes.int32)
      l = list_ops.tensor_list_reserve(
          element_shape=(dim, 15), num_elements=20,
          element_dtype=dtypes.float32)
      e32 = list_ops.tensor_list_element_shape(l, shape_type=dtypes.int32)
      e64 = list_ops.tensor_list_element_shape(l, shape_type=dtypes.int64)
      self.assertAllEqual(sess.run(e32, {dim: 10}), (10, 15))
      self.assertAllEqual(sess.run(e64, {dim: 7}), (7, 15))

  def testPushPop(self):
    with self.cached_session() as sess, self.test_scope():
      num = array_ops.placeholder(dtypes.int32)
      l = list_ops.tensor_list_reserve(
          element_shape=(7, 15), num_elements=num, element_dtype=dtypes.float32)
      l = list_ops.tensor_list_push_back(
          l, constant_op.constant(1.0, shape=(7, 15)))
      l = list_ops.tensor_list_push_back(
          l, constant_op.constant(2.0, shape=(7, 15)))
      l, e2 = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
      _, e1 = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
      self.assertAllEqual(sess.run(e2, {num: 10}), 2.0 * np.ones((7, 15)))
      self.assertAllEqual(sess.run(e1, {num: 10}), 1.0 * np.ones((7, 15)))

  def testPushPopSeparateLists(self):
    with self.cached_session() as sess, self.test_scope():
      num = array_ops.placeholder(dtypes.int32)
      l = list_ops.tensor_list_reserve(
          element_shape=scalar_shape(),
          num_elements=num,
          element_dtype=dtypes.float32)
      l = list_ops.tensor_list_push_back(l, constant_op.constant(1.0))
      l2 = list_ops.tensor_list_push_back(l, constant_op.constant(2.0))
      l3 = list_ops.tensor_list_push_back(l, constant_op.constant(3.0))
      _, e11 = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
      l2, e21 = list_ops.tensor_list_pop_back(l2, element_dtype=dtypes.float32)
      l2, e22 = list_ops.tensor_list_pop_back(l2, element_dtype=dtypes.float32)
      l3, e31 = list_ops.tensor_list_pop_back(l3, element_dtype=dtypes.float32)
      l3, e32 = list_ops.tensor_list_pop_back(l3, element_dtype=dtypes.float32)
      result = sess.run([e11, [e21, e22], [e31, e32]], {num: 20})
      self.assertEqual(result, [1.0, [2.0, 1.0], [3.0, 1.0]])

  def testEmptyTensorList(self):
    dim = 7
    with self.cached_session() as sess, self.test_scope():
      p = array_ops.placeholder(dtypes.int32)
      l = list_ops.empty_tensor_list(
          element_shape=(p, 15), element_dtype=dtypes.float32)
      l = list_ops.tensor_list_push_back(
          l, constant_op.constant(1.0, shape=(dim, 15)))
      _, e = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   "Use TensorListReserve instead"):
        self.assertEqual(sess.run(e, {p: dim}), 1.0 * np.ones((dim, 15)))


if __name__ == "__main__":
  test.main()
