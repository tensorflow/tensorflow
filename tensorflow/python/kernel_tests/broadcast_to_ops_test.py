# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for broadcast_to ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test as test_lib


class BroadcastToTest(test_util.TensorFlowTestCase):

  def testBroadcastToBasic(self):
    for dtype in [np.uint8, np.uint16, np.int8, np.int16, np.int32, np.int64]:
      with self.test_session(use_gpu=True):
        x = np.array([1, 2, 3], dtype=dtype)
        v_tf = array_ops.broadcast_to(constant_op.constant(x), [3, 3])
        v_np = np.broadcast_to(x, [3, 3])
        self.assertAllEqual(v_tf.eval(), v_np)

  def testBroadcastToString(self):
    with self.test_session(use_gpu=True):
      x = np.array([b"1", b"2", b"3"])
      v_tf = array_ops.broadcast_to(constant_op.constant(x), [3, 3])
      v_np = np.broadcast_to(x, [3, 3])
      self.assertAllEqual(v_tf.eval(), v_np)

  def testBroadcastToBool(self):
    with self.test_session(use_gpu=True):
      x = np.array([True, False, True], dtype=np.bool)
      v_tf = array_ops.broadcast_to(constant_op.constant(x), [3, 3])
      v_np = np.broadcast_to(x, [3, 3])
      self.assertAllEqual(v_tf.eval(), v_np)

  def testBroadcastToShape(self):
    for input_dim in range(1, 6):
      for output_dim in range(input_dim, 6):
        with self.test_session(use_gpu=True):
          input_shape = [2] * input_dim
          output_shape = [2] * output_dim
          x = np.array(np.random.randint(5, size=input_shape), dtype=np.int32)
          v_tf = array_ops.broadcast_to(constant_op.constant(x), output_shape)
          v_np = np.broadcast_to(x, output_shape)
          self.assertAllEqual(v_tf.eval(), v_np)

  def testBroadcastToScalar(self):
    with self.test_session(use_gpu=True):
      x = np.array(1, dtype=np.int32)
      v_tf = array_ops.broadcast_to(constant_op.constant(x), [3, 3])
      v_np = np.broadcast_to(x, [3, 3])
      self.assertAllEqual(v_tf.eval(), v_np)

  def testBroadcastToShapeTypeAndInference(self):
    for dtype in [dtypes.int32, dtypes.int64]:
      with self.test_session(use_gpu=True):
        x = np.array([1, 2, 3])
        v_tf = array_ops.broadcast_to(
            constant_op.constant(x),
            constant_op.constant([3, 3], dtype=dtype))
        shape = v_tf.get_shape().as_list()
        v_np = np.broadcast_to(x, [3, 3])
        self.assertAllEqual(v_tf.eval(), v_np)
        # check shape inference when shape input is constant
        self.assertAllEqual(shape, v_np.shape)

if __name__ == "__main__":
  test_lib.main()
