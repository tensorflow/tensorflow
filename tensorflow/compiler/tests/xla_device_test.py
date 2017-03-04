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
"""Test cases for XLA devices."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class XlaDeviceTest(test.TestCase):

  def testCopies(self):
    """Tests that copies between GPU and XLA devices work."""
    if not test.is_gpu_available():
      return

    with session_lib.Session() as sess:
      x = array_ops.placeholder(dtypes.float32, [2])
      with ops.device("GPU"):
        y = x * 2
      with ops.device("device:XLA_CPU:0"):
        z = y * y
      with ops.device("GPU"):
        w = y + z
      result = sess.run(w, {x: [1.5, 0.5]})
    self.assertAllClose(result, [12., 2.], rtol=1e-3)

  def testLoops(self):
    """Tests that loops work on XLA devices."""

    with session_lib.Session() as session:
      x = array_ops.placeholder(dtypes.float32)
      with ops.device("device:XLA_CPU:0"):
        c = lambda i, _: math_ops.less(i, 5)
        b = lambda i, x: (i + 1, x * 2.0 + 1.0)
        _, y = control_flow_ops.while_loop(c, b, (constant_op.constant(0), x))

      result = session.run(y, {x: np.float32(2)})
      self.assertAllClose(result, np.float32(95), rtol=1e-3)

  def testCond(self):
    """Tests that tf.cond works on XLA devices."""

    with session_lib.Session() as session:
      x = array_ops.placeholder(dtypes.float32)
      y = array_ops.placeholder(dtypes.float32)
      c = array_ops.placeholder(dtypes.bool)
      with ops.device("device:XLA_CPU:0"):
        z = x + 1.0
        w = control_flow_ops.cond(c, lambda: z, lambda: y)
        t = math_ops.add(z, w)

      result = session.run(t, {x: np.float32(2), y: np.float32(4), c: True})
      self.assertAllClose(result, np.float32(6), rtol=1e-3)


if __name__ == "__main__":
  test.main()
