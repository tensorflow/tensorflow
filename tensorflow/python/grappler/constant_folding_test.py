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
"""Tests for Grappler Constant Folding."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class ConstantFoldingTest(test.TestCase):

  # See b/76008022.
  def testScanInsideWhile(self):

    def loop_cond(idx_step, *unused_args):
      return idx_step < 1

    def loop_body(idx_step, y):
      x = array_ops.zeros([10, 20, 30], dtype=dtypes.float32)
      x = functional_ops.scan(
          math_ops.add,
          x,
          initializer=array_ops.zeros([20, 30], dtype=dtypes.float32),
          back_prop=False,
          parallel_iterations=1)

      with ops.device('/cpu:0'):
        y = array_ops.identity(x)

        return idx_step + 1, y

    if test.is_gpu_available(cuda_only=True):
      init_y = array_ops.zeros([10, 20, 30], dtype=dtypes.float32)
      _, y = control_flow_ops.while_loop(
          loop_cond,
          loop_body,
          loop_vars=[0, init_y],
          back_prop=False,
          parallel_iterations=1)
      with session.Session() as sess:
        y_v = sess.run(y)
        self.assertAllEqual(np.zeros([10, 20, 30]), y_v)


if __name__ == '__main__':
  test.main()
