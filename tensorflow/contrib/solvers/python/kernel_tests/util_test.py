# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.solvers.python.ops import util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class UtilTest(test.TestCase):

  def _testCreateOperator(self, use_static_shape_):
    for dtype in np.float32, np.float64:
      a_np = np.array([[1., 2.], [3., 4.], [5., 6.]], dtype=dtype)
      x_np = np.array([[2.], [-3.]], dtype=dtype)
      y_np = np.array([[2], [-3.], [5.]], dtype=dtype)
      with self.test_session() as sess:
        if use_static_shape_:
          a = constant_op.constant(a_np, dtype=dtype)
          x = constant_op.constant(x_np, dtype=dtype)
          y = constant_op.constant(y_np, dtype=dtype)
        else:
          a = array_ops.placeholder(dtype)
          x = array_ops.placeholder(dtype)
          y = array_ops.placeholder(dtype)
        op = util.create_operator(a)
        ax = op.apply(x)
        aty = op.apply_adjoint(y)
        op_shape = ops.convert_to_tensor(op.shape)
        if use_static_shape_:
          op_shape_val, ax_val, aty_val = sess.run([op_shape, ax, aty])
        else:
          op_shape_val, ax_val, aty_val = sess.run(
              [op_shape, ax, aty], feed_dict={a: a_np,
                                              x: x_np,
                                              y: y_np})
      self.assertAllEqual(op_shape_val, [3, 2])
      self.assertAllClose(ax_val, np.dot(a_np, x_np))
      self.assertAllClose(aty_val, np.dot(a_np.T, y_np))

  def testCreateOperator(self):
    self._testCreateOperator(True)

  def testCreateOperatorUnknownShape(self):
    self._testCreateOperator(False)

  def testL2Norm(self):
    with self.test_session():
      x_np = np.array([[2], [-3.], [5.]])
      x_norm_np = np.linalg.norm(x_np)
      x_normalized_np = x_np / x_norm_np
      x = constant_op.constant(x_np)
      l2norm = util.l2norm(x)
      l2norm_squared = util.l2norm_squared(x)
      x_normalized, x_norm = util.l2normalize(x)
      self.assertAllClose(l2norm.eval(), x_norm_np)
      self.assertAllClose(l2norm_squared.eval(), np.square(x_norm_np))
      self.assertAllClose(x_norm.eval(), x_norm_np)
      self.assertAllClose(x_normalized.eval(), x_normalized_np)


if __name__ == '__main__':
  test.main()
