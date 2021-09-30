# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Test cases for ftrl ("follow the regularized leader") operations."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import googletest
from tensorflow.python.training import training_ops


class ResourceApplyFtrlTest(xla_test.XLATestCase):
  """Test cases for ftrl ops."""

  def setUp(self):
    super().setUp()
    self.rewrite_ops_for_tpu = ("TPU" in self.device and
                                test_util.is_mlir_bridge_enabled())

  def _eval(self, var, accum, linear, grad, lr, l1, l2, l2_shrinkage=0,
            lr_power=1, multiply_linear_by_lr=False):
    dtype = np.float32
    var = np.array(var, dtype=dtype)
    accum = np.array(accum, dtype=dtype)
    linear = np.array(linear, dtype=dtype)
    grad = np.array(grad, dtype=dtype)
    use_v2 = bool(l2_shrinkage)
    with self.session() as session:
      with self.test_scope():
        lr = constant_op.constant(lr, dtype=dtype)
        l1 = constant_op.constant(l1, dtype=dtype)
        l2 = constant_op.constant(l2, dtype=dtype)
        l2_shrinkage = constant_op.constant(l2_shrinkage, dtype=dtype)
        lr_power = constant_op.constant(lr_power, dtype=dtype)
        v_var = resource_variable_ops.ResourceVariable(var, dtype=dtype)
        v_accum = resource_variable_ops.ResourceVariable(accum, dtype=dtype)
        v_linear = resource_variable_ops.ResourceVariable(linear, dtype=dtype)
        session.run(v_var.create)
        session.run(v_accum.create)
        session.run(v_linear.create)
        assert not (use_v2 and multiply_linear_by_lr)
        if use_v2:
          session.run(training_ops.resource_apply_ftrl_v2(
              v_var.handle, v_accum.handle, v_linear.handle,
              grad, lr, l1, l2, l2_shrinkage, lr_power,
              multiply_linear_by_lr=multiply_linear_by_lr))
        else:
          session.run(training_ops.resource_apply_ftrl(
              v_var.handle, v_accum.handle, v_linear.handle,
              grad, lr, l1, l2, lr_power,
              multiply_linear_by_lr=multiply_linear_by_lr))
        return (v_var.read_value().eval().reshape(var.shape),
                v_accum.read_value().eval().reshape(accum.shape),
                v_linear.read_value().eval().reshape(linear.shape))

  def testAccum(self):
    """Test that accum is updated with grad^2."""
    accum = np.array([[[1, 3], [2, 5], [6, 8]]])
    grad = np.array([[[1, 3], [2, 5], [6, 8]]])
    _, new_accum, _ = self._eval(
        var=np.zeros((1, 3, 2)),
        accum=accum,
        linear=np.zeros((1, 3, 2)),
        grad=grad,
        lr=7, l1=3, l2=7, lr_power=2)
    self.assertAllClose(accum + grad*grad, new_accum)

  def testLinearNoGradient(self):
    """Test that if accum_new == accum, linear doesn't change."""
    _, _, linear = self._eval(
        var=np.ones((1, 3, 2)),
        accum=[[[1, 3], [2, 5], [6, 8]]],
        linear=[[[1, 2], [3, 4], [5, 6]]],
        grad=np.zeros((1, 3, 2)),  # make accum_new == acum
        lr=1, l1=3, l2=7, lr_power=2)
    self.assertAllClose([[[1, 2], [3, 4], [5, 6]]], linear)

  def testLinear(self):
    """Test the linear update for new_linear=2 and linear=1."""
    _, _, linear = self._eval(
        var=np.ones((1, 3, 2)),
        accum=np.ones((1, 3, 2)),
        linear=np.zeros((1, 3, 2)),
        grad=np.ones((1, 3, 2)),
        lr=1, l1=3, l2=7, lr_power=2)
    self.assertAllClose(1.75 * np.ones((1, 3, 2)), linear)

  def testLR(self):
    """Test that the linear update is divided by lr."""
    _, _, linear = self._eval(
        var=np.ones((1, 3, 2)),
        accum=np.ones((1, 3, 2)),
        linear=np.zeros((1, 3, 2)),
        grad=np.ones((1, 3, 2)),
        lr=5, l1=3, l2=7, lr_power=-1)
    self.assertAllClose(0.8 * np.ones((1, 3, 2)), linear)

  def testVar(self):
    """Test computation of var with linear=1.5, quadratic=1."""
    var, _, _ = self._eval(
        var=np.ones((1, 3, 2)),
        accum=np.ones((1, 3, 2)),
        linear=np.zeros((1, 3, 2)),
        grad=np.ones((1, 3, 2)),
        lr=1, l1=1, l2=0.25, lr_power=1)
    self.assertAllClose(-0.5 * np.ones((1, 3, 2)), var)

  def testVarClipped(self):
    """Test that var becomes 0 if |linear| < l1."""
    var, _, _ = self._eval(
        var=np.ones((1, 3, 2)),
        accum=np.ones((1, 3, 2)),
        linear=np.zeros((1, 3, 2)),
        grad=np.ones((1, 3, 2)),
        lr=1, l1=1.6, l2=0.25, lr_power=1)
    self.assertAllClose(np.zeros((1, 3, 2)), var)

  def testQuadratic(self):
    """Test that quadratic (here: -2) is the divisor of var."""
    var, _, _ = self._eval(
        var=np.ones((1, 3, 2)),
        accum=np.ones((1, 3, 2)),
        linear=np.zeros((1, 3, 2)),
        grad=np.ones((1, 3, 2)),
        lr=1, l1=1, l2=-1.25, lr_power=1)
    self.assertAllClose(0.25 * np.ones((1, 3, 2)), var)

  def testL2Shrinkage(self):
    """Test that 2 * l2_shrinkage * var is *not* added to the gradient."""
    _, accum, _ = self._eval(
        var=np.ones((1, 3, 2)),
        accum=np.zeros((1, 3, 2)),
        linear=np.zeros((1, 3, 2)),
        grad=np.zeros((1, 3, 2)),
        lr=7, l1=3, l2=7, lr_power=2, l2_shrinkage=0.5)
    self.assertAllClose(np.zeros((1, 3, 2)), accum)

  def testL2ShrinkageOnLinear(self):
    """Test that 2 * l2_shrinkage * var is added to linear."""
    _, _, linear = self._eval(
        var=np.ones((1, 3, 2)),
        accum=np.zeros((1, 3, 2)),
        linear=np.zeros((1, 3, 2)),
        grad=np.zeros((1, 3, 2)),
        lr=2, l1=3, l2=7, lr_power=0, l2_shrinkage=11)
    self.assertAllClose(22 * np.ones((1, 3, 2)), linear)

  def testMultiplyLinearByLR(self):
    """Test multiply_linear_by_lr = true for the linear variable."""
    _, _, linear = self._eval(
        var=np.zeros((1, 3, 2)),
        accum=np.zeros((1, 3, 2)),
        linear=np.ones((1, 3, 2)),
        grad=np.ones((1, 3, 2)),
        lr=6, l1=1, l2=-1.25, lr_power=0,
        multiply_linear_by_lr=True)
    self.assertAllClose(7 * np.ones((1, 3, 2)), linear)

  def testMultiplyLinearByLRClipping(self):
    """Test that multiply_linear_by_lr = true scales the clip margins."""
    var, _, _ = self._eval(
        var=np.ones((1, 3, 2)),
        accum=np.ones((1, 3, 2)),
        linear=np.zeros((1, 3, 2)),
        grad=np.ones((1, 3, 2)),
        lr=3, l1=1.0, l2=0.25, lr_power=1,
        multiply_linear_by_lr=True)
    self.assertAllClose(-0.25 * np.ones((1, 3, 2)), var)

  def testMultiplyLinearByLRClipZero(self):
    """Test that multiply_linear_by_lr = true still clips to 0."""
    var, _, _ = self._eval(
        var=np.ones((1, 3, 2)),
        accum=np.ones((1, 3, 2)),
        linear=np.zeros((1, 3, 2)),
        grad=np.ones((1, 3, 2)),
        lr=3, l1=1.2, l2=0.25, lr_power=1,
        multiply_linear_by_lr=True)
    self.assertAllClose(np.zeros((1, 3, 2)), var)


if __name__ == "__main__":
  googletest.main()
