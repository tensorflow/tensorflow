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
"""Functional tests for Ftrl operations."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import ftrl
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adagrad
from tensorflow.python.training import gradient_descent


class FtrlOptimizerTest(test.TestCase):

  def doTestFtrlwithoutRegularization(self, use_resource=False):
    # TODO(tanzheny, omalleyt): Fix test in eager mode.
    for dtype in [dtypes.float32]:
      with ops.Graph().as_default(), self.cached_session():
        if use_resource:
          var0 = variables.Variable([0.0, 0.0], dtype=dtype)
          var1 = variables.Variable([0.0, 0.0], dtype=dtype)
        else:
          var0 = variables.Variable([0.0, 0.0], dtype=dtype)
          var1 = variables.Variable([0.0, 0.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)
        opt = ftrl.Ftrl(
            3.0,
            initial_accumulator_value=0.1,
            l1_regularization_strength=0.0,
            l2_regularization_strength=0.0)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())

        v0_val, v1_val = self.evaluate([var0, var1])
        self.assertAllClose([0.0, 0.0], v0_val)
        self.assertAllClose([0.0, 0.0], v1_val)

        # Run 3 steps FTRL
        for _ in range(3):
          update.run()

        v0_val, v1_val = self.evaluate([var0, var1])
        self.assertAllCloseAccordingToType(
            np.array([-2.60260963, -4.29698515]), v0_val)
        self.assertAllCloseAccordingToType(
            np.array([-0.28432083, -0.56694895]), v1_val)

  def testFtrlWithoutRegularization(self):
    self.doTestFtrlwithoutRegularization(use_resource=False)

  def testResourceFtrlWithoutRegularization(self):
    self.doTestFtrlwithoutRegularization(use_resource=True)

  def testFtrlwithoutRegularization2(self):
    # TODO(tanzheny, omalleyt): Fix test in eager mode.
    for dtype in [dtypes.half, dtypes.float32]:
      with ops.Graph().as_default(), self.cached_session():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([4.0, 3.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)

        opt = ftrl.Ftrl(
            3.0,
            initial_accumulator_value=0.1,
            l1_regularization_strength=0.0,
            l2_regularization_strength=0.0)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())

        v0_val, v1_val = self.evaluate([var0, var1])
        self.assertAllCloseAccordingToType([1.0, 2.0], v0_val)
        self.assertAllCloseAccordingToType([4.0, 3.0], v1_val)

        # Run 3 steps FTRL
        for _ in range(3):
          update.run()
        v0_val, v1_val = self.evaluate([var0, var1])
        self.assertAllCloseAccordingToType(
            np.array([-2.55607247, -3.98729396]), v0_val)
        self.assertAllCloseAccordingToType(
            np.array([-0.28232238, -0.56096673]), v1_val)

  def testMinimizeSparseResourceVariable(self):
    # TODO(tanzheny, omalleyt): Fix test in eager mode.
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with ops.Graph().as_default(), self.cached_session():
        var0 = variables.Variable([[1.0, 2.0]], dtype=dtype)
        x = constant_op.constant([[4.0], [5.0]], dtype=dtype)

        def loss():
          pred = math_ops.matmul(embedding_ops.embedding_lookup([var0], [0]), x)  # pylint: disable=cell-var-from-loop
          return pred * pred

        sgd_op = ftrl.Ftrl(1.0).minimize(loss, var_list=[var0])
        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([[1.0, 2.0]], self.evaluate(var0))
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        self.assertAllCloseAccordingToType([[0, 1]],
                                           self.evaluate(var0),
                                           atol=0.01)

  def testFtrlWithL1(self):
    # TODO(tanzheny, omalleyt): Fix test in eager mode.
    for dtype in [dtypes.half, dtypes.float32]:
      with ops.Graph().as_default(), self.cached_session():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([4.0, 3.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)

        opt = ftrl.Ftrl(
            3.0,
            initial_accumulator_value=0.1,
            l1_regularization_strength=0.001,
            l2_regularization_strength=0.0)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())

        v0_val, v1_val = self.evaluate([var0, var1])
        self.assertAllCloseAccordingToType([1.0, 2.0], v0_val)
        self.assertAllCloseAccordingToType([4.0, 3.0], v1_val)

        # Run 10 steps FTRL
        for _ in range(10):
          update.run()
        v0_val, v1_val = self.evaluate([var0, var1])
        self.assertAllCloseAccordingToType(
            np.array([-7.66718769, -10.91273689]), v0_val)
        self.assertAllCloseAccordingToType(
            np.array([-0.93460727, -1.86147261]), v1_val)

  def testFtrlWithBeta(self):
    # TODO(tanzheny, omalleyt): Fix test in eager mode.
    for dtype in [dtypes.half, dtypes.float32]:
      with ops.Graph().as_default(), self.cached_session():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([4.0, 3.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)

        opt = ftrl.Ftrl(3.0, initial_accumulator_value=0.1, beta=0.1)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())

        v0_val, v1_val = self.evaluate([var0, var1])
        self.assertAllCloseAccordingToType([1.0, 2.0], v0_val)
        self.assertAllCloseAccordingToType([4.0, 3.0], v1_val)

        # Run 10 steps FTRL
        for _ in range(10):
          update.run()
        v0_val, v1_val = self.evaluate([var0, var1])
        self.assertAllCloseAccordingToType(
            np.array([-6.096838, -9.162214]), v0_val)
        self.assertAllCloseAccordingToType(
            np.array([-0.717741, -1.425132]), v1_val)

  def testFtrlWithL2_Beta(self):
    # TODO(tanzheny, omalleyt): Fix test in eager mode.
    for dtype in [dtypes.half, dtypes.float32]:
      with ops.Graph().as_default(), self.cached_session():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([4.0, 3.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)

        opt = ftrl.Ftrl(
            3.0,
            initial_accumulator_value=0.1,
            l1_regularization_strength=0.0,
            l2_regularization_strength=0.1,
            beta=0.1)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())

        v0_val, v1_val = self.evaluate([var0, var1])
        self.assertAllCloseAccordingToType([1.0, 2.0], v0_val)
        self.assertAllCloseAccordingToType([4.0, 3.0], v1_val)

        # Run 10 steps FTRL
        for _ in range(10):
          update.run()
        v0_val, v1_val = self.evaluate([var0, var1])
        self.assertAllCloseAccordingToType(
            np.array([-2.735487, -4.704625]), v0_val)
        self.assertAllCloseAccordingToType(
            np.array([-0.294335, -0.586556]), v1_val)

  def testFtrlWithL1_L2(self):
    # TODO(tanzheny, omalleyt): Fix test in eager mode.
    for dtype in [dtypes.half, dtypes.float32]:
      with ops.Graph().as_default(), self.cached_session():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([4.0, 3.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)

        opt = ftrl.Ftrl(
            3.0,
            initial_accumulator_value=0.1,
            l1_regularization_strength=0.001,
            l2_regularization_strength=2.0)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())

        v0_val, v1_val = self.evaluate([var0, var1])
        self.assertAllCloseAccordingToType([1.0, 2.0], v0_val)
        self.assertAllCloseAccordingToType([4.0, 3.0], v1_val)

        # Run 10 steps FTRL
        for _ in range(10):
          update.run()

        v0_val, v1_val = self.evaluate([var0, var1])
        self.assertAllCloseAccordingToType(
            np.array([-0.24059935, -0.46829352]), v0_val)
        self.assertAllCloseAccordingToType(
            np.array([-0.02406147, -0.04830509]), v1_val)

  def testFtrlWithL1_L2_L2Shrinkage(self):
    """Test the new FTRL op with support for l2 shrinkage.

    The addition of this parameter which places a constant pressure on weights
    towards the origin causes the gradient descent trajectory to differ. The
    weights will tend to have smaller magnitudes with this parameter set.
    """
    # TODO(tanzheny, omalleyt): Fix test in eager mode.
    for dtype in [dtypes.half, dtypes.float32]:
      with ops.Graph().as_default(), self.cached_session():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([4.0, 3.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)

        opt = ftrl.Ftrl(
            3.0,
            initial_accumulator_value=0.1,
            l1_regularization_strength=0.001,
            l2_regularization_strength=2.0,
            l2_shrinkage_regularization_strength=0.1)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())

        v0_val, v1_val = self.evaluate([var0, var1])
        self.assertAllCloseAccordingToType([1.0, 2.0], v0_val)
        self.assertAllCloseAccordingToType([4.0, 3.0], v1_val)

        # Run 10 steps FTRL
        for _ in range(10):
          update.run()

        v0_val, v1_val = self.evaluate([var0, var1])
        self.assertAllCloseAccordingToType(
            np.array([-0.22578995, -0.44345796]), v0_val)
        self.assertAllCloseAccordingToType(
            np.array([-0.14378493, -0.13229476]), v1_val)

  def testFtrlWithL1_L2_L2ShrinkageSparse(self):
    """Tests the new FTRL op with support for l2 shrinkage on sparse grads."""
    # TODO(tanzheny, omalleyt): Fix test in eager mode.
    for dtype in [dtypes.half, dtypes.float32]:
      with ops.Graph().as_default(), self.cached_session():
        var0 = variables.Variable([[1.0], [2.0]], dtype=dtype)
        var1 = variables.Variable([[4.0], [3.0]], dtype=dtype)
        grads0 = ops.IndexedSlices(
            constant_op.constant([0.1], shape=[1, 1], dtype=dtype),
            constant_op.constant([0]), constant_op.constant([2, 1]))
        grads1 = ops.IndexedSlices(
            constant_op.constant([0.02], shape=[1, 1], dtype=dtype),
            constant_op.constant([1]), constant_op.constant([2, 1]))

        opt = ftrl.Ftrl(
            3.0,
            initial_accumulator_value=0.1,
            l1_regularization_strength=0.001,
            l2_regularization_strength=2.0,
            l2_shrinkage_regularization_strength=0.1)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())

        v0_val, v1_val = self.evaluate([var0, var1])
        self.assertAllCloseAccordingToType([[1.0], [2.0]], v0_val)
        self.assertAllCloseAccordingToType([[4.0], [3.0]], v1_val)

        # Run 10 steps FTRL
        for _ in range(10):
          update.run()

        v0_val, v1_val = self.evaluate([var0, var1])
        self.assertAllCloseAccordingToType([[-0.22578995], [2.]], v0_val)
        self.assertAllCloseAccordingToType([[4.], [-0.13229476]], v1_val)

  def testFtrlWithL2ShrinkageDoesNotChangeLrSchedule(self):
    """Verifies that l2 shrinkage in FTRL does not change lr schedule."""
    # TODO(tanzheny, omalleyt): Fix test in eager mode.
    for dtype in [dtypes.half, dtypes.float32]:
      with ops.Graph().as_default(), self.cached_session() as sess:
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([1.0, 2.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
        grads1 = constant_op.constant([0.1, 0.2], dtype=dtype)

        opt0 = ftrl.Ftrl(
            3.0,
            initial_accumulator_value=0.1,
            l1_regularization_strength=0.001,
            l2_regularization_strength=2.0,
            l2_shrinkage_regularization_strength=0.1)
        opt1 = ftrl.Ftrl(
            3.0,
            initial_accumulator_value=0.1,
            l1_regularization_strength=0.001,
            l2_regularization_strength=2.0)
        update0 = opt0.apply_gradients([(grads0, var0)])
        update1 = opt1.apply_gradients([(grads1, var1)])
        self.evaluate(variables.global_variables_initializer())

        v0_val, v1_val = self.evaluate([var0, var1])
        self.assertAllCloseAccordingToType([1.0, 2.0], v0_val)
        self.assertAllCloseAccordingToType([1.0, 2.0], v1_val)

        # Run 10 steps FTRL
        for _ in range(10):
          update0.run()
          update1.run()

        v0_val, v1_val = self.evaluate([var0, var1])
        # var0 is experiencing L2 shrinkage so it should be smaller than var1
        # in magnitude.
        self.assertTrue((v0_val**2 < v1_val**2).all())
        accum0 = sess.run(opt0.get_slot(var0, "accumulator"))
        accum1 = sess.run(opt1.get_slot(var1, "accumulator"))
        # L2 shrinkage should not change how we update grad accumulator.
        self.assertAllCloseAccordingToType(accum0, accum1)

  def applyOptimizer(self, opt, dtype, steps=5, is_sparse=False):
    if is_sparse:
      var0 = variables.Variable([[0.0], [0.0]], dtype=dtype)
      var1 = variables.Variable([[0.0], [0.0]], dtype=dtype)
      grads0 = ops.IndexedSlices(
          constant_op.constant([0.1], shape=[1, 1], dtype=dtype),
          constant_op.constant([0]), constant_op.constant([2, 1]))
      grads1 = ops.IndexedSlices(
          constant_op.constant([0.02], shape=[1, 1], dtype=dtype),
          constant_op.constant([1]), constant_op.constant([2, 1]))
    else:
      var0 = variables.Variable([0.0, 0.0], dtype=dtype)
      var1 = variables.Variable([0.0, 0.0], dtype=dtype)
      grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
      grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)

    update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
    self.evaluate(variables.global_variables_initializer())

    v0_val, v1_val = self.evaluate([var0, var1])
    if is_sparse:
      self.assertAllCloseAccordingToType([[0.0], [0.0]], v0_val)
      self.assertAllCloseAccordingToType([[0.0], [0.0]], v1_val)
    else:
      self.assertAllCloseAccordingToType([0.0, 0.0], v0_val)
      self.assertAllCloseAccordingToType([0.0, 0.0], v1_val)

    # Run Ftrl for a few steps
    for _ in range(steps):
      update.run()

    v0_val, v1_val = self.evaluate([var0, var1])
    return v0_val, v1_val

  # When variables are initialized with Zero, FTRL-Proximal has two properties:
  # 1. Without L1&L2 but with fixed learning rate, FTRL-Proximal is identical
  # with GradientDescent.
  # 2. Without L1&L2 but with adaptive learning rate, FTRL-Proximal is identical
  # with Adagrad.
  # So, basing on these two properties, we test if our implementation of
  # FTRL-Proximal performs same updates as Adagrad or GradientDescent.
  def testEquivAdagradwithoutRegularization(self):
    # TODO(tanzheny, omalleyt): Fix test in eager mode.
    for dtype in [dtypes.half, dtypes.float32]:
      with ops.Graph().as_default(), self.cached_session():
        val0, val1 = self.applyOptimizer(
            ftrl.Ftrl(
                3.0,
                # Adagrad learning rate
                learning_rate_power=-0.5,
                initial_accumulator_value=0.1,
                l1_regularization_strength=0.0,
                l2_regularization_strength=0.0),
            dtype)

      with ops.Graph().as_default(), self.cached_session():
        val2, val3 = self.applyOptimizer(
            adagrad.AdagradOptimizer(3.0, initial_accumulator_value=0.1), dtype)

      self.assertAllCloseAccordingToType(val0, val2)
      self.assertAllCloseAccordingToType(val1, val3)

  def testEquivSparseAdagradwithoutRegularization(self):
    # TODO(tanzheny, omalleyt): Fix test in eager mode.
    for dtype in [dtypes.half, dtypes.float32]:
      with ops.Graph().as_default(), self.cached_session():
        val0, val1 = self.applyOptimizer(
            ftrl.Ftrl(
                3.0,
                # Adagrad learning rate
                learning_rate_power=-0.5,
                initial_accumulator_value=0.1,
                l1_regularization_strength=0.0,
                l2_regularization_strength=0.0),
            dtype,
            is_sparse=True)

      with ops.Graph().as_default(), self.cached_session():
        val2, val3 = self.applyOptimizer(
            adagrad.AdagradOptimizer(3.0, initial_accumulator_value=0.1),
            dtype,
            is_sparse=True)

      self.assertAllCloseAccordingToType(val0, val2)
      self.assertAllCloseAccordingToType(val1, val3)

  def testEquivSparseGradientDescentwithoutRegularization(self):
    # TODO(tanzheny, omalleyt): Fix test in eager mode.
    for dtype in [dtypes.half, dtypes.float32]:
      with ops.Graph().as_default(), self.cached_session():
        val0, val1 = self.applyOptimizer(
            ftrl.Ftrl(
                3.0,
                # Fixed learning rate
                learning_rate_power=-0.0,
                initial_accumulator_value=0.1,
                l1_regularization_strength=0.0,
                l2_regularization_strength=0.0),
            dtype,
            is_sparse=True)

      with ops.Graph().as_default(), self.cached_session():
        val2, val3 = self.applyOptimizer(
            gradient_descent.GradientDescentOptimizer(3.0),
            dtype,
            is_sparse=True)

      self.assertAllCloseAccordingToType(val0, val2)
      self.assertAllCloseAccordingToType(val1, val3)

  def testEquivGradientDescentwithoutRegularization(self):
    # TODO(tanzheny, omalleyt): Fix test in eager mode.
    for dtype in [dtypes.half, dtypes.float32]:
      with ops.Graph().as_default(), self.cached_session():
        val0, val1 = self.applyOptimizer(
            ftrl.Ftrl(
                3.0,
                # Fixed learning rate
                learning_rate_power=-0.0,
                initial_accumulator_value=0.1,
                l1_regularization_strength=0.0,
                l2_regularization_strength=0.0),
            dtype)

      with ops.Graph().as_default(), self.cached_session():
        val2, val3 = self.applyOptimizer(
            gradient_descent.GradientDescentOptimizer(3.0), dtype)

      self.assertAllCloseAccordingToType(val0, val2)
      self.assertAllCloseAccordingToType(val1, val3)


if __name__ == "__main__":
  test.main()
