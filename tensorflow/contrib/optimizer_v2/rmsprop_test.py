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
"""Tests for rmsprop optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

from absl.testing import parameterized
import numpy as np

from tensorflow.contrib.optimizer_v2 import rmsprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

_DATA_TYPES = [dtypes.half, dtypes.float32]

_TEST_PARAM_VALUES = [
    # learning_rate, decay, momentum, epsilon, centered, use_resource
    [0.5, 0.9, 0.0, 1.0, True, False],
    [0.5, 0.9, 0.0, 1.0, False, False],
    [0.5, 0.9, 0.0, 1.0, True, True],
    [0.5, 0.9, 0.0, 1.0, False, True],
    [0.1, 0.9, 0.0, 1.0, True, False],
    [0.5, 0.95, 0.0, 1.0, False, False],
    [0.5, 0.8, 0.0, 1e-3, True, False],
    [0.5, 0.8, 0.9, 1e-3, True, False],
]


class RMSPropOptimizerTest(test.TestCase, parameterized.TestCase):

  def _rmsprop_update_numpy(self, var, g, mg, rms, mom, lr, decay, momentum,
                            centered):
    rms_t = rms * decay + (1 - decay) * g * g
    if centered:
      mg_t = mg * decay + (1 - decay) * g
      denom_t = rms_t - mg_t * mg_t
    else:
      mg_t = mg
      denom_t = rms_t
    mom_t = momentum * mom + lr * g / np.sqrt(denom_t, dtype=denom_t.dtype)
    var_t = var - mom_t
    return var_t, mg_t, rms_t, mom_t

  def _sparse_rmsprop_update_numpy(self, var, gindexs, gvalues, mg, rms, mom,
                                   lr, decay, momentum, centered):
    mg_t = copy.deepcopy(mg)
    rms_t = copy.deepcopy(rms)
    mom_t = copy.deepcopy(mom)
    var_t = copy.deepcopy(var)
    for i in range(len(gindexs)):
      gindex = gindexs[i]
      gvalue = gvalues[i]
      rms_t[gindex] = rms[gindex] * decay + (1 - decay) * gvalue * gvalue
      denom_t = rms_t[gindex]
      if centered:
        mg_t[gindex] = mg_t[gindex] * decay + (1 - decay) * gvalue
        denom_t -= mg_t[gindex] * mg_t[gindex]
      mom_t[gindex] = momentum * mom[gindex] + lr * gvalue / np.sqrt(denom_t)
      var_t[gindex] = var[gindex] - mom_t[gindex]
    return var_t, mg_t, rms_t, mom_t

  @parameterized.named_parameters(
      *test_util.generate_combinations_with_testcase_name(
          dtype=_DATA_TYPES, param_value=_TEST_PARAM_VALUES))
  def testDense(self, dtype, param_value):
    (learning_rate, decay, momentum, epsilon, centered, use_resource) = tuple(
        param_value)
    with self.session(use_gpu=True):
      # Initialize variables for numpy implementation.
      var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
      grads0_np = np.array([0.1, 0.2], dtype=dtype.as_numpy_dtype)
      var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
      grads1_np = np.array([0.01, 0.2], dtype=dtype.as_numpy_dtype)

      if use_resource:
        var0 = resource_variable_ops.ResourceVariable(var0_np)
        var1 = resource_variable_ops.ResourceVariable(var1_np)
      else:
        var0 = variables.Variable(var0_np)
        var1 = variables.Variable(var1_np)
      grads0 = constant_op.constant(grads0_np)
      grads1 = constant_op.constant(grads1_np)
      opt = rmsprop.RMSPropOptimizer(
          learning_rate=learning_rate,
          decay=decay,
          momentum=momentum,
          epsilon=epsilon,
          centered=centered)

      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      variables.global_variables_initializer().run()

      mg0 = opt.get_slot(var0, "mg")
      self.assertEqual(mg0 is not None, centered)
      mg1 = opt.get_slot(var1, "mg")
      self.assertEqual(mg1 is not None, centered)
      rms0 = opt.get_slot(var0, "rms")
      self.assertIsNotNone(rms0)
      rms1 = opt.get_slot(var1, "rms")
      self.assertIsNotNone(rms1)
      mom0 = opt.get_slot(var0, "momentum")
      self.assertIsNotNone(mom0)
      mom1 = opt.get_slot(var1, "momentum")
      self.assertIsNotNone(mom1)

      mg0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
      mg1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
      rms0_np = np.array([epsilon, epsilon], dtype=dtype.as_numpy_dtype)
      rms1_np = np.array([epsilon, epsilon], dtype=dtype.as_numpy_dtype)
      mom0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
      mom1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)

      # Fetch params to validate initial values
      self.assertAllClose([1.0, 2.0], var0.eval())
      self.assertAllClose([3.0, 4.0], var1.eval())

      # Run 4 steps of RMSProp
      for _ in range(4):
        update.run()

        var0_np, mg0_np, rms0_np, mom0_np = self._rmsprop_update_numpy(
            var0_np, grads0_np, mg0_np, rms0_np, mom0_np, learning_rate,
            decay, momentum, centered)
        var1_np, mg1_np, rms1_np, mom1_np = self._rmsprop_update_numpy(
            var1_np, grads1_np, mg1_np, rms1_np, mom1_np, learning_rate,
            decay, momentum, centered)

        # Validate updated params
        if centered:
          self.assertAllCloseAccordingToType(mg0_np, mg0.eval())
          self.assertAllCloseAccordingToType(mg1_np, mg1.eval())
        self.assertAllCloseAccordingToType(rms0_np, rms0.eval())
        self.assertAllCloseAccordingToType(rms1_np, rms1.eval())
        self.assertAllCloseAccordingToType(mom0_np, mom0.eval())
        self.assertAllCloseAccordingToType(mom1_np, mom1.eval())
        # TODO(b/117393988): Reduce tolerances for float16.
        self.assertAllCloseAccordingToType(
            var0_np, var0.eval(), half_rtol=3e-3, half_atol=3e-3)
        self.assertAllCloseAccordingToType(
            var1_np, var1.eval(), half_rtol=3e-3, half_atol=3e-3)

  @parameterized.parameters([dtypes.float32, dtypes.float64])
  def testMinimizeSparseResourceVariable(self, dtype):
    with self.cached_session():
      var0 = resource_variable_ops.ResourceVariable([[1.0, 2.0]], dtype=dtype)
      x = constant_op.constant([[4.0], [5.0]], dtype=dtype)
      pred = math_ops.matmul(embedding_ops.embedding_lookup([var0], [0]), x)
      loss = pred * pred
      sgd_op = rmsprop.RMSPropOptimizer(
          learning_rate=1.0,
          decay=0.0,
          momentum=0.0,
          epsilon=0.0,
          centered=False).minimize(loss)
      variables.global_variables_initializer().run()
      # Fetch params to validate initial values
      self.assertAllCloseAccordingToType([[1.0, 2.0]], var0.eval())
      # Run 1 step of sgd
      sgd_op.run()
      # Validate updated params
      self.assertAllCloseAccordingToType(
          [[0., 1.]], var0.eval(), atol=0.01)

  @parameterized.parameters([dtypes.float32, dtypes.float64])
  def testMinimizeSparseResourceVariableCentered(self, dtype):
    with self.cached_session():
      var0 = resource_variable_ops.ResourceVariable([[1.0, 2.0]], dtype=dtype)
      x = constant_op.constant([[4.0], [5.0]], dtype=dtype)
      pred = math_ops.matmul(embedding_ops.embedding_lookup([var0], [0]), x)
      loss = pred * pred
      sgd_op = rmsprop.RMSPropOptimizer(
          learning_rate=1.0,
          decay=0.1,
          momentum=0.0,
          epsilon=1.0,
          centered=True).minimize(loss)
      variables.global_variables_initializer().run()
      # Fetch params to validate initial values
      self.assertAllCloseAccordingToType([[1.0, 2.0]], var0.eval())
      # Run 1 step of sgd
      sgd_op.run()
      # Validate updated params
      self.assertAllCloseAccordingToType(
          [[-7/3.0, -4/3.0]], var0.eval(), atol=0.01)

  @parameterized.named_parameters(
      *test_util.generate_combinations_with_testcase_name(
          dtype=_DATA_TYPES, param_value=_TEST_PARAM_VALUES))
  def testSparse(self, dtype, param_value):
    (learning_rate, decay, momentum, epsilon, centered, _) = tuple(
        param_value)
    with self.session(use_gpu=True):
      # Initialize variables for numpy implementation.
      var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
      grads0_np = np.array([0.1], dtype=dtype.as_numpy_dtype)
      var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
      grads1_np = np.array([0.01], dtype=dtype.as_numpy_dtype)

      var0 = variables.Variable(var0_np)
      var1 = variables.Variable(var1_np)
      grads0_np_indices = np.array([0], dtype=np.int32)
      grads0 = ops.IndexedSlices(
          constant_op.constant(grads0_np),
          constant_op.constant(grads0_np_indices), constant_op.constant([1]))
      grads1_np_indices = np.array([1], dtype=np.int32)
      grads1 = ops.IndexedSlices(
          constant_op.constant(grads1_np),
          constant_op.constant(grads1_np_indices), constant_op.constant([1]))
      opt = rmsprop.RMSPropOptimizer(
          learning_rate=learning_rate,
          decay=decay,
          momentum=momentum,
          epsilon=epsilon,
          centered=centered)
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      variables.global_variables_initializer().run()

      mg0 = opt.get_slot(var0, "mg")
      self.assertEqual(mg0 is not None, centered)
      mg1 = opt.get_slot(var1, "mg")
      self.assertEqual(mg1 is not None, centered)
      rms0 = opt.get_slot(var0, "rms")
      self.assertIsNotNone(rms0)
      rms1 = opt.get_slot(var1, "rms")
      self.assertIsNotNone(rms1)
      mom0 = opt.get_slot(var0, "momentum")
      self.assertIsNotNone(mom0)
      mom1 = opt.get_slot(var1, "momentum")
      self.assertIsNotNone(mom1)

      mg0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
      mg1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
      rms0_np = np.array([epsilon, epsilon], dtype=dtype.as_numpy_dtype)
      rms1_np = np.array([epsilon, epsilon], dtype=dtype.as_numpy_dtype)
      mom0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
      mom1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)

      # Fetch params to validate initial values
      self.assertAllClose([1.0, 2.0], var0.eval())
      self.assertAllClose([3.0, 4.0], var1.eval())

      # Run 4 steps of RMSProp
      for _ in range(4):
        update.run()

        var0_np, mg0_np, rms0_np, mom0_np = self._sparse_rmsprop_update_numpy(
            var0_np, grads0_np_indices, grads0_np, mg0_np, rms0_np, mom0_np,
            learning_rate, decay, momentum, centered)
        var1_np, mg1_np, rms1_np, mom1_np = self._sparse_rmsprop_update_numpy(
            var1_np, grads1_np_indices, grads1_np, mg1_np, rms1_np, mom1_np,
            learning_rate, decay, momentum, centered)

        # Validate updated params
        if centered:
          self.assertAllCloseAccordingToType(mg0_np, mg0.eval())
          self.assertAllCloseAccordingToType(mg1_np, mg1.eval())
        self.assertAllCloseAccordingToType(rms0_np, rms0.eval())
        self.assertAllCloseAccordingToType(rms1_np, rms1.eval())
        self.assertAllCloseAccordingToType(mom0_np, mom0.eval())
        self.assertAllCloseAccordingToType(mom1_np, mom1.eval())
        self.assertAllCloseAccordingToType(var0_np, var0.eval())
        self.assertAllCloseAccordingToType(var1_np, var1.eval())

  @parameterized.parameters(_DATA_TYPES)
  def testWithoutMomentum(self, dtype):
    with self.session(use_gpu=True):
      var0 = variables.Variable([1.0, 2.0], dtype=dtype)
      var1 = variables.Variable([3.0, 4.0], dtype=dtype)
      grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
      grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
      opt = rmsprop.RMSPropOptimizer(
          learning_rate=2.0, decay=0.9, momentum=0.0, epsilon=1.0)
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      variables.global_variables_initializer().run()

      rms0 = opt.get_slot(var0, "rms")
      self.assertIsNotNone(rms0)
      rms1 = opt.get_slot(var1, "rms")
      self.assertIsNotNone(rms1)
      mom0 = opt.get_slot(var0, "momentum")
      self.assertIsNotNone(mom0)
      mom1 = opt.get_slot(var1, "momentum")
      self.assertIsNotNone(mom1)

      # Fetch params to validate initial values
      self.assertAllClose([1.0, 2.0], var0.eval())
      self.assertAllClose([3.0, 4.0], var1.eval())
      # Step 1: the rms accumulators where 1. So we should see a normal
      # update: v -= grad * learning_rate
      update.run()
      # Check the root mean square accumulators.
      self.assertAllCloseAccordingToType(
          np.array([0.901, 0.901]), rms0.eval())
      self.assertAllCloseAccordingToType(
          np.array([0.90001, 0.90001]), rms1.eval())
      # Check the parameters.
      self.assertAllCloseAccordingToType(
          np.array([
              1.0 - (0.1 * 2.0 / math.sqrt(0.901)),
              2.0 - (0.1 * 2.0 / math.sqrt(0.901))
          ]), var0.eval())
      self.assertAllCloseAccordingToType(
          np.array([
              3.0 - (0.01 * 2.0 / math.sqrt(0.90001)),
              4.0 - (0.01 * 2.0 / math.sqrt(0.90001))
          ]), var1.eval())
      # Step 2: the root mean square accumulators contain the previous update.
      update.run()
      # Check the rms accumulators.
      self.assertAllCloseAccordingToType(
          np.array([0.901 * 0.9 + 0.001, 0.901 * 0.9 + 0.001]), rms0.eval())
      self.assertAllCloseAccordingToType(
          np.array([0.90001 * 0.9 + 1e-5, 0.90001 * 0.9 + 1e-5]), rms1.eval())
      # Check the parameters.
      self.assertAllCloseAccordingToType(
          np.array([
              1.0 - (0.1 * 2.0 / math.sqrt(0.901)) -
              (0.1 * 2.0 / math.sqrt(0.901 * 0.9 + 0.001)),
              2.0 - (0.1 * 2.0 / math.sqrt(0.901)) -
              (0.1 * 2.0 / math.sqrt(0.901 * 0.9 + 0.001))
          ]), var0.eval())
      self.assertAllCloseAccordingToType(
          np.array([
              3.0 - (0.01 * 2.0 / math.sqrt(0.90001)) -
              (0.01 * 2.0 / math.sqrt(0.90001 * 0.9 + 1e-5)),
              4.0 - (0.01 * 2.0 / math.sqrt(0.90001)) -
              (0.01 * 2.0 / math.sqrt(0.90001 * 0.9 + 1e-5))
          ]), var1.eval())

  @parameterized.parameters(_DATA_TYPES)
  def testWithMomentum(self, dtype):
    with self.session(use_gpu=True):
      var0 = variables.Variable([1.0, 2.0], dtype=dtype)
      var1 = variables.Variable([3.0, 4.0], dtype=dtype)
      grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
      grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)

      opt = rmsprop.RMSPropOptimizer(
          learning_rate=2.0, decay=0.9, momentum=0.5, epsilon=1.0)
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      variables.global_variables_initializer().run()

      rms0 = opt.get_slot(var0, "rms")
      self.assertIsNotNone(rms0)
      rms1 = opt.get_slot(var1, "rms")
      self.assertIsNotNone(rms1)
      mom0 = opt.get_slot(var0, "momentum")
      self.assertIsNotNone(mom0)
      mom1 = opt.get_slot(var1, "momentum")
      self.assertIsNotNone(mom1)

      # Fetch params to validate initial values
      self.assertAllClose([1.0, 2.0], var0.eval())
      self.assertAllClose([3.0, 4.0], var1.eval())
      # Step 1: rms = 1, mom = 0. So we should see a normal
      # update: v -= grad * learning_rate
      update.run()
      # Check the root mean square accumulators.
      self.assertAllCloseAccordingToType(
          np.array([0.901, 0.901]), rms0.eval())
      self.assertAllCloseAccordingToType(
          np.array([0.90001, 0.90001]), rms1.eval())
      # Check the momentum accumulators
      self.assertAllCloseAccordingToType(
          np.array([(0.1 * 2.0 / math.sqrt(0.901)),
                    (0.1 * 2.0 / math.sqrt(0.901))]), mom0.eval())
      self.assertAllCloseAccordingToType(
          np.array([(0.01 * 2.0 / math.sqrt(0.90001)),
                    (0.01 * 2.0 / math.sqrt(0.90001))]), mom1.eval())

      # Check that the parameters.
      self.assertAllCloseAccordingToType(
          np.array([
              1.0 - (0.1 * 2.0 / math.sqrt(0.901)),
              2.0 - (0.1 * 2.0 / math.sqrt(0.901))
          ]), var0.eval())
      self.assertAllCloseAccordingToType(
          np.array([
              3.0 - (0.01 * 2.0 / math.sqrt(0.90001)),
              4.0 - (0.01 * 2.0 / math.sqrt(0.90001))
          ]), var1.eval())

      # Step 2: the root mean square accumulators contain the previous update.
      update.run()
      # Check the rms accumulators.
      self.assertAllCloseAccordingToType(
          np.array([0.901 * 0.9 + 0.001, 0.901 * 0.9 + 0.001]), rms0.eval())
      self.assertAllCloseAccordingToType(
          np.array([0.90001 * 0.9 + 1e-5, 0.90001 * 0.9 + 1e-5]), rms1.eval())
      self.assertAllCloseAccordingToType(
          np.array([
              0.5 * (0.1 * 2.0 / math.sqrt(0.901)) +
              (0.1 * 2.0 / math.sqrt(0.901 * 0.9 + 0.001)),
              0.5 * (0.1 * 2.0 / math.sqrt(0.901)) +
              (0.1 * 2.0 / math.sqrt(0.901 * 0.9 + 0.001))
          ]), mom0.eval())
      self.assertAllCloseAccordingToType(
          np.array([
              0.5 * (0.01 * 2.0 / math.sqrt(0.90001)) +
              (0.01 * 2.0 / math.sqrt(0.90001 * 0.9 + 1e-5)),
              0.5 * (0.01 * 2.0 / math.sqrt(0.90001)) +
              (0.01 * 2.0 / math.sqrt(0.90001 * 0.9 + 1e-5))
          ]), mom1.eval())

      # Check the parameters.
      self.assertAllCloseAccordingToType(
          np.array([
              1.0 - (0.1 * 2.0 / math.sqrt(0.901)) -
              (0.5 * (0.1 * 2.0 / math.sqrt(0.901)) +
               (0.1 * 2.0 / math.sqrt(0.901 * 0.9 + 0.001))),
              2.0 - (0.1 * 2.0 / math.sqrt(0.901)) -
              (0.5 * (0.1 * 2.0 / math.sqrt(0.901)) +
               (0.1 * 2.0 / math.sqrt(0.901 * 0.9 + 0.001)))
          ]), var0.eval())

      self.assertAllCloseAccordingToType(
          np.array([
              3.0 - (0.01 * 2.0 / math.sqrt(0.90001)) -
              (0.5 * (0.01 * 2.0 / math.sqrt(0.90001)) +
               (0.01 * 2.0 / math.sqrt(0.90001 * 0.9 + 1e-5))),
              4.0 - (0.01 * 2.0 / math.sqrt(0.90001)) -
              (0.5 * (0.01 * 2.0 / math.sqrt(0.90001)) +
               (0.01 * 2.0 / math.sqrt(0.90001 * 0.9 + 1e-5)))
          ]), var1.eval())


class SlotColocationTest(test.TestCase, parameterized.TestCase):

  @parameterized.parameters([True, False])
  @test_util.run_in_graph_and_eager_modes
  def testRunMinimizeOnGPUForCPUVariables(self, use_resource):
    if not context.context().num_gpus():
      self.skipTest("No GPUs found")

    with ops.device("/device:CPU:0"):
      if use_resource:
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0],
                                                      dtype=dtypes.float32)
        var1 = resource_variable_ops.ResourceVariable([3.0, 4.0],
                                                      dtype=dtypes.float32)
        global_step = resource_variable_ops.ResourceVariable(
            array_ops.zeros([], dtypes.int64), name="global_step")
      else:
        var0 = variables.Variable([1.0, 2.0], dtype=dtypes.float32)
        var1 = variables.Variable([3.0, 4.0], dtype=dtypes.float32)
        global_step = variables.Variable(
            array_ops.zeros([], dtypes.int64), name="global_step")

    def loss():
      return 5 * var0 + 3 * var1

    opt = rmsprop.RMSPropOptimizer(
        learning_rate=1.0, decay=0.9, momentum=0.5, epsilon=1.0)

    # Fetch params to validate initial values
    self.evaluate(variables.global_variables_initializer())
    self.assertAllClose([1.0, 2.0], self.evaluate(var0))
    self.assertAllClose([3.0, 4.0], self.evaluate(var1))

    # Run 1 step through optimizer on GPU.
    # Slot variables are created the first time optimizer is used on some
    # variable. This tests that slot variables will be colocated with the base
    # variable.
    with ops.device("/device:GPU:0"):
      # Note that for eager execution, minimize expects a function instead of a
      # Tensor.
      opt_op = opt.minimize(loss, global_step, [var0, var1])
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(opt_op)

    # Validate updated params, All variables should have decreased.
    self.assertTrue(all(v < 0.0 for v in self.evaluate(var0)),
                    msg="updated variables: %s" % self.evaluate(var0))
    self.assertTrue(all(v < 2.0 for v in self.evaluate(var1)),
                    msg="updated variables: %s" % self.evaluate(var1))


if __name__ == "__main__":
  test.main()
