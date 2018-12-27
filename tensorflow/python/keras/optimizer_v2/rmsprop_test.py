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
"""Tests for rmsprop."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import itertools
import math

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras.optimizer_v2 import rmsprop
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

_DATA_TYPES = [dtypes.half, dtypes.float32]

_TEST_PARAM_VALUES = [
    # learning_rate, rho, momentum, epsilon, centered
    [0.05, 0.9, 0.0, 1e-3, True],
    [0.05, 0.9, 0.0, 1e-3, False],
    [0.1, 0.9, 0.0, 1e-3, True],
    [0.01, 0.9, 0.0, 1e-5, True],
    [0.01, 0.9, 0.9, 1e-5, True],
]

_TESTPARAMS = [
    [data_type] + values
    for data_type, values in itertools.product(_DATA_TYPES, _TEST_PARAM_VALUES)
]


class RMSpropOptimizerTest(test.TestCase):

  def _rmsprop_update_numpy(self, var, g, mg, rms, mom, lr, rho, momentum,
                            epsilon, centered):
    rms_t = rms * rho + (1 - rho) * g * g
    denom_t = rms_t + epsilon
    if centered:
      mg_t = mg * rho + (1 - rho) * g
      denom_t -= mg_t * mg_t
    else:
      mg_t = mg
    mom_t = momentum * mom + lr * g / np.sqrt(denom_t, dtype=denom_t.dtype)
    var_t = var - mom_t
    return var_t, mg_t, rms_t, mom_t

  def _sparse_rmsprop_update_numpy(self, var, gindexs, gvalues, mg, rms, mom,
                                   lr, rho, momentum, epsilon, centered):
    mg_t = copy.deepcopy(mg)
    rms_t = copy.deepcopy(rms)
    mom_t = copy.deepcopy(mom)
    var_t = copy.deepcopy(var)
    for i in range(len(gindexs)):
      gindex = gindexs[i]
      gvalue = gvalues[i]
      rms_t[gindex] = rms[gindex] * rho + (1 - rho) * gvalue * gvalue
      denom_t = rms_t[gindex] + epsilon
      if centered:
        mg_t[gindex] = mg_t[gindex] * rho + (1 - rho) * gvalue
        denom_t -= mg_t[gindex] * mg_t[gindex]
      mom_t[gindex] = momentum * mom[gindex] + lr * gvalue / np.sqrt(denom_t)
      var_t[gindex] = var[gindex] - mom_t[gindex]
    return var_t, mg_t, rms_t, mom_t

  @test_util.run_deprecated_v1
  def testDense(self):
    for (dtype, learning_rate, rho, momentum, epsilon, centered) in _TESTPARAMS:
      with test_util.use_gpu():
        # Initialize variables for numpy implementation.
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.2], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.2], dtype=dtype.as_numpy_dtype)

        var0 = resource_variable_ops.ResourceVariable(var0_np, dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable(var1_np, dtype=dtype)
        grads0 = constant_op.constant(grads0_np, dtype=dtype)
        grads1 = constant_op.constant(grads1_np, dtype=dtype)
        opt = rmsprop.RMSprop(
            learning_rate=learning_rate,
            rho=rho,
            momentum=momentum,
            epsilon=epsilon,
            centered=centered)

        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())

        if centered:
          mg0 = opt.get_slot(var0, "mg")
          mg1 = opt.get_slot(var1, "mg")
        else:
          mg0 = None
          mg1 = None

        rms0 = opt.get_slot(var0, "rms")
        self.assertTrue(rms0 is not None)
        rms1 = opt.get_slot(var1, "rms")
        self.assertTrue(rms1 is not None)
        mom0 = opt.get_slot(var0, "momentum")
        self.assertTrue(mom0 is not None)
        mom1 = opt.get_slot(var1, "momentum")
        self.assertTrue(mom1 is not None)

        mg0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        mg1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        rms0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        rms1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        mom0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        mom1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))

        # Run 4 steps of RMSprop
        for _ in range(1, 5):
          self.evaluate(update)

          var0_np, mg0_np, rms0_np, mom0_np = self._rmsprop_update_numpy(
              var0_np, grads0_np, mg0_np, rms0_np, mom0_np, learning_rate, rho,
              momentum, epsilon, centered)
          var1_np, mg1_np, rms1_np, mom1_np = self._rmsprop_update_numpy(
              var1_np, grads1_np, mg1_np, rms1_np, mom1_np, learning_rate, rho,
              momentum, epsilon, centered)

          # Validate updated params
          if centered:
            self.assertAllCloseAccordingToType(mg0_np, self.evaluate(mg0))
            self.assertAllCloseAccordingToType(mg1_np, self.evaluate(mg1))
          self.assertAllCloseAccordingToType(rms0_np, self.evaluate(rms0))
          self.assertAllCloseAccordingToType(rms1_np, self.evaluate(rms1))
          self.assertAllCloseAccordingToType(mom0_np, self.evaluate(mom0))
          self.assertAllCloseAccordingToType(mom1_np, self.evaluate(mom1))
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  @test_util.run_deprecated_v1
  def testDenseWithLearningRateDecay(self):
    var0_np = np.array([1.0, 2.0])
    grads0_np = np.array([0.1, 0.2])
    var1_np = np.array([3.0, 4.0])
    grads1_np = np.array([0.01, 0.2])

    var0 = resource_variable_ops.ResourceVariable(var0_np)
    var1 = resource_variable_ops.ResourceVariable(var1_np)
    grads0 = constant_op.constant(grads0_np)
    grads1 = constant_op.constant(grads1_np)
    learning_rate = 0.01
    rho = 0.9
    momentum = 0.0
    epsilon = 1e-7
    centered = False
    decay = 0.5
    opt = rmsprop.RMSprop(
        learning_rate=learning_rate,
        rho=rho,
        momentum=momentum,
        epsilon=epsilon,
        centered=centered,
        decay=decay)

    update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
    self.evaluate(variables.global_variables_initializer())

    rms0 = opt.get_slot(var0, "rms")
    self.assertTrue(rms0 is not None)
    rms1 = opt.get_slot(var1, "rms")
    self.assertTrue(rms1 is not None)
    mom0 = opt.get_slot(var0, "momentum")
    self.assertTrue(mom0 is not None)
    mom1 = opt.get_slot(var1, "momentum")
    self.assertTrue(mom1 is not None)

    mg0_np = np.array([0.0, 0.0])
    mg1_np = np.array([0.0, 0.0])
    rms0_np = np.array([0.0, 0.0])
    rms1_np = np.array([0.0, 0.0])
    mom0_np = np.array([0.0, 0.0])
    mom1_np = np.array([0.0, 0.0])

    # Fetch params to validate initial values
    self.assertAllClose([1.0, 2.0], self.evaluate(var0))
    self.assertAllClose([3.0, 4.0], self.evaluate(var1))

    # Run 4 steps of RMSprop
    for t in range(2):
      self.evaluate(update)

      lr = learning_rate / (1 + decay * t)
      var0_np, mg0_np, rms0_np, mom0_np = self._rmsprop_update_numpy(
          var0_np, grads0_np, mg0_np, rms0_np, mom0_np, lr, rho, momentum,
          epsilon, centered)
      var1_np, mg1_np, rms1_np, mom1_np = self._rmsprop_update_numpy(
          var1_np, grads1_np, mg1_np, rms1_np, mom1_np, lr, rho, momentum,
          epsilon, centered)

      # Validate updated params
      self.assertAllCloseAccordingToType(rms0_np, self.evaluate(rms0))
      self.assertAllCloseAccordingToType(rms1_np, self.evaluate(rms1))
      self.assertAllCloseAccordingToType(mom0_np, self.evaluate(mom0))
      self.assertAllCloseAccordingToType(mom1_np, self.evaluate(mom1))
      self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
      self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  @test_util.run_deprecated_v1
  def testMinimizeSparseResourceVariable(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      with self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([[1.0, 2.0]], dtype=dtype)
        x = constant_op.constant([[4.0], [5.0]], dtype=dtype)

        def loss():
          pred = math_ops.matmul(embedding_ops.embedding_lookup([var0], [0]), x)  # pylint: disable=cell-var-from-loop
          return pred * pred

        sgd_op = rmsprop.RMSprop(
            learning_rate=1.0,
            rho=0.0,
            momentum=0.0,
            epsilon=0.0,
            centered=False).minimize(
                loss, var_list=[var0])
        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([[1.0, 2.0]], self.evaluate(var0))
        # Run 1 step of sgd
        self.evaluate(sgd_op)
        # Validate updated params
        self.assertAllCloseAccordingToType([[0., 1.]],
                                           self.evaluate(var0),
                                           atol=0.01)

  @test_util.run_deprecated_v1
  def testMinimizeSparseResourceVariableCentered(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      with self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([[1.0, 2.0]], dtype=dtype)
        x = constant_op.constant([[4.0], [5.0]], dtype=dtype)

        def loss():
          pred = math_ops.matmul(embedding_ops.embedding_lookup([var0], [0]), x)  # pylint: disable=cell-var-from-loop
          return pred * pred

        # loss = lambda: pred * pred  # pylint: disable=cell-var-from-loop
        sgd_op = rmsprop.RMSprop(
            learning_rate=1.0,
            rho=0.0,
            momentum=0.0,
            epsilon=1.0,
            centered=True).minimize(
                loss, var_list=[var0])
        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([[1.0, 2.0]], self.evaluate(var0))
        # Run 1 step of sgd
        self.evaluate(sgd_op)
        # Validate updated params
        self.assertAllCloseAccordingToType([[-111, -138]],
                                           self.evaluate(var0),
                                           atol=0.01)

  @test_util.run_deprecated_v1
  def testSparse(self):
    for (dtype, learning_rate, rho, momentum, epsilon, centered) in _TESTPARAMS:
      with test_util.use_gpu():
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
        opt = rmsprop.RMSprop(
            learning_rate=learning_rate,
            rho=rho,
            momentum=momentum,
            epsilon=epsilon,
            centered=centered)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())

        if centered:
          mg0 = opt.get_slot(var0, "mg")
          self.assertEqual(mg0 is not None, centered)
          mg1 = opt.get_slot(var1, "mg")
          self.assertEqual(mg1 is not None, centered)
        else:
          mg0 = None
          mg1 = None
        rms0 = opt.get_slot(var0, "rms")
        self.assertTrue(rms0 is not None)
        rms1 = opt.get_slot(var1, "rms")
        self.assertTrue(rms1 is not None)
        mom0 = opt.get_slot(var0, "momentum")
        self.assertTrue(mom0 is not None)
        mom1 = opt.get_slot(var1, "momentum")
        self.assertTrue(mom1 is not None)

        mg0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        mg1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        rms0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        rms1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        mom0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        mom1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))

        # Run 4 steps of RMSprop
        for _ in range(1, 5):
          self.evaluate(update)

          var0_np, mg0_np, rms0_np, mom0_np = self._sparse_rmsprop_update_numpy(
              var0_np, grads0_np_indices, grads0_np, mg0_np, rms0_np, mom0_np,
              learning_rate, rho, momentum, epsilon, centered)
          var1_np, mg1_np, rms1_np, mom1_np = self._sparse_rmsprop_update_numpy(
              var1_np, grads1_np_indices, grads1_np, mg1_np, rms1_np, mom1_np,
              learning_rate, rho, momentum, epsilon, centered)

          # Validate updated params
          if centered:
            self.assertAllCloseAccordingToType(mg0_np, self.evaluate(mg0))
            self.assertAllCloseAccordingToType(mg1_np, self.evaluate(mg1))
          self.assertAllCloseAccordingToType(rms0_np, self.evaluate(rms0))
          self.assertAllCloseAccordingToType(rms1_np, self.evaluate(rms1))
          self.assertAllCloseAccordingToType(mom0_np, self.evaluate(mom0))
          self.assertAllCloseAccordingToType(mom1_np, self.evaluate(mom1))
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  def testCallableParams(self):
    with context.eager_mode():
      for dtype in [dtypes.half, dtypes.float32]:
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)

        learning_rate = lambda: 2.0
        rho = lambda: 0.9
        momentum = lambda: 0.0
        epsilon = lambda: 1.0
        opt = rmsprop.RMSprop(learning_rate, rho, momentum, epsilon)

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))
        # Step 1: the rms accumulators where 1. So we should see a normal
        # update: v -= grad * learning_rate
        opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        # Check the parameters.
        self.assertAllCloseAccordingToType(
            np.array([
                1.0 - (0.1 * 2.0 / math.sqrt(0.001 + 1.0)),
                2.0 - (0.1 * 2.0 / math.sqrt(0.001 + 1.0))
            ]), self.evaluate(var0))
        self.assertAllCloseAccordingToType(
            np.array([
                3.0 - (0.01 * 2.0 / math.sqrt(0.00001 + 1.0)),
                4.0 - (0.01 * 2.0 / math.sqrt(0.00001 + 1.0))
            ]), self.evaluate(var1))
        # Step 2: the root mean square accumulators contain the previous update.
        opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        # Check the parameters.
        self.assertAllCloseAccordingToType(
            np.array([
                1.0 - (0.1 * 2.0 / math.sqrt(0.001 + 1.0)) -
                (0.1 * 2.0 / math.sqrt(0.001 * 0.9 + 0.001 + 1.0)),
                2.0 - (0.1 * 2.0 / math.sqrt(0.001 + 1.0)) -
                (0.1 * 2.0 / math.sqrt(0.001 * 0.9 + 0.001 + 1.0))
            ]), self.evaluate(var0))
        self.assertAllCloseAccordingToType(
            np.array([
                3.0 - (0.01 * 2.0 / math.sqrt(0.00001 + 1.0)) -
                (0.01 * 2.0 / math.sqrt(0.00001 * 0.9 + 1e-5 + 1.0)),
                4.0 - (0.01 * 2.0 / math.sqrt(0.00001 + 1.0)) -
                (0.01 * 2.0 / math.sqrt(0.00001 * 0.9 + 1e-5 + 1.0))
            ]), self.evaluate(var1))

  def testConstructRMSpropWithLR(self):
    opt = rmsprop.RMSprop(lr=1.0)
    self.assertEqual(opt.lr, 1.0)
    opt_2 = rmsprop.RMSprop(learning_rate=0.1, lr=1.0)
    self.assertEqual(opt_2.lr, 1.0)
    opt_3 = rmsprop.RMSprop(learning_rate=0.1)
    self.assertEqual(opt_3.lr, 0.1)


if __name__ == "__main__":
  test.main()
