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

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras import combinations
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.keras.optimizer_v2 import rmsprop
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

_DATA_TYPES = [dtypes.half, dtypes.float32, dtypes.float64]
# TODO(b/143684500): Eigen to support complex sqrt
if not test_util.IsBuiltWithNvcc():
  _DATA_TYPES += [dtypes.complex64, dtypes.complex128]

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
    if centered:
      mg_t = mg * rho + (1 - rho) * g
      denom_t = rms_t - mg_t * mg_t
    else:
      mg_t = mg
      denom_t = rms_t
    if momentum > 0.:
      mom_t = momentum * mom + lr * g / (np.sqrt(denom_t + epsilon))
      var_t = var - mom_t
    else:
      mom_t = mom
      var_t = var - lr * g / (np.sqrt(denom_t) + epsilon)
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
      if centered:
        mg_t[gindex] = mg_t[gindex] * rho + (1 - rho) * gvalue
        denom_t = rms_t[gindex] - mg_t[gindex] * mg_t[gindex]
      else:
        denom_t = rms_t[gindex]
      if momentum > 0.:
        mom_t[gindex] = momentum * mom[gindex] + lr * gvalue / np.sqrt(denom_t +
                                                                       epsilon)
        var_t[gindex] = var[gindex] - mom_t[gindex]
      else:
        mom_t[gindex] = mom[gindex]
        var_t[gindex] = var[gindex] - lr * gvalue / (np.sqrt(denom_t) + epsilon)
    return var_t, mg_t, rms_t, mom_t

  def testDense(self):
    # TODO(tanzheny, omalleyt): Fix test in eager mode.
    for (dtype, learning_rate, rho, momentum, epsilon, centered) in _TESTPARAMS:
      with ops.get_default_graph().as_default(), test_util.use_gpu():
        # Initialize variables for numpy implementation.
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.2], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.2], dtype=dtype.as_numpy_dtype)

        var0 = variables.Variable(var0_np, dtype=dtype)
        var1 = variables.Variable(var1_np, dtype=dtype)
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

        if momentum > 0.:
          mom0 = opt.get_slot(var0, "momentum")
          mom1 = opt.get_slot(var1, "momentum")
        else:
          mom0 = None
          mom1 = None

        rms0 = opt.get_slot(var0, "rms")
        self.assertIsNotNone(rms0)
        rms1 = opt.get_slot(var1, "rms")
        self.assertIsNotNone(rms1)

        mg0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        mg1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        rms0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        rms1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        mom0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        mom1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))

        # Run 3 steps of RMSprop
        for _ in range(1, 4):
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
          if momentum > 0.:
            self.assertAllCloseAccordingToType(mom0_np, self.evaluate(mom0))
            self.assertAllCloseAccordingToType(mom1_np, self.evaluate(mom1))
          self.assertAllCloseAccordingToType(rms0_np, self.evaluate(rms0))
          self.assertAllCloseAccordingToType(rms1_np, self.evaluate(rms1))
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  def testDenseWithLearningRateDecay(self):
    # TODO(tanzheny, omalleyt): Fix test in eager mode.
    with ops.Graph().as_default():
      var0_np = np.array([1.0, 2.0])
      grads0_np = np.array([0.1, 0.2])
      var1_np = np.array([3.0, 4.0])
      grads1_np = np.array([0.01, 0.2])

      var0 = variables.Variable(var0_np)
      var1 = variables.Variable(var1_np)
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
      self.assertIsNotNone(rms0)
      rms1 = opt.get_slot(var1, "rms")
      self.assertIsNotNone(rms1)
      if momentum > 0.:
        mom0 = opt.get_slot(var0, "momentum")
        mom1 = opt.get_slot(var1, "momentum")
      else:
        mom0 = None
        mom1 = None

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
        if momentum > 0.:
          self.assertAllCloseAccordingToType(mom0_np, self.evaluate(mom0))
          self.assertAllCloseAccordingToType(mom1_np, self.evaluate(mom1))
        self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
        self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  def testDenseWithLearningRateInverseTimeDecay(self):
    # TODO(tanzheny, omalleyt): Fix test in eager mode.
    with ops.Graph().as_default():
      var0_np = np.array([1.0, 2.0])
      grads0_np = np.array([0.1, 0.2])
      var1_np = np.array([3.0, 4.0])
      grads1_np = np.array([0.01, 0.2])

      var0 = variables.Variable(var0_np)
      var1 = variables.Variable(var1_np)
      grads0 = constant_op.constant(grads0_np)
      grads1 = constant_op.constant(grads1_np)
      learning_rate = 0.01
      rho = 0.9
      momentum = 0.0
      epsilon = 1e-7
      centered = False
      decay = 0.5
      lr_schedule = learning_rate_schedule.InverseTimeDecay(
          learning_rate, decay_steps=1.0, decay_rate=decay)
      opt = rmsprop.RMSprop(
          learning_rate=lr_schedule,
          rho=rho,
          momentum=momentum,
          epsilon=epsilon,
          centered=centered)

      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      self.evaluate(variables.global_variables_initializer())

      rms0 = opt.get_slot(var0, "rms")
      self.assertIsNotNone(rms0)
      rms1 = opt.get_slot(var1, "rms")
      self.assertIsNotNone(rms1)
      if momentum > 0.:
        mom0 = opt.get_slot(var0, "momentum")
        mom1 = opt.get_slot(var1, "momentum")
      else:
        mom0 = None
        mom1 = None

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
        if momentum > 0.:
          self.assertAllCloseAccordingToType(mom0_np, self.evaluate(mom0))
          self.assertAllCloseAccordingToType(mom1_np, self.evaluate(mom1))
        self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
        self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  def testMinimizeSparseResourceVariable(self):
    # TODO(tanzheny, omalleyt): Fix test in eager mode.
    with ops.Graph().as_default():
      for dtype in _DATA_TYPES:
        var0 = variables.Variable([[1.0, 2.0]], dtype=dtype)
        x = constant_op.constant([[4.0], [5.0]], dtype=dtype)

        def loss():
          pred = math_ops.matmul(embedding_ops.embedding_lookup([var0], [0]), x)  # pylint: disable=cell-var-from-loop
          return pred * pred

        sgd_op = rmsprop.RMSprop(
            learning_rate=1.0, rho=0.0, momentum=0.0, epsilon=0.0,
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

  def testMinimizeSparseResourceVariableCentered(self):
    # TODO(tanzheny, omalleyt): Fix test in eager mode.
    with ops.Graph().as_default():
      for dtype in _DATA_TYPES:
        if test_util.is_xla_enabled() and dtype.is_complex:
          self.skipTest("b/143578550")
        var0 = variables.Variable([[1.0, 2.0]], dtype=dtype)
        x = constant_op.constant([[4.0], [5.0]], dtype=dtype)

        def loss():
          pred = math_ops.matmul(embedding_ops.embedding_lookup([var0], [0]), x)  # pylint: disable=cell-var-from-loop
          return pred * pred

        # loss = lambda: pred * pred  # pylint: disable=cell-var-from-loop
        sgd_op = rmsprop.RMSprop(
            learning_rate=1.0, rho=0.0, momentum=0.0, epsilon=1.0,
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

  def testSparse(self):
    # TODO(tanzheny, omalleyt): Fix test in eager mode.
    for (dtype, learning_rate, rho, momentum, epsilon, centered) in _TESTPARAMS:
      with ops.get_default_graph().as_default(), test_util.use_gpu():
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
        self.assertIsNotNone(rms0)
        rms1 = opt.get_slot(var1, "rms")
        self.assertIsNotNone(rms1)
        if momentum > 0.:
          mom0 = opt.get_slot(var0, "momentum")
          mom1 = opt.get_slot(var1, "momentum")
        else:
          mom0 = None
          mom1 = None

        mg0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        mg1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        rms0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        rms1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        mom0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        mom1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))

        # Run 3 steps of RMSprop
        for _ in range(1, 4):
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
          if momentum > 0.:
            self.assertAllCloseAccordingToType(mom0_np, self.evaluate(mom0))
            self.assertAllCloseAccordingToType(mom1_np, self.evaluate(mom1))
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  def testCallableParams(self):
    with context.eager_mode():
      for dtype in _DATA_TYPES:
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)

        learning_rate = lambda: 2.0
        rho = lambda: 0.9
        momentum = lambda: 0.0
        epsilon = 1.0
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
    opt_2 = rmsprop.RMSprop(learning_rate=0.1, lr=1.0)
    opt_3 = rmsprop.RMSprop(learning_rate=0.1)
    self.assertIsInstance(opt.lr, variables.Variable)
    self.assertIsInstance(opt_2.lr, variables.Variable)
    self.assertIsInstance(opt_3.lr, variables.Variable)

    self.evaluate(variables.global_variables_initializer())
    self.assertAllClose(self.evaluate(opt.lr), (1.0))
    self.assertAllClose(self.evaluate(opt_2.lr), (1.0))
    self.assertAllClose(self.evaluate(opt_3.lr), (0.1))

  def testSlotsUniqueEager(self):
    with context.eager_mode():
      v1 = variables.Variable(1.)
      v2 = variables.Variable(1.)

      opt = rmsprop.RMSprop(1., momentum=0., centered=False)
      opt.minimize(lambda: v1 + v2, var_list=[v1, v2])
      # There should be iteration, and one unique slot variable for v1 and v2.
      self.assertEqual(3, len(set({id(v) for v in opt.variables()})))
      self.assertEqual(
          self.evaluate(opt.variables()[0]), self.evaluate(opt.iterations))

      opt = rmsprop.RMSprop(learning_rate=1., momentum=0.2, centered=False)
      opt.minimize(lambda: v1 + v2, var_list=[v1, v2])
      # There should be iteration, and two unique slot variables for v1 and v2.
      self.assertEqual(5, len(set({id(v) for v in opt.variables()})))
      self.assertEqual(
          self.evaluate(opt.variables()[0]), self.evaluate(opt.iterations))

      opt = rmsprop.RMSprop(learning_rate=1., momentum=0.2, centered=True)
      opt.minimize(lambda: v1 + v2, var_list=[v1, v2])
      # There should be iteration, and three unique slot variables for v1 and v2
      self.assertEqual(7, len(set({id(v) for v in opt.variables()})))
      self.assertEqual(
          self.evaluate(opt.variables()[0]), self.evaluate(opt.iterations))


@combinations.generate(combinations.combine(mode=["graph", "eager"]))
class SlotColocationTest(test.TestCase, parameterized.TestCase):

  @parameterized.parameters([True, False])
  @test_util.run_gpu_only
  def testRunMinimizeOnGPUForCPUVariables(self, use_resource):
    with ops.device("/device:CPU:0"):
      if use_resource:
        var0 = variables.Variable([1.0, 2.0], dtype=dtypes.float32)
        var1 = variables.Variable([3.0, 4.0], dtype=dtypes.float32)
      else:
        var0 = variables.Variable([1.0, 2.0], dtype=dtypes.float32)
        var1 = variables.Variable([3.0, 4.0], dtype=dtypes.float32)

    def loss():
      return 5 * var0 + 3 * var1

    opt = rmsprop.RMSprop(
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
      opt_op = opt.minimize(loss, [var0, var1])
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(opt_op)

    # Validate updated params, All variables should have decreased.
    self.assertTrue(all(v < 0.0 for v in self.evaluate(var0)),
                    msg="updated variables: %s" % self.evaluate(var0))
    self.assertTrue(all(v < 2.0 for v in self.evaluate(var1)),
                    msg="updated variables: %s" % self.evaluate(var1))

if __name__ == "__main__":
  test.main()
