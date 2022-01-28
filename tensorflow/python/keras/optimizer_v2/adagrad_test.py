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
"""Functional tests for aggregate operations."""

import copy

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.keras import combinations
from tensorflow.python.keras.optimizer_v2 import adagrad
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

_DATA_TYPES = [
    dtypes.half, dtypes.float32, dtypes.float64, dtypes.complex64,
    dtypes.complex128
]


def adagrad_update_numpy(param, accum, g_t, lr=0.001, epsilon=1e-7):
  accum_t = accum + g_t * g_t
  param_t = param - lr * g_t / (np.sqrt(accum_t) + epsilon)
  return param_t, accum_t


def sparse_adagrad_update_numpy(param,
                                accum,
                                gindexs,
                                gvalues,
                                lr=0.001,
                                epsilon=1e-7):
  accum_t = copy.deepcopy(accum)
  param_t = copy.deepcopy(param)
  # first loop accumulates repeated indices if necessary.
  for i in range(len(gindexs)):
    gindex = gindexs[i]
    gvalue = gvalues[i]
    accum_t[gindex] = accum_t[gindex] + gvalue * gvalue
  for i in range(len(gindexs)):
    gindex = gindexs[i]
    gvalue = gvalues[i]
    param_t[gindex] = param_t[gindex] - lr * gvalue / (
        np.sqrt(accum_t[gindex]) + epsilon)
  return param_t, accum_t


class AdagradOptimizerTest(test.TestCase, parameterized.TestCase):

  def doTestBasic(self, use_callable_params=False):
    for dtype in _DATA_TYPES:
      var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
      var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
      grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
      grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)
      var0 = variables.Variable(var0_np)
      var1 = variables.Variable(var1_np)
      grads0 = constant_op.constant(grads0_np)
      grads1 = constant_op.constant(grads1_np)

      learning_rate = lambda: 3.0
      if not use_callable_params:
        learning_rate = learning_rate()

      ada_opt = adagrad.Adagrad(learning_rate)

      accum0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
      accum1_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)

      if not context.executing_eagerly():
        ada_update = ada_opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())

      # Fetch params to validate initial values
      v0_val, v1_val = self.evaluate([var0, var1])
      self.assertAllClose([1.0, 2.0], v0_val)
      self.assertAllClose([3.0, 4.0], v1_val)

      # Run 3 steps of adagrad
      for _ in range(3):
        if not context.executing_eagerly():
          self.evaluate(ada_update)
        else:
          ada_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        var0_np, accum0_np = adagrad_update_numpy(var0_np, accum0_np, grads0_np,
                                                  3.0)
        var1_np, accum1_np = adagrad_update_numpy(var1_np, accum1_np, grads1_np,
                                                  3.0)
        self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
        self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  @combinations.generate(combinations.combine(mode=["graph", "eager"]))
  def testBasic(self):
    self.doTestBasic()

  @combinations.generate(combinations.combine(mode=["eager"]))
  def testBasicCallableParams(self):
    self.doTestBasic(use_callable_params=True)

  def testBasicWithLearningRateDecay(self):
    for dtype in _DATA_TYPES:
      var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
      var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
      grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
      grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)
      var0 = variables.Variable(var0_np)
      var1 = variables.Variable(var1_np)
      grads0 = constant_op.constant(grads0_np)
      grads1 = constant_op.constant(grads1_np)

      learning_rate = 3.0
      decay = 0.5

      ada_opt = adagrad.Adagrad(learning_rate, decay=decay)

      accum0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
      accum1_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)

      if not context.executing_eagerly():
        ada_update = ada_opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())

      # Fetch params to validate initial values
      v0_val, v1_val = self.evaluate([var0, var1])
      self.assertAllClose([1.0, 2.0], v0_val)
      self.assertAllClose([3.0, 4.0], v1_val)

      # Run 3 steps of adagrad
      for t in range(3):
        if not context.executing_eagerly():
          self.evaluate(ada_update)
        else:
          ada_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        lr_np = learning_rate / (1 + decay * t)
        var0_np, accum0_np = adagrad_update_numpy(var0_np, accum0_np, grads0_np,
                                                  lr_np)
        var1_np, accum1_np = adagrad_update_numpy(var1_np, accum1_np, grads1_np,
                                                  lr_np)
        self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
        self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  def testBasicWithLargeEpsilon(self):
    var0_np = np.array([1.0, 2.0])
    var1_np = np.array([3.0, 4.0])
    grads0_np = np.array([0.1, 0.1])
    grads1_np = np.array([0.01, 0.01])
    var0 = variables.Variable(var0_np)
    var1 = variables.Variable(var1_np)
    grads0 = constant_op.constant(grads0_np)
    grads1 = constant_op.constant(grads1_np)

    learning_rate = 3.0

    ada_opt = adagrad.Adagrad(learning_rate, epsilon=1.0)

    accum0_np = np.array([0.1, 0.1])
    accum1_np = np.array([0.1, 0.1])

    if not context.executing_eagerly():
      ada_update = ada_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      self.evaluate(variables.global_variables_initializer())

    # Fetch params to validate initial values
    v0_val, v1_val = self.evaluate([var0, var1])
    self.assertAllClose([1.0, 2.0], v0_val)
    self.assertAllClose([3.0, 4.0], v1_val)

    # Run 3 steps of adagrad
    for _ in range(3):
      if not context.executing_eagerly():
        self.evaluate(ada_update)
      else:
        ada_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      var0_np, accum0_np = adagrad_update_numpy(var0_np, accum0_np, grads0_np,
                                                3.0, 1.0)
      var1_np, accum1_np = adagrad_update_numpy(var1_np, accum1_np, grads1_np,
                                                3.0, 1.0)
      self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
      self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  def testBasicWithLearningRateInverseTimeDecay(self):
    for dtype in _DATA_TYPES:
      var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
      var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
      grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
      grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)
      var0 = variables.Variable(var0_np)
      var1 = variables.Variable(var1_np)
      grads0 = constant_op.constant(grads0_np)
      grads1 = constant_op.constant(grads1_np)

      learning_rate = 3.0
      decay = 0.5
      lr_schedule = learning_rate_schedule.InverseTimeDecay(
          learning_rate, decay_steps=1.0, decay_rate=decay)

      ada_opt = adagrad.Adagrad(lr_schedule)

      accum0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
      accum1_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)

      if not context.executing_eagerly():
        ada_update = ada_opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())

      # Fetch params to validate initial values
      v0_val, v1_val = self.evaluate([var0, var1])
      self.assertAllClose([1.0, 2.0], v0_val)
      self.assertAllClose([3.0, 4.0], v1_val)

      # Run 3 steps of adagrad
      for t in range(3):
        if not context.executing_eagerly():
          self.evaluate(ada_update)
        else:
          ada_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        lr_np = learning_rate / (1 + decay * t)
        var0_np, accum0_np = adagrad_update_numpy(var0_np, accum0_np, grads0_np,
                                                  lr_np)
        var1_np, accum1_np = adagrad_update_numpy(var1_np, accum1_np, grads1_np,
                                                  lr_np)
        self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
        self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  def testMinimizeSparseResourceVariable(self):
    # TODO(tanzheny, omalleyt): Fix test in eager mode.
    with ops.Graph().as_default():
      for dtype in _DATA_TYPES:
        var0 = variables.Variable([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
        x = constant_op.constant([[4.0], [5.0]], dtype=dtype)

        def loss():
          pred = math_ops.matmul(embedding_ops.embedding_lookup([var0], [0]), x)  # pylint: disable=cell-var-from-loop
          return pred * pred

        sgd_op = adagrad.Adagrad(1.0).minimize(loss, var_list=[var0])
        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([[1.0, 2.0], [3.0, 4.0]],
                                           self.evaluate(var0))
        # Run 1 step of sgd
        self.evaluate(sgd_op)
        # Validate updated params
        self.assertAllCloseAccordingToType([[0, 1], [3, 4]],
                                           self.evaluate(var0),
                                           atol=0.01)

  def testTensorLearningRate(self):
    # TODO(tanzheny, omalleyt): Fix test in eager mode.
    with ops.Graph().as_default():
      for dtype in _DATA_TYPES:
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)
        var0 = variables.Variable(var0_np)
        var1 = variables.Variable(var1_np)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)

        learning_rate = constant_op.constant(3.0)
        ada_opt = adagrad.Adagrad(learning_rate)
        ada_update = ada_opt.apply_gradients(zip([grads0, grads1],
                                                 [var0, var1]))
        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))
        accum0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        accum1_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        # Run 3 steps of adagrad
        for _ in range(3):
          self.evaluate(ada_update)
          var0_np, accum0_np = adagrad_update_numpy(
              var0_np, accum0_np, grads0_np, learning_rate)
          var1_np, accum1_np = adagrad_update_numpy(
              var1_np, accum1_np, grads1_np, learning_rate)
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  def testSparseBasic(self):
    # TODO(tanzheny, omalleyt): Fix test in eager mode.
    with ops.Graph().as_default():
      for dtype in _DATA_TYPES:
        var0_np = np.array([1.0, 1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0, 0.01], dtype=dtype.as_numpy_dtype)

        var0 = variables.Variable(var0_np)
        var1 = variables.Variable(var1_np)
        grads0_np_indices = np.array([0, 2], dtype=np.int32)
        grads0 = indexed_slices.IndexedSlices(
            constant_op.constant(grads0_np[grads0_np_indices]),
            constant_op.constant(grads0_np_indices), constant_op.constant([3]))
        grads1_np_indices = np.array([0, 2], dtype=np.int32)
        grads1 = indexed_slices.IndexedSlices(
            constant_op.constant(grads1_np[grads1_np_indices]),
            constant_op.constant(grads1_np_indices), constant_op.constant([3]))
        learning_rate = 3.0
        ada_opt = adagrad.Adagrad(learning_rate)
        ada_update = ada_opt.apply_gradients(zip([grads0, grads1],
                                                 [var0, var1]))
        self.evaluate(variables.global_variables_initializer())

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 3.0, 4.0], self.evaluate(var1))

        accum0_np = np.array([0.1, 0.1, 0.1], dtype=dtype.as_numpy_dtype)
        accum1_np = np.array([0.1, 0.1, 0.1], dtype=dtype.as_numpy_dtype)

        # Run 3 step of sgd
        for _ in range(3):
          self.evaluate(ada_update)

          var0_np, accum0_np = sparse_adagrad_update_numpy(
              var0_np, accum0_np, grads0_np_indices,
              grads0_np[grads0_np_indices], learning_rate)
          var1_np, accum1_np = sparse_adagrad_update_numpy(
              var1_np, accum1_np, grads1_np_indices,
              grads1_np[grads1_np_indices], learning_rate)
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  def testSparseSingleVarDim(self):
    # TODO(tanzheny, omalleyt): Fix test in eager mode.
    with ops.Graph().as_default():
      for dtype in _DATA_TYPES:
        var0_np = np.array([1.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1], dtype=dtype.as_numpy_dtype)

        var0 = variables.Variable(var0_np)
        grads0_np_indices = np.array([0], dtype=np.int32)
        grads0 = indexed_slices.IndexedSlices(
            constant_op.constant(grads0_np[grads0_np_indices]),
            constant_op.constant(grads0_np_indices), constant_op.constant([3]))
        learning_rate = 3.0
        ada_opt = adagrad.Adagrad(learning_rate, epsilon=1.)
        ada_update = ada_opt.apply_gradients(zip([grads0], [var0]))
        self.evaluate(variables.global_variables_initializer())

        # Fetch params to validate initial values
        self.assertAllClose([1.0], self.evaluate(var0))

        accum0_np = np.array([0.1], dtype=dtype.as_numpy_dtype)

        # Run 3 step of sgd
        for _ in range(3):
          self.evaluate(ada_update)

          var0_np, accum0_np = sparse_adagrad_update_numpy(
              var0_np,
              accum0_np,
              grads0_np_indices,
              grads0_np[grads0_np_indices],
              learning_rate,
              epsilon=1.)
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))

  def testSparseRepeatedIndices(self):
    # TODO(tanzheny, omalleyt): Fix test in eager mode.
    with ops.Graph().as_default():
      for dtype in _DATA_TYPES:
        var_np = np.array([[1.0], [2.0]], dtype=dtype.as_numpy_dtype)

        repeated_index_update_var = variables.Variable(
            var_np, dtype=dtype)
        aggregated_update_var = variables.Variable(
            var_np, dtype=dtype)
        grad_repeated_index = indexed_slices.IndexedSlices(
            constant_op.constant([0.1, 0.1], shape=[2, 1], dtype=dtype),
            constant_op.constant([1, 1]), constant_op.constant([2, 1]))
        grad_aggregated = indexed_slices.IndexedSlices(
            constant_op.constant([0.2], shape=[1, 1], dtype=dtype),
            constant_op.constant([1]), constant_op.constant([2, 1]))
        repeated_update = adagrad.Adagrad(3.0).apply_gradients([
            (grad_repeated_index, repeated_index_update_var)
        ])
        aggregated_update = adagrad.Adagrad(3.0).apply_gradients([
            (grad_aggregated, aggregated_update_var)
        ])
        self.evaluate(variables.global_variables_initializer())
        self.assertAllClose(
            self.evaluate(aggregated_update_var),
            self.evaluate(repeated_index_update_var))
        for _ in range(3):
          self.evaluate(repeated_update)
          self.evaluate(aggregated_update)
          self.assertAllClose(
              self.evaluate(aggregated_update_var),
              self.evaluate(repeated_index_update_var))

  def testSparseRepeatedIndicesByEmbeddingLookUp(self):
    # TODO(tanzheny, omalleyt): Fix test in eager mode.
    with ops.Graph().as_default():
      for dtype in _DATA_TYPES:
        var_repeated = variables.Variable([1.0, 2.0], dtype=dtype)
        loss_repeated = lambda: math_ops.reduce_sum(  # pylint: disable=g-long-lambda
            embedding_ops.embedding_lookup(var_repeated, [0, 0]))  # pylint: disable=cell-var-from-loop
        var_aggregated = variables.Variable([1.0, 2.0], dtype=dtype)
        loss_aggregated = lambda: 2 * math_ops.reduce_sum(  # pylint: disable=g-long-lambda
            embedding_ops.embedding_lookup(var_aggregated, [0]))  # pylint: disable=cell-var-from-loop
        update_op_repeated = adagrad.Adagrad(2.0).minimize(
            loss_repeated, var_list=[var_repeated])
        update_op_aggregated = adagrad.Adagrad(2.0).minimize(
            loss_aggregated, var_list=[var_aggregated])
        self.evaluate(variables.global_variables_initializer())
        self.assertAllCloseAccordingToType(
            self.evaluate(var_repeated), self.evaluate(var_aggregated))
        for _ in range(3):
          self.evaluate(update_op_repeated)
          self.evaluate(update_op_aggregated)
          self.assertAllCloseAccordingToType(
              self.evaluate(var_repeated), self.evaluate(var_aggregated))

  def testSparseStability(self):
    # TODO(tanzheny, omalleyt): Fix test in eager mode.
    with ops.Graph().as_default():
      for dtype in [dtypes.half]:
        shape = [1, 6]
        var0_np = np.array([[0.00872496, -0.106952, 0.110467,
                             0.226505, -0.0147257, -0.0105945]],
                           dtype=dtype.as_numpy_dtype)
        var0 = variables.Variable(var0_np)
        grads0_np = np.array([[
            -5.91278e-05, 5.31673e-05, -2.5779e-06, 4.29153e-05, -8.4877e-05,
            -9.48906e-05
        ]],
                             dtype=dtype.as_numpy_dtype)
        grads0 = indexed_slices.IndexedSlices(
            constant_op.constant(grads0_np), constant_op.constant([0]),
            constant_op.constant(shape))
        ada_opt = adagrad.Adagrad(1.0)
        ada_update = ada_opt.apply_gradients(zip([grads0], [var0]))
        slot0 = ada_opt.get_slot(var0, "accumulator")
        init = variables.global_variables_initializer()
        for _ in range(100):
          self.evaluate(init)
          self.evaluate(ada_update)
          self.assertAllCloseAccordingToType(
              np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]), self.evaluate(slot0))
          self.assertAllCloseAccordingToType(
              np.array([[
                  0.00891194, -0.10712013, 0.11047515, 0.22636929, -0.0144573,
                  -0.01029443
              ]]), self.evaluate(var0))

  def testSharing(self):
    # TODO(tanzheny, omalleyt): Fix test in eager mode.
    with ops.Graph().as_default():
      for dtype in _DATA_TYPES:
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        var0 = variables.Variable(var0_np)
        var1 = variables.Variable(var1_np)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)

        learning_rate = 3.0
        ada_opt = adagrad.Adagrad(learning_rate)
        # Apply the optimizer twice.  Both applications will use
        # the same accums.
        ada_update1 = ada_opt.apply_gradients(zip([grads0, grads1],
                                                  [var0, var1]))
        ada_update2 = ada_opt.apply_gradients(zip([grads0, grads1],
                                                  [var0, var1]))
        slot0 = ada_opt.get_slot(var0, "accumulator")
        self.assertEqual(slot0.shape, var0.shape)
        slot1 = ada_opt.get_slot(var1, "accumulator")
        self.assertEqual(slot1.shape, var1.shape)
        self.evaluate(variables.global_variables_initializer())

        # Fetch params to validate initial values.
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))
        # Mix the first and the second adagrad for 3 steps.
        self.evaluate(ada_update1)
        self.evaluate(ada_update2)
        self.evaluate(ada_update1)

        accum0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        accum1_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        for _ in range(3):
          var0_np, accum0_np = adagrad_update_numpy(
              var0_np, accum0_np, grads0_np, learning_rate)
          var1_np, accum1_np = adagrad_update_numpy(
              var1_np, accum1_np, grads1_np, learning_rate)
        self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
        self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  def testConstructAdagradWithLR(self):
    opt = adagrad.Adagrad(lr=1.0)
    opt_2 = adagrad.Adagrad(learning_rate=0.1, lr=1.0)
    opt_3 = adagrad.Adagrad(learning_rate=0.1)
    self.assertIsInstance(opt.lr, variables.Variable)
    self.assertIsInstance(opt_2.lr, variables.Variable)
    self.assertIsInstance(opt_3.lr, variables.Variable)

    self.evaluate(variables.global_variables_initializer())
    self.assertAllClose(self.evaluate(opt.lr), (1.0))
    self.assertAllClose(self.evaluate(opt_2.lr), (1.0))
    self.assertAllClose(self.evaluate(opt_3.lr), (0.1))


if __name__ == "__main__":
  test.main()
