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
"""Tests for Adam."""

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.compiler.xla.experimental import xla_sharding
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adam


def adam_update_numpy(param,
                      g_t,
                      t,
                      m,
                      v,
                      alpha=0.001,
                      beta1=0.9,
                      beta2=0.999,
                      epsilon=1e-8):
  alpha_t = alpha * np.sqrt(1 - beta2**t) / (1 - beta1**t)

  m_t = beta1 * m + (1 - beta1) * g_t
  v_t = beta2 * v + (1 - beta2) * g_t * g_t

  param_t = param - alpha_t * m_t / (np.sqrt(v_t) + epsilon)
  return param_t, m_t, v_t


class AdamOptimizerTest(test.TestCase):

  def doTestSparse(self, use_resource=False):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        if use_resource:
          var0 = resource_variable_ops.ResourceVariable(var0_np)
          var1 = resource_variable_ops.ResourceVariable(var1_np)
        else:
          var0 = ref_variable.RefVariable(var0_np)
          var1 = ref_variable.RefVariable(var1_np)
        grads0_np_indices = np.array([0, 1], dtype=np.int32)
        grads0 = indexed_slices.IndexedSlices(
            constant_op.constant(grads0_np),
            constant_op.constant(grads0_np_indices), constant_op.constant([2]))
        grads1_np_indices = np.array([0, 1], dtype=np.int32)
        grads1 = indexed_slices.IndexedSlices(
            constant_op.constant(grads1_np),
            constant_op.constant(grads1_np_indices), constant_op.constant([2]))
        opt = adam.AdamOptimizer()
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))

        beta1_power, beta2_power = opt._get_beta_accumulators()

        # Run 3 steps of Adam
        for t in range(1, 4):
          self.assertAllCloseAccordingToType(0.9**t, self.evaluate(beta1_power))
          self.assertAllCloseAccordingToType(0.999**t,
                                             self.evaluate(beta2_power))
          update.run()

          var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  def testSparse(self):
    with ops.Graph().as_default():
      self.doTestSparse(use_resource=False)

  def testResourceSparse(self):
    with ops.Graph().as_default():
      self.doTestSparse(use_resource=True)

  def testSparseDevicePlacement(self):
    with ops.Graph().as_default():
      for index_dtype in [dtypes.int32, dtypes.int64]:
        with self.cached_session(force_gpu=test.is_gpu_available()):
          # If a GPU is available, tests that all optimizer ops can be placed on
          # it (i.e. they have GPU kernels).
          var = variables.Variable([[1.0], [2.0]])
          indices = constant_op.constant([0, 1], dtype=index_dtype)
          gathered_sum = math_ops.reduce_sum(array_ops.gather(var, indices))
          optimizer = adam.AdamOptimizer(3.0)
          minimize_op = optimizer.minimize(gathered_sum)
          self.evaluate(variables.global_variables_initializer())
          minimize_op.run()

  def testSparseRepeatedIndices(self):
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
        with self.cached_session():
          repeated_index_update_var = variables.Variable(
              [[1.0], [2.0]], dtype=dtype)
          aggregated_update_var = variables.Variable(
              [[1.0], [2.0]], dtype=dtype)
          grad_repeated_index = indexed_slices.IndexedSlices(
              constant_op.constant(
                  [0.1, 0.1], shape=[2, 1], dtype=dtype),
              constant_op.constant([1, 1]),
              constant_op.constant([2, 1]))
          grad_aggregated = indexed_slices.IndexedSlices(
              constant_op.constant(
                  [0.2], shape=[1, 1], dtype=dtype),
              constant_op.constant([1]),
              constant_op.constant([2, 1]))
          repeated_update = adam.AdamOptimizer().apply_gradients(
              [(grad_repeated_index, repeated_index_update_var)])
          aggregated_update = adam.AdamOptimizer().apply_gradients(
              [(grad_aggregated, aggregated_update_var)])
          self.evaluate(variables.global_variables_initializer())
          self.assertAllClose(aggregated_update_var,
                              self.evaluate(repeated_index_update_var))
          for _ in range(3):
            repeated_update.run()
            aggregated_update.run()
            self.assertAllClose(aggregated_update_var,
                                self.evaluate(repeated_index_update_var))

  def doTestBasic(self, use_resource=False, use_callable_params=False):
    if context.executing_eagerly() and not use_resource:
      self.skipTest(
          "Skipping test with use_resource=False and executing eagerly.")
    for i, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      with self.session(graph=ops.Graph()):
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        if use_resource:
          var0 = resource_variable_ops.ResourceVariable(
              var0_np, name="var0_%d" % i)
          var1 = resource_variable_ops.ResourceVariable(
              var1_np, name="var1_%d" % i)
        else:
          var0 = ref_variable.RefVariable(var0_np)
          var1 = ref_variable.RefVariable(var1_np)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)

        learning_rate = lambda: 0.001
        beta1 = lambda: 0.9
        beta2 = lambda: 0.999
        epsilon = lambda: 1e-8
        if not use_callable_params:
          learning_rate = learning_rate()
          beta1 = beta1()
          beta2 = beta2()
          epsilon = epsilon()

        opt = adam.AdamOptimizer(learning_rate=learning_rate)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        opt_variables = opt.variables()
        beta1_power, beta2_power = opt._get_beta_accumulators()
        self.assertTrue(beta1_power is not None)
        self.assertTrue(beta2_power is not None)
        self.assertIn(beta1_power, opt_variables)
        self.assertIn(beta2_power, opt_variables)
        # Ensure that non-slot variables are the same type as the requested
        # variables.
        self.assertEqual(
            use_resource,
            resource_variable_ops.is_resource_variable(beta1_power))
        self.assertEqual(
            use_resource,
            resource_variable_ops.is_resource_variable(beta2_power))

        if not context.executing_eagerly():
          with ops.Graph().as_default():
            # Shouldn't return non-slot variables from other graphs.
            self.assertEqual(0, len(opt.variables()))
          self.evaluate(variables.global_variables_initializer())
          # Fetch params to validate initial values
          self.assertAllClose([1.0, 2.0], self.evaluate(var0))
          self.assertAllClose([3.0, 4.0], self.evaluate(var1))

        beta1_power, beta2_power = opt._get_beta_accumulators()

        # Run 3 steps of Adam
        for t in range(1, 4):
          if not context.executing_eagerly():
            self.evaluate(update)
          elif t > 1:
            opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

          self.assertAllCloseAccordingToType(0.9**(t + 1),
                                             self.evaluate(beta1_power))
          self.assertAllCloseAccordingToType(0.999**(t + 1),
                                             self.evaluate(beta2_power))

          var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))
          if use_resource:
            self.assertEqual("var0_%d/Adam:0" % (i,),
                             opt.get_slot(var=var0, name="m").name)

  def testBasic(self):
    with self.cached_session():
      self.doTestBasic(use_resource=False)

  @test_util.run_in_graph_and_eager_modes
  @test_util.disable_tfrt("b/168527439: invalid runtime fallback "
                          "resource variable reference on GPU.")
  def testResourceBasic(self):
    self.doTestBasic(use_resource=True)

  @test_util.disable_tfrt("b/153089059: cannot create half tensor on GPU.")
  def testBasicCallableParams(self):
    with context.eager_mode():
      self.doTestBasic(use_resource=True, use_callable_params=True)

  def testTensorLearningRate(self):
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
        with self.cached_session():
          # Initialize variables for numpy implementation.
          m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
          var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
          grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
          var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
          grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

          var0 = variables.Variable(var0_np)
          var1 = variables.Variable(var1_np)
          grads0 = constant_op.constant(grads0_np)
          grads1 = constant_op.constant(grads1_np)
          opt = adam.AdamOptimizer(constant_op.constant(0.001))
          update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
          self.evaluate(variables.global_variables_initializer())

          # Fetch params to validate initial values
          self.assertAllClose([1.0, 2.0], self.evaluate(var0))
          self.assertAllClose([3.0, 4.0], self.evaluate(var1))

          beta1_power, beta2_power = opt._get_beta_accumulators()

          # Run 3 steps of Adam
          for t in range(1, 4):
            self.assertAllCloseAccordingToType(0.9**t,
                                               self.evaluate(beta1_power))
            self.assertAllCloseAccordingToType(0.999**t,
                                               self.evaluate(beta2_power))
            update.run()

            var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
            var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

            # Validate updated params
            self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
            self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  def testSharing(self):
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
        with self.cached_session():
          # Initialize variables for numpy implementation.
          m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
          var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
          grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
          var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
          grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

          var0 = variables.Variable(var0_np)
          var1 = variables.Variable(var1_np)
          grads0 = constant_op.constant(grads0_np)
          grads1 = constant_op.constant(grads1_np)
          opt = adam.AdamOptimizer()
          update1 = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
          update2 = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
          self.evaluate(variables.global_variables_initializer())

          beta1_power, beta2_power = opt._get_beta_accumulators()

          # Fetch params to validate initial values
          self.assertAllClose([1.0, 2.0], self.evaluate(var0))
          self.assertAllClose([3.0, 4.0], self.evaluate(var1))

          # Run 3 steps of intertwined Adam1 and Adam2.
          for t in range(1, 4):
            self.assertAllCloseAccordingToType(0.9**t,
                                               self.evaluate(beta1_power))
            self.assertAllCloseAccordingToType(0.999**t,
                                               self.evaluate(beta2_power))
            if t % 2 == 0:
              update1.run()
            else:
              update2.run()

            var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
            var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

            # Validate updated params
            self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
            self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  @test_util.disable_tfrt("b/168527439: invalid runtime fallback "
                          "resource variable reference on GPU.")
  def testTwoSessions(self):
    optimizer = adam.AdamOptimizer()

    with context.eager_mode():
      var0 = variables.Variable(np.array([1.0, 2.0]), name="v0")
      grads0 = constant_op.constant(np.array([0.1, 0.1]))
      optimizer.apply_gradients([(grads0, var0)])

    g = ops.Graph()
    with g.as_default():
      with session.Session():
        var0 = variables.Variable(np.array([1.0, 2.0]), name="v0")
        grads0 = constant_op.constant(np.array([0.1, 0.1]))
        optimizer.apply_gradients([(grads0, var0)])

    gg = ops.Graph()
    with gg.as_default():
      with session.Session():
        var0 = variables.Variable(np.array([1.0, 2.0]), name="v0")
        grads0 = constant_op.constant(np.array([0.1, 0.1]))

        # If the optimizer saves any state not keyed by graph the following line
        # fails.
        optimizer.apply_gradients([(grads0, var0)])

  @test_util.disable_tfrt("b/168527439: invalid runtime fallback "
                          "resource variable reference on GPU.")
  def testSlotsUniqueEager(self):
    with context.eager_mode():
      v1 = resource_variable_ops.ResourceVariable(1.)
      v2 = resource_variable_ops.ResourceVariable(1.)
      opt = adam.AdamOptimizer(1.)
      opt.minimize(lambda: v1 + v2)
      # There should be two non-slot variables, and two unique slot variables
      # for v1 and v2 respectively.
      self.assertEqual(6, len({id(v) for v in opt.variables()}))

  @test_util.deprecated_graph_mode_only
  def testXlaSharding(self):
    dtype = dtypes.float32
    with self.session(graph=ops.Graph()):
      # Initialize variables for numpy implementation.
      var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
      grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
      var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
      grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

      var0 = resource_variable_ops.ResourceVariable(var0_np, name="var0")
      var1 = resource_variable_ops.ResourceVariable(var1_np, name="var1")
      var0, var1 = [
          xla_sharding.mesh_split(
              v, np.array([0, 1]), [0], use_sharding_op=False)
          for v in (var0, var1)
      ]
      grads0 = constant_op.constant(grads0_np)
      grads1 = constant_op.constant(grads1_np)

      learning_rate = lambda: 0.001

      opt = adam.AdamOptimizer(learning_rate=learning_rate)
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

      self.evaluate(variables.global_variables_initializer())
      self.evaluate(update)
      # The beta accumulators are not sharded.
      beta1_power, beta2_power = opt._get_beta_accumulators()
      self.assertIsNone(xla_sharding.get_tensor_sharding(beta1_power))
      self.assertIsNone(xla_sharding.get_tensor_sharding(beta2_power))

      # Variables and slots are sharded.
      for v in (var0, var1):
        self.assertIsNotNone(xla_sharding.get_tensor_sharding(v))
        for slot_name in ("m", "v"):
          slot = opt.get_slot(v, slot_name)
          self.assertIsNotNone(xla_sharding.get_tensor_sharding(slot))


if __name__ == "__main__":
  test.main()
