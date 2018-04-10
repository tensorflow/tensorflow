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
"""Tests for Momentum."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

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
from tensorflow.python.training import momentum as momentum_lib


class MomentumOptimizerTest(test.TestCase):

  def _update_nesterov_momentum_numpy(self, var, accum, g, lr, momentum):
    var = var + accum * lr * momentum
    accum = accum * momentum + g
    var = var - lr * accum
    var = var - accum * lr * momentum
    return var, accum

  def doTestBasic(self, use_resource=False, use_callable_params=False):
    for i, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      if use_resource:
        var0 = resource_variable_ops.ResourceVariable(
            [1.0, 2.0], dtype=dtype, name="var0_%d" % i)
        var1 = resource_variable_ops.ResourceVariable(
            [3.0, 4.0], dtype=dtype, name="var1_%d" % i)
      else:
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
      grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
      grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
      learning_rate = lambda: 2.0
      momentum = lambda: 0.9
      if not use_callable_params:
        learning_rate = learning_rate()
        momentum = momentum()
      mom_opt = momentum_lib.MomentumOptimizer(
          learning_rate=learning_rate, momentum=momentum)
      mom_update = mom_opt.apply_gradients(
          zip([grads0, grads1], [var0, var1]))

      if not context.executing_eagerly():
        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))

      # Check we have slots
      self.assertEqual(["momentum"], mom_opt.get_slot_names())
      slot0 = mom_opt.get_slot(var0, "momentum")
      self.assertEquals(slot0.get_shape(), var0.get_shape())
      slot1 = mom_opt.get_slot(var1, "momentum")
      self.assertEquals(slot1.get_shape(), var1.get_shape())
      if not context.executing_eagerly():
        self.assertFalse(slot0 in variables.trainable_variables())
        self.assertFalse(slot1 in variables.trainable_variables())

      # Step 1: the momentum accumulators where 0. So we should see a normal
      # update: v -= grad * learning_rate
      if not context.executing_eagerly():
        self.evaluate(mom_update)
      # Check that the momentum accumulators have been updated.
      self.assertAllCloseAccordingToType(np.array([0.1, 0.1]),
                                         self.evaluate(slot0))
      self.assertAllCloseAccordingToType(np.array([0.01, 0.01]),
                                         self.evaluate(slot1))
      # Check that the parameters have been updated.
      self.assertAllCloseAccordingToType(
          np.array([1.0 - (0.1 * 2.0), 2.0 - (0.1 * 2.0)]),
          self.evaluate(var0))
      self.assertAllCloseAccordingToType(
          np.array([3.0 - (0.01 * 2.0), 4.0 - (0.01 * 2.0)]),
          self.evaluate(var1))
      # Step 2: the momentum accumulators contain the previous update.
      if context.executing_eagerly():
        mom_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      else:
        self.evaluate(mom_update)
      # Check that the momentum accumulators have been updated.
      self.assertAllCloseAccordingToType(
          np.array([(0.9 * 0.1 + 0.1), (0.9 * 0.1 + 0.1)]),
          self.evaluate(slot0))
      self.assertAllCloseAccordingToType(
          np.array([(0.9 * 0.01 + 0.01), (0.9 * 0.01 + 0.01)]),
          self.evaluate(slot1))
      # Check that the parameters have been updated.
      self.assertAllCloseAccordingToType(
          np.array([
              1.0 - (0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0),
              2.0 - (0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0)
          ]), self.evaluate(var0))
      self.assertAllCloseAccordingToType(
          np.array([
              2.98 - ((0.9 * 0.01 + 0.01) * 2.0), 3.98 - (
                  (0.9 * 0.01 + 0.01) * 2.0)
          ]), self.evaluate(var1))

  def testBasic(self):
    with self.test_session():
      self.doTestBasic(use_resource=False)

  @test_util.run_in_graph_and_eager_modes(reset_test=True)
  def testResourceBasic(self):
    self.doTestBasic(use_resource=True)

  def testBasicCallableParams(self):
    with context.eager_mode():
      self.doTestBasic(use_resource=True, use_callable_params=True)

  @test_util.run_in_graph_and_eager_modes(reset_test=True)
  def testVariablesAcrossGraphs(self):
    optimizer = momentum_lib.MomentumOptimizer(0.01, 0.5)
    with ops.Graph().as_default():
      var0 = resource_variable_ops.ResourceVariable(
          [1.0, 2.0], dtype=dtypes.float32, name="var0")
      var1 = resource_variable_ops.ResourceVariable(
          [3.0, 4.0], dtype=dtypes.float32, name="var1")
      if context.executing_eagerly():
        loss = lambda: math_ops.reduce_sum(var0 + var1)
      else:
        loss = math_ops.reduce_sum(var0 + var1)
      optimizer.minimize(loss)
      optimizer_variables = optimizer.variables()
      self.assertStartsWith(optimizer_variables[0].name, "var0")
      self.assertStartsWith(optimizer_variables[1].name, "var1")
      self.assertEquals(2, len(optimizer_variables))

    with ops.Graph().as_default():
      var2 = resource_variable_ops.ResourceVariable(
          [1.0, 2.0], dtype=dtypes.float32, name="var2")
      var3 = resource_variable_ops.ResourceVariable(
          [3.0, 4.0], dtype=dtypes.float32, name="var3")
      if context.executing_eagerly():
        loss = lambda: math_ops.reduce_sum(var2 + var3)
      else:
        loss = math_ops.reduce_sum(var2 + var3)
      optimizer.minimize(loss)
      optimizer_variables = optimizer.variables()
      self.assertStartsWith(optimizer_variables[0].name, "var2")
      self.assertStartsWith(optimizer_variables[1].name, "var3")
      self.assertEquals(2, len(optimizer_variables))

  def testNesterovMomentum(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      with self.test_session():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        accum0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        accum1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        cost = 5 * var0 * var0 + 3 * var1
        global_step = variables.Variable(
            array_ops.zeros([], dtypes.int64), name="global_step")
        mom_op = momentum_lib.MomentumOptimizer(
            learning_rate=2.0, momentum=0.9, use_nesterov=True)
        opt_op = mom_op.minimize(cost, global_step, [var0, var1])
        variables.global_variables_initializer().run()
        for t in range(1, 5):
          opt_op.run()
          var0_np, accum0_np = self._update_nesterov_momentum_numpy(
              var0_np, accum0_np, var0_np * 10, 2.0, 0.9)
          var1_np, accum1_np = self._update_nesterov_momentum_numpy(var1_np,
                                                                    accum1_np,
                                                                    3, 2.0, 0.9)
          self.assertAllClose(var0_np, var0.eval())
          self.assertAllClose(var1_np, var1.eval())

  def testSparseNesterovMomentum(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      with self.test_session():
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        accum0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        accum1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        grads = []
        for t in range(1, 5):
          grads.append(var0_np * 10)
          var0_np, accum0_np = self._update_nesterov_momentum_numpy(
              var0_np, accum0_np, var0_np * 10, 2.0, 0.9)
          var1_np, accum1_np = self._update_nesterov_momentum_numpy(var1_np,
                                                                    accum1_np,
                                                                    3, 2.0, 0.9)
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        accum0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        accum1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        var0 = variables.Variable(var0_np)
        var1 = variables.Variable(var1_np)
        loss = 5 * var0 * var0 + 3 * var1
        mom_op = momentum_lib.MomentumOptimizer(
            learning_rate=2.0, momentum=0.9, use_nesterov=True)
        x_feed = array_ops.placeholder(dtype)
        y_feed = ops.IndexedSlices(
            x_feed, constant_op.constant([0, 1]), constant_op.constant([2]))
        grads_and_vars = [(y_feed, var0), (constant_op.constant(
            [3.0, 3.0], dtype=dtype), var1)]
        opt_update = mom_op.apply_gradients(grads_and_vars)
        variables.global_variables_initializer().run()
        for t in range(1, 5):
          opt_update.run(feed_dict={x_feed: grads[t - 1]})
          var0_np, accum0_np = self._update_nesterov_momentum_numpy(
              var0_np, accum0_np, var0_np * 10, 2.0, 0.9)
          var1_np, accum1_np = self._update_nesterov_momentum_numpy(var1_np,
                                                                    accum1_np,
                                                                    3, 2.0, 0.9)
          self.assertAllClose(var0_np, var0.eval())
          self.assertAllClose(var1_np, var1.eval())

  @test_util.run_in_graph_and_eager_modes(reset_test=True)
  def testMinimizeSparseResourceVariable(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      # This test invokes the ResourceSparseApplyMomentum operation, which
      # did not have a registered GPU kernel as of April 2018. With graph
      # execution, the placement algorithm notices this and automatically
      # places the variable in CPU (host) memory. With eager execution,
      # the variable would be placed in GPU memory if available, which
      # would then conflict with the future invocation of the
      # ResourceSparseApplyMomentum operation.
      # To work around this discrepancy, for now we force the variable
      # to be placed on CPU.
      with ops.device("/cpu:0"):
        var0 = resource_variable_ops.ResourceVariable([[1.0, 2.0]], dtype=dtype)

      # pylint: disable=cell-var-from-loop
      def loss():
        x = constant_op.constant([[4.0], [5.0]], dtype=dtype)
        pred = math_ops.matmul(embedding_ops.embedding_lookup([var0], [0]), x)
        return pred * pred
      # pylint: enable=cell-var-from-loop

      opt = momentum_lib.MomentumOptimizer(learning_rate=1.0, momentum=0.0)
      sgd_op = opt.minimize(loss)
      self.evaluate(variables.global_variables_initializer())
      # Run 1 step of sgd
      self.evaluate(sgd_op)
      # Validate updated params
      self.assertAllCloseAccordingToType([[-111, -138]], self.evaluate(var0))

  @test_util.run_in_graph_and_eager_modes(reset_test=True)
  def testMinimizeWith2DIndiciesForEmbeddingLookup(self):
    # This test invokes the ResourceSparseApplyMomentum operation, which
    # did not have a registered GPU kernel as of April 2018. With graph
    # execution, the placement algorithm notices this and automatically
    # places the variable in CPU (host) memory. With eager execution,
    # the variable would be placed in GPU memory if available, which
    # would then conflict with the future invocation of the
    # ResourceSparseApplyMomentum operation.
    # To work around this discrepancy, for now we force the variable
    # to be placed on CPU.
    with ops.device("/cpu:0"):
      var0 = resource_variable_ops.ResourceVariable(array_ops.ones([2, 2]))

    def loss():
      return math_ops.reduce_sum(embedding_ops.embedding_lookup(var0, [[1]]))

    opt = momentum_lib.MomentumOptimizer(learning_rate=1.0, momentum=0.0)
    sgd_op = opt.minimize(loss)
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(sgd_op)
    self.assertAllCloseAccordingToType([[1, 1], [0, 0]], self.evaluate(var0))

  def testTensorLearningRateAndMomentum(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.test_session():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        mom_opt = momentum_lib.MomentumOptimizer(
            learning_rate=constant_op.constant(2.0),
            momentum=constant_op.constant(0.9))
        mom_update = mom_opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()
        # Check we have slots
        self.assertEqual(["momentum"], mom_opt.get_slot_names())
        slot0 = mom_opt.get_slot(var0, "momentum")
        self.assertEquals(slot0.get_shape(), var0.get_shape())
        self.assertFalse(slot0 in variables.trainable_variables())
        slot1 = mom_opt.get_slot(var1, "momentum")
        self.assertEquals(slot1.get_shape(), var1.get_shape())
        self.assertFalse(slot1 in variables.trainable_variables())

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 4.0], var1.eval())
        # Step 1: the momentum accumulators where 0. So we should see a normal
        # update: v -= grad * learning_rate
        mom_update.run()
        # Check that the momentum accumulators have been updated.
        self.assertAllCloseAccordingToType(np.array([0.1, 0.1]), slot0.eval())
        self.assertAllCloseAccordingToType(np.array([0.01, 0.01]), slot1.eval())
        # Check that the parameters have been updated.
        self.assertAllCloseAccordingToType(
            np.array([1.0 - (0.1 * 2.0), 2.0 - (0.1 * 2.0)]), var0.eval())
        self.assertAllCloseAccordingToType(
            np.array([3.0 - (0.01 * 2.0), 4.0 - (0.01 * 2.0)]), var1.eval())
        # Step 2: the momentum accumulators contain the previous update.
        mom_update.run()
        # Check that the momentum accumulators have been updated.
        self.assertAllCloseAccordingToType(
            np.array([(0.9 * 0.1 + 0.1), (0.9 * 0.1 + 0.1)]), slot0.eval())
        self.assertAllCloseAccordingToType(
            np.array([(0.9 * 0.01 + 0.01), (0.9 * 0.01 + 0.01)]), slot1.eval())
        # Check that the parameters have been updated.
        self.assertAllCloseAccordingToType(
            np.array([
                1.0 - (0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0),
                2.0 - (0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0)
            ]), var0.eval())
        self.assertAllCloseAccordingToType(
            np.array([
                2.98 - ((0.9 * 0.01 + 0.01) * 2.0), 3.98 - (
                    (0.9 * 0.01 + 0.01) * 2.0)
            ]), var1.eval())

  def _dbParamsMom01(self):
    """Return dist-belief momentum values.

    Return values been generated from the dist-belief momentum unittest,
    running with a learning rate of 0.1 and a momentum of 0.1.

    These values record how a parameter vector of size 10, initialized with 0.0,
    gets updated with 10 consecutive momentum steps.  It uses random gradients.

    Returns:
      db_grad: The gradients to apply
      db_out: The parameters after the momentum update.
    """
    db_grad = [[]] * 10
    db_out = [[]] * 10
    # pylint: disable=line-too-long
    db_grad[0] = [
        0.00096264342, 0.17914793, 0.93945462, 0.41396621, 0.53037018,
        0.93197989, 0.78648776, 0.50036013, 0.55345792, 0.96722615
    ]
    db_out[0] = [
        -9.6264346e-05, -0.017914793, -0.093945466, -0.041396622, -0.053037018,
        -0.093197994, -0.078648776, -0.050036013, -0.055345792, -0.096722618
    ]
    db_grad[1] = [
        0.17075552, 0.88821375, 0.20873757, 0.25236958, 0.57578111, 0.15312378,
        0.5513742, 0.94687688, 0.16012503, 0.22159521
    ]
    db_out[1] = [
        -0.017181443, -0.10852765, -0.12421377, -0.070773244, -0.11591884,
        -0.11783017, -0.14165108, -0.14972731, -0.076892875, -0.1285544
    ]
    db_grad[2] = [
        0.35077485, 0.47304362, 0.44412705, 0.44368884, 0.078527533, 0.81223965,
        0.31168157, 0.43203235, 0.16792089, 0.24644311
    ]
    db_out[2] = [
        -0.053967446, -0.1648933, -0.1716533, -0.1180798, -0.13005978,
        -0.20151734, -0.17911947, -0.20289968, -0.095839672, -0.15638189
    ]
    db_grad[3] = [
        0.9694621, 0.75035888, 0.28171822, 0.83813518, 0.53807181, 0.3728098,
        0.81454384, 0.03848977, 0.89759839, 0.93665648
    ]
    db_out[3] = [
        -0.15459226, -0.24556576, -0.20456907, -0.20662397, -0.18528105,
        -0.24716705, -0.2643207, -0.21206589, -0.18749419, -0.2528303
    ]
    db_grad[4] = [
        0.38578293, 0.8536852, 0.88722926, 0.66276771, 0.13678469, 0.94036359,
        0.69107032, 0.81897682, 0.5433259, 0.67860287
    ]
    db_out[4] = [
        -0.20323303, -0.33900154, -0.29658359, -0.28175515, -0.20448165,
        -0.34576839, -0.34194785, -0.29488021, -0.25099224, -0.33033544
    ]
    db_grad[5] = [
        0.27885768, 0.76100707, 0.24625534, 0.81354135, 0.18959245, 0.48038563,
        0.84163809, 0.41172323, 0.83259648, 0.44941229
    ]
    db_out[5] = [
        -0.23598288, -0.42444581, -0.33041057, -0.3706224, -0.22536094,
        -0.40366709, -0.43387437, -0.34433398, -0.34060168, -0.38302717
    ]
    db_grad[6] = [
        0.27233034, 0.056316052, 0.5039115, 0.24105175, 0.35697976, 0.75913221,
        0.73577434, 0.16014607, 0.57500273, 0.071136251
    ]
    db_out[6] = [
        -0.26649091, -0.43862185, -0.38418442, -0.40361428, -0.26314685,
        -0.48537019, -0.51664448, -0.36529395, -0.40706289, -0.39540997
    ]
    db_grad[7] = [
        0.58697265, 0.2494842, 0.08106143, 0.39954534, 0.15892942, 0.12683646,
        0.74053431, 0.16033, 0.66625422, 0.73515922
    ]
    db_out[7] = [
        -0.32823896, -0.46498787, -0.39766794, -0.446868, -0.28281838,
        -0.50622416, -0.59897494, -0.38342294, -0.48033443, -0.47016418
    ]
    db_grad[8] = [
        0.8215279, 0.41994119, 0.95172721, 0.68000203, 0.79439718, 0.43384039,
        0.55561525, 0.22567581, 0.93331909, 0.29438227
    ]
    db_out[8] = [
        -0.41656655, -0.50961858, -0.49418902, -0.51919359, -0.36422527,
        -0.55169362, -0.6627695, -0.40780342, -0.58099347, -0.50707781
    ]
    db_grad[9] = [
        0.68297005, 0.67758518, 0.1748755, 0.13266537, 0.70697063, 0.055731893,
        0.68593478, 0.50580865, 0.12602448, 0.093537711
    ]
    db_out[9] = [
        -0.49369633, -0.58184016, -0.52132869, -0.5396927, -0.44306302,
        -0.56181377, -0.73774242, -0.46082234, -0.60366184, -0.52012295
    ]
    # pylint: enable=line-too-long
    return db_grad, db_out

  def testLikeDistBeliefMom01(self):
    with self.test_session():
      db_grad, db_out = self._dbParamsMom01()
      num_samples = len(db_grad)
      var0 = variables.Variable([0.0] * num_samples)
      grads0 = constant_op.constant([0.0] * num_samples)
      mom_opt = momentum_lib.MomentumOptimizer(learning_rate=0.1, momentum=0.1)
      mom_update = mom_opt.apply_gradients(zip([grads0], [var0]))
      variables.global_variables_initializer().run()
      for i in xrange(num_samples):
        mom_update.run(feed_dict={grads0: db_grad[i]})
        self.assertAllClose(np.array(db_out[i]), var0.eval())

  def testSparse(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.test_session():
        var0 = variables.Variable(array_ops.zeros([4, 2], dtype=dtype))
        var1 = variables.Variable(constant_op.constant(1.0, dtype, [4, 2]))
        grads0 = ops.IndexedSlices(
            constant_op.constant(
                [[.1, .1]], dtype=dtype),
            constant_op.constant([1]),
            constant_op.constant([4, 2]))
        grads1 = ops.IndexedSlices(
            constant_op.constant(
                [[.01, .01], [.01, .01]], dtype=dtype),
            constant_op.constant([2, 3]),
            constant_op.constant([4, 2]))
        mom_opt = momentum_lib.MomentumOptimizer(
            learning_rate=2.0, momentum=0.9)
        mom_update = mom_opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        # Check we have slots
        self.assertEqual(["momentum"], mom_opt.get_slot_names())
        slot0 = mom_opt.get_slot(var0, "momentum")
        self.assertEquals(slot0.get_shape(), var0.get_shape())
        slot1 = mom_opt.get_slot(var1, "momentum")
        self.assertEquals(slot1.get_shape(), var1.get_shape())

        # Fetch params to validate initial values
        self.assertAllClose([0, 0], var0.eval()[0])
        self.assertAllClose([0, 0], var0.eval()[1])
        self.assertAllClose([1, 1], var1.eval()[2])

        # Step 1: the momentum accumulators are 0. So we should see a normal
        # update: v -= grad * learning_rate
        mom_update.run()
        # Check that the momentum accumulators have been updated.
        self.assertAllCloseAccordingToType(np.array([0, 0]), slot0.eval()[0])
        self.assertAllCloseAccordingToType(np.array([.1, .1]), slot0.eval()[1])
        self.assertAllCloseAccordingToType(
            np.array([.01, .01]), slot1.eval()[2])
        # Check that the parameters have been updated.
        self.assertAllCloseAccordingToType(np.array([0, 0]), var0.eval()[0])
        self.assertAllCloseAccordingToType(
            np.array([-(0.1 * 2.0), -(0.1 * 2.0)]), var0.eval()[1])
        self.assertAllCloseAccordingToType(
            np.array([1.0 - (0.01 * 2.0), 1.0 - (0.01 * 2.0)]), var1.eval()[2])
        # Step 2: the momentum accumulators contain the previous update.
        mom_update.run()
        # Check that the momentum accumulators have been updated.
        self.assertAllClose(np.array([0, 0]), slot0.eval()[0])
        self.assertAllCloseAccordingToType(
            np.array([(0.9 * 0.1 + 0.1), (0.9 * 0.1 + 0.1)]), slot0.eval()[1])
        self.assertAllCloseAccordingToType(
            np.array([(0.9 * 0.01 + 0.01), (0.9 * 0.01 + 0.01)]),
            slot1.eval()[2])
        # Check that the parameters have been updated.
        self.assertAllClose(np.array([0, 0]), var0.eval()[0])
        self.assertAllCloseAccordingToType(
            np.array([
                -(0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0), -(0.1 * 2.0) - (
                    (0.9 * 0.1 + 0.1) * 2.0)
            ]), var0.eval()[1])
        self.assertAllCloseAccordingToType(
            np.array([
                0.98 - ((0.9 * 0.01 + 0.01) * 2.0), 0.98 - (
                    (0.9 * 0.01 + 0.01) * 2.0)
            ]), var1.eval()[2])

  def testSharing(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.test_session():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        mom_opt = momentum_lib.MomentumOptimizer(
            learning_rate=2.0, momentum=0.9)
        mom_update1 = mom_opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        mom_update2 = mom_opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        self.assertEqual(["momentum"], mom_opt.get_slot_names())
        slot0 = mom_opt.get_slot(var0, "momentum")
        self.assertEquals(slot0.get_shape(), var0.get_shape())
        slot1 = mom_opt.get_slot(var1, "momentum")
        self.assertEquals(slot1.get_shape(), var1.get_shape())

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 4.0], var1.eval())
        # Step 1: the momentum accumulators where 0. So we should see a normal
        # update: v -= grad * learning_rate
        mom_update1.run()
        # Check that the momentum accumulators have been updated.
        self.assertAllCloseAccordingToType(np.array([0.1, 0.1]), slot0.eval())
        self.assertAllCloseAccordingToType(np.array([0.01, 0.01]), slot1.eval())
        # Check that the parameters have been updated.
        self.assertAllCloseAccordingToType(
            np.array([1.0 - (0.1 * 2.0), 2.0 - (0.1 * 2.0)]), var0.eval())
        self.assertAllCloseAccordingToType(
            np.array([3.0 - (0.01 * 2.0), 4.0 - (0.01 * 2.0)]), var1.eval())
        # Step 2: the second momentum accumulators contain the previous update.
        mom_update2.run()
        # Check that the momentum accumulators have been updated.
        self.assertAllCloseAccordingToType(
            np.array([(0.9 * 0.1 + 0.1), (0.9 * 0.1 + 0.1)]), slot0.eval())
        self.assertAllCloseAccordingToType(
            np.array([(0.9 * 0.01 + 0.01), (0.9 * 0.01 + 0.01)]), slot1.eval())
        # Check that the parameters have been updated.
        self.assertAllCloseAccordingToType(
            np.array([
                1.0 - (0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0),
                2.0 - (0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0)
            ]), var0.eval())
        self.assertAllCloseAccordingToType(
            np.array([
                2.98 - ((0.9 * 0.01 + 0.01) * 2.0), 3.98 - (
                    (0.9 * 0.01 + 0.01) * 2.0)
            ]), var1.eval())


if __name__ == "__main__":
  test.main()
