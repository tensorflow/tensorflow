# Copyright 2019 Vishnu sai rao suresh Lokhande & Pengyu Kan. All Rights Reserved.
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
"""Tests for Conditional Gradient."""

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
from tensorflow.python.training import conditional_gradient as cg_lib


class ConditionalGradientTest(test.TestCase):


  def _update_conditional_gradient_numpy(self, var, norm, g, lr, lamda):
    var = var * lr - (1 - lr) * lamda * g /norm
    return var

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
      norm0 = math_ops.reduce_sum(grads0 ** 2) ** 0.5
      norm1 = math_ops.reduce_sum(grads1 ** 2) ** 0.5
      learning_rate = lambda: 0.5
      lamda = lambda: 0.01
      if not use_callable_params:
        learning_rate = learning_rate()
        lamda = lamda()
      cg_opt = cg_lib.ConditionalGradientOptimizer(
          learning_rate=learning_rate, lamda=lamda)
      cg_update = cg_opt.apply_gradients(
          zip([grads0, grads1], [var0, var1]))

      if not context.executing_eagerly():
        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))

      # Check we have slots
      self.assertEqual(["conditional_gradient"], cg_opt.get_slot_names())
      slot0 = cg_opt.get_slot(var0, "conditional_gradient")
      self.assertEquals(slot0.get_shape(), var0.get_shape())
      slot1 = cg_opt.get_slot(var1, "conditional_gradient")
      self.assertEquals(slot1.get_shape(), var1.get_shape())
      if not context.executing_eagerly():
        self.assertFalse(slot0 in variables.trainable_variables())
        self.assertFalse(slot1 in variables.trainable_variables())

      if not context.executing_eagerly():
        self.evaluate(cg_update)

      # Check that the parameters have been updated.
      norm0 = self.evaluate(norm0)
      norm1 = self.evaluate(norm1)
      self.assertAllCloseAccordingToType(
          np.array([
              1.0 * 0.5 - (1-0.5) * 0.01 * 0.1 / norm0,
              2.0 * 0.5 - (1-0.5) * 0.01 * 0.1 / norm0
          ]), self.evaluate(var0))
      self.assertAllCloseAccordingToType(
          np.array([
              3.0 * 0.5 - (1-0.5) * 0.01 * 0.01 / norm1,
              4.0 * 0.5 - (1-0.5) * 0.01 * 0.01 / norm1
          ]), self.evaluate(var1))

      # Step 2: the conditional_gradient contain the previous update.
      if context.executing_eagerly():
        cg_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      else:
        self.evaluate(cg_update)
      self.assertAllCloseAccordingToType(
          np.array([
              (1.0 * 0.5 - (1 - 0.5) * 0.01 * 0.1 / norm0) * 0.5  \
                - (1 - 0.5) * 0.01 * 0.1 / norm0,
              (2.0 * 0.5 - (1 - 0.5) * 0.01 * 0.1 / norm0) * 0.5  \
                - (1 - 0.5) * 0.01 * 0.1 / norm0
          ]), self.evaluate(var0))
      self.assertAllCloseAccordingToType(
          np.array([
              (3.0 * 0.5 - (1 - 0.5) * 0.01 * 0.01 / norm1) * 0.5  \
                - (1 - 0.5) * 0.01 * 0.01 / norm1,
              (4.0 * 0.5 - (1 - 0.5) * 0.01 * 0.01 / norm1) * 0.5  \
                - (1 - 0.5) * 0.01 * 0.01 / norm1
          ]), self.evaluate(var1))


  def testBasic(self):
    with self.cached_session():
      self.doTestBasic(use_resource=False)


  @test_util.run_in_graph_and_eager_modes(reset_test=True)
  def testResourceBasic(self):
    self.doTestBasic(use_resource=True)

  def testBasicCallableParams(self):
    with context.eager_mode():
      self.doTestBasic(use_resource=True, use_callable_params=True)

  def testVariablesAcrossGraphs(self):
    optimizer = cg_lib.ConditionalGradientOptimizer(0.01, 0.5)
    with ops.Graph().as_default():
      var0 = resource_variable_ops.ResourceVariable(
          [1.0, 2.0], dtype=dtypes.float32, name="var0")
      var1 = resource_variable_ops.ResourceVariable(
          [3.0, 4.0], dtype=dtypes.float32, name="var1")
      loss = math_ops.reduce_sum(var0 + var1)
      optimizer.minimize(loss)
      optimizer_variables = optimizer.variables()
      self.assertStartsWith(optimizer_variables[0].name, "var0")
      self.assertStartsWith(optimizer_variables[1].name, "var1")
      self.assertEqual(2, len(optimizer_variables))

    with ops.Graph().as_default():
      var2 = resource_variable_ops.ResourceVariable(
          [1.0, 2.0], dtype=dtypes.float32, name="var2")
      var3 = resource_variable_ops.ResourceVariable(
          [3.0, 4.0], dtype=dtypes.float32, name="var3")
      loss = math_ops.reduce_sum(var2 + var3)
      optimizer.minimize(loss)
      optimizer_variables = optimizer.variables()
      self.assertStartsWith(optimizer_variables[0].name, "var2")
      self.assertStartsWith(optimizer_variables[1].name, "var3")
      self.assertEqual(2, len(optimizer_variables))


  @test_util.run_in_graph_and_eager_modes(reset_test=True)
  def testMinimizeSparseResourceVariable(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      # This test invokes the ResourceSparseApplyConditionalGradient operation, which
      # did not have a registered GPU kernel as of April 2018. With graph
      # execution, the placement algorithm notices this and automatically
      # places the variable in CPU (host) memory. With eager execution,
      # the variable would be placed in GPU memory if available, which
      # would then conflict with the future invocation of the
      # ResourceSparseApplyConditionalGradient operation.
      # To work around this discrepancy, for now we force the variable
      # to be placed on CPU.
      with ops.device("/cpu:0"):
        var0 = resource_variable_ops.ResourceVariable([[1.0, 2.0]], dtype=dtype)

      # pylint: disable=cell-var-from-loop
      def loss():
        x = constant_op.constant([[4.0], [5.0]], dtype=dtype)
        pred = math_ops.matmul(embedding_ops.embedding_lookup([var0], [0]), x)
        return pred * pred

      #the gradient based on the current loss function
      grads0_0 = 32 * 1.0 + 40 * 2.0
      grads0_1 = 40 * 1.0 + 50 * 2.0
      grads0 = constant_op.constant([[grads0_0, grads0_1]], dtype=dtype)
      norm0 = math_ops.reduce_sum(grads0 ** 2) ** 0.5

      # pylint: enable=cell-var-from-loop
      learning_rate = 0.1
      lamda = 0.1
      opt = cg_lib.ConditionalGradientOptimizer(learning_rate=learning_rate, \
                                                lamda=lamda)
      cg_op = opt.minimize(loss)
      self.evaluate(variables.global_variables_initializer())

      # Run 1 step of cg_op
      self.evaluate(cg_op)

      # Validate updated params
      norm0 = self.evaluate(norm0)
      self.assertAllCloseAccordingToType([
          [1.0 * learning_rate - (1-learning_rate)*lamda*grads0_0/norm0,
           2.0 * learning_rate - (1-learning_rate)*lamda*grads0_1/norm0]
      ], self.evaluate(var0))


  @test_util.run_in_graph_and_eager_modes(reset_test=True)
  def testMinimizeWith2DIndiciesForEmbeddingLookup(self):
    # This test invokes the ResourceSparseApplyConditionalGradient operation, which
    # did not have a registered GPU kernel as of April 2018. With graph
    # execution, the placement algorithm notices this and automatically
    # places the variable in CPU (host) memory. With eager execution,
    # the variable would be placed in GPU memory if available, which
    # would then conflict with the future invocation of the
    # ResourceSparseApplyConditionalGradient operation.
    # To work around this discrepancy, for now we force the variable
    # to be placed on CPU.
    with ops.device("/cpu:0"):
      var0 = resource_variable_ops.ResourceVariable(array_ops.ones([2, 2]))

    def loss():
      return math_ops.reduce_sum(embedding_ops.embedding_lookup(var0, [[1]]))

    # the gradient for this loss function:
    grads0 = constant_op.constant([[0, 0], [1, 1]], dtype=dtypes.float32)
    norm0 = math_ops.reduce_sum(grads0 ** 2) ** 0.5

    learning_rate = 0.1
    lamda = 0.1
    opt = cg_lib.ConditionalGradientOptimizer(learning_rate=learning_rate,  \
                                              lamda=lamda)
    cg_op = opt.minimize(loss)
    self.evaluate(variables.global_variables_initializer())

    # Run 1 step of cg_op
    self.evaluate(cg_op)
    norm0 = self.evaluate(norm0)
    '''
        This is the test case we need to pass, if we only want to update the
        sparse dimension's entry of the var.
    '''
    self.assertAllCloseAccordingToType([
        [1,
         1],
        [learning_rate * 1 - (1-learning_rate)*lamda*1/norm0,
         learning_rate * 1 - (1-learning_rate)*lamda*1/norm0]
    ], self.evaluate(var0))

  @test_util.run_deprecated_v1
  def testTensorLearningRateAndConditionalGradient(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        norm0 = math_ops.reduce_sum(grads0 ** 2) ** 0.5
        norm1 = math_ops.reduce_sum(grads1 ** 2) ** 0.5
        cg_opt = cg_lib.ConditionalGradientOptimizer(
            learning_rate=constant_op.constant(0.5),
            lamda=constant_op.constant(0.01))
        cg_update = cg_opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()
        # Check we have slots
        self.assertEqual(["conditional_gradient"], cg_opt.get_slot_names())
        slot0 = cg_opt.get_slot(var0, "conditional_gradient")
        self.assertEquals(slot0.get_shape(), var0.get_shape())
        slot1 = cg_opt.get_slot(var1, "conditional_gradient")
        self.assertEquals(slot1.get_shape(), var1.get_shape())
        if not context.executing_eagerly():
          self.assertFalse(slot0 in variables.trainable_variables())
          self.assertFalse(slot1 in variables.trainable_variables())

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))

        cg_update.run()
        # Check that the parameters have been updated.
        norm0 = self.evaluate(norm0)
        norm1 = self.evaluate(norm1)
        self.assertAllCloseAccordingToType(
            np.array([
                1.0 * 0.5 - (1-0.5) * 0.01 * 0.1 / norm0,
                2.0 * 0.5 - (1-0.5) * 0.01 * 0.1 / norm0
            ]), self.evaluate(var0))
        self.assertAllCloseAccordingToType(
            np.array([
                3.0 * 0.5 - (1-0.5) * 0.01 * 0.01 / norm1,
                4.0 * 0.5 - (1-0.5) * 0.01 * 0.01 / norm1
            ]), self.evaluate(var1))
        # Step 2: the conditional_gradient contain the previous update.
        cg_update.run()
        # Check that the parameters have been updated.
        self.assertAllCloseAccordingToType(
            np.array([
                (1.0 * 0.5 - (1 - 0.5) * 0.01 * 0.1 / norm0) * 0.5  \
                        - (1 - 0.5) * 0.01 * 0.1 / norm0,
                (2.0 * 0.5 - (1 - 0.5) * 0.01 * 0.1 / norm0) * 0.5  \
                        - (1 - 0.5) * 0.01 * 0.1 / norm0
            ]), self.evaluate(var0))
        self.assertAllCloseAccordingToType(
            np.array([
                (3.0 * 0.5 - (1 - 0.5) * 0.01 * 0.01 / norm1) * 0.5 \
                        - (1 - 0.5) * 0.01 * 0.01 / norm1,
                (4.0 * 0.5 - (1 - 0.5) * 0.01 * 0.01 / norm1) * 0.5 \
                        - (1 - 0.5) * 0.01 * 0.01 / norm1
            ]), self.evaluate(var1))

  def _dbParamsCG01(self):
    """Return dist-belief conditional_gradient values.

    Return values been generated from the dist-belief conditional_gradient
    unittest, running with a learning rate of 0.1 and a lamda of 0.1.

    These values record how a parameter vector of size 10, initialized with 0.0,
    gets updated with 10 consecutive conditional_gradient steps.
    It uses random gradients.

    Returns:
      db_grad: The gradients to apply
      db_out: The parameters after the conditional_gradient update.
    """
    db_grad = [[]] * 10
    db_out = [[]] * 10
    # pylint: disable=line-too-long
    db_grad[0] = [
        0.00096264342, 0.17914793, 0.93945462, 0.41396621, 0.53037018,
        0.93197989, 0.78648776, 0.50036013, 0.55345792, 0.96722615
    ]
    db_out[0] = [
        -4.1555551e-05, -7.7334875e-03, -4.0554531e-02, -1.7870162e-02,
        -2.2895107e-02, -4.0231861e-02, -3.3951234e-02, -2.1599628e-02,
        -2.3891762e-02, -4.1753378e-02
    ]
    db_grad[1] = [
        0.17075552, 0.88821375, 0.20873757, 0.25236958, 0.57578111, 0.15312378,
        0.5513742, 0.94687688, 0.16012503, 0.22159521
    ]
    db_out[1] = [
        -0.00961733, -0.0507779, -0.01580694, -0.01599489, -0.03470477, -0.01264373,
        -0.03443632, -0.05546713, -0.01140388, -0.01665068
    ]
    db_grad[2] = [
        0.35077485, 0.47304362, 0.44412705, 0.44368884, 0.078527533, 0.81223965,
        0.31168157, 0.43203235, 0.16792089, 0.24644311
    ]
    db_out[2] = [
        -0.02462724, -0.03699233, -0.03154434, -0.03153357, -0.00876844, -0.05606323,
        -0.02447166, -0.03469437, -0.0124694, -0.01829169
    ]
    db_grad[3] = [
        0.9694621, 0.75035888, 0.28171822, 0.83813518, 0.53807181, 0.3728098,
        0.81454384, 0.03848977, 0.89759839, 0.93665648
    ]
    db_out[3] = [
        -0.04124615, -0.03371741, -0.0144246, -0.03668303, -0.02240246, -0.02052062,
        -0.03503307, -0.00500922, -0.03715545, -0.0393002
    ]
    db_grad[4] = [
        0.38578293, 0.8536852, 0.88722926, 0.66276771, 0.13678469, 0.94036359,
        0.69107032, 0.81897682, 0.5433259, 0.67860287
    ]
    db_out[4] = [
        -0.01979208, -0.0380417, -0.03747472, -0.0305847, -0.00779536, -0.04024222,
        -0.03156913, -0.0337613, -0.02578116, -0.03148952
    ]
    db_grad[5] = [
        0.27885768, 0.76100707, 0.24625534, 0.81354135, 0.18959245, 0.48038563,
        0.84163809, 0.41172323, 0.83259648, 0.44941229
    ]
    db_out[5] = [
        -0.01555188, -0.04084422, -0.01573331, -0.04265549, -0.01000746, -0.02740575,
        -0.04412147, -0.02341569, -0.0431026, -0.02502293
    ]
    db_grad[6] = [
        0.27233034, 0.056316052, 0.5039115, 0.24105175, 0.35697976, 0.75913221,
        0.73577434, 0.16014607, 0.57500273, 0.071136251
    ]
    db_out[6] = [
        -0.01890448, -0.00767214, -0.03367592, -0.01962219, -0.02374279, -0.05110247,
        -0.05128598, -0.01254396, -0.04094185, -0.00703416
    ]
    db_grad[7] = [
        0.58697265, 0.2494842, 0.08106143, 0.39954534, 0.15892942, 0.12683646,
        0.74053431, 0.16033, 0.66625422, 0.73515922
    ]
    db_out[7] = [
        -0.03772914, -0.01599993, -0.00831695, -0.02635719, -0.01207801, -0.01285448,
        -0.05034328, -0.01104364, -0.04477356, -0.04558991
    ]
    db_grad[8] = [
        0.8215279, 0.41994119, 0.95172721, 0.68000203, 0.79439718, 0.43384039,
        0.55561525, 0.22567581, 0.93331909, 0.29438227
    ]
    db_out[8] = [
        -0.03919835, -0.01970845, -0.04187151, -0.03195836, -0.03546333, -0.01999326,
        -0.02899324, -0.01083582, -0.04472339, -0.01725317
    ]
    db_grad[9] = [
        0.68297005, 0.67758518, 0.1748755, 0.13266537, 0.70697063, 0.055731893,
        0.68593478, 0.50580865, 0.12602448, 0.093537711
    ]
    db_out[9] = [
        -0.04510314, -0.04282944, -0.0147322, -0.0111956, -0.04617687, -0.00535998,
        -0.0442614, -0.03158399, -0.01207165, -0.00736567
    ]
    # pylint: enable=line-too-long
    return db_grad, db_out

  @test_util.run_deprecated_v1
  def testLikeDistBeliefCG01(self):
    with self.cached_session():
      db_grad, db_out = self._dbParamsCG01()
      num_samples = len(db_grad)
      var0 = variables.Variable([0.0] * num_samples)
      grads0 = constant_op.constant([0.0] * num_samples)
      cg_opt = cg_lib.ConditionalGradientOptimizer(learning_rate=0.1, lamda=0.1)
      cg_update = cg_opt.apply_gradients(zip([grads0], [var0]))
      variables.global_variables_initializer().run()
      for i in xrange(num_samples):
        cg_update.run(feed_dict={grads0: db_grad[i]})
        self.assertAllClose(np.array(db_out[i]), self.evaluate(var0))


  @test_util.run_deprecated_v1
  def testSparse(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
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
        norm0 = math_ops.reduce_sum(math_ops.multiply(grads0, grads0)) ** 0.5
        norm1 = math_ops.reduce_sum(math_ops.multiply(grads1, grads1)) ** 0.5
        learning_rate = 0.1
        lamda = 0.1
        cg_opt = cg_lib.ConditionalGradientOptimizer(
            learning_rate=learning_rate, lamda=lamda)
        cg_update = cg_opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()
        # Check we have slots
        self.assertEqual(["conditional_gradient"], cg_opt.get_slot_names())
        slot0 = cg_opt.get_slot(var0, "conditional_gradient")
        self.assertEquals(slot0.get_shape(), var0.get_shape())
        slot1 = cg_opt.get_slot(var1, "conditional_gradient")
        self.assertEquals(slot1.get_shape(), var1.get_shape())
        if not context.executing_eagerly():
          self.assertFalse(slot0 in variables.trainable_variables())
          self.assertFalse(slot1 in variables.trainable_variables())

        # Fetch params to validate initial values
        self.assertAllClose([0, 0], self.evaluate(var0)[0])
        self.assertAllClose([0, 0], self.evaluate(var0)[1])
        self.assertAllClose([1, 1], self.evaluate(var1)[2])

        # Step 1:
        cg_update.run()
        # Check that the parameters have been updated.
        norm0 = self.evaluate(norm0)
        norm1 = self.evaluate(norm1)
        self.assertAllCloseAccordingToType(
            np.array([0 - (1 - learning_rate) * lamda * 0 / norm0,
                      0 - (1 - learning_rate) * lamda * 0 / norm0]),
            self.evaluate(var0)[0])
        self.assertAllCloseAccordingToType(
            np.array([0 - (1 - learning_rate) * lamda * 0.1 / norm0,
                      0 - (1 - learning_rate) * lamda * 0.1 / norm0]),
            self.evaluate(var0)[1])
        self.assertAllCloseAccordingToType(
            np.array([1.0 * learning_rate- (1 - learning_rate)  \
                      * lamda * 0.01 / norm1,
                      1.0 * learning_rate- (1 - learning_rate)  \
                      * lamda * 0.01 / norm1]),
            self.evaluate(var1)[2])
        # Step 2: the conditional_gradient contain the previous update.
        cg_update.run()
        # Check that the parameters have been updated.
        self.assertAllClose(np.array([0, 0]), self.evaluate(var0)[0])
        self.assertAllCloseAccordingToType(
            np.array([
                (0 - (1 - learning_rate) * lamda * 0.1 / norm0) \
                      * learning_rate                           \
                      - (1 - learning_rate) * lamda * 0.1 / norm0,
                (0 - (1 - learning_rate) * lamda * 0.1 / norm0) \
                      * learning_rate                           \
                      - (1 - learning_rate) * lamda * 0.1 / norm0]),
            self.evaluate(var0)[1])
        self.assertAllCloseAccordingToType(
            np.array([
                (1.0 * learning_rate -                          \
                 (1 - learning_rate) * lamda * 0.01 / norm1)    \
                    * learning_rate - (1 - learning_rate)       \
                    * lamda * 0.01 / norm1,
                (1.0 * learning_rate-                           \
                 (1 - learning_rate) * lamda * 0.01 / norm1)    \
                    * learning_rate - (1 - learning_rate)       \
                    * lamda * 0.01 / norm1]),
            self.evaluate(var1)[2])


  @test_util.run_deprecated_v1
  def testSharing(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        norm0 = math_ops.reduce_sum(grads0 ** 2) ** 0.5
        norm1 = math_ops.reduce_sum(grads1 ** 2) ** 0.5
        learning_rate = 0.1
        lamda = 0.1
        cg_opt = cg_lib.ConditionalGradientOptimizer(
            learning_rate=learning_rate, lamda=lamda)
        cg_update1 = cg_opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        cg_update2 = cg_opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()
        # Check we have slots
        self.assertEqual(["conditional_gradient"], cg_opt.get_slot_names())
        slot0 = cg_opt.get_slot(var0, "conditional_gradient")
        self.assertEquals(slot0.get_shape(), var0.get_shape())
        slot1 = cg_opt.get_slot(var1, "conditional_gradient")
        self.assertEquals(slot1.get_shape(), var1.get_shape())
        if not context.executing_eagerly():
          self.assertFalse(slot0 in variables.trainable_variables())
          self.assertFalse(slot1 in variables.trainable_variables())
        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))

        cg_update1.run()
        # Check that the parameters have been updated.
        norm0 = self.evaluate(norm0)
        norm1 = self.evaluate(norm1)
        self.assertAllCloseAccordingToType(
            np.array(
                [1.0 * learning_rate - (1-learning_rate)    \
                    * lamda * 0.1 / norm0,
                 2.0 * learning_rate - (1-learning_rate)    \
                    * lamda * 0.1 / norm0]),
            self.evaluate(var0))
        self.assertAllCloseAccordingToType(
            np.array(
                [3.0 * learning_rate - (1-learning_rate)    \
                    * lamda * 0.01 / norm1,
                 4.0 * learning_rate - (1-learning_rate)    \
                    * lamda * 0.01 / norm1]),
            self.evaluate(var1))

        # Step 2: the second conditional_gradient contain the previous update.
        cg_update2.run()

        # Check that the parameters have been updated.
        self.assertAllCloseAccordingToType(
            np.array([
                (1.0 * learning_rate - (1-learning_rate)             \
                        * lamda * 0.1 / norm0) * learning_rate       \
                        - (1 - learning_rate) * lamda * 0.1 / norm0,
                (2.0 * learning_rate - (1-learning_rate)             \
                        * lamda * 0.1 / norm0) * learning_rate       \
                        - (1 - learning_rate) * lamda * 0.1 / norm0
            ]), self.evaluate(var0))
        self.assertAllCloseAccordingToType(
            np.array([
                (3.0 * learning_rate - (1-learning_rate)             \
                       * lamda * 0.01 / norm1) * learning_rate       \
                       - (1 - learning_rate) * lamda * 0.01 / norm1,
                (4.0 * learning_rate - (1-learning_rate)             \
                       * lamda * 0.01 / norm1) * learning_rate       \
                       - (1 - learning_rate) * lamda * 0.01 / norm1
            ]), self.evaluate(var1))

if __name__ == "__main__":
  test.main()
