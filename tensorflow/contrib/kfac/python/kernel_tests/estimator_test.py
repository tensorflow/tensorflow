# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.contrib.kfac.estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.kfac.python.ops import estimator
from tensorflow.contrib.kfac.python.ops import layer_collection as lc
from tensorflow.contrib.kfac.python.ops import utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import training_util

_ALL_ESTIMATION_MODES = ["gradients", "empirical", "curvature_prop", "exact"]


class EstimatorTest(test.TestCase):

  def setUp(self):
    self._graph = ops.Graph()
    with self._graph.as_default():
      self.layer_collection = lc.LayerCollection()

      self.inputs = random_ops.random_normal((2, 2), dtype=dtypes.float32)
      self.weights = variable_scope.get_variable(
          "w", shape=(2, 2), dtype=dtypes.float32)
      self.bias = variable_scope.get_variable(
          "b", initializer=init_ops.zeros_initializer(), shape=(2, 1))
      self.output = math_ops.matmul(self.inputs, self.weights) + self.bias

      # Only register the weights.
      self.layer_collection.register_fully_connected(
          params=(self.weights,), inputs=self.inputs, outputs=self.output)

      self.outputs = math_ops.tanh(self.output)
      self.targets = array_ops.zeros_like(self.outputs)
      self.layer_collection.register_categorical_predictive_distribution(
          logits=self.outputs, targets=self.targets)

  def testEstimatorInitManualRegistration(self):
    with self._graph.as_default():
      # We should be able to build an estimator for only the registered vars.
      estimator.FisherEstimatorRoundRobin(
          variables=[self.weights],
          cov_ema_decay=0.1,
          damping=0.2,
          layer_collection=self.layer_collection
      )

      # Check that we throw an error if we try to build an estimator for vars
      # that were not manually registered.
      with self.assertRaises(ValueError):
        est = estimator.FisherEstimatorRoundRobin(
            variables=[self.weights, self.bias],
            cov_ema_decay=0.1,
            damping=0.2,
            layer_collection=self.layer_collection
        )
        est.make_vars_and_create_op_thunks()

      # Check that we throw an error if we don't include registered variables,
      # i.e. self.weights
      with self.assertRaises(ValueError):
        est = estimator.FisherEstimatorRoundRobin(
            variables=[],
            cov_ema_decay=0.1,
            damping=0.2,
            layer_collection=self.layer_collection)
        est.make_vars_and_create_op_thunks()

  @test.mock.patch.object(utils.SubGraph, "variable_uses", return_value=42)
  def testVariableWrongNumberOfUses(self, mock_uses):
    with self.assertRaises(ValueError):
      est = estimator.FisherEstimatorRoundRobin(
          variables=[self.weights],
          cov_ema_decay=0.1,
          damping=0.2,
          layer_collection=self.layer_collection)
      est.make_vars_and_create_op_thunks()

  def testInvalidEstimationMode(self):
    with self.assertRaises(ValueError):
      est = estimator.FisherEstimatorRoundRobin(
          variables=[self.weights],
          cov_ema_decay=0.1,
          damping=0.2,
          layer_collection=self.layer_collection,
          estimation_mode="not_a_real_mode")
      est.make_vars_and_create_op_thunks()

  def testGradientsModeBuild(self):
    with self._graph.as_default():
      est = estimator.FisherEstimatorRoundRobin(
          variables=[self.weights],
          cov_ema_decay=0.1,
          damping=0.2,
          layer_collection=self.layer_collection,
          estimation_mode="gradients")
      est.make_vars_and_create_op_thunks()

  def testEmpiricalModeBuild(self):
    with self._graph.as_default():
      est = estimator.FisherEstimatorRoundRobin(
          variables=[self.weights],
          cov_ema_decay=0.1,
          damping=0.2,
          layer_collection=self.layer_collection,
          estimation_mode="empirical")
      est.make_vars_and_create_op_thunks()

  def testCurvaturePropModeBuild(self):
    with self._graph.as_default():
      est = estimator.FisherEstimatorRoundRobin(
          variables=[self.weights],
          cov_ema_decay=0.1,
          damping=0.2,
          layer_collection=self.layer_collection,
          estimation_mode="curvature_prop")
      est.make_vars_and_create_op_thunks()

  def testExactModeBuild(self):
    with self._graph.as_default():
      est = estimator.FisherEstimatorRoundRobin(
          variables=[self.weights],
          cov_ema_decay=0.1,
          damping=0.2,
          layer_collection=self.layer_collection,
          estimation_mode="exact")
      est.make_vars_and_create_op_thunks()

  def test_cov_update_thunks(self):
    """Ensures covariance update ops run once per global_step."""
    with self._graph.as_default(), self.test_session() as sess:
      fisher_estimator = estimator.FisherEstimatorRoundRobin(
          variables=[self.weights],
          layer_collection=self.layer_collection,
          damping=0.2,
          cov_ema_decay=0.0)

      # Construct an op that executes one covariance update per step.
      global_step = training_util.get_or_create_global_step()
      (cov_variable_thunks, cov_update_op_thunks, _,
       _) = fisher_estimator.create_ops_and_vars_thunks()
      for thunk in cov_variable_thunks:
        thunk()
      cov_matrices = [
          fisher_factor.get_cov()
          for fisher_factor in self.layer_collection.get_factors()
      ]
      cov_update_op = control_flow_ops.case(
          [(math_ops.equal(global_step, i), thunk)
           for i, thunk in enumerate(cov_update_op_thunks)])
      increment_global_step = global_step.assign_add(1)

      sess.run(variables.global_variables_initializer())
      initial_cov_values = sess.run(cov_matrices)

      # Ensure there's one update per covariance matrix.
      self.assertEqual(len(cov_matrices), len(cov_update_op_thunks))

      # Test is no-op if only 1 covariance matrix.
      assert len(cov_matrices) > 1

      for i in range(len(cov_matrices)):
        # Compare new and old covariance values
        new_cov_values = sess.run(cov_matrices)
        is_cov_equal = [
            np.allclose(initial_cov_value, new_cov_value)
            for (initial_cov_value,
                 new_cov_value) in zip(initial_cov_values, new_cov_values)
        ]
        num_cov_equal = sum(is_cov_equal)

        # Ensure exactly one covariance matrix changes per step.
        self.assertEqual(num_cov_equal, len(cov_matrices) - i)

        # Run all covariance update ops.
        sess.run(cov_update_op)
        sess.run(increment_global_step)

  def test_round_robin_placement(self):
    """Check if the ops and variables are placed on devices correctly."""
    with self._graph.as_default():
      fisher_estimator = estimator.FisherEstimatorRoundRobin(
          variables=[self.weights],
          layer_collection=self.layer_collection,
          damping=0.2,
          cov_ema_decay=0.0,
          cov_devices=["/cpu:{}".format(i) for i in range(2)],
          inv_devices=["/cpu:{}".format(i) for i in range(2)])

      # Construct an op that executes one covariance update per step.
      (cov_update_thunks,
       inv_update_thunks) = fisher_estimator.make_vars_and_create_op_thunks(
           scope="test")
      cov_update_ops = tuple(thunk() for thunk in cov_update_thunks)
      inv_update_ops = tuple(thunk() for thunk in inv_update_thunks)
      self.assertEqual(cov_update_ops[0].device, "/device:CPU:0")
      self.assertEqual(cov_update_ops[1].device, "/device:CPU:1")
      self.assertEqual(inv_update_ops[0].device, "/device:CPU:0")
      self.assertEqual(inv_update_ops[1].device, "/device:CPU:1")
      cov_matrices = [
          fisher_factor.get_cov()
          for fisher_factor in self.layer_collection.get_factors()
      ]
      inv_matrices = [
          matrix
          for fisher_factor in self.layer_collection.get_factors()
          for matrix in fisher_factor._matpower_by_exp_and_damping.values()
      ]
      self.assertEqual(cov_matrices[0].device, "/device:CPU:0")
      self.assertEqual(cov_matrices[1].device, "/device:CPU:1")
      # Inverse matrices need to be explicitly placed.
      self.assertEqual(inv_matrices[0].device, "")
      self.assertEqual(inv_matrices[1].device, "")

  def test_inv_update_thunks(self):
    """Ensures inverse update ops run once per global_step."""
    with self._graph.as_default(), self.test_session() as sess:
      fisher_estimator = estimator.FisherEstimatorRoundRobin(
          variables=[self.weights],
          layer_collection=self.layer_collection,
          damping=0.2,
          cov_ema_decay=0.0)

      # Construct op that updates one inverse per global step.
      global_step = training_util.get_or_create_global_step()
      (cov_variable_thunks, _, inv_variable_thunks,
       inv_update_op_thunks) = fisher_estimator.create_ops_and_vars_thunks()
      for thunk in cov_variable_thunks:
        thunk()
      for thunk in inv_variable_thunks:
        thunk()
      inv_matrices = [
          matrix
          for fisher_factor in self.layer_collection.get_factors()
          for matrix in fisher_factor._matpower_by_exp_and_damping.values()
      ]
      inv_update_op = control_flow_ops.case(
          [(math_ops.equal(global_step, i), thunk)
           for i, thunk in enumerate(inv_update_op_thunks)])
      increment_global_step = global_step.assign_add(1)

      sess.run(variables.global_variables_initializer())
      initial_inv_values = sess.run(inv_matrices)

      # Ensure there's one update per inverse matrix. This is true as long as
      # there's no fan-in/fan-out or parameter re-use.
      self.assertEqual(len(inv_matrices), len(inv_update_op_thunks))

      # Test is no-op if only 1 invariance matrix.
      assert len(inv_matrices) > 1

      # Assign each covariance matrix a value other than the identity. This
      # ensures that the inverse matrices are updated to something different as
      # well.
      cov_matrices = [
          fisher_factor.get_cov()
          for fisher_factor in self.layer_collection.get_factors()
      ]
      sess.run([
          cov_matrix.assign(2 * linalg_ops.eye(int(cov_matrix.shape[0])))
          for cov_matrix in cov_matrices
      ])

      for i in range(len(inv_matrices)):
        # Compare new and old inverse values
        new_inv_values = sess.run(inv_matrices)
        is_inv_equal = [
            np.allclose(initial_inv_value, new_inv_value)
            for (initial_inv_value,
                 new_inv_value) in zip(initial_inv_values, new_inv_values)
        ]
        num_inv_equal = sum(is_inv_equal)

        # Ensure exactly one inverse matrix changes per step.
        self.assertEqual(num_inv_equal, len(inv_matrices) - i)

        # Run all inverse update ops.
        sess.run(inv_update_op)
        sess.run(increment_global_step)


if __name__ == "__main__":
  test.main()
