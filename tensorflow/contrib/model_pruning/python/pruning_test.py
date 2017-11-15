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
"""Tests for the key functions in pruning library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.model_pruning.python import pruning
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import training_util


class PruningHParamsTest(test.TestCase):
  PARAM_LIST = [
      "name=test", "threshold_decay=0.9", "pruning_frequency=10",
      "do_not_prune=[conv1,conv2]", "sparsity_function_end_step=100",
      "target_sparsity=0.9"
  ]
  TEST_HPARAMS = ",".join(PARAM_LIST)

  def setUp(self):
    super(PruningHParamsTest, self).setUp()
    # Add global step variable to the graph
    self.global_step = training_util.get_or_create_global_step()
    # Add sparsity
    self.sparsity = variables.Variable(0.5, name="sparsity")
    # Parse hparams
    self.pruning_hparams = pruning.get_pruning_hparams().parse(
        self.TEST_HPARAMS)

  def testInit(self):
    p = pruning.Pruning(self.pruning_hparams)
    self.assertEqual(p._spec.name, "test")
    self.assertAlmostEqual(p._spec.threshold_decay, 0.9)
    self.assertEqual(p._spec.pruning_frequency, 10)
    self.assertAllEqual(p._spec.do_not_prune, ["conv1", "conv2"])
    self.assertEqual(p._spec.sparsity_function_end_step, 100)
    self.assertAlmostEqual(p._spec.target_sparsity, 0.9)

  def testInitWithExternalSparsity(self):
    with self.test_session():
      p = pruning.Pruning(spec=self.pruning_hparams, sparsity=self.sparsity)
      variables.global_variables_initializer().run()
      sparsity = p._sparsity.eval()
      self.assertAlmostEqual(sparsity, 0.5)

  def testInitWithVariableReuse(self):
    with self.test_session():
      p = pruning.Pruning(spec=self.pruning_hparams, sparsity=self.sparsity)
      p_copy = pruning.Pruning(
          spec=self.pruning_hparams, sparsity=self.sparsity)
      variables.global_variables_initializer().run()
      sparsity = p._sparsity.eval()
      self.assertAlmostEqual(sparsity, 0.5)
      self.assertEqual(p._sparsity.eval(), p_copy._sparsity.eval())


class PruningTest(test.TestCase):

  def setUp(self):
    super(PruningTest, self).setUp()
    self.global_step = training_util.get_or_create_global_step()

  def testCreateMask2D(self):
    width = 10
    height = 20
    with self.test_session():
      weights = variables.Variable(
          random_ops.random_normal([width, height], stddev=1), name="weights")
      masked_weights = pruning.apply_mask(weights,
                                          variable_scope.get_variable_scope())
      variables.global_variables_initializer().run()
      weights_val = weights.eval()
      masked_weights_val = masked_weights.eval()
      self.assertAllEqual(weights_val, masked_weights_val)

  def testUpdateSingleMask(self):
    with self.test_session() as session:
      weights = variables.Variable(
          math_ops.linspace(1.0, 100.0, 100), name="weights")
      masked_weights = pruning.apply_mask(weights)
      sparsity = variables.Variable(0.5, name="sparsity")
      p = pruning.Pruning(sparsity=sparsity)
      p._spec.threshold_decay = 0.0
      mask_update_op = p.mask_update_op()
      variables.global_variables_initializer().run()
      masked_weights_val = masked_weights.eval()
      self.assertAllEqual(np.count_nonzero(masked_weights_val), 100)
      session.run(mask_update_op)
      masked_weights_val = masked_weights.eval()
      self.assertAllEqual(np.count_nonzero(masked_weights_val), 51)

  def testPartitionedVariableMasking(self):
    partitioner = partitioned_variables.variable_axis_size_partitioner(40)
    with self.test_session() as session:
      with variable_scope.variable_scope("", partitioner=partitioner):
        sparsity = variables.Variable(0.5, name="Sparsity")
        weights = variable_scope.get_variable(
            "weights", initializer=math_ops.linspace(1.0, 100.0, 100))
        masked_weights = pruning.apply_mask(
            weights, scope=variable_scope.get_variable_scope())
      p = pruning.Pruning(sparsity=sparsity, partitioner=partitioner)
      p._spec.threshold_decay = 0.0
      mask_update_op = p.mask_update_op()
      variables.global_variables_initializer().run()
      masked_weights_val = masked_weights.eval()
      session.run(mask_update_op)
      masked_weights_val = masked_weights.eval()
      self.assertAllEqual(np.count_nonzero(masked_weights_val), 51)

  def testConditionalMaskUpdate(self):
    param_list = [
        "pruning_frequency=2", "begin_pruning_step=1", "end_pruning_step=6"
    ]
    test_spec = ",".join(param_list)
    pruning_hparams = pruning.get_pruning_hparams().parse(test_spec)
    weights = variables.Variable(
        math_ops.linspace(1.0, 100.0, 100), name="weights")
    masked_weights = pruning.apply_mask(weights)
    sparsity = variables.Variable(0.00, name="sparsity")
    # Set up pruning
    p = pruning.Pruning(pruning_hparams, sparsity=sparsity)
    p._spec.threshold_decay = 0.0
    mask_update_op = p.conditional_mask_update_op()
    sparsity_val = math_ops.linspace(0.0, 0.9, 10)
    increment_global_step = state_ops.assign_add(self.global_step, 1)
    non_zero_count = []
    with self.test_session() as session:
      variables.global_variables_initializer().run()
      for i in range(10):
        session.run(state_ops.assign(sparsity, sparsity_val[i]))
        session.run(mask_update_op)
        session.run(increment_global_step)
        non_zero_count.append(np.count_nonzero(masked_weights.eval()))
    # Weights pruned at steps 0,2,4,and,6
    expected_non_zero_count = [100, 100, 80, 80, 60, 60, 40, 40, 40, 40]
    self.assertAllEqual(expected_non_zero_count, non_zero_count)


if __name__ == "__main__":
  test.main()
