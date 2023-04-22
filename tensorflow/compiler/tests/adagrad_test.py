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
"""Tests for Adagrad."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adagrad


class AdagradOptimizerTest(xla_test.XLATestCase):

  def testBasic(self):
    for dtype in self.float_types | self.complex_types:
      with self.session(), self.test_scope():
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        ada_opt = adagrad.AdagradOptimizer(3.0, initial_accumulator_value=0.1)
        ada_update = ada_opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))
        # Run 3 steps of adagrad
        for _ in range(3):
          ada_update.run()
        # Validate updated params
        self.assertAllCloseAccordingToType(
            np.array([-1.6026098728179932, -0.6026098728179932]),
            self.evaluate(var0),
            float_rtol=1e-5)
        self.assertAllCloseAccordingToType(
            np.array([2.715679168701172, 3.715679168701172]),
            self.evaluate(var1),
            float_rtol=1e-5)

  def testTensorLearningRate(self):
    for dtype in self.float_types:
      with self.session(), self.test_scope():
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        ada_opt = adagrad.AdagradOptimizer(
            constant_op.constant(3.0), initial_accumulator_value=0.1)
        ada_update = ada_opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))
        # Run 3 steps of adagrad
        for _ in range(3):
          ada_update.run()
        # Validate updated params
        self.assertAllCloseAccordingToType(
            np.array([-1.6026098728179932, -0.6026098728179932]),
            self.evaluate(var0),
            float_rtol=1e-5)
        self.assertAllCloseAccordingToType(
            np.array([2.715679168701172, 3.715679168701172]),
            self.evaluate(var1),
            float_rtol=1e-5)

  def testSharing(self):
    for dtype in self.float_types:
      with self.session(), self.test_scope():
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        ada_opt = adagrad.AdagradOptimizer(3.0)
        # Apply the optimizer twice.  Both applications will use
        # the same accums.
        ada_update1 = ada_opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        ada_update2 = ada_opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        self.assertEqual(["accumulator"], ada_opt.get_slot_names())
        slot0 = ada_opt.get_slot(var0, "accumulator")
        self.assertEqual(slot0.get_shape(), var0.get_shape())
        slot1 = ada_opt.get_slot(var1, "accumulator")
        self.assertEqual(slot1.get_shape(), var1.get_shape())
        self.evaluate(variables.global_variables_initializer())

        # Fetch params to validate initial values.
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))
        # Mix the first and the second adagrad for 3 steps.
        ada_update1.run()
        ada_update2.run()
        ada_update1.run()
        # Validate updated params (the same as with only 1 Adagrad).
        self.assertAllCloseAccordingToType(
            np.array([-1.6026098728179932, -0.6026098728179932]),
            self.evaluate(var0),
            float_rtol=1e-5)
        self.assertAllCloseAccordingToType(
            np.array([2.715679168701172, 3.715679168701172]),
            self.evaluate(var1),
            float_rtol=1e-5)


if __name__ == "__main__":
  test.main()
