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
"""Tests for PowerSign optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.opt.python.training import power_sign_optimizer
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def np_power_sign_update(var, grad, m, decay=0.9, lr=1e-4):
  m_t = (grad - m) * decay + m
  same_sign = np.sign(m_t) * np.sign(grad)
  delta = np.exp(same_sign) * grad
  var_t = var - lr * delta
  return var_t, m_t


class PowerSignOptimizerTest(test.TestCase):

  def doTestBasic(self, use_resource=False):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.test_session():
        # Initialize variables for numpy implementation.
        m0, m1 = 0, 0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        if use_resource:
          var0 = resource_variable_ops.ResourceVariable(var0_np)
          var1 = resource_variable_ops.ResourceVariable(var1_np)
        else:
          var0 = variables.Variable(var0_np)
          var1 = variables.Variable(var1_np)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)

        opt = power_sign_optimizer.PowerSignOptimizer()

        update = opt.apply_gradients(tuple(zip([grads0, grads1], [var0, var1])))
        variables.global_variables_initializer().run()

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 4.0], var1.eval())

        # Run 3 steps of Nadam
        for t in range(1, 4):
          update.run()
          var0_np, m0 = np_power_sign_update(var0_np, grads0_np, m0)
          var1_np, m1 = np_power_sign_update(var1_np, grads1_np, m1)

        var0_tf = var0.eval() 
        var1_tf = var1.eval() 
        
        # Validate updated params
        self.assertAllCloseAccordingToType(var0_tf, var0_np)
        self.assertAllCloseAccordingToType(var1_tf, var1_np)

  def doTestSparse(self, use_resource):
    with self.test_session():
      # Initialize variables for numpy implementation.
      m0, m1 = 0, 0
      var0_np = np.array([1.0, 2.0], dtype=np.float64)
      grads0_np = np.array([0.1, 0.1], dtype=np.float64)
      var1_np = np.array([3.0, 4.0], dtype=np.float64)
      grads1_np = np.array([0.01, 0.01], dtype=np.float64)

      if use_resource:
        var0 = resource_variable_ops.ResourceVariable(var0_np)
        var1 = resource_variable_ops.ResourceVariable(var1_np)
      else:
        var0 = variables.Variable(var0_np)
        var1 = variables.Variable(var1_np)

      grads0_np_indices = np.array([0, 1], dtype=np.int32)
      grads0 = ops.IndexedSlices(
              constant_op.constant(grads0_np),
                            constant_op.constant(grads0_np_indices),
                            constant_op.constant([2]))
      grads1_np_indices = np.array([0, 1], dtype=np.int32)
      grads1 = ops.IndexedSlices(
              constant_op.constant(grads1_np),
                            constant_op.constant(grads1_np_indices),
                            constant_op.constant([2]))
      opt = power_sign_optimizer.PowerSignOptimizer()
      update = opt.apply_gradients(tuple(zip([grads0, grads1], [var0, var1])))
      variables.global_variables_initializer().run()

      # Fetch params to validate initial values
      self.assertAllClose([1.0, 2.0], var0.eval())
      self.assertAllClose([3.0, 4.0], var1.eval())

      # Run 3 steps of Nadam
      for t in range(1, 4):
        update.run()
        var0_np, m0 = np_power_sign_update(var0_np, grads0_np, m0)
        var1_np, m1 = np_power_sign_update(var1_np, grads1_np, m1)

      var0_tf = var0.eval() 
      var1_tf = var1.eval() 
      
      # Validate updated params
      self.assertAllCloseAccordingToType(var0_tf, var0_np)
      self.assertAllCloseAccordingToType(var1_tf, var1_np)


  def testBasic(self):
    self.doTestBasic(use_resource=False)

  def testResourceBasic(self):
    self.doTestBasic(use_resource=True)

  def testSparse(self):
    self.doTestSparse(use_resource=False)

  def testResourceSparse(self):
    self.doTestSparse(use_resource=True)


if __name__ == "__main__":

  test.main()

