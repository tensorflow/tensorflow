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
"""Functional test for DelayCompensatedGradientDescentOptimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.contrib.opt.python.training import delay_compensated_gradient_descent


class DelayCompensatedGradientDescentOptimizerTest(test.TestCase):

  def testBasic(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.test_session():
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        optimizer = (delay_compensated_gradient_descent.
                     DelayCompensatedGradientDescentOptimizer)(
                         learning_rate=3.0,
                         variance_parameter=2.0,
                         num_workers=1)
        sgd_op = optimizer.apply_gradients(
            zip([grads0, grads1], [var0, var1]), worker_index=0)
        variables.global_variables_initializer().run()
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([1.0, 2.0], var0.eval())
        self.assertAllCloseAccordingToType([3.0, 4.0], var1.eval())
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        self.assertAllCloseAccordingToType(
            [1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1], var0.eval())
        self.assertAllCloseAccordingToType(
            [3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01], var1.eval())

  def testTensorLearningRate(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.test_session():
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        lrate = constant_op.constant(3.0)
        optimizer = (delay_compensated_gradient_descent.
                     DelayCompensatedGradientDescentOptimizer)(
                         learning_rate=3.0,
                         variance_parameter=2.0,
                         num_workers=1)
        sgd_op = optimizer.apply_gradients(
            zip([grads0, grads1], [var0, var1]), worker_index=0)
        variables.global_variables_initializer().run()
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([1.0, 2.0], var0.eval())
        self.assertAllCloseAccordingToType([3.0, 4.0], var1.eval())
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        self.assertAllCloseAccordingToType(
            [1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1], var0.eval())
        self.assertAllCloseAccordingToType(
            [3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01], var1.eval())

    def testGradWrtRef(self):
      for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
        with self.test_session():
          optimizer = (delay_compensated_gradient_descent.
                       DelayCompensatedGradientDescentOptimizer)(
                           learning_rate=3.0,
                           variance_parameter=2.0,
                           num_workers=1)
          values = [1.0, 3.0]
          vars_ = [variables.Variable([v], dtype=dtype) for v in values]
          grads_and_vars = optimizer.compute_gradients(
              vars_[0] + vars_[1], vars_)
          variables.global_variables_initializer().run()
          for grad, _ in grads_and_vars:
            self.assertAllCloseAccordingToType([1.0], grad.eval())

  def testWithGlobalStep(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.test_session():
        global_step = variables.Variable(0, trainable=False)
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        optimizer = (delay_compensated_gradient_descent.
                     DelayCompensatedGradientDescentOptimizer)(
                         learning_rate=3.0,
                         variance_parameter=2.0,
                         num_workers=1)
        sgd_op = optimizer.apply_gradients(
            zip([grads0, grads1], [var0, var1]),
            global_step=global_step,
            worker_index=0)
        variables.global_variables_initializer().run()
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([1.0, 2.0], var0.eval())
        self.assertAllCloseAccordingToType([3.0, 4.0], var1.eval())
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params and global_step
        self.assertAllCloseAccordingToType(
            [1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1], var0.eval())
        self.assertAllCloseAccordingToType(
            [3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01], var1.eval())
        self.assertAllCloseAccordingToType(1, global_step.eval())


if __name__ == "__main__":
  test.main()
