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

"""Functional test for GradientDescent."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import resources


class GradientDescentOptimizerTest(tf.test.TestCase):

  def testBasic(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        var0 = tf.Variable([1.0, 2.0], dtype=dtype)
        var1 = tf.Variable([3.0, 4.0], dtype=dtype)
        grads0 = tf.constant([0.1, 0.1], dtype=dtype)
        grads1 = tf.constant([0.01, 0.01], dtype=dtype)
        sgd_op = tf.train.GradientDescentOptimizer(3.0).apply_gradients(zip(
            [grads0, grads1], [var0, var1]))
        tf.global_variables_initializer().run()
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

  def testBasicResourceVariable(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        var0 = resource_variable_ops.ResourceVariable(
            [1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable(
            [3.0, 4.0], dtype=dtype)
        grads0 = tf.constant([0.1, 0.1], dtype=dtype)
        grads1 = tf.constant([0.01, 0.01], dtype=dtype)
        sgd_op = tf.train.GradientDescentOptimizer(3.0).apply_gradients(zip(
            [grads0, grads1], [var0, var1]))
        # TODO(apassos) calling initialize_resources on all resources here
        # doesn't work because the sessions and graph are reused across unit
        # tests and this would mean trying to reinitialize variables. Figure out
        # a long-term solution for this.
        resources.initialize_resources([var0, var1]).run()
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

  def testMinimizeResourceVariable(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        var0 = resource_variable_ops.ResourceVariable(
            [[1.0, 2.0]], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable(
            [3.0], dtype=dtype)
        x = tf.constant([[4.0], [5.0]], dtype=dtype)
        pred = tf.matmul(var0, x) + var1
        loss = pred*pred
        sgd_op = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
        # TODO(apassos) calling initialize_resources on all resources here
        # doesn't work because the sessions and graph are reused across unit
        # tests and this would mean trying to reinitialize variables. Figure out
        # a long-term solution for this.
        resources.initialize_resources([var0, var1]).run()
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([[1.0, 2.0]], var0.eval())
        self.assertAllCloseAccordingToType([3.0], var1.eval())
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        np_pred = 1.0 * 4.0 + 2.0 * 5.0 + 3.0
        np_grad = 2 * np_pred
        self.assertAllCloseAccordingToType(
            [[1.0 - np_grad * 4.0, 2.0 - np_grad * 5.0]], var0.eval())
        self.assertAllCloseAccordingToType(
            [3.0 - np_grad], var1.eval())

  def testMinimizeSparseResourceVariable(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        var0 = resource_variable_ops.ResourceVariable(
            [[1.0, 2.0]], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable(
            [3.0], dtype=dtype)
        x = tf.constant([[4.0], [5.0]], dtype=dtype)
        pred = tf.matmul(tf.nn.embedding_lookup([var0], [0]), x)
        pred = tf.matmul(var0, x) + var1
        loss = pred*pred
        sgd_op = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
        # TODO(apassos) calling initialize_resources on all resources here
        # doesn't work because the sessions and graph are reused across unit
        # tests and this would mean trying to reinitialize variables. Figure out
        # a long-term solution for this.
        resources.initialize_resources([var0, var1]).run()
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([[1.0, 2.0]], var0.eval())
        self.assertAllCloseAccordingToType([3.0], var1.eval())
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        np_pred = 1.0 * 4.0 + 2.0 * 5.0 + 3.0
        np_grad = 2 * np_pred
        self.assertAllCloseAccordingToType(
            [[1.0 - np_grad * 4.0, 2.0 - np_grad * 5.0]], var0.eval())
        self.assertAllCloseAccordingToType(
            [3.0 - np_grad], var1.eval())

  def testTensorLearningRate(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        var0 = tf.Variable([1.0, 2.0], dtype=dtype)
        var1 = tf.Variable([3.0, 4.0], dtype=dtype)
        grads0 = tf.constant([0.1, 0.1], dtype=dtype)
        grads1 = tf.constant([0.01, 0.01], dtype=dtype)
        lrate = tf.constant(3.0)
        sgd_op = tf.train.GradientDescentOptimizer(lrate).apply_gradients(zip(
            [grads0, grads1], [var0, var1]))
        tf.global_variables_initializer().run()
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
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        opt = tf.train.GradientDescentOptimizer(3.0)
        values = [1.0, 3.0]
        vars_ = [tf.Variable([v], dtype=dtype) for v in values]
        grads_and_vars = opt.compute_gradients(vars_[0] + vars_[1], vars_)
        tf.global_variables_initializer().run()
        for grad, _ in grads_and_vars:
          self.assertAllCloseAccordingToType([1.0], grad.eval())

  def testWithGlobalStep(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        global_step = tf.Variable(0, trainable=False)
        var0 = tf.Variable([1.0, 2.0], dtype=dtype)
        var1 = tf.Variable([3.0, 4.0], dtype=dtype)
        grads0 = tf.constant([0.1, 0.1], dtype=dtype)
        grads1 = tf.constant([0.01, 0.01], dtype=dtype)
        sgd_op = tf.train.GradientDescentOptimizer(3.0).apply_gradients(
            zip([grads0, grads1], [var0, var1]),
            global_step=global_step)
        tf.global_variables_initializer().run()
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

  def testSparseBasic(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session():
        var0 = tf.Variable([[1.0], [2.0]], dtype=dtype)
        var1 = tf.Variable([[3.0], [4.0]], dtype=dtype)
        grads0 = tf.IndexedSlices(
            tf.constant([0.1], shape=[1, 1], dtype=dtype),
            tf.constant([0]),
            tf.constant([2, 1]))
        grads1 = tf.IndexedSlices(
            tf.constant([0.01], shape=[1, 1], dtype=dtype),
            tf.constant([1]),
            tf.constant([2, 1]))
        sgd_op = tf.train.GradientDescentOptimizer(3.0).apply_gradients(
            zip([grads0, grads1], [var0, var1]))
        tf.global_variables_initializer().run()
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([[1.0], [2.0]], var0.eval())
        self.assertAllCloseAccordingToType([[3.0], [4.0]], var1.eval())
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        self.assertAllCloseAccordingToType(
            [[1.0 - 3.0 * 0.1], [2.0]], var0.eval())
        self.assertAllCloseAccordingToType(
            [[3.0], [4.0 - 3.0 * 0.01]], var1.eval())


if __name__ == "__main__":
  tf.test.main()
