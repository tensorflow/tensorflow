# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for moving_average_optimizer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import tempfile

import six
from tensorflow.contrib.opt.python.training import moving_average_optimizer
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import saver


class MovingAverageOptimizerTest(test.TestCase):

  def testRun(self):
    self._helpTestRun(use_resource=False)

  def testRunUseResource(self):
    # Test that MovingAverageOptimizer works with resource variables.
    self._helpTestRun(use_resource=True)

  def testRunUsePartitionedVars(self):
    # Test that MovingAverageOptimizer works with partitioned variables.
    self._helpTestRun(use_partitioned_vars=True)

  def testRunUseResourcePartitionedVars(self):
    # Test that MovingAverageOptimizer works with resource and partitioned
    # variables.
    self._helpTestRun(use_partitioned_vars=True, use_resource=True)

  def _helpTestRun(self, use_resource=False, use_partitioned_vars=False):
    # Partitioned variables are represented as a "collection" of partitions.
    # To simplify the test and reuse as much code as possible we employ
    # following test strategy for partitioned variables.
    #
    # In the case of non-partitioned variables test runs on variables with
    # shape [2].
    #
    # In the case of partitioned variables we use shape [4] with two partitions,
    # thus each partition has shape [2].
    # For partitioned variables the test is run twice (for loop over
    # variable_part_names), first time on the first partition of each variable,
    # second time on the second partition of each variable.
    variable_part_names = ['part_0', 'part_1'] if use_partitioned_vars else ['']
    for sequential_update in [True, False]:
      for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
        for var_part_name in variable_part_names:
          with self.session(graph=ops.Graph()) as sess:
            orig_val0 = [1.0, 2.0]
            orig_val1 = [3.0, 4.0]
            grads0 = [0.1, 0.1]
            grads1 = [0.01, 0.01]
            if use_partitioned_vars:
              # Use partitioned variables.
              # Create partitioned and duplicate each value used as initial
              # value of variables.
              partitioner = partitioned_variables.fixed_size_partitioner(
                  num_shards=2)
              orig_val0 = orig_val0 * 2
              orig_val1 = orig_val1 * 2
              grads0 = grads0 * 2
              grads1 = grads1 * 2
            else:
              # Regular (non-partitioned) variables.
              partitioner = None
            var0 = variable_scope.get_variable(
                'var0',
                initializer=constant_op.constant(orig_val0, dtype=dtype),
                use_resource=use_resource,
                partitioner=partitioner)
            var1 = variable_scope.get_variable(
                'var1',
                initializer=constant_op.constant(orig_val1, dtype=dtype),
                use_resource=use_resource,
                partitioner=partitioner)
            # Make a fake loss, such that gradient(loss, var0) == grads0
            # and gradient(loss, var1) == grads1
            grads0 = constant_op.constant(grads0, dtype=dtype)
            grads1 = constant_op.constant(grads1, dtype=dtype)
            loss = (math_ops.reduce_sum(grads0 * var0)
                    + math_ops.reduce_sum(grads1 * var1))

            opt = moving_average_optimizer.MovingAverageOptimizer(
                gradient_descent.GradientDescentOptimizer(learning_rate=2.0),
                average_decay=0.5,
                sequential_update=sequential_update)
            save_dir = tempfile.mkdtemp(
                prefix=os.path.join(self.get_temp_dir(), 'run_1'))
            save_path = os.path.join(save_dir, 'model')

            update = opt.minimize(loss)

            # Get variables and their EMAs. In case of partitioned variables
            # get proper part of each variable.
            def _get_variable(var_name, part_name, ema):
              """Returns variable of it's moving average by name."""
              matches = [
                  v for v in variables.global_variables()
                  if ((var_name in v.op.name)
                      and (part_name in v.op.name)
                      and (('ExponentialMovingAverage' in v.op.name) == ema))
              ]
              self.assertEqual(len(matches), 1)
              return matches[0]
            var0 = _get_variable('var0', var_part_name, ema=False)
            var1 = _get_variable('var1', var_part_name, ema=False)
            ema_var0 = _get_variable('var0', var_part_name, ema=True)
            ema_var1 = _get_variable('var1', var_part_name, ema=True)

            perturb = control_flow_ops.group([
                state_ops.assign_add(var0, [1.0, 1.0]),
                state_ops.assign_add(var1, [2.0, 2.0]),
                state_ops.assign_add(ema_var0, [3.0, 3.0]),
                state_ops.assign_add(ema_var1, [4.0, 4.0])
            ])

            # Test that saver with missing ema variables will fail.
            with self.assertRaisesRegexp(ValueError, r'Variable to swap'):
              opt.swapping_saver(var_list=[var0])

            train_saver = opt.swapping_saver()
            train_saver_subset = opt.swapping_saver(var_list=[var0, ema_var0])
            inference_saver = saver.Saver()
            variables.global_variables_initializer().run()
            # Step 1.
            update.run()
            self.assertAllCloseAccordingToType([0.8, 1.8], var0.eval())
            self.assertAllCloseAccordingToType([2.98, 3.98], var1.eval())
            if sequential_update:
              self.assertAllCloseAccordingToType([0.9, 1.9], ema_var0.eval())
              self.assertAllCloseAccordingToType([2.99, 3.99], ema_var1.eval())
            # Test that the swapping saver save/restore operation is identity.
            train_saver.save(sess, save_path)
            train_saver.restore(sess, save_path)
            self.assertAllCloseAccordingToType([0.8, 1.8], var0.eval())
            self.assertAllCloseAccordingToType([2.98, 3.98], var1.eval())
            if sequential_update:
              self.assertAllCloseAccordingToType([0.9, 1.9], ema_var0.eval())
              self.assertAllCloseAccordingToType([2.99, 3.99], ema_var1.eval())
            # Test that the subset saver saves the EMA variable as well.
            if sequential_update:
              subset_save_path = save_path + '_subset'
              train_saver_subset.save(sess, subset_save_path)
              perturb.run()
              self.assertAllCloseAccordingToType([1.8, 2.8], var0.eval())
              self.assertAllCloseAccordingToType([3.9, 4.9], ema_var0.eval())
              self.assertAllCloseAccordingToType([4.98, 5.98], var1.eval())
              self.assertAllCloseAccordingToType([6.99, 7.99], ema_var1.eval())
              # Restoring should only restore var0 and ema_var0.
              train_saver_subset.restore(sess, subset_save_path)
              self.assertAllCloseAccordingToType([0.8, 1.8], var0.eval())
              self.assertAllCloseAccordingToType([0.9, 1.9], ema_var0.eval())
              self.assertAllCloseAccordingToType([4.98, 5.98], var1.eval())
              self.assertAllCloseAccordingToType([6.99, 7.99], ema_var1.eval())
              # Restore back to previous state.
              train_saver.restore(sess, save_path)

            # If updates are parallel,
            # this is not always true after the 1st step.
            if sequential_update:
              # Test that the normal saver will have the averaged variables.
              # We test that the average values are between the original value
              # and the most recent variable values (since they are an average
              # of the two).
              val0 = var0.eval()
              val1 = var1.eval()
              train_saver.save(sess, save_path)
              inference_saver.restore(sess, save_path)
              avg_val0 = var0.eval()
              avg_val1 = var1.eval()
              for i in six.moves.range(len(val0)):
                self.assertLess(val0[i], avg_val0[i])
                self.assertLess(avg_val0[i], orig_val0[i])
                self.assertLess(val1[i], avg_val1[i])
                self.assertLess(avg_val1[i], orig_val1[i])
              train_saver.restore(sess, save_path)
            # Step 2.
            update.run()
            # Test that the normal saver will have the averaged variables.
            # We test that the average values are between the original value and
            # the most recent variable values (since they are an average of the
            # two).
            val0 = var0.eval()
            val1 = var1.eval()
            self.assertAllCloseAccordingToType([0.6, 1.6], val0)
            self.assertAllCloseAccordingToType([2.96, 3.96], val1)
            train_saver.save(sess, save_path)
            inference_saver.restore(sess, save_path)
            avg_val0 = var0.eval()
            avg_val1 = var1.eval()
            for i in six.moves.range(len(val0)):
              self.assertLess(val0[i], avg_val0[i])
              self.assertLess(avg_val0[i], orig_val0[i])
              self.assertLess(val1[i], avg_val1[i])
              self.assertLess(avg_val1[i], orig_val1[i])

  def testFailWhenSaverCreatedBeforeInitialized(self):
    with self.cached_session():
      var = variables.Variable([1.0], name='var', dtype=dtypes.float32)
      opt = moving_average_optimizer.MovingAverageOptimizer(
          gradient_descent.GradientDescentOptimizer(learning_rate=2.0))
      # We didn't call apply_gradients yet.
      # This will raise an exception.
      with self.assertRaises(RuntimeError):
        _ = opt.swapping_saver([var])

  def testCorrectOverride(self):

    class WrapperOptimizer(gradient_descent.GradientDescentOptimizer):

      def compute_gradients(self, *args, **kwargs):
        self.compute_gradients_called = True
        return super(WrapperOptimizer, self).compute_gradients(
            *args, **kwargs)

      def apply_gradients(self, *args, **kwargs):
        self.apply_gradients_called = True
        return super(WrapperOptimizer, self).apply_gradients(*args, **kwargs)

    with self.cached_session() as sess:
      var = variables.Variable([1.2], name='var', dtype=dtypes.float32)
      loss = var ** 2
      wrapper_opt = WrapperOptimizer(learning_rate=2.0)
      opt = moving_average_optimizer.MovingAverageOptimizer(wrapper_opt)
      train_op = opt.minimize(loss)

      # Check that both methods are called on the underlying optimizer.
      self.assertTrue(wrapper_opt.compute_gradients_called)
      self.assertTrue(wrapper_opt.apply_gradients_called)

      # Run train_op once, and verify that we've updated the variable.
      variables.global_variables_initializer().run()
      sess.run(train_op)
      var_value = sess.run(var)
      # Started at 1.2, gradient is 2*1.2=2.4, lr=2, so should now be -3.6.
      self.assertNear(-3.6, var_value, 1e-6)


if __name__ == '__main__':
  test.main()
