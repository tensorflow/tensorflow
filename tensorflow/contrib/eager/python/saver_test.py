"""Tests for eager mode Saver."""
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.contrib.eager.python import saver as _saver
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import adam
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import momentum
from tensorflow.python.training import rmsprop


class SaverTest(test.TestCase):

  def _dev(self):
    return '/device:GPU:0' if context.num_gpus() else '/device:CPU:0'

  def testBasics(self):
    with ops.device(self._dev()):
      v1 = resource_variable_ops.ResourceVariable(1.0, name='v1')
      def model():
        return array_ops.constant(2.0) * v1

      ckpt_prefix = os.path.join(test.get_temp_dir(), 'ckpt')

      _ = model()
      saver = _saver.Saver([v1])
      saver.save(ckpt_prefix)
      v1.assign(2.0)
      self.assertEqual(v1.read_value().numpy(), 2.0)

      saver.restore(ckpt_prefix)
      self.assertEqual(v1.read_value().numpy(), 1.0)

  def testSameNameNoClobbering(self):
    with ops.device(self._dev()):
      v1 = resource_variable_ops.ResourceVariable(1.0, name='v1')
      v2 = resource_variable_ops.ResourceVariable(2.0, name='v1')
      saver = _saver.Saver([v1, v2])
      ckpt_prefix = os.path.join(test.get_temp_dir(), 'ckpt')
      with self.assertRaisesRegexp(ValueError, 'v1'):
        saver.save(ckpt_prefix)

  def testSameObjectOK(self):
    with ops.device(self._dev()):
      v1 = resource_variable_ops.ResourceVariable(1.0, name='v1')
      # While different objects with the same shared_name are not good, passing
      # in the same object multiple times is fine.
      saver = _saver.Saver([v1, v1])
      ckpt_prefix = os.path.join(test.get_temp_dir(), 'ckpt')
      saver.save(ckpt_prefix)

  def testSaveByDict(self):
    with ops.device(self._dev()):
      v1 = resource_variable_ops.ResourceVariable(1.0, name='v1')
      v2 = resource_variable_ops.ResourceVariable(1.0, name='v2')
      def model():
        return array_ops.constant(2.0) * v1 * v2

      ckpt_prefix = os.path.join(test.get_temp_dir(), 'ckpt')

      # Save the variables under different names.
      _ = model()
      saver = _saver.Saver({'ckpt/v1': v1, 'ckpt/v2': v2})
      saver.save(ckpt_prefix)
      v1.assign(2.0)
      v2.assign(2.0)
      self.assertEqual(v1.read_value().numpy(), 2.0)
      self.assertEqual(v2.read_value().numpy(), 2.0)
      # Can still restore it.
      saver.restore(ckpt_prefix)
      self.assertEqual(v1.read_value().numpy(), 1.0)
      # However, cannot restore it with default name.
      with self.assertRaisesOpError('not found in checkpoint'):
        saver = _saver.Saver([v1, v2]).restore(ckpt_prefix)

      # Can specify which variable in ckpt to restore to which variable.
      def map_func(x):
        return {'v3': 'ckpt/v1', 'v4': 'ckpt/v2'}.get(x, x)
      with _saver.restore_variables_on_create(ckpt_prefix, map_func):
        v3 = resource_variable_ops.ResourceVariable(2.0, name='v3')
        v4 = resource_variable_ops.ResourceVariable(2.0, name='v4')
      self.assertEqual(v3.read_value().numpy(), 1.0)
      self.assertEqual(v4.read_value().numpy(), 1.0)

  def testRestoreOnCreate(self):
    with ops.device(self._dev()):
      def model(init_val):
        v1 = resource_variable_ops.ResourceVariable(init_val, name='v1')
        return array_ops.constant(1.0) * v1, v1

      ckpt_prefix = os.path.join(test.get_temp_dir(), 'ckpt')
      _, v1 = model(1.0)
      saver = _saver.Saver([v1])
      saver.save(ckpt_prefix)

      saver = _saver.Saver([v1])
      with _saver.restore_variables_on_create(ckpt_prefix):
        # Value is from checkpoint, but not from argument.
        ret, _ = model(2.0)
        self.assertEqual(ret.numpy(), 1.0)

  def testRestoreNotFound(self):
    with ops.device(self._dev()):
      def model(v):
        return array_ops.constant(1.0) * v

      ckpt_prefix = os.path.join(test.get_temp_dir(), 'ckpt')
      v = resource_variable_ops.ResourceVariable(1.0, name='v1')
      _ = model(v)
      saver = _saver.Saver([v])
      saver.save(ckpt_prefix)

      with self.assertRaisesRegexp(errors.NotFoundError,
                                   'v2 not found in checkpoint'):
        with _saver.restore_variables_on_create(ckpt_prefix):
          _ = model(resource_variable_ops.ResourceVariable(1.0, name='v2'))


class GetOptimizerTests(test.TestCase):

  def _optimizer_test_template(self, optimizer):
    """Checks save and restore. Returns the optimizer variables."""
    v = resource_variable_ops.ResourceVariable([[2., 3.]], name='v')
    loss_fn = lambda: v[0, 0] ** 2 + v[0, 1] ** 2
    optimizer.minimize(loss_fn)
    optimizer_variables = _saver.get_optimizer_variables(optimizer)
    saver = _saver.Saver(optimizer_variables + [v])
    checkpoint_path = saver.save(self.get_temp_dir())
    optimizer.minimize(loss_fn)
    after_first_minimize = v.numpy()
    # After we restore, the next step should be exactly the same as the one we
    # just did.
    saver.restore(checkpoint_path)
    optimizer.minimize(loss_fn)
    self.assertAllEqual(after_first_minimize, v.numpy())
    return optimizer_variables

  def testAdam(self):
    optimizer = adam.AdamOptimizer(0.1)
    self._optimizer_test_template(optimizer)

  def testGradientDescent(self):
    optimizer = gradient_descent.GradientDescentOptimizer(0.02)
    self.assertEqual(0, len(self._optimizer_test_template(optimizer)))

  def testMomentum(self):
    optimizer = momentum.MomentumOptimizer(
        learning_rate=0.03,
        momentum=0.5)
    self._optimizer_test_template(optimizer)

  def testRMSProp(self):
    optimizer = rmsprop.RMSPropOptimizer(0.01)
    self._optimizer_test_template(optimizer)

if __name__ == '__main__':
  test.main()
