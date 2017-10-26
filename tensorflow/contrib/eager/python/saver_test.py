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
from tensorflow.python.eager import graph_callable
from tensorflow.python.eager import test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope


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
    with context.eager_mode(), ops.device(self._dev()):
      # Note that this test purposefully uses Graphs rather than
      # IsolateTest. Users are more likely to accidentally create the same
      # variable name this way.
      first_graph = ops.Graph()
      with first_graph.as_default():
        v1_first_graph = resource_variable_ops.ResourceVariable(1.0, name='v1')
      with ops.Graph().as_default():
        v1_second_graph = resource_variable_ops.ResourceVariable(2.0, name='v1')
        saver = _saver.Saver([v1_first_graph, v1_second_graph])
      ckpt_prefix = os.path.join(test.get_temp_dir(), 'ckpt')
      with self.assertRaisesRegexp(ValueError, 'v1'):
        saver.save(ckpt_prefix)

  def testDifferentGraphError(self):
    with context.eager_mode(), ops.device(self._dev()):
      with ops.Graph().as_default():
        v1 = resource_variable_ops.ResourceVariable(1.0, name='v1')
      with ops.Graph().as_default():
        saver = _saver.Saver([v1])
        ckpt_prefix = os.path.join(test.get_temp_dir(), 'ckpt')
        with self.assertRaisesRegexp(ValueError, 'Graph'):
          saver.save(ckpt_prefix)

  def testSameObjectOK(self):
    with context.eager_mode(), ops.device(self._dev()):
      v1 = resource_variable_ops.ResourceVariable(1.0, name='v1')
      # While different objects with the same shared_name are not good, passing
      # in the same object multiple times is fine.
      saver = _saver.Saver([v1, v1])
      ckpt_prefix = os.path.join(test.get_temp_dir(), 'ckpt')
      saver.save(ckpt_prefix)

  def testRestoreOnCreate(self):
    with ops.device(self._dev()):
      def model(init_val):
        v1 = resource_variable_ops.ResourceVariable(init_val, name='v1')
        return array_ops.constant(1.0) * v1, v1

      ckpt_prefix = os.path.join(test.get_temp_dir(), 'ckpt')
      _, v1 = model(1.0)
      saver = _saver.Saver([v1])
      saver.save(ckpt_prefix)

      with ops.Graph().as_default():
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

  def testSaveRestoreGraphCallable(self):
    with ops.device(self._dev()):
      @graph_callable.graph_callable(
          [graph_callable.ShapeAndDtype(shape=(), dtype=dtypes.float32)])
      def model(x):
        v = variable_scope.get_variable(
            'v', initializer=init_ops.zeros_initializer(), shape=())
        return v + x

      # Default 2 + 0 = 2
      self.assertEqual(
          2, model(array_ops.constant(2, dtype=dtypes.float32)).numpy())

      # Save the variable value 0.
      ckpt_prefix = os.path.join(test.get_temp_dir(), 'ckpt')
      _saver.Saver(model.variables).save(ckpt_prefix)

      # update variable to 1, so that 2 + 1 = 3
      model.variables[0].assign(1.)
      self.assertEqual(
          3, model(array_ops.constant(2, dtype=dtypes.float32)).numpy())

      # load the variable value 0, so that 2 + 0 = 2
      _saver.Saver(model.variables).restore(ckpt_prefix)
      self.assertEqual(
          2, model(array_ops.constant(2, dtype=dtypes.float32)).numpy())

      # update checkpoint variable to 1 and memory value to 2.
      model.variables[0].assign(1.)
      _saver.Saver(model.variables).save(ckpt_prefix)
      model.variables[0].assign(2.)
      self.assertEqual(
          4, model(array_ops.constant(2, dtype=dtypes.float32)).numpy())

      # reset the graph and reload on create, so that 1 + 2 = 3
      with ops.Graph().as_default():
        with _saver.restore_variables_on_create(ckpt_prefix):
          @graph_callable.graph_callable(
              [graph_callable.ShapeAndDtype(shape=(), dtype=dtypes.float32)])
          def model2(x):
            v = variable_scope.get_variable(
                'v', initializer=init_ops.zeros_initializer(), shape=())
            return v + x

          self.assertEqual(
              3, model2(array_ops.constant(2, dtype=dtypes.float32)).numpy())


if __name__ == '__main__':
  test.main()
