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
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test


class SaverTest(test.TestCase):

  def testBasics(self):
    with context.eager_mode():
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

  def testRestoreOnCreate(self):
    with context.eager_mode():
      def model(init_val):
        v1 = resource_variable_ops.ResourceVariable(init_val, name='v1')
        return array_ops.constant(1.0) * v1, v1

      ckpt_prefix = os.path.join(test.get_temp_dir(), 'ckpt')
      _, v1 = model(1.0)
      saver = _saver.Saver([v1])
      saver.save(ckpt_prefix)

      with ops.Graph().as_default():
        saver = _saver.Saver([v1])
        with saver.maybe_restore_on_create(ckpt_prefix):
          # Value is from checkpoint, but not from argument.
          ret, _ = model(2.0)
          self.assertEqual(ret.numpy(), 1.0)
          # Create it a second time won't re-assign the checkpoint value.
          v1_2 = resource_variable_ops.ResourceVariable(3.0, name='v1')
          self.assertEqual(v1_2.read_value().numpy(), 3.0)

  def testRestoreNotFound(self):
    with context.eager_mode():
      def model(v):
        return array_ops.constant(1.0) * v

      ckpt_prefix = os.path.join(test.get_temp_dir(), 'ckpt')
      v = resource_variable_ops.ResourceVariable(1.0, name='v1')
      _ = model(v)
      saver = _saver.Saver([v])
      saver.save(ckpt_prefix)

      with self.assertRaisesRegexp(errors.NotFoundError,
                                   'v2 not found in checkpoint'):
        with saver.maybe_restore_on_create(ckpt_prefix):
          _ = model(resource_variable_ops.ResourceVariable(1.0, name='v2'))


if __name__ == '__main__':
  test.main()
