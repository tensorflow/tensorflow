# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for ShardedVariable."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.training.tracking import util


class ShardedVariableTest(test.TestCase):

  def test_sharded_variable_simple(self):
    v0 = variables_lib.Variable([0])
    v1 = variables_lib.Variable([1])
    s = sharded_variable.ShardedVariable([v0, v1], name='s')
    self.assertEqual(s.variables[0], v0)
    self.assertEqual(s.variables[1], v1)
    self.assertEqual(s.shape.as_list(), [2])
    self.assertEqual(s.dtype, v0.dtype)
    self.assertEqual(s.name, 's')

  def test_save_restore(self):
    fname = os.path.join(self.get_temp_dir(), 'checkpoint')
    variables = [
        variables_lib.Variable([0]),
        variables_lib.Variable([1]),
        variables_lib.Variable([2]),
        variables_lib.Variable([3])
    ]
    s = sharded_variable.ShardedVariable(variables, name='s')

    cp = util.Checkpoint(s=s)
    self.assertEqual(self.evaluate(cp.s.variables[0]), [0])
    cp.write(fname)

    self.evaluate(cp.s.variables[0].assign([4]))
    self.assertEqual(self.evaluate(cp.s.variables[0]), [4])

    cp.restore(fname)
    # Tests that the original weights are restored.
    self.assertEqual(self.evaluate(cp.s.variables[0]), [0])

  def test_save_restore_different_partitions(self):
    fname = os.path.join(self.get_temp_dir(), 'checkpoint')
    variables = [
        variables_lib.Variable([0]),
        variables_lib.Variable([1]),
        variables_lib.Variable([2]),
        variables_lib.Variable([3])
    ]
    s = sharded_variable.ShardedVariable(variables, name='s')

    cp = util.Checkpoint(s=s)
    cp.write(fname)

    variables2 = [variables_lib.Variable([0, 0, 0, 0])]
    s2 = sharded_variable.ShardedVariable(variables2, name='s')

    # Restore from 4 partitions into 1.
    cp2 = util.Checkpoint(s=s2)
    cp2.restore(fname)
    self.assertAllEqual(self.evaluate(cp2.s.variables[0]), [0, 1, 2, 3])

    self.evaluate(cp2.s.variables[0].assign([5, 10, 15, 20]))
    cp2.write(fname)

    # Restore 1 partition into 4.
    cp.restore(fname)
    self.assertEqual(self.evaluate(cp.s.variables[0]), [5])
    self.assertEqual(self.evaluate(cp.s.variables[1]), [10])
    self.assertEqual(self.evaluate(cp.s.variables[2]), [15])
    self.assertEqual(self.evaluate(cp.s.variables[3]), [20])

  def test_save_restore_4_to_2_partitions(self):
    fname = os.path.join(self.get_temp_dir(), 'checkpoint')
    variables = [
        variables_lib.Variable([0]),
        variables_lib.Variable([1]),
        variables_lib.Variable([2]),
        variables_lib.Variable([3])
    ]
    s = sharded_variable.ShardedVariable(variables, name='s')
    cp = util.Checkpoint(s=s)
    cp.write(fname)

    variables2 = [
        variables_lib.Variable([0, 0]),
        variables_lib.Variable([0, 0])
    ]
    s2 = sharded_variable.ShardedVariable(variables2, name='s')
    cp2 = util.Checkpoint(s=s2)
    cp2.restore(fname)
    # Assert that weights from the 4 partitions were loaded here.
    self.assertLen(cp2.s.variables, 2)
    self.assertAllEqual(self.evaluate(cp2.s.variables[0]), [0, 1])
    self.assertAllEqual(self.evaluate(cp2.s.variables[1]), [2, 3])

  def test_validation_errors(self):
    with self.assertRaisesRegexp(ValueError, 'Expected a list of '):
      sharded_variable.ShardedVariable(
          [variables_lib.Variable([0]), 'not-a-variable'])

    with self.assertRaisesRegexp(ValueError, 'must have the same dtype'):
      sharded_variable.ShardedVariable([
          variables_lib.Variable([0], dtype='int64'),
          variables_lib.Variable([1], dtype='int32')
      ])

    with self.assertRaisesRegexp(ValueError, 'the same shapes except'):
      sharded_variable.ShardedVariable([
          variables_lib.Variable(array_ops.ones((5, 10))),
          variables_lib.Variable(array_ops.ones((5, 20)))
      ])

    with self.assertRaisesRegexp(ValueError, '`SaveSliceInfo` should not'):
      v = variables_lib.Variable([0])
      v._set_save_slice_info(
          variables_lib.Variable.SaveSliceInfo(
              full_name='s', full_shape=[2], var_offset=[0], var_shape=[1]))
      sharded_variable.ShardedVariable([v])


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
