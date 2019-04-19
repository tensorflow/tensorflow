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
"""Tests for AutoCastVariable."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import parameterized
import numpy as np

from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras.mixed_precision.experimental import autocast_variable

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training.tracking import util as trackable_utils

TESTCASES = ({
    'testcase_name': 'base',
    'distribute': False
}, {
    'testcase_name': 'distribute',
    'distribute': True
})


def get_distribute_scope(distribute):

  class DummyContextManager(object):

    def __enter__(self):
      pass

    def __exit__(self, *args):
      pass

  if distribute:
    return mirrored_strategy.MirroredStrategy(['cpu:0']).scope()
  else:
    return DummyContextManager()


def get_autocast_var(var, distribute):
  if distribute:
    return autocast_variable.AutoCastDistributedVariable(var)
  else:
    return autocast_variable.AutoCastVariable(var)


def get_var(val, dtype):
  return variables.VariableV1(val, use_resource=True, dtype=dtype)


@test_util.run_all_in_graph_and_eager_modes
class AutoCastVariableTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(*TESTCASES)
  def test_read(self, distribute):
    with get_distribute_scope(distribute):
      x = get_var(1., dtypes.float32)
      x = get_autocast_var(x, distribute)
      self.evaluate(x.initializer)

      # outside of auto cast scope.
      self.assertEqual(x.dtype, dtypes.float32)
      self.assertEqual(x.value().dtype, dtypes.float32)
      self.assertEqual(x.read_value().dtype, dtypes.float32)
      self.assertEqual(array_ops.identity(x).dtype, dtypes.float32)

      # within auto cast scope of different dtype
      with ops.get_default_graph()._enable_auto_casting_variables(
          dtypes.float16):
        self.assertEqual(x.dtype, dtypes.float16)
        self.assertEqual(x.value().dtype, dtypes.float16)
        self.assertEqual(x.read_value().dtype, dtypes.float16)
        self.assertEqual(array_ops.identity(x).dtype, dtypes.float16)

      # within auto cast scope of same dtype
      with ops.get_default_graph()._enable_auto_casting_variables(
          dtypes.float32):
        self.assertEqual(x.dtype, dtypes.float32)
        self.assertEqual(x.value().dtype, dtypes.float32)
        self.assertEqual(x.read_value().dtype, dtypes.float32)
        self.assertEqual(array_ops.identity(x).dtype, dtypes.float32)

  @parameterized.named_parameters(*TESTCASES)
  def test_read_nested_scopes(self, distribute):
    with get_distribute_scope(distribute):
      x = get_var(1., dtypes.float32)
      x = get_autocast_var(x, distribute)
      self.evaluate(x.initializer)

      with ops.get_default_graph()._enable_auto_casting_variables(
          dtypes.float16):
        self.assertEqual(x.dtype, dtypes.float16)
        self.assertEqual(x.read_value().dtype, dtypes.float16)

        with ops.get_default_graph()._enable_auto_casting_variables(
            dtypes.float32):
          self.assertEqual(x.dtype, dtypes.float32)
          self.assertEqual(x.read_value().dtype, dtypes.float32)

        self.assertEqual(x.dtype, dtypes.float16)
        self.assertEqual(x.read_value().dtype, dtypes.float16)

  @parameterized.named_parameters(*TESTCASES)
  def test_operator_overloads(self, distribute):
    with get_distribute_scope(distribute):
      x = get_var(1., dtypes.float32)
      x = get_autocast_var(x, distribute)
      self.evaluate(x.initializer)

    v1 = constant_op.constant(2., dtype=dtypes.float32)
    v2 = constant_op.constant(2., dtype=dtypes.float16)

    # Because autocast variables do not yet define operator overloads, the
    # operator is defined by the non-variable tensor

    # Test variable as the LHS. Currently, this is not supported with
    # distributed autocast variables
    if not distribute:
      self.assertEqual(self.evaluate(x + v1), 3.)

      with ops.get_default_graph()._enable_auto_casting_variables(
          dtypes.float16):
        self.assertEqual(self.evaluate(x + v2), 3.)

    # Test variable as the RHS
    self.assertEqual(self.evaluate(v1 + x), 3.)

    with ops.get_default_graph()._enable_auto_casting_variables(
        dtypes.float16):
      self.assertEqual(self.evaluate(v2 + x), 3.)

  @parameterized.named_parameters(*TESTCASES)
  def test_assign(self, distribute):
    with get_distribute_scope(distribute):
      x = get_var(0., dtypes.float32)
      x = get_autocast_var(x, distribute)
      self.evaluate(x.initializer)

      # outside of auto cast scope.
      v1 = constant_op.constant(3.14, dtype=dtypes.float32)
      v2 = constant_op.constant(3.14, dtype=dtypes.float16)

      def run_and_check():
        # Assign float32 values
        self.assertAllClose(3.14, self.evaluate(x.assign(v1)))
        self.assertAllClose(3.14 * 2, self.evaluate(x.assign_add(v1)))
        self.assertAllClose(3.14, self.evaluate(x.assign_sub(v1)))

        # Attempt to assign float16 values
        with self.assertRaisesRegexp(
            ValueError,
            'conversion requested dtype float32 for Tensor with dtype float16'):
          self.evaluate(x.assign(v2))
        with self.assertRaisesRegexp(
            ValueError,
            'conversion requested dtype float32 for Tensor with dtype float16'):
          self.evaluate(x.assign_add(v2))
        with self.assertRaisesRegexp(
            ValueError,
            'conversion requested dtype float32 for Tensor with dtype float16'):
          self.evaluate(x.assign_sub(v2))

        # Assign Python floats
        self.assertAllClose(3.14, self.evaluate(x.assign(3.14)))
        self.assertAllClose(3.14 * 2, self.evaluate(x.assign_add(3.14)))
        self.assertAllClose(3.14, self.evaluate(x.assign_sub(3.14)))

      run_and_check()
      # reset x
      self.evaluate(x.assign(0.))
      # within auto cast scope.
      with ops.get_default_graph()._enable_auto_casting_variables(
          dtypes.float16):
        # assign still expect float32 value even if in float16 scope
        run_and_check()

  @parameterized.named_parameters(*TESTCASES)
  def test_assign_stays_in_true_dtype(self, distribute):
    with get_distribute_scope(distribute):
      x = get_var(1., dtypes.float32)
      x = get_autocast_var(x, distribute)
      self.evaluate(x.initializer)
      # small_val is a value such that 1.0 + small_val == 1.0 in fp16, but not
      # in fp32
      small_val = np.finfo('float16').eps / 2
      small_tensor = constant_op.constant(small_val, dtype=dtypes.float32)
      with ops.get_default_graph()._enable_auto_casting_variables(
          dtypes.float16):
        # Variable should be increased, despite it appearing to be the same
        # float16 value.
        self.assertEqual(1. + small_val,
                         self.evaluate(x.assign(1. + small_tensor)))
        self.assertEqual(1., self.evaluate(x.value()))
      self.assertEqual(1. + small_val, self.evaluate(x.value()))

      self.evaluate(x.assign(1.))
      with ops.get_default_graph()._enable_auto_casting_variables(
          dtypes.float16):
        self.assertEqual(1. + small_val,
                         self.evaluate(x.assign_add(small_tensor)))
        self.assertEqual(1., self.evaluate(x.value()))
      self.assertEqual(1. + small_val, self.evaluate(x.value()))

  @parameterized.named_parameters(*TESTCASES)
  def test_checkpoint(self, distribute):
    with get_distribute_scope(distribute):
      x = get_var(1., dtypes.float32)
      x = get_autocast_var(x, distribute)
    self.evaluate(x.initializer)
    self.evaluate(x.assign(123.))

    checkpoint = trackable_utils.Checkpoint(x=x)
    prefix = os.path.join(self.get_temp_dir(), 'ckpt')
    save_path = checkpoint.save(prefix)
    self.evaluate(x.assign(234.))
    checkpoint.restore(save_path).assert_consumed().run_restore_ops()
    self.assertEqual(self.evaluate(x), 123.)

  @parameterized.named_parameters(*TESTCASES)
  def test_invalid_wrapped_variable(self, distribute):
    with get_distribute_scope(distribute):
      # Wrap a non-variable
      with self.assertRaisesRegexp(ValueError, 'variable must be of type'):
        x = constant_op.constant([1.], dtype=dtypes.float32)
        get_autocast_var(x, distribute)

      # Wrap a non-floating point variable
      with self.assertRaisesRegexp(ValueError,
                                   'variable must be a floating point'):
        x = get_var(1, dtypes.int32)
        get_autocast_var(x, distribute)

    if distribute:
      # Wrap a non-distributed variable with AutoCastDistributedVariable
      with self.assertRaisesRegexp(ValueError, 'variable must be of type'):
        x = get_var(1., dtypes.float32)
        get_autocast_var(x, distribute)


if __name__ == '__main__':
  test.main()
