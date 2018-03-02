# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for utility functions related to managing `tf.Variable`s."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import numpy as np

from tensorflow.contrib.bayesflow.python.ops import variable_utils

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope as varscope_ops
from tensorflow.python.ops import variables as variables_ops
from tensorflow.python.platform import test


def test_fn(x):
  x = ops.convert_to_tensor(x, name="x")
  dtype = x.dtype.as_numpy_dtype
  s = x.shape.as_list()
  z = varscope_ops.get_variable(
      name="z",
      dtype=dtype,
      initializer=np.arange(np.prod(s)).reshape(s).astype(dtype))
  y = varscope_ops.get_variable(
      name="y",
      dtype=dtype,
      initializer=np.arange(np.prod(s)).reshape(s).astype(dtype)**2)
  return x + y + z


class _WrapCallableTest(object):

  def testDefaultArgsWorkCorrectly(self):
    with self.test_session():
      x = constant_op.constant(self.dtype([0.1, 0.2]))
      wrapped_fn, vars_args = variable_utils.externalize_variables_as_args(
          test_fn, [x])

      varscope_ops.get_variable_scope().reuse_variables()

      result = wrapped_fn(self.dtype(2), [3, 4, 5], 0.5)

      y_actual = varscope_ops.get_variable("y", dtype=self.dtype)
      z_actual = varscope_ops.get_variable("z", dtype=self.dtype)

      variables_ops.global_variables_initializer().run()
      result_ = result.eval()

      self.assertEqual(self.dtype, result_.dtype)
      self.assertAllEqual([5.5, 6.5, 7.5], result_)
      self.assertAllEqual([y_actual, z_actual], vars_args)

  def testNonDefaultArgsWorkCorrectly(self):
    with self.test_session():
      x = constant_op.constant(self.dtype([0.1, 0.2]))

      _ = test_fn(self.dtype([0., 0.]))   # Needed to create vars.
      varscope_ops.get_variable_scope().reuse_variables()

      y_actual = varscope_ops.get_variable("y", dtype=self.dtype)

      wrapped_fn, vars_args = variable_utils.externalize_variables_as_args(
          test_fn, [x], possible_ancestor_vars=[y_actual])

      result = wrapped_fn(self.dtype([2, 3]), 0.5)  # x, y

      variables_ops.global_variables_initializer().run()
      result_ = result.eval()

      self.assertEqual(self.dtype, result_.dtype)
      self.assertAllEqual([2.5, 4.5], result_)
      self.assertAllEqual([y_actual], vars_args)

  def testWarnings(self):
    with self.test_session():
      x = constant_op.constant(self.dtype([0.1, 0.2]))
      wrapped_fn, _ = variable_utils.externalize_variables_as_args(
          test_fn, [x], possible_ancestor_vars=[])
      varscope_ops.get_variable_scope().reuse_variables()
      with warnings.catch_warnings(record=True) as w:
        wrapped_fn(self.dtype(2))
      w = sorted(w, key=lambda w: str(w.message))
      self.assertEqual(2, len(w))
      self.assertRegexpMatches(
          str(w[0].message),
          r"Variable .* 'y:0' .* not found in bypass dict.")
      self.assertRegexpMatches(
          str(w[1].message),
          r"Variable .* 'z:0' .* not found in bypass dict.")

  def testExceptions(self):
    with self.test_session():
      x = constant_op.constant(self.dtype([0.1, 0.2]))
      wrapped_fn, _ = variable_utils.externalize_variables_as_args(
          test_fn,
          [x],
          possible_ancestor_vars=[],
          assert_variable_override=True)
      varscope_ops.get_variable_scope().reuse_variables()
      with self.assertRaisesRegexp(ValueError, r"not found"):
        wrapped_fn(self.dtype(2))


class WrapCallableTest16(test.TestCase, _WrapCallableTest):
  dtype = np.float16


class WrapCallableTest32(test.TestCase, _WrapCallableTest):
  dtype = np.float32


class WrapCallableTest64(test.TestCase, _WrapCallableTest):
  dtype = np.float64


if __name__ == "__main__":
  test.main()
