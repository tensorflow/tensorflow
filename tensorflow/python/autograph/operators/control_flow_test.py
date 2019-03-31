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
"""Tests for control_flow module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import sys

import six

from tensorflow.python.autograph.operators import control_flow
from tensorflow.python.autograph.pyct import errors
from tensorflow.python.autograph.utils import ag_logging
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class ForLoopTest(test.TestCase):

  @test_util.run_deprecated_v1
  def test_tensor(self):
    s = control_flow.for_stmt(
        constant_op.constant([1, 2, 3, 4]),
        extra_test=lambda s: True,
        body=lambda i, s: (s + i,),
        init_state=(0,))
    with self.cached_session():
      self.assertEqual((10,), self.evaluate(s))

  def test_python(self):
    s = control_flow.for_stmt(
        range(5),
        extra_test=lambda s: True,
        body=lambda i, s: (s + i,),
        init_state=(0,))
    self.assertEqual((10,), s)

  def test_dataset(self):
    s = control_flow.for_stmt(
        dataset_ops.Dataset.range(5),
        extra_test=None,
        body=lambda i, s: (s + i,),
        init_state=(constant_op.constant(0, dtype=dtypes.int64),))
    self.assertEqual(self.evaluate(s), (10,))

  @test_util.run_v2_only
  def test_dataset_no_state(self):
    v = variables.Variable(0, dtype=dtypes.int64)
    def stateless_with_side_effects(i):
      v.assign(v.read_value() + i)
    s = control_flow.for_stmt(
        dataset_ops.Dataset.range(5),
        extra_test=None,
        body=stateless_with_side_effects,
        init_state=())
    self.evaluate(s)
    self.assertEqual(self.evaluate(v.read_value()), 10)


class WhileLoopTest(test.TestCase):

  @test_util.run_deprecated_v1
  def test_tensor(self):
    n = constant_op.constant(5)
    results = control_flow.while_stmt(
        test=lambda i, s: i < n,
        body=lambda i, s: (i + 1, s + i),
        init_state=(0, 0))
    self.assertEqual((5, 10), self.evaluate(results))

  def test_tensor_with_tf_side_effects_in_cond(self):

    n = constant_op.constant(5, dtype=dtypes.int64)
    v = variables.Variable(0, dtype=dtypes.int64)

    def get_and_increment(v):
      v.assign(v.read_value() + 1)
      return v.read_value()

    # function is important here, for its automatic control deps.
    @def_function.function(autograph=False)
    def test_fn():
      return control_flow.while_stmt(
          test=lambda i: get_and_increment(v) < n,
          body=lambda i: (i + 1,),
          init_state=(0,))

    results = test_fn()

    self.evaluate(v.initializer)
    self.assertEqual(self.evaluate(results), (4,))
    self.assertEqual(self.evaluate(v), (5,))

  @test_util.run_deprecated_v1
  def test_python_with_tensor_state(self):
    n = 5
    results = control_flow.while_stmt(
        test=lambda i, s: i < n,
        body=lambda i, s: (i + 1, s + i),
        init_state=(0, constant_op.constant(0)))
    result_i, result_s = results
    self.assertEqual(5, result_i)
    self.assertEqual(10, self.evaluate(result_s))

  def test_python(self):
    n = 5
    results = control_flow.while_stmt(
        test=lambda i, s: i < n,
        body=lambda i, s: (i + 1, s + i),
        init_state=(0, 0))
    self.assertEqual((5, 10), results)

  def test_python_infinite_loop(self):
    if __debug__:
      with test.mock.patch.object(control_flow, 'PYTHON_MAX_ITERATIONS', 1000):
        with self.assertRaisesRegexp(errors.ExecutionError, 'iteration limit'):
          control_flow.while_stmt(
              test=lambda _: True,
              body=lambda i: (i + 1,),
              init_state=(0,))

  def test_python_long_loop_unroll_warning(self):
    if __debug__:
      with ops.Graph().as_default():
        out_capturer = six.StringIO()
        with test.mock.patch.object(sys, 'stdout', out_capturer):
          ag_logging.echo_log_to_stdout = True
          sys.stdout = out_capturer
          control_flow.while_stmt(
              test=lambda i, _: i < 10000,
              body=lambda i, _: (i + 1, gen_math_ops.add(i, 1),),
              init_state=(0, None))
        self.assertTrue(re.match(
            r'.*ops.*loop.*large.*iterations.*Add.*', out_capturer.getvalue()))


class IfStmtTest(test.TestCase):

  def single_return_if_stmt(self, cond):
    return control_flow.if_stmt(
        cond=cond,
        body=lambda: 1,
        orelse=lambda: -1,
        get_state=lambda: (),
        set_state=lambda _: None)

  def multi_return_if_stmt(self, cond):
    return control_flow.if_stmt(
        cond=cond,
        body=lambda: (1, 2),
        orelse=lambda: (-1, -2),
        get_state=lambda: (),
        set_state=lambda _: None)

  @test_util.run_deprecated_v1
  def test_tensor(self):
    with self.cached_session():
      t = self.single_return_if_stmt(constant_op.constant(True))
      self.assertEqual(1, self.evaluate(t))
      t = self.single_return_if_stmt(constant_op.constant(False))
      self.assertEqual(-1, self.evaluate(t))

  def test_python(self):
    self.assertEqual(1, self.single_return_if_stmt(True))
    self.assertEqual(-1, self.single_return_if_stmt(False))

  @test_util.run_deprecated_v1
  def test_tensor_multiple_returns(self):
    with self.cached_session():
      t = self.multi_return_if_stmt(constant_op.constant(True))
      self.assertAllEqual([1, 2], self.evaluate(t))
      t = self.multi_return_if_stmt(constant_op.constant(False))
      self.assertAllEqual([-1, -2], self.evaluate(t))

  def test_python_multiple_returns(self):
    self.assertEqual((1, 2), self.multi_return_if_stmt(True))
    self.assertEqual((-1, -2), self.multi_return_if_stmt(False))


if __name__ == '__main__':
  test.main()
