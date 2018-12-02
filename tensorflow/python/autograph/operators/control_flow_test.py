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

from tensorflow.python.autograph.operators import control_flow
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class ForLoopTest(test.TestCase):

  @test_util.run_deprecated_v1
  def test_tensor(self):
    s = control_flow.for_stmt(
        constant_op.constant([1, 2, 3, 4]),
        extra_test=lambda s: True,
        body=lambda i, s: (s + i,),
        init_state=(0,))
    with self.cached_session() as sess:
      self.assertEqual((10,), self.evaluate(s))

  def test_python(self):
    s = control_flow.for_stmt(
        range(5),
        extra_test=lambda s: True,
        body=lambda i, s: (s + i,),
        init_state=(0,))
    self.assertEqual(10, s)

  @test_util.run_deprecated_v1
  def test_dataset(self):
    to_int32 = lambda i: math_ops.cast(i, dtypes.int32)
    s = control_flow.for_stmt(
        dataset_ops.Dataset.range(5).map(to_int32),
        extra_test=lambda s: True,
        body=lambda i, s: (s + i,),
        init_state=(0,))
    with self.cached_session() as sess:
      self.assertEqual((10,), self.evaluate(s))


class WhileLoopTest(test.TestCase):

  @test_util.run_deprecated_v1
  def test_tensor(self):
    n = constant_op.constant(5)
    results = control_flow.while_stmt(
        test=lambda i, s: i < n,
        body=lambda i, s: (i + 1, s + i,),
        init_state=(0, 0),
        extra_deps=(n,))
    with self.cached_session() as sess:
      self.assertEqual((5, 10), self.evaluate(results))

  def test_python(self):
    n = 5
    results = control_flow.while_stmt(
        test=lambda i, s: i < n,
        body=lambda i, s: (i + 1, s + i),
        init_state=(0, 0),
        extra_deps=(n,))
    self.assertEqual((5, 10), results)


class IfStmtTest(test.TestCase):

  def single_return_if_stmt(self, cond):
    return control_flow.if_stmt(cond=cond, body=lambda: 1, orelse=lambda: -1)

  def multi_return_if_stmt(self, cond):
    return control_flow.if_stmt(
        cond=cond, body=lambda: (1, 2), orelse=lambda: (-1, -2))

  @test_util.run_deprecated_v1
  def test_tensor(self):
    with self.cached_session() as sess:
      t = self.single_return_if_stmt(constant_op.constant(True))
      self.assertEqual(1, self.evaluate(t))
      t = self.single_return_if_stmt(constant_op.constant(False))
      self.assertEqual(-1, self.evaluate(t))

  def test_python(self):
    self.assertEqual(1, self.single_return_if_stmt(True))
    self.assertEqual(-1, self.single_return_if_stmt(False))

  @test_util.run_deprecated_v1
  def test_tensor_multiple_returns(self):
    with self.cached_session() as sess:
      t = self.multi_return_if_stmt(constant_op.constant(True))
      self.assertAllEqual([1, 2], self.evaluate(t))
      t = self.multi_return_if_stmt(constant_op.constant(False))
      self.assertAllEqual([-1, -2], self.evaluate(t))

  def test_python_multiple_returns(self):
    self.assertEqual((1, 2), self.multi_return_if_stmt(True))
    self.assertEqual((-1, -2), self.multi_return_if_stmt(False))


if __name__ == '__main__':
  test.main()
