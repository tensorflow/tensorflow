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

from tensorflow.contrib.py2tf.converters import control_flow
from tensorflow.contrib.py2tf.converters import converter_test_base
from tensorflow.contrib.py2tf.pyct import compiler
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import test


class TestNamer(control_flow.SymbolNamer):

  def new_symbol(self, name_root, used):
    i = 0
    while True:
      name = '%s%d' % (name_root, i)
      if name not in used:
        return name
      i += 1


class ControlFlowTest(converter_test_base.TestCase):

  def test_simple_while(self):

    def test_fn(n):
      i = 0
      s = 0
      while i < n:
        s += i
        i += 1
      return s, i, n

    node = self.parse_and_analyze(test_fn, {})
    node = control_flow.transform(node, TestNamer())
    result = compiler.ast_to_object(node)
    setattr(result, 'tf', control_flow_ops)

    with self.test_session() as sess:
      self.assertEqual((10, 5, 5),
                       sess.run(result.test_fn(constant_op.constant(5))))

  def test_while_single_var(self):

    def test_fn(n):
      while n > 0:
        n -= 1
      return n

    node = self.parse_and_analyze(test_fn, {})
    node = control_flow.transform(node, TestNamer())
    result = compiler.ast_to_object(node)
    setattr(result, 'tf', control_flow_ops)

    with self.test_session() as sess:
      self.assertEqual(0, sess.run(result.test_fn(constant_op.constant(5))))

  def test_simple_if(self):

    def test_fn(n):
      a = 0
      b = 0
      if n > 0:
        a = -n
      else:
        b = 2 * n
      return a, b

    node = self.parse_and_analyze(test_fn, {})
    node = control_flow.transform(node, TestNamer())
    result = compiler.ast_to_object(node)
    setattr(result, 'tf', control_flow_ops)

    with self.test_session() as sess:
      self.assertEqual((-1, 0), sess.run(
          result.test_fn(constant_op.constant(1))))
      self.assertEqual((0, -2),
                       sess.run(result.test_fn(constant_op.constant(-1))))

  def test_if_single_var(self):

    def test_fn(n):
      if n > 0:
        n = -n
      return n

    node = self.parse_and_analyze(test_fn, {})
    node = control_flow.transform(node, TestNamer())
    result = compiler.ast_to_object(node)
    setattr(result, 'tf', control_flow_ops)

    with self.test_session() as sess:
      self.assertEqual(-1, sess.run(result.test_fn(constant_op.constant(1))))


if __name__ == '__main__':
  test.main()
