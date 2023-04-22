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
"""Tests for directives module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.converters import directives as directives_converter
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.autograph.lang import directives
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.platform import test


class DirectivesTest(converter_testing.TestCase):

  def test_local_target(self):

    def f():
      l = []
      string_var = 0
      directives.set_element_type(l, 'a', string_var)

    _, node, _ = self.transform(f, directives_converter, include_ast=True)

    def_, = anno.getanno(node.body[0].targets[0],
                         anno.Static.DEFINITIONS)
    d = def_.directives[directives.set_element_type]
    self.assertEqual(d['dtype'].value, 'a')
    self.assertEqual(d['shape'].id, 'string_var')

  def test_argument_target(self):

    def f(a):
      directives.set_element_type(a, 1, shape=2)
      pass

    _, node, _ = self.transform(f, directives_converter, include_ast=True)

    def_, = anno.getanno(node.args.args[0], anno.Static.DEFINITIONS)
    d = def_.directives[directives.set_element_type]
    self.assertEqual(d['dtype'].value, 1)
    self.assertEqual(d['shape'].value, 2)

  def test_loop_target(self):

    def f():
      a = True
      while True:
        directives.set_loop_options(parallel_iterations=10, back_prop=a)  # pylint: disable=unexpected-keyword-arg
        pass

    _, node, _ = self.transform(f, directives_converter, include_ast=True)

    d = anno.getanno(node.body[1], anno.Basic.DIRECTIVES)
    d = d[directives.set_loop_options]
    self.assertEqual(d['parallel_iterations'].value, 10)
    self.assertEqual(d['back_prop'].id, 'a')
    self.assertNotIn('swap_memory', d)

  def test_loop_target_no_loop(self):

    def f():
      directives.set_loop_options()
      pass

    with self.assertRaisesRegex(ValueError, 'must be used inside a statement'):
      self.transform(f, directives_converter, include_ast=True)

  def test_loop_target_not_first(self):

    def f():
      a = 1
      while True:
        a = 2
        directives.set_loop_options(parallel_iterations=10, back_prop=a)  # pylint: disable=unexpected-keyword-arg

    with self.assertRaisesRegex(ValueError, 'must be the first statement'):
      self.transform(f, directives_converter, include_ast=True)

  def test_value_verification_does_not_trigger_properties(self):

    self_test = self

    class TestClass(object):

      @property
      def b(self):
        self_test.fail('This should never be evaluated')

    tc = TestClass()

    def f():
      return tc.b + 1

    _, node, _ = self.transform(f, directives_converter, include_ast=True)

    self.assertIsNotNone(node)

  def test_value_verification_does_not_trigger_getattr(self):

    class TestClass(object):

      def __init__(self):
        self.getattr_called = False

      def __getattr__(self, _):
        # Note: seems that any exception raised here is absorbed by hasattr.
        # So we can't call test.fail or raise.
        self.getattr_called = True

    tc = TestClass()

    def f():
      return tc.b + 1

    _, node, _ = self.transform(f, directives_converter, include_ast=True)

    self.assertIsNotNone(node)
    self.assertFalse(tc.getattr_called)


if __name__ == '__main__':
  test.main()
