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
"""Tests for decorators module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import textwrap

from tensorflow.contrib.py2tf.converters import converter_test_base
from tensorflow.contrib.py2tf.converters import decorators
from tensorflow.contrib.py2tf.pyct import compiler
from tensorflow.python.platform import test
from tensorflow.python.util import tf_inspect


class DecoratorsTest(converter_test_base.TestCase):

  def test_function_decorator(self):

    def function_decorator():

      def decorator(f):
        return lambda a: f(a) + 1

      return decorator

    # The Python parser does capture decorators into the AST.
    # However, the interpreter desugars them on load, and refering to the
    # decorated function at runtime usually loses any trace of the decorator.
    # Below is an example when that doesn't happen.
    def static_wrapper():

      @function_decorator()
      def test_fn(a):  # pylint:disable=unused-variable
        return a

    node = self.parse_and_analyze(static_wrapper,
                                  {'function_decorator': function_decorator})
    node = node.body[0].body[0]

    node = decorators.transform(node, remove_decorators=())
    result = compiler.ast_to_object(
        node,
        source_prefix=textwrap.dedent(tf_inspect.getsource(function_decorator)))
    self.assertEqual(2, result.test_fn(1))

    node = decorators.transform(node, remove_decorators=(function_decorator,))
    result = compiler.ast_to_object(node)
    self.assertEqual(1, result.test_fn(1))

  def test_simple_decorator(self):

    def simple_decorator(f):
      return lambda a: f(a) + 1

    # The Python parser does capture decorators into the AST.
    # However, the interpreter desugars them upon load, and refering to the
    # decorated function at runtime usually loses any trace of the decorator.
    # Below is an example when that doesn't happen.
    def static_wrapper():

      @simple_decorator
      def test_fn(a):  # pylint:disable=unused-variable
        return a

    node = self.parse_and_analyze(static_wrapper,
                                  {'simple_decorator': simple_decorator})
    node = node.body[0].body[0]

    node = decorators.transform(node, remove_decorators=())
    result = compiler.ast_to_object(
        node,
        source_prefix=textwrap.dedent(tf_inspect.getsource(simple_decorator)))
    self.assertEqual(2, result.test_fn(1))

    node = decorators.transform(node, remove_decorators=(simple_decorator,))
    result = compiler.ast_to_object(node)
    self.assertEqual(1, result.test_fn(1))


if __name__ == '__main__':
  test.main()
