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
"""Tests for templates module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import imp

import gast

from tensorflow.contrib.py2tf.pyct import compiler
from tensorflow.contrib.py2tf.pyct import templates
from tensorflow.python.platform import test


class TemplatesTest(test.TestCase):

  def test_replace_tuple(self):
    template = """
      def test_fn(a, c):
        return b,
    """

    node = templates.replace(template, b=('a', 'c'))[0]
    result, _ = compiler.ast_to_object(node)

    self.assertEquals((2, 3), result.test_fn(2, 3))

  def test_replace_variable(self):
    template = """
      def test_fn(a):
        a += 1
        a = 2 * a + 1
        return b
    """

    node = templates.replace(template, a='b')[0]
    result, _ = compiler.ast_to_object(node)
    self.assertEquals(7, result.test_fn(2))

  def test_replace_function_name(self):
    template = """
      def fname(a):
        a += 1
        a = 2 * a + 1
        return a
    """

    node = templates.replace(template, fname='test_fn')[0]
    result, _ = compiler.ast_to_object(node)
    self.assertEquals(7, result.test_fn(2))

  def test_replace_code_block(self):
    template = """
      def test_fn(a):
        block
        return a
    """

    node = templates.replace(
        template,
        block=[
            gast.Assign([
                gast.Name('a', None, None)
            ], gast.BinOp(gast.Name('a', None, None), gast.Add(), gast.Num(1))),
        ] * 2)[0]
    result, _ = compiler.ast_to_object(node)
    self.assertEquals(3, result.test_fn(1))

  def test_replace_attribute(self):
    template = """
      def test_fn(a):
        return a.foo
    """

    node = templates.replace(template, foo='b')[0]
    result, _ = compiler.ast_to_object(node)
    mod = imp.new_module('test')
    mod.b = 3
    self.assertEquals(3, result.test_fn(mod))

    with self.assertRaises(ValueError):
      templates.replace(template, foo=1)


if __name__ == '__main__':
  test.main()
