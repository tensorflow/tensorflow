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

import gast

from tensorflow.contrib.py2tf.pyct import compiler
from tensorflow.contrib.py2tf.pyct import templates
from tensorflow.python.platform import test


class TemplatesTest(test.TestCase):

  def test_replace_variable(self):
    def template(a):  # pylint:disable=unused-argument
      def test_fn(a):  # pylint:disable=unused-variable
        a += 1
        a = 2 * a + 1
        return b  # pylint:disable=undefined-variable

    node = templates.replace(
        template, a=gast.Name('b', gast.Load(), None))[0]
    result = compiler.ast_to_object(node)
    self.assertEquals(7, result.test_fn(2))

  def test_replace_function_name(self):
    def template(fname):  # pylint:disable=unused-argument
      def fname(a):  # pylint:disable=function-redefined
        a += 1
        a = 2 * a + 1
        return a

    node = templates.replace(
        template, fname=gast.Name('test_fn', gast.Load(), None))[0]
    result = compiler.ast_to_object(node)
    self.assertEquals(7, result.test_fn(2))

  def test_code_block(self):
    def template(block):  # pylint:disable=unused-argument
      def test_fn(a):  # pylint:disable=unused-variable
        block  # pylint:disable=pointless-statement
        return a

    node = templates.replace(
        template,
        block=[
            gast.Assign(
                [
                    gast.Name('a', gast.Store(), None)
                ],
                gast.BinOp(
                    gast.Name('a', gast.Load(), None),
                    gast.Add(),
                    gast.Num(1))),
        ] * 2)[0]
    result = compiler.ast_to_object(node)
    self.assertEquals(3, result.test_fn(1))


if __name__ == '__main__':
  test.main()
