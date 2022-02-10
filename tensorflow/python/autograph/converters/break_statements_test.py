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
"""Tests for break_statements module."""

from tensorflow.python.autograph.converters import break_statements
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.platform import test


class BreakCanonicalizationTest(converter_testing.TestCase):

  def assertTransformedEquivalent(self, f, *inputs):
    tr = self.transform(f, break_statements)
    self.assertEqual(f(*inputs), tr(*inputs))

  def test_while_loop(self):

    def f(x):
      v = []
      while x > 0:
        x -= 1
        if x % 2 == 0:
          break
        v.append(x)
      return v

    self.assertTransformedEquivalent(f, 0)
    self.assertTransformedEquivalent(f, 1)
    self.assertTransformedEquivalent(f, 4)

  def test_while_loop_preserves_directives(self):

    def f(x):
      while x > 0:
        x -= 1
        if x % 2 == 0:
          break

    _, node, ctx = self.transform(f, (), include_ast=True)
    fake_annotation = object()
    anno.setanno(node.body[0], anno.Basic.DIRECTIVES, fake_annotation)
    node = break_statements.transform(node, ctx)

    self.assertIs(
        anno.getanno(node.body[1], anno.Basic.DIRECTIVES), fake_annotation)

  def test_for_loop(self):

    def f(a):
      v = []
      for x in a:
        x -= 1
        if x % 2 == 0:
          break
        v.append(x)
      return v

    tr = self.transform(f, break_statements)

    self.assertEqual([3], tr([5, 4]))

  def test_for_loop_preserves_directives(self):

    def f(a):
      for x in a:
        if x % 2 == 0:
          break

    _, node, ctx = self.transform(f, (), include_ast=True)
    fake_annotation = object()
    anno.setanno(node.body[0], anno.Basic.DIRECTIVES, fake_annotation)
    node = break_statements.transform(node, ctx)
    self.assertIs(
        anno.getanno(node.body[1], anno.Basic.DIRECTIVES), fake_annotation)

  def test_nested(self):

    def f(x):
      v = []
      u = []
      w = []
      while x > 0:
        x -= 1
        if x % 2 == 0:
          if x % 3 != 0:
            u.append(x)
          else:
            w.append(x)
            break
        v.append(x)
      return v, u, w

    self.assertTransformedEquivalent(f, 0)
    self.assertTransformedEquivalent(f, 3)
    self.assertTransformedEquivalent(f, 11)

  def test_nested_loops(self):

    def f(x):
      v = []
      u = []
      while x > 0:
        x -= 1
        y = x
        while y > 0:
          y -= 1
          if y % 2 == 0:
            break
          u.append(y)
        if x == 0:
          break
        v.append(x)
      return v, u

    self.assertTransformedEquivalent(f, 0)
    self.assertTransformedEquivalent(f, 2)
    self.assertTransformedEquivalent(f, 3)
    self.assertTransformedEquivalent(f, 5)

  def test_loop_orelse(self):

    def f(x):
      v = []
      u = []
      while x > 0:
        x -= 1
        y = x
        while y > 1:
          break
        else:
          u.append(y)
          break
        v.append(x)
      return v, u

    self.assertTransformedEquivalent(f, 0)
    self.assertTransformedEquivalent(f, 2)
    self.assertTransformedEquivalent(f, 3)

  def test_multiple_correlated_breaks_with_side_effects(self):
    def f(cond1):
      lst = []
      while True:
        if cond1:
          lst.append(1)
        else:
          break
        if lst[-1] > 0:  # lst always has an element here
          break
      return lst

    self.assertTransformedEquivalent(f, True)
    self.assertTransformedEquivalent(f, False)


if __name__ == '__main__':
  test.main()
