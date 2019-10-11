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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.converters import break_statements
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test


class BreakCanonicalizationTest(converter_testing.TestCase):

  def assertTransformedEquivalent(self, test_fn, *inputs):
    with self.converted(test_fn, break_statements, {},
                        (constant_op.constant,)) as result:
      self.assertEqual(test_fn(*inputs), result.test_fn(*inputs))

  def test_while_loop(self):

    def test_fn(x):
      v = []
      while x > 0:
        x -= 1
        if x % 2 == 0:
          break
        v.append(x)
      return v

    self.assertTransformedEquivalent(test_fn, 0)
    self.assertTransformedEquivalent(test_fn, 1)
    self.assertTransformedEquivalent(test_fn, 4)

  def test_for_loop(self):

    def test_fn(a):
      v = []
      for x in a:
        x -= 1
        if x % 2 == 0:
          break
        v.append(x)
      return v

    with self.converted(test_fn, break_statements, {},
                        (constant_op.constant,)) as result:
      # The break is incompletely canonicalized. The loop will not interrupt,
      # but the section following the break will be skipped.
      self.assertEqual([3], result.test_fn([5, 4]))

  def test_nested(self):

    def test_fn(x):
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

    self.assertTransformedEquivalent(test_fn, 0)
    self.assertTransformedEquivalent(test_fn, 3)
    self.assertTransformedEquivalent(test_fn, 11)

  def test_nested_loops(self):

    def test_fn(x):
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

    self.assertTransformedEquivalent(test_fn, 0)
    self.assertTransformedEquivalent(test_fn, 2)
    self.assertTransformedEquivalent(test_fn, 3)
    self.assertTransformedEquivalent(test_fn, 5)

  def test_loop_orelse(self):

    def test_fn(x):
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

    self.assertTransformedEquivalent(test_fn, 0)
    self.assertTransformedEquivalent(test_fn, 2)
    self.assertTransformedEquivalent(test_fn, 3)

  def test_multiple_correlated_breaks_with_side_effects(self):
    def test_fn(cond1):
      lst = []
      while True:
        if cond1:
          lst.append(1)
        else:
          break
        if lst[-1] > 0:  # lst always has an element here
          break
      return lst

    self.assertTransformedEquivalent(test_fn, True)
    self.assertTransformedEquivalent(test_fn, False)

if __name__ == '__main__':
  test.main()
