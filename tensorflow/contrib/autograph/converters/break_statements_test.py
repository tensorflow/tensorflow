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

from tensorflow.contrib.autograph.converters import break_statements
from tensorflow.contrib.autograph.converters import converter_test_base
from tensorflow.python.platform import test


class BreakCanonicalizationTest(converter_test_base.TestCase):

  def test_basic_while(self):

    def test_fn(x):
      v = []
      while x > 0:
        x -= 1
        if x % 2 == 0:
          break
        v.append(x)
      return v

    node = self.parse_and_analyze(test_fn, {})
    node = break_statements.transform(node, self.ctx)

    with self.compiled(node) as result:
      self.assertEqual([], result.test_fn(0))
      self.assertEqual([], result.test_fn(1))
      self.assertEqual([3], result.test_fn(4))

  def test_basic_for(self):

    def test_fn(a):
      v = []
      for x in a:
        x -= 1
        if x % 2 == 0:
          break
        v.append(x)
      return v

    node = self.parse_and_analyze(test_fn, {})
    node = break_statements.transform(node, self.ctx)

    with self.compiled(node) as result:
      # The break is incompletely canonicalized. The loop will not interrupt,
      # but the section following the break will be skipped.
      self.assertEqual([], result.test_fn([]))
      self.assertEqual([3, 3], result.test_fn([4, 4]))
      self.assertEqual([3], result.test_fn([4, 5]))
      self.assertEqual([3], result.test_fn([5, 4]))

  def test_deeply_nested(self):

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

    node = self.parse_and_analyze(test_fn, {})
    node = break_statements.transform(node, self.ctx)

    with self.compiled(node) as result:
      self.assertEqual(([], [], []), result.test_fn(0))
      self.assertEqual(([2, 1], [2], [0]), result.test_fn(3))
      self.assertEqual(([10, 9, 8, 7], [10, 8], [6]), result.test_fn(11))

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

    node = self.parse_and_analyze(test_fn, {})
    node = break_statements.transform(node, self.ctx)

    with self.compiled(node) as result:
      self.assertEqual(([], []), result.test_fn(0))
      self.assertEqual(([1], []), result.test_fn(2))
      self.assertEqual(([2, 1], [1]), result.test_fn(3))
      self.assertEqual(([4, 3, 2, 1], [3, 1]), result.test_fn(5))

  def test_loop_else(self):

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

    node = self.parse_and_analyze(test_fn, {})
    node = break_statements.transform(node, self.ctx)

    with self.compiled(node) as result:
      self.assertEqual(([], []), result.test_fn(0))
      self.assertEqual(([], [1]), result.test_fn(2))
      self.assertEqual(([2], [1]), result.test_fn(3))


if __name__ == '__main__':
  test.main()
