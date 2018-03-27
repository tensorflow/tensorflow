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

  def test_basic_break(self):

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
      self.assertEqual(test_fn(0), result.test_fn(0))
      self.assertEqual(test_fn(1), result.test_fn(1))
      self.assertEqual(test_fn(2), result.test_fn(2))
      self.assertEqual(test_fn(3), result.test_fn(3))
      self.assertEqual(test_fn(4), result.test_fn(4))

  def test_basic_break_for_loop(self):

    def test_fn(a):
      v = []
      for x in a:
        x -= 1
        if x % 2 == 0:
          break
        v.append(x)
      return v

    # The break is incompletely canonicalized for for loops. Everything is
    # in place except for the condition verification.
    def test_equiv_fn(a):
      v = []
      for x in a:
        x -= 1
        if x % 2 == 0:
          continue
        v.append(x)
      return v

    node = self.parse_and_analyze(test_fn, {})
    node = break_statements.transform(node, self.ctx)

    with self.compiled(node) as result:
      # The break is incompletely canonicalized. Everything is in place, but
      # the loop does not break.
      self.assertEqual(test_equiv_fn([]), result.test_fn([]))
      self.assertEqual(test_equiv_fn([1]), result.test_fn([1]))
      self.assertEqual(test_equiv_fn([2]), result.test_fn([2]))
      self.assertEqual(
          test_equiv_fn([1, 2, 3, 4]), result.test_fn([1, 2, 3, 4]))

  def test_continue_deeply_nested(self):

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
            continue
        v.append(x)
      return v, u, w

    node = self.parse_and_analyze(test_fn, {})
    node = break_statements.transform(node, self.ctx)

    with self.compiled(node) as result:
      self.assertEqual(test_fn(0), result.test_fn(0))
      self.assertEqual(test_fn(1), result.test_fn(1))
      self.assertEqual(test_fn(2), result.test_fn(2))
      self.assertEqual(test_fn(3), result.test_fn(3))
      self.assertEqual(test_fn(4), result.test_fn(4))


if __name__ == '__main__':
  test.main()
