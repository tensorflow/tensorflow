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
"""Tests for continue_canonicalization module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.py2tf.convert import continue_canonicalization
from tensorflow.contrib.py2tf.convert import control_flow
from tensorflow.contrib.py2tf.pyct import compiler
from tensorflow.contrib.py2tf.pyct import parser
from tensorflow.contrib.py2tf.pyct.static_analysis import access
from tensorflow.python.platform import test


class TestNamer(control_flow.SymbolNamer):

  def new_symbol(self, name_root, _):
    return name_root


class ContinueCanonicalizationTest(test.TestCase):

  def _parse_and_analyze(self, test_fn, namespace):
    node = parser.parse_object(test_fn)
    node = access.resolve(node)
    return node

  def test_basic_continue(self):

    def test_fn(x):
      v = []
      while x > 0:
        x -= 1
        if x % 2 == 0:
          continue
        v.append(x)
      return v

    node = self._parse_and_analyze(test_fn, {})
    node = continue_canonicalization.transform(node, TestNamer())
    result = compiler.ast_to_object(node)

    self.assertEqual(test_fn(0), result.test_fn(0))
    self.assertEqual(test_fn(1), result.test_fn(1))
    self.assertEqual(test_fn(2), result.test_fn(2))
    self.assertEqual(test_fn(3), result.test_fn(3))
    self.assertEqual(test_fn(4), result.test_fn(4))

  def test_basic_continue_for_loop(self):

    def test_fn(a):
      v = []
      for x in a:
        x -= 1
        if x % 2 == 0:
          continue
        v.append(x)
      return v

    node = self._parse_and_analyze(test_fn, {})
    node = continue_canonicalization.transform(node, TestNamer())
    result = compiler.ast_to_object(node)

    self.assertEqual(test_fn([]), result.test_fn([]))
    self.assertEqual(test_fn([1]), result.test_fn([1]))
    self.assertEqual(test_fn([2]), result.test_fn([2]))
    self.assertEqual(test_fn([1, 2, 3]), result.test_fn([1, 2, 3]))

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

    node = self._parse_and_analyze(test_fn, {})
    node = continue_canonicalization.transform(node, TestNamer())
    result = compiler.ast_to_object(node)

    self.assertEqual(test_fn(0), result.test_fn(0))
    self.assertEqual(test_fn(1), result.test_fn(1))
    self.assertEqual(test_fn(2), result.test_fn(2))
    self.assertEqual(test_fn(3), result.test_fn(3))
    self.assertEqual(test_fn(4), result.test_fn(4))


if __name__ == '__main__':
  test.main()
