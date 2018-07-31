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
"""Tests for continue_statements module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.autograph.converters import continue_statements
from tensorflow.contrib.autograph.core import converter_testing
from tensorflow.python.platform import test


class ContinueCanonicalizationTest(converter_testing.TestCase):

  def assertTransformedEquivalent(self, test_fn, *inputs):
    with self.converted(test_fn, continue_statements, {}) as result:
      self.assertEqual(test_fn(*inputs), result.test_fn(*inputs))

  def test_basic(self):

    def test_fn(x):
      v = []
      while x > 0:
        x -= 1
        if x % 2 == 0:
          continue
        v.append(x)
      return v

    self.assertTransformedEquivalent(test_fn, 0)
    self.assertTransformedEquivalent(test_fn, 1)
    self.assertTransformedEquivalent(test_fn, 3)
    self.assertTransformedEquivalent(test_fn, 4)

  def test_for_loop(self):

    def test_fn(a):
      v = []
      for x in a:
        x -= 1
        if x % 2 == 0:
          continue
        v.append(x)
      return v

    self.assertTransformedEquivalent(test_fn, [])
    self.assertTransformedEquivalent(test_fn, [1])
    self.assertTransformedEquivalent(test_fn, [2])
    self.assertTransformedEquivalent(test_fn, [1, 2, 3])

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
            continue
        v.append(x)
      return v, u, w

    self.assertTransformedEquivalent(test_fn, 0)
    self.assertTransformedEquivalent(test_fn, 1)
    self.assertTransformedEquivalent(test_fn, 3)
    self.assertTransformedEquivalent(test_fn, 4)


if __name__ == '__main__':
  test.main()
