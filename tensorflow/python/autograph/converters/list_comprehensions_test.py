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
"""Tests for list_comprehensions module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.converters import list_comprehensions
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.platform import test


class ListCompTest(converter_testing.TestCase):

  def assertTransformedEquivalent(self, test_fn, *inputs):
    with self.converted(test_fn, list_comprehensions, {}) as result:
      self.assertEqual(test_fn(*inputs), result.test_fn(*inputs))

  def test_basic(self):

    def test_fn(l):
      s = [e * e for e in l]
      return s

    self.assertTransformedEquivalent(test_fn, [])
    self.assertTransformedEquivalent(test_fn, [1, 2, 3])

  def test_multiple_generators(self):

    def test_fn(l):
      s = [e * e for sublist in l for e in sublist]
      return s

    self.assertTransformedEquivalent(test_fn, [])
    self.assertTransformedEquivalent(test_fn, [[1], [2], [3]])

  def test_cond(self):

    def test_fn(l):
      s = [e * e for e in l if e > 1]
      return s

    self.assertTransformedEquivalent(test_fn, [])
    self.assertTransformedEquivalent(test_fn, [1, 2, 3])


if __name__ == '__main__':
  test.main()
