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
"""Tests for list_comprehension module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.autograph.converters import converter_test_base
from tensorflow.contrib.autograph.converters import list_comprehension
from tensorflow.python.platform import test


class ListCompTest(converter_test_base.TestCase):

  def test_basic(self):

    def test_fn(l):
      s = [e * e for e in l]
      return s

    node = self.parse_and_analyze(test_fn, {})
    node = list_comprehension.transform(node, self.ctx)

    with self.compiled(node) as result:
      l = [1, 2, 3]
      self.assertEqual(test_fn(l), result.test_fn(l))
      l = []
      self.assertEqual(test_fn(l), result.test_fn(l))

  def test_multiple_generators(self):

    def test_fn(l):
      s = [e * e for sublist in l for e in sublist]
      return s

    node = self.parse_and_analyze(test_fn, {})
    node = list_comprehension.transform(node, self.ctx)

    with self.compiled(node) as result:
      l = [[1], [2], [3]]
      self.assertEqual(test_fn(l), result.test_fn(l))
      l = []
      self.assertEqual(test_fn(l), result.test_fn(l))

  def test_conds(self):

    def test_fn(l):
      s = [e * e for e in l if e > 1]
      return s

    node = self.parse_and_analyze(test_fn, {})
    node = list_comprehension.transform(node, self.ctx)

    with self.compiled(node) as result:
      l = [1, 2, 3]
      self.assertEqual(test_fn(l), result.test_fn(l))
      l = []
      self.assertEqual(test_fn(l), result.test_fn(l))


if __name__ == '__main__':
  test.main()
