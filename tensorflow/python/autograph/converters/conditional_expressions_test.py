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
"""Tests for conditional_expressions module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.converters import conditional_expressions
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.platform import test


class ConditionalExpressionsTest(converter_testing.TestCase):

  def assertTransformedEquivalent(self, test_fn, *inputs):
    ns = {}
    with self.converted(test_fn, conditional_expressions, ns) as result:
      self.assertEqual(test_fn(*inputs), result.test_fn(*inputs))

  def test_basic(self):

    def test_fn(x):
      return 1 if x else 0

    self.assertTransformedEquivalent(test_fn, 0)
    self.assertTransformedEquivalent(test_fn, 3)

  def test_nested_orelse(self):

    def test_fn(x):
      y = x * x if x > 0 else x if x else 1
      return y

    self.assertTransformedEquivalent(test_fn, -2)
    self.assertTransformedEquivalent(test_fn, 0)
    self.assertTransformedEquivalent(test_fn, 2)


if __name__ == '__main__':
  test.main()
