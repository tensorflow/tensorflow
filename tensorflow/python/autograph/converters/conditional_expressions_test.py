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

  def assertTransformedEquivalent(self, f, *inputs):
    tr = self.transform(f, conditional_expressions)
    self.assertEqual(f(*inputs), tr(*inputs))

  def test_basic(self):

    def f(x):
      return 1 if x else 0

    self.assertTransformedEquivalent(f, 0)
    self.assertTransformedEquivalent(f, 3)

  def test_nested_orelse(self):

    def f(x):
      y = x * x if x > 0 else x if x else 1
      return y

    self.assertTransformedEquivalent(f, -2)
    self.assertTransformedEquivalent(f, 0)
    self.assertTransformedEquivalent(f, 2)


if __name__ == '__main__':
  test.main()
