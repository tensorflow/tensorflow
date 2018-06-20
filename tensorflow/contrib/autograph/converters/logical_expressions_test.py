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
"""Tests for logical_expressions module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.autograph.converters import converter_test_base
from tensorflow.contrib.autograph.converters import logical_expressions
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class GradientsFunctionTest(converter_test_base.TestCase):

  def test_equals(self):

    def test_fn(a, b):
      return a == b

    node = self.parse_and_analyze(test_fn, {})
    node = logical_expressions.transform(node, self.ctx)

    with self.compiled(node, math_ops.equal) as result:
      with self.test_session() as sess:
        self.assertTrue(sess.run(result.test_fn(1, 1)))
        self.assertFalse(sess.run(result.test_fn(1, 2)))

  def test_bool_ops(self):

    def test_fn(a, b, c):
      return (a or b) and (a or b or c)

    node = self.parse_and_analyze(test_fn, {})
    node = logical_expressions.transform(node, self.ctx)

    with self.compiled(node, math_ops.logical_or,
                       math_ops.logical_and) as result:
      with self.test_session() as sess:
        self.assertTrue(sess.run(result.test_fn(True, False, True)))


if __name__ == '__main__':
  test.main()
