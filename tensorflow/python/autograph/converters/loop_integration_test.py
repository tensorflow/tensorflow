# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Integration Tests for loop."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.converters import break_statements
from tensorflow.python.autograph.converters import continue_statements
from tensorflow.python.autograph.converters import control_flow
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test


class LoopIntegrationTest(converter_testing.TestCase):

  def assertTransformedEquivalent(self, test_fn, *inputs):
    with self.converted(test_fn,
                        [break_statements, continue_statements, control_flow],
                        {}, (constant_op.constant,)) as result:
      self.assertEqual(test_fn(*inputs), result.test_fn(*inputs))

  def test_while_loop_with_else(self):

    def test_fn(x):
      while x > 2:
        x /= 2
      else:
        x += 1
      return x

    self.assertTransformedEquivalent(test_fn, 4)
    self.assertTransformedEquivalent(test_fn, 2)

  def test_while_loop_with_else_and_break(self):

    def test_fn(cond1):
      x = 8
      while x > 2:
        x /= 2
        if cond1:
          break
      else:
        x += 1
      return x

    self.assertTransformedEquivalent(test_fn, True)
    self.assertTransformedEquivalent(test_fn, False)

  def test_for_loop_with_else(self):

    def test_fn(l):
      res = 0
      for x in l:
        res += x
      else:
        res += 1
      return res

    self.assertTransformedEquivalent(test_fn, [])
    self.assertTransformedEquivalent(test_fn, [1, 2])

  def test_for_loop_with_else_and_break(self):

    def test_fn(flag):
      l = [1, 2, 3]
      res = 0
      for x in l:
        res += x
        if flag:
          break
      else:
        res += 1
      return res

    self.assertTransformedEquivalent(test_fn, True)
    self.assertTransformedEquivalent(test_fn, False)


if __name__ == '__main__':
  test.main()
