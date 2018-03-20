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
"""Tests for for_loops module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.py2tf.converters import converter_test_base
from tensorflow.contrib.py2tf.converters import for_loops
from tensorflow.python.platform import test


class ControlFlowTest(converter_test_base.TestCase):

  def test_basic_for(self):

    def test_fn(l):
      s = 0
      for e in l:
        s += e
      return s

    node = self.parse_and_analyze(test_fn, {})
    node = for_loops.transform(node, self.ctx)

    with self.compiled(node) as result:
      l = [1, 2, 3]
      self.assertEqual(test_fn(l), result.test_fn(l))
      l = []
      self.assertEqual(test_fn(l), result.test_fn(l))

  def test_for_with_iterated_expression(self):

    eval_count = [0]

    def count_evals(x):
      eval_count[0] += 1
      return x

    def test_fn(n):
      s = 0
      for e in count_evals(range(n)):
        s += e
      return s

    node = self.parse_and_analyze(test_fn, {'count_evals': count_evals})
    node = for_loops.transform(node, self.ctx)

    with self.compiled(node) as result:
      result.count_evals = count_evals
      self.assertEqual(test_fn(5), result.test_fn(5))
      # count_evals ran twice, once for test_fn and another for result.test_fn
      self.assertEqual(eval_count[0], 2)


if __name__ == '__main__':
  test.main()
