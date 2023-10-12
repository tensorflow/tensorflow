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
"""Loop control statements (e.g. break, return) in illegal patterns.

Meant to verify that:
  * break/return on a dynamic condition raises error inside Python loop
"""

import itertools
import re

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


def tf_break_in_py_for(l):
  s = 0
  for c in l:
    if tf.greater(c % 2, 0):
      break
    s += c
  return s


def tf_return_in_py_for(l):
  s = 0
  for c in l:
    if tf.greater(c % 2, 0):
      return s
    else:
      return s
    s += c
  return s


def tf_break_in_py_while(x):
  s = 0
  while x > 0:
    x -= 1
    if tf.greater(x % 2, 0):
      break
    s += x
  return s


def tf_return_in_py_while(x):
  s = 0
  while x > 0:
    x -= 1
    if tf.greater(x % 2, 0):
      return s
    else:
      return s
    s += x
  return s


class LoopControlFlowIllegalCasesTest(reference_test_base.TestCase,
                                      parameterized.TestCase):

  @parameterized.parameters(*itertools.product(
      (
          [1],
          [1, 2],
          [1, 2, 3],
      ),
      (
          tf_break_in_py_for,
          tf_return_in_py_for,
      ),
  ))
  def test_tf_control_flow_in_py_for(self, l, target):
    with self.assertRaisesRegex(NotImplementedError,
                                'not supported in Python for'):
      tf.function(target)(l)

  @parameterized.parameters(*itertools.product(
      (
          1,
          2,
          3,
      ),
      (
          tf_break_in_py_while,
          tf_return_in_py_while,
      ),
  ))
  def test_tf_control_flow_in_py_while(self, n, target):
    with self.assertRaisesRegex(
        NotImplementedError,
        re.compile(
            r'.*condition of while loop started as non\-Tensor,'
            r' then changed to Tensor.*', re.DOTALL)):
      tf.function(target)(n)


if __name__ == '__main__':
  tf.test.main()
