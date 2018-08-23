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
"""Tests for builtin_functions module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.contrib.autograph.converters import builtin_functions
from tensorflow.contrib.autograph.core import converter_testing
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class BuiltinFunctionsTest(converter_testing.TestCase):

  def test_len(self):

    def test_fn(a):
      return len(a)

    with self.converted(test_fn, builtin_functions, {'len': len},
                        array_ops.shape) as result:
      with self.cached_session() as sess:
        ops = result.test_fn(constant_op.constant([0, 0, 0]))
        self.assertEqual(sess.run(ops), 3)

  def test_print(self):

    if six.PY2:
      return

    def test_fn(a):
      return print(a)

    with self.converted(test_fn, builtin_functions, {'print': print}) as result:
      with self.cached_session() as sess:
        with self.assertPrints('a\n'):
          sess.run(result.test_fn('a'))

  def test_print_multiple_values(self):

    if six.PY2:
      return

    def test_fn(a, b, c):
      return print(a, b, c)

    with self.converted(test_fn, builtin_functions, {'print': print}) as result:
      with self.cached_session() as sess:
        with self.assertPrints('a 1 [2, 3]\n'):
          sess.run(
              result.test_fn(
                  constant_op.constant('a'), constant_op.constant(1), [2, 3]))


if __name__ == '__main__':
  test.main()
