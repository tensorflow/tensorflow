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

from tensorflow.contrib.py2tf.convert import builtin_functions
from tensorflow.contrib.py2tf.pyct import compiler
from tensorflow.contrib.py2tf.pyct import parser
from tensorflow.contrib.py2tf.pyct.static_analysis import access
from tensorflow.contrib.py2tf.pyct.static_analysis import live_values
from tensorflow.contrib.py2tf.pyct.static_analysis import type_info
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class BuiltinFunctionsTest(test.TestCase):

  def _parse_and_analyze(self, test_fn, namespace):
    node = parser.parse_object(test_fn)
    node = access.resolve(node)
    node = live_values.resolve(node, namespace, {})
    node = type_info.resolve(node, {})
    return node

  def test_len(self):

    def test_fn(a):
      return len(a)

    node = self._parse_and_analyze(test_fn, {'len': len})
    node = builtin_functions.transform(node)
    result = compiler.ast_to_object(node)
    setattr(result, 'tf', array_ops)

    with self.test_session() as sess:
      self.assertEqual(3,
                       sess.run(
                           result.test_fn(constant_op.constant([0, 0, 0]))))


if __name__ == '__main__':
  test.main()
