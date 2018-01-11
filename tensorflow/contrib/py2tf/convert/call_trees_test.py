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
"""Tests for call_trees module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.py2tf.convert import call_trees
from tensorflow.contrib.py2tf.pyct import compiler
from tensorflow.contrib.py2tf.pyct import parser
from tensorflow.contrib.py2tf.pyct.static_analysis import access
from tensorflow.contrib.py2tf.pyct.static_analysis import live_values
from tensorflow.contrib.py2tf.pyct.static_analysis import type_info
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class TestNamer(call_trees.FunctionNamer):

  def compiled_function_name(self, original_name, live_object=None):
    return 'renamed_%s' % original_name


class CallTreesTest(test.TestCase):

  def _parse_and_analyze(self, test_fn, namespace):
    node = parser.parse_object(test_fn)
    node = access.resolve(node)
    node = live_values.resolve(node, namespace, {})
    node = type_info.resolve(node, None)
    return node

  def test_basic(self):

    def test_fn_1(_):
      raise ValueError('This should not be called in the compiled verison.')

    def renamed_test_fn_1(a):
      return a + 1

    def test_fn_2(a):
      return test_fn_1(a) + 1

    node = self._parse_and_analyze(test_fn_2, {'test_fn_1': test_fn_1})
    node = call_trees.transform(node, TestNamer(), set())
    result = compiler.ast_to_object(node)
    # Only test_fn_2 is transformed, so we'll insert renamed_test_fn_1 manually.
    setattr(result, 'renamed_test_fn_1', renamed_test_fn_1)

    self.assertEquals(3, result.renamed_test_fn_2(1))

  def test_uncompiled_modules(self):

    def test_fn(a):
      a = math_ops.multiply(a, constant_op.constant(2))
      a = math_ops.add(a, constant_op.constant(1))
      return a

    node = self._parse_and_analyze(test_fn, {
        'math_ops': math_ops,
        'constant_op': constant_op
    })
    node = call_trees.transform(node, TestNamer(),
                                set(((math_ops.__name__,),
                                     (constant_op.__name__,))))
    result = compiler.ast_to_object(node)
    setattr(result, 'math_ops', math_ops)
    setattr(result, 'constant_op', constant_op)

    with self.test_session() as sess:
      result_tensor = result.renamed_test_fn(constant_op.constant(1))
      result_val = sess.run(result_tensor)

    self.assertEquals(3, result_val)


if __name__ == '__main__':
  test.main()
