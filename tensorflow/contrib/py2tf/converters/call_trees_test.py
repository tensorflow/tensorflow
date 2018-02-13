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

from tensorflow.contrib.py2tf.converters import call_trees
from tensorflow.contrib.py2tf.converters import converter_test_base
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class CallTreesTest(converter_test_base.TestCase):

  def test_basic(self):

    def test_fn_1(_):
      raise ValueError('This should not be called in the compiled verison.')

    def renamed_test_fn_1(a):
      return a + 1

    def test_fn_2(a):
      return test_fn_1(a) + 1

    node = self.parse_and_analyze(test_fn_2, {'test_fn_1': test_fn_1})
    node = call_trees.transform(node, self.ctx, (), ())

    with self.compiled(node) as result:
      # Only test_fn_2 is transformed, so we'll insert renamed_test_fn_1
      # manually.
      result.renamed_test_fn_1 = renamed_test_fn_1
      self.assertEquals(3, result.test_fn_2(1))

  def test_simple_methods(self):

    class TestClass(object):

      def test_fn_1(self, a):
        return a + 1

      def test_fn_2(self, a):
        return self.test_fn_1(a) + 1

    node = self.parse_and_analyze(
        TestClass.test_fn_2, {'TestClass': TestClass},
        arg_types={'self': (TestClass.__name__, TestClass)})
    node = call_trees.transform(node, self.ctx, (), ())

    with self.compiled(node) as result:
      tc = TestClass()
      self.assertEquals(3, result.test_fn_2(tc, 1))

  def test_uncompiled_modules(self):

    def test_fn(a):
      a = math_ops.multiply(a, constant_op.constant(2))
      a = math_ops.add(a, constant_op.constant(1))
      return a

    node = self.parse_and_analyze(test_fn, {
        'math_ops': math_ops,
        'constant_op': constant_op
    })
    node = call_trees.transform(node, self.ctx,
                                set(((math_ops.__name__,),
                                     (constant_op.__name__,))), ())

    with self.compiled(node) as result:
      result.math_ops = math_ops
      result.constant_op = constant_op
      with self.test_session() as sess:
        # Not renamed, because the converter doesn't rename the definition
        # itself (the caller is responsible for that).
        result_tensor = result.test_fn(constant_op.constant(1))
        self.assertEquals(3, sess.run(result_tensor))


if __name__ == '__main__':
  test.main()
