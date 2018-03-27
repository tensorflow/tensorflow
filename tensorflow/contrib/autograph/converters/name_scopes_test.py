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
"""Tests for for_canonicalization module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.autograph.converters import converter_test_base
from tensorflow.contrib.autograph.converters import name_scopes
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


class FunctionNameScopeTransformer(converter_test_base.TestCase):

  def test_basic_name(self):

    def test_fn(l):
      a = 5
      l += a
      return l

    node = self.parse_and_analyze(test_fn, {})
    node = name_scopes.transform(node, self.ctx)

    with self.compiled(node, ops.name_scope) as result:
      result_op = result.test_fn(constant_op.constant([1, 2, 3]))
      self.assertIn('test_fn/', result_op.op.name)

  def test_nested_name(self):

    def test_fn(l):

      def body(i):
        return i**2

      l += [4]
      return body(l)

    node = self.parse_and_analyze(test_fn, {})
    node = name_scopes.transform(node, self.ctx)

    with self.compiled(node, ops.name_scope) as result:
      result_op = result.test_fn(constant_op.constant([1, 2, 3]))
      first_result_input_name = result_op.op.inputs[0].name
      second_result_input_name = result_op.op.inputs[1].name
      self.assertIn('test_fn/', first_result_input_name)
      self.assertNotIn('body/', first_result_input_name)
      self.assertIn('test_fn/body/', second_result_input_name)

  def test_class_name(self):

    class TestClass(object):

      def test_fn(self, l):

        def body(i):
          return i**2

        l += [4]
        return body(l)

    # Note that 'TestClass' was needed in the namespace here.
    node = self.parse_and_analyze(
        TestClass, {'TestClass': TestClass}, owner_type=TestClass)
    node = name_scopes.transform(node, self.ctx)

    with self.compiled(node, ops.name_scope) as result:
      result_op = result.TestClass().test_fn(constant_op.constant([1, 2, 3]))
      first_result_input_name = result_op.op.inputs[0].name
      second_result_input_name = result_op.op.inputs[1].name
      self.assertIn('TestClass/test_fn/', first_result_input_name)
      self.assertNotIn('body/', first_result_input_name)
      self.assertIn('TestClass/test_fn/body/', second_result_input_name)


if __name__ == '__main__':
  test.main()
