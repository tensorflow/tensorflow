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
"""Tests for templates module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.autograph.pyct import anno
from tensorflow.contrib.autograph.pyct import context
from tensorflow.contrib.autograph.pyct import parser
from tensorflow.contrib.autograph.pyct import transformer
from tensorflow.python.platform import test


class TransformerTest(test.TestCase):

  def test_entity_scope_tracking(self):

    class TestTransformer(transformer.Base):

      # The choice of note to assign to is arbitrary. Using Assign because it's
      # easy to find in the tree.
      def visit_Assign(self, node):
        anno.setanno(node, 'enclosing_entities', self.enclosing_entities)
        return self.generic_visit(node)

      # This will show up in the lambda function.
      def visit_BinOp(self, node):
        anno.setanno(node, 'enclosing_entities', self.enclosing_entities)
        return self.generic_visit(node)

    tr = TestTransformer(
        context.EntityContext(
            namer=None,
            source_code=None,
            source_file=None,
            namespace=None,
            arg_values=None,
            arg_types=None,
            owner_type=None,
            recursive=False))

    def test_function():
      a = 0

      class TestClass(object):

        def test_method(self):
          b = 0
          def inner_function(x):
            c = 0
            d = lambda y: (x + y)
            return c, d
          return b, inner_function
      return a, TestClass

    node, _ = parser.parse_entity(test_function)
    node = tr.visit(node)

    test_function_node = node.body[0]
    test_class = test_function_node.body[1]
    test_method = test_class.body[0]
    inner_function = test_method.body[1]
    lambda_node = inner_function.body[1].value

    a = test_function_node.body[0]
    b = test_method.body[0]
    c = inner_function.body[0]
    lambda_expr = lambda_node.body

    self.assertEqual(
        (test_function_node,), anno.getanno(a, 'enclosing_entities'))
    self.assertEqual((test_function_node, test_class, test_method),
                     anno.getanno(b, 'enclosing_entities'))
    self.assertEqual(
        (test_function_node, test_class, test_method, inner_function),
        anno.getanno(c, 'enclosing_entities'))
    self.assertEqual((test_function_node, test_class, test_method,
                      inner_function, lambda_node),
                     anno.getanno(lambda_expr, 'enclosing_entities'))


if __name__ == '__main__':
  test.main()
