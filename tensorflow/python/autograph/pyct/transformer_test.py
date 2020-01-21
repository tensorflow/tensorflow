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

import gast

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.platform import test


class TransformerTest(test.TestCase):

  def _simple_context(self):
    entity_info = transformer.EntityInfo(
        source_code=None, source_file=None, future_features=(), namespace=None)
    return transformer.Context(entity_info)

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

    tr = TestTransformer(self._simple_context())

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

    node, _ = parser.parse_entity(test_function, future_features=())
    node = tr.visit(node)

    test_function_node = node
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

  def assertSameAnno(self, first, second, key):
    self.assertIs(anno.getanno(first, key), anno.getanno(second, key))

  def assertDifferentAnno(self, first, second, key):
    self.assertIsNot(anno.getanno(first, key), anno.getanno(second, key))

  def test_state_tracking(self):

    class LoopState(object):
      pass

    class CondState(object):
      pass

    class TestTransformer(transformer.Base):

      def visit(self, node):
        anno.setanno(node, 'loop_state', self.state[LoopState].value)
        anno.setanno(node, 'cond_state', self.state[CondState].value)
        return super(TestTransformer, self).visit(node)

      def visit_While(self, node):
        self.state[LoopState].enter()
        node = self.generic_visit(node)
        self.state[LoopState].exit()
        return node

      def visit_If(self, node):
        self.state[CondState].enter()
        node = self.generic_visit(node)
        self.state[CondState].exit()
        return node

    tr = TestTransformer(self._simple_context())

    def test_function(a):
      a = 1
      while a:
        _ = 'a'
        if a > 2:
          _ = 'b'
          while True:
            raise '1'
        if a > 3:
          _ = 'c'
          while True:
            raise '1'

    node, _ = parser.parse_entity(test_function, future_features=())
    node = tr.visit(node)

    fn_body = node.body
    outer_while_body = fn_body[1].body
    self.assertSameAnno(fn_body[0], outer_while_body[0], 'cond_state')
    self.assertDifferentAnno(fn_body[0], outer_while_body[0], 'loop_state')

    first_if_body = outer_while_body[1].body
    self.assertDifferentAnno(outer_while_body[0], first_if_body[0],
                             'cond_state')
    self.assertSameAnno(outer_while_body[0], first_if_body[0], 'loop_state')

    first_inner_while_body = first_if_body[1].body
    self.assertSameAnno(first_if_body[0], first_inner_while_body[0],
                        'cond_state')
    self.assertDifferentAnno(first_if_body[0], first_inner_while_body[0],
                             'loop_state')

    second_if_body = outer_while_body[2].body
    self.assertDifferentAnno(first_if_body[0], second_if_body[0], 'cond_state')
    self.assertSameAnno(first_if_body[0], second_if_body[0], 'loop_state')

    second_inner_while_body = second_if_body[1].body
    self.assertDifferentAnno(first_inner_while_body[0],
                             second_inner_while_body[0], 'cond_state')
    self.assertDifferentAnno(first_inner_while_body[0],
                             second_inner_while_body[0], 'loop_state')

  def test_state_tracking_context_manager(self):

    class CondState(object):
      pass

    class TestTransformer(transformer.Base):

      def visit(self, node):
        anno.setanno(node, 'cond_state', self.state[CondState].value)
        return super(TestTransformer, self).visit(node)

      def visit_If(self, node):
        with self.state[CondState]:
          return self.generic_visit(node)

    tr = TestTransformer(self._simple_context())

    def test_function(a):
      a = 1
      if a > 2:
        _ = 'b'
        if a < 5:
          _ = 'c'
        _ = 'd'

    node, _ = parser.parse_entity(test_function, future_features=())
    node = tr.visit(node)

    fn_body = node.body
    outer_if_body = fn_body[1].body
    self.assertDifferentAnno(fn_body[0], outer_if_body[0], 'cond_state')
    self.assertSameAnno(outer_if_body[0], outer_if_body[2], 'cond_state')

    inner_if_body = outer_if_body[1].body
    self.assertDifferentAnno(inner_if_body[0], outer_if_body[0], 'cond_state')

  def test_local_scope_info_stack(self):

    class TestTransformer(transformer.Base):

      # Extract all string constants from the block.
      def visit_Constant(self, node):
        self.set_local(
            'string', self.get_local('string', default='') + str(node.value))
        return self.generic_visit(node)

      def _annotate_result(self, node):
        self.enter_local_scope()
        node = self.generic_visit(node)
        anno.setanno(node, 'test', self.get_local('string'))
        self.exit_local_scope()
        return node

      def visit_While(self, node):
        return self._annotate_result(node)

      def visit_For(self, node):
        return self._annotate_result(node)

    tr = TestTransformer(self._simple_context())

    def test_function(a):
      """Docstring."""
      assert a == 'This should not be counted'
      for i in range(3):
        _ = 'a'
        if i > 2:
          return 'b'
        else:
          _ = 'c'
          while 4:
            raise '1'
      return 'nor this'

    node, _ = parser.parse_entity(test_function, future_features=())
    node = tr.visit(node)

    for_node = node.body[2]
    while_node = for_node.body[1].orelse[1]

    self.assertFalse(anno.hasanno(for_node, 'string'))
    self.assertEqual('3a2bc', anno.getanno(for_node, 'test'))
    self.assertFalse(anno.hasanno(while_node, 'string'))
    self.assertEqual('41', anno.getanno(while_node, 'test'))

  def test_local_scope_info_stack_checks_integrity(self):

    class TestTransformer(transformer.Base):

      def visit_If(self, node):
        self.enter_local_scope()
        return self.generic_visit(node)

      def visit_For(self, node):
        node = self.generic_visit(node)
        self.exit_local_scope()
        return node

    tr = TestTransformer(self._simple_context())

    def no_exit(a):
      if a > 0:
        print(a)
      return None

    node, _ = parser.parse_entity(no_exit, future_features=())
    with self.assertRaises(AssertionError):
      tr.visit(node)

    def no_entry(a):
      for _ in a:
        print(a)

    node, _ = parser.parse_entity(no_entry, future_features=())
    with self.assertRaises(AssertionError):
      tr.visit(node)

  def test_visit_block_postprocessing(self):

    class TestTransformer(transformer.Base):

      def _process_body_item(self, node):
        if isinstance(node, gast.Assign) and (node.value.id == 'y'):
          if_node = gast.If(
              gast.Name(
                  'x', ctx=gast.Load(), annotation=None, type_comment=None),
              [node], [])
          return if_node, if_node.body
        return node, None

      def visit_FunctionDef(self, node):
        node.body = self.visit_block(
            node.body, after_visit=self._process_body_item)
        return node

    def test_function(x, y):
      z = x
      z = y
      return z

    tr = TestTransformer(self._simple_context())

    node, _ = parser.parse_entity(test_function, future_features=())
    node = tr.visit(node)

    self.assertEqual(len(node.body), 2)
    self.assertTrue(isinstance(node.body[0], gast.Assign))
    self.assertTrue(isinstance(node.body[1], gast.If))
    self.assertTrue(isinstance(node.body[1].body[0], gast.Assign))
    self.assertTrue(isinstance(node.body[1].body[1], gast.Return))

  def test_robust_error_on_list_visit(self):

    class BrokenTransformer(transformer.Base):

      def visit_If(self, node):
        # This is broken because visit expects a single node, not a list, and
        # the body of an if is a list.
        # Importantly, the default error handling in visit also expects a single
        # node.  Therefore, mistakes like this need to trigger a type error
        # before the visit called here installs its error handler.
        # That type error can then be caught by the enclosing call to visit,
        # and correctly blame the If node.
        self.visit(node.body)
        return node

    def test_function(x):
      if x > 0:
        return x

    tr = BrokenTransformer(self._simple_context())

    node, _ = parser.parse_entity(test_function, future_features=())
    with self.assertRaises(ValueError) as cm:
      node = tr.visit(node)
    obtained_message = str(cm.exception)
    expected_message = r'expected "ast.AST", got "\<(type|class) \'list\'\>"'
    self.assertRegexpMatches(obtained_message, expected_message)

  def test_robust_error_on_ast_corruption(self):
    # A child class should not be able to be so broken that it causes the error
    # handling in `transformer.Base` to raise an exception.  Why not?  Because
    # then the original error location is dropped, and an error handler higher
    # up in the call stack gives misleading information.

    # Here we test that the error handling in `visit` completes, and blames the
    # correct original exception, even if the AST gets corrupted.

    class NotANode(object):
      pass

    class BrokenTransformer(transformer.Base):

      def visit_If(self, node):
        node.body = NotANode()
        raise ValueError('I blew up')

    def test_function(x):
      if x > 0:
        return x

    tr = BrokenTransformer(self._simple_context())

    node, _ = parser.parse_entity(test_function, future_features=())
    with self.assertRaises(ValueError) as cm:
      node = tr.visit(node)
    obtained_message = str(cm.exception)
    # The message should reference the exception actually raised, not anything
    # from the exception handler.
    expected_substring = 'I blew up'
    self.assertTrue(expected_substring in obtained_message, obtained_message)

  def test_origin_info_propagated_to_new_nodes(self):

    class TestTransformer(transformer.Base):

      def visit_If(self, node):
        return gast.Pass()

    tr = TestTransformer(self._simple_context())

    def test_fn():
      x = 1
      if x > 0:
        x = 1
      return x

    node, source = parser.parse_entity(test_fn, future_features=())
    origin_info.resolve(node, source, 'test_file', 100, 0)
    node = tr.visit(node)

    created_pass_node = node.body[1]
    # Takes the line number of the if statement.
    self.assertEqual(
        anno.getanno(created_pass_node, anno.Basic.ORIGIN).loc.lineno, 102)

  def test_origin_info_preserved_in_moved_nodes(self):

    class TestTransformer(transformer.Base):

      def visit_If(self, node):
        return node.body

    tr = TestTransformer(self._simple_context())

    def test_fn():
      x = 1
      if x > 0:
        x = 1
        x += 3
      return x

    node, source = parser.parse_entity(test_fn, future_features=())
    origin_info.resolve(node, source, 'test_file', 100, 0)
    node = tr.visit(node)

    assign_node = node.body[1]
    aug_assign_node = node.body[2]
    # Keep their original line numbers.
    self.assertEqual(
        anno.getanno(assign_node, anno.Basic.ORIGIN).loc.lineno, 103)
    self.assertEqual(
        anno.getanno(aug_assign_node, anno.Basic.ORIGIN).loc.lineno, 104)


if __name__ == '__main__':
  test.main()
