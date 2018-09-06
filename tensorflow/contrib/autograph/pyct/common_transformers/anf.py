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
"""Conversion to A-normal form.

The general idea of A-normal form is that every intermediate value is
explicitly named with a variable.  For more, see
https://en.wikipedia.org/wiki/A-normal_form.

The specific converters used here are based on Python AST semantics as
documented at https://greentreesnakes.readthedocs.io/en/latest/.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast
import six

from tensorflow.contrib.autograph.pyct import templates
from tensorflow.contrib.autograph.pyct import transformer


class DummyGensym(object):
  """A dumb gensym that suffixes a stem by sequential numbers from 1000."""

  def __init__(self, entity_info):
    del entity_info
    # A proper implementation needs to account for:
    #   * entity_info.namespace
    #   * all the symbols defined in the AST
    #   * the symbols generated so far
    self._idx = 0

  def new_name(self, stem='tmp'):
    self._idx += 1
    return stem + '_' + str(1000 + self._idx)


class AnfTransformer(transformer.Base):
  """Performs the conversion to A-normal form (ANF)."""

  # The algorithm is a postorder recursive tree walk.  Any given node A may, in
  # general, require creation of a series B of Assign statements, which compute
  # and explicitly name the intermediate values needed to compute the value of
  # A.  If A was already a statement, it can be replaced with the sequence B +
  # [A].  If A was an expression, B needs to be propagated up the tree until a
  # statement is encountered.  Since the `ast.NodeTransformer` framework makes
  # no provision for subtraversals returning side information, this class
  # accumulates the sequence B in an instance variable.

  # The only other subtlety is that some Python statements (like `if`) have both
  # expression fields (`test`) and statement list fields (`body` and `orelse`).
  # Any additional assignments needed to name all the intermediate values in the
  # `test` can be prepended to the `if` node, but assignments produced by
  # processing the `body` and the `orelse` need to be kept together with them,
  # and not accidentally lifted out of the `if`.

  def __init__(self, entity_info, gensym_source=None):
    """Creates an ANF transformer.

    Args:
      entity_info: transformer.EntityInfo
      gensym_source: An optional object with the same interface as `DummyGensym`
        for generating unique names
    """
    super(AnfTransformer, self).__init__(entity_info)
    if gensym_source is None:
      self._gensym = DummyGensym(entity_info)
    else:
      self._gensym = gensym_source(entity_info)
    self._pending_statements = []

  def _consume_pending_statements(self):
    ans = self._pending_statements
    self._pending_statements = []
    return ans

  def _add_pending_statement(self, stmt):
    self._pending_statements.append(stmt)

  _trivial_nodes = (
      # Non-nodes that show up as AST fields
      bool, six.string_types,
      # Leaf nodes that are already in A-normal form
      gast.expr_context, gast.Name, gast.Num, gast.Str, gast.Bytes,
      gast.NameConstant, gast.Ellipsis,
      # Binary operators
      gast.Add, gast.Sub, gast.Mult, gast.Div, gast.Mod, gast.Pow, gast.LShift,
      gast.RShift, gast.BitOr, gast.BitXor, gast.BitAnd, gast.FloorDiv,
      # Unary operators
      gast.Invert, gast.Not, gast.UAdd, gast.USub,
      # Comparison operators
      gast.Eq, gast.NotEq, gast.Lt, gast.LtE, gast.Gt, gast.GtE,
      gast.Is, gast.IsNot, gast.In, gast.NotIn,
  )

  def _is_node_trivial(self, node):
    if node is None:
      return True
    elif isinstance(node, self._trivial_nodes):
      return True
    elif isinstance(node, gast.keyword):
      return self._is_node_trivial(node.value)
    elif isinstance(node, (gast.Starred, gast.withitem, gast.slice)):
      return self._are_children_trivial(node)
    return False

  def _are_children_trivial(self, node):
    for field in node._fields:
      if not field.startswith('__'):
        if not self._is_node_trivial(getattr(node, field)):
          return False
    return True

  def _ensure_node_is_trivial(self, node):
    if node is None:
      return node
    elif isinstance(node, self._trivial_nodes):
      return node
    elif isinstance(node, list):
      # If something's field was actually a list, e.g., variadic arguments.
      return [self._ensure_node_is_trivial(n) for n in node]
    elif isinstance(node, gast.keyword):
      node.value = self._ensure_node_is_trivial(node.value)
      return node
    elif isinstance(node, (gast.Starred, gast.withitem, gast.slice)):
      return self._ensure_fields_trivial(node)
    elif isinstance(node, gast.expr):
      temp_name = self._gensym.new_name()
      temp_assign = templates.replace(
          'temp_name = expr', temp_name=temp_name, expr=node)[0]
      self._add_pending_statement(temp_assign)
      answer = templates.replace('temp_name', temp_name=temp_name)[0]
      return answer
    else:
      raise ValueError('Do not know how to treat {}'.format(node))

  def _ensure_fields_trivial(self, node):
    for field in node._fields:
      if field.startswith('__'):
        continue
      setattr(node, field, self._ensure_node_is_trivial(getattr(node, field)))
    return node

  def _visit_strict_statement(self, node, trivialize_children=True):
    assert not self._pending_statements
    node = self.generic_visit(node)
    if trivialize_children:
      self._ensure_fields_trivial(node)
    results = self._consume_pending_statements()
    results.append(node)
    return results

  def _visit_strict_expression(self, node):
    node = self.generic_visit(node)
    self._ensure_fields_trivial(node)
    return node

  # Note on code order: These are listed in the same order as the grammar
  # elements on https://github.com/serge-sans-paille/gast

  # FunctionDef, AsyncFunctionDef, and ClassDef should be correct by default.

  def visit_Return(self, node):
    return self._visit_strict_statement(node)

  def visit_Delete(self, node):
    return self._visit_strict_statement(node, trivialize_children=False)

  def visit_Assign(self, node):
    return self._visit_strict_statement(node, trivialize_children=False)

  def visit_AugAssign(self, node):
    return self._visit_strict_statement(node, trivialize_children=False)

  def visit_Print(self, node):
    return self._visit_strict_statement(node)

  def visit_For(self, node):
    assert not self._pending_statements
    # It's important to visit node.iter first, because any statements created
    # thereby need to live outside the body.
    self.visit(node.iter)
    node.iter = self._ensure_node_is_trivial(node.iter)
    iter_stmts = self._consume_pending_statements()
    # This generic_visit will revisit node.iter, but that is both correct and
    # cheap because by this point node.iter is trivial.
    node = self.generic_visit(node)
    assert not self._pending_statements
    iter_stmts.append(node)
    return iter_stmts

  def visit_AsyncFor(self, node):
    if not self._are_children_trivial(node):
      msg = ('Nontrivial AsyncFor nodes not supported yet '
             '(need to think through the semantics).')
      raise ValueError(msg)
    return self.generic_visit(node)

  def visit_While(self, node):
    if not self._is_node_trivial(node.test):
      msg = ('While with nontrivial test not supported yet '
             '(need to avoid precomputing the test).')
      raise ValueError(msg)
    return self.generic_visit(node)

  def visit_If(self, node):
    assert not self._pending_statements
    # It's important to visit node.test first, because any statements created
    # thereby need to live outside the body.
    self.visit(node.test)
    node.test = self._ensure_node_is_trivial(node.test)
    condition_stmts = self._consume_pending_statements()
    # This generic_visit will revisit node.test, but that is both correct and
    # cheap because by this point node.test is trivial.
    node = self.generic_visit(node)
    assert not self._pending_statements
    condition_stmts.append(node)
    return condition_stmts

  def visit_With(self, node):
    assert not self._pending_statements
    # It's important to visit node.items first, because any statements created
    # thereby need to live outside the body.
    for item in node.items:
      self.visit(item)
    node.items = [self._ensure_node_is_trivial(n) for n in node.items]
    contexts_stmts = self._consume_pending_statements()
    # This generic_visit will revisit node.items, but that is both correct and
    # cheap because by this point node.items is trivial.
    node = self.generic_visit(node)
    assert not self._pending_statements
    contexts_stmts.append(node)
    return contexts_stmts

  def visit_AsyncWith(self, node):
    if not self._are_children_trivial(node):
      msg = ('Nontrivial AsyncWith nodes not supported yet '
             '(need to think through the semantics).')
      raise ValueError(msg)
    return self.generic_visit(node)

  def visit_Raise(self, node):
    return self._visit_strict_statement(node)

  # Try should be correct by default.

  def visit_Assert(self, node):
    if not self._are_children_trivial(node):
      msg = ('Nontrivial Assert nodes not supported yet '
             '(need to avoid computing the test when assertions are off, and '
             'avoid computing the irritant when the assertion does not fire).')
      raise ValueError(msg)
    return self.generic_visit(node)

  # Import and ImportFrom should be correct by default.

  def visit_Exec(self, node):
    return self._visit_strict_statement(node)

  # Global and Nonlocal should be correct by default.

  def visit_Expr(self, node):
    return self._visit_strict_statement(node, trivialize_children=False)

  # Pass, Break, and Continue should be correct by default.

  def visit_BoolOp(self, node):
    if not self._are_children_trivial(node):
      msg = ('Nontrivial BoolOp nodes not supported yet '
             '(need to preserve short-circuiting semantics).')
      raise ValueError(msg)
    return self.generic_visit(node)

  def visit_BinOp(self, node):
    return self._visit_strict_expression(node)

  def visit_UnaryOp(self, node):
    return self._visit_strict_expression(node)

  def visit_Lambda(self, node):
    if not self._are_children_trivial(node):
      msg = ('Nontrivial Lambda nodes not supported '
             '(cannot insert statements into lambda bodies).')
      raise ValueError(msg)
    return self.generic_visit(node)

  def visit_IfExp(self, node):
    if not self._are_children_trivial(node):
      msg = ('Nontrivial IfExp nodes not supported yet '
             '(need to convert to If statement, to evaluate branches lazily '
             'and insert statements into them).')
      raise ValueError(msg)
    return self.generic_visit(node)

  def visit_Dict(self, node):
    return self._visit_strict_expression(node)

  def visit_Set(self, node):
    return self._visit_strict_expression(node)

  def visit_ListComp(self, node):
    msg = ('ListComp nodes not supported '
           '(need to convert to a form that tolerates '
           'assignment statements in clause bodies).')
    raise ValueError(msg)

  def visit_SetComp(self, node):
    msg = ('SetComp nodes not supported '
           '(need to convert to a form that tolerates '
           'assignment statements in clause bodies).')
    raise ValueError(msg)

  def visit_DictComp(self, node):
    msg = ('DictComp nodes not supported '
           '(need to convert to a form that tolerates '
           'assignment statements in clause bodies).')
    raise ValueError(msg)

  def visit_GeneratorExp(self, node):
    msg = ('GeneratorExp nodes not supported '
           '(need to convert to a form that tolerates '
           'assignment statements in clause bodies).')
    raise ValueError(msg)

  def visit_Await(self, node):
    if not self._are_children_trivial(node):
      msg = ('Nontrivial Await nodes not supported yet '
             '(need to think through the semantics).')
      raise ValueError(msg)
    return self.generic_visit(node)

  def visit_Yield(self, node):
    return self._visit_strict_expression(node)

  def visit_YieldFrom(self, node):
    if not self._are_children_trivial(node):
      msg = ('Nontrivial YieldFrom nodes not supported yet '
             '(need to unit-test them in Python 2).')
      raise ValueError(msg)
    return self.generic_visit(node)

  def visit_Compare(self, node):
    if len(node.ops) > 1:
      msg = ('Multi-ary compare nodes not supported yet '
             '(need to preserve short-circuiting semantics).')
      raise ValueError(msg)
    return self._visit_strict_expression(node)

  def visit_Call(self, node):
    return self._visit_strict_expression(node)

  def visit_Repr(self, node):
    if not self._are_children_trivial(node):
      msg = ('Nontrivial Repr nodes not supported yet '
             '(need to research their syntax and semantics).')
      raise ValueError(msg)
    return self.generic_visit(node)

  def visit_FormattedValue(self, node):
    if not self._are_children_trivial(node):
      msg = ('Nontrivial FormattedValue nodes not supported yet '
             '(need to unit-test them in Python 2).')
      raise ValueError(msg)
    return self.generic_visit(node)

  def visit_JoinedStr(self, node):
    if not self._are_children_trivial(node):
      msg = ('Nontrivial JoinedStr nodes not supported yet '
             '(need to unit-test them in Python 2).')
      raise ValueError(msg)
    return self.generic_visit(node)

  def visit_Attribute(self, node):
    return self._visit_strict_expression(node)

  def visit_Subscript(self, node):
    return self._visit_strict_expression(node)

  # Starred and Name are correct by default, because the right thing to do is to
  # just recur.

  def visit_List(self, node):
    node = self.generic_visit(node)
    if not isinstance(node.ctx, gast.Store):
      self._ensure_fields_trivial(node)
    return node

  def visit_Tuple(self, node):
    node = self.generic_visit(node)
    if not isinstance(node.ctx, gast.Store):
      self._ensure_fields_trivial(node)
    return node


def transform(node, entity_info, gensym_source=None):
  """Converts the given node to A-normal form (ANF).

  The general idea of A-normal form: https://en.wikipedia.org/wiki/A-normal_form

  The specific converters used here are based on Python AST semantics as
  documented at https://greentreesnakes.readthedocs.io/en/latest/.

  Args:
    node: The node to transform.
    entity_info: transformer.EntityInfo.  TODO(mdan): What information does this
      argument provide?
    gensym_source: An optional object with the same interface as `DummyGensym`
      for generating unique names.
  """
  return AnfTransformer(entity_info, gensym_source=gensym_source).visit(node)
