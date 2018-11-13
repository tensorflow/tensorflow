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
"""AST conversion templates.

Adapted from Tangent.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import textwrap

import gast

from tensorflow.contrib.autograph.pyct import anno
from tensorflow.contrib.autograph.pyct import ast_util
from tensorflow.contrib.autograph.pyct import parser
from tensorflow.contrib.autograph.pyct import qual_names


class ReplaceTransformer(gast.NodeTransformer):
  """Replace AST nodes."""

  def __init__(self, replacements):
    """Create a new ReplaceTransformer.

    Args:
      replacements: A mapping from placeholder names to (lists of) AST nodes
          that these placeholders will be replaced by.
    """
    self.replacements = replacements
    self.in_replacements = False
    self.preserved_annos = {
        anno.Basic.ORIGIN,
        anno.Basic.SKIP_PROCESSING,
        anno.Static.ORIG_DEFINITIONS,
    }

  def _prepare_replacement(self, replaced, key):
    """Prepares a replacement AST that's safe to swap in for a node.

    Args:
      replaced: ast.AST, the node being replaced
      key: Hashable, the key of the replacement AST
    Returns:
      ast.AST, the replacement AST
    """
    repl = self.replacements[key]

    new_nodes = ast_util.copy_clean(repl, preserve_annos=self.preserved_annos)
    if isinstance(new_nodes, gast.AST):
      new_nodes = [new_nodes]

    return new_nodes

  def visit_Expr(self, node):
    # When replacing a placeholder with an entire statement, the replacement
    # must stand on its own and not be wrapped in an Expr.
    new_value = self.visit(node.value)
    if new_value is node.value:
      return node
    return new_value

  def visit_keyword(self, node):
    if node.arg not in self.replacements:
      return self.generic_visit(node)

    repl = self._prepare_replacement(node, node.arg)
    if isinstance(repl, gast.keyword):
      return repl
    elif (repl and isinstance(repl, (list, tuple)) and
          all(isinstance(r, gast.keyword) for r in repl)):
      return repl
    # TODO(mdan): We may allow replacing with a string as well.
    # For example, if one wanted to replace foo with bar in foo=baz, then
    # we could allow changing just node arg, so that we end up with bar=baz.
    raise ValueError(
        'a keyword argument may only be replaced by another keyword or a '
        'non-empty list of keywords. Found: %s' % repl)

  def visit_FunctionDef(self, node):
    node = self.generic_visit(node)
    if node.name not in self.replacements:
      return node

    repl = self.replacements[node.name]
    if not isinstance(repl, (gast.Name, ast.Name)):
      raise ValueError(
          'a function name can only be replaced by a Name node. Found: %s' %
          repl)
    node.name = repl.id
    return node

  def _check_has_context(self, node):
    if not node.ctx:
      raise ValueError('node %s is missing ctx value' % node)

  def _check_inner_children_have_context(self, node):
    if isinstance(node, gast.Attribute):
      self._check_inner_children_have_context(node.value)
      self._check_has_context(node)
    elif isinstance(node, gast.Tuple):
      for e in node.elts:
        self._check_inner_children_have_context(e)
      self._check_has_context(node)
    elif isinstance(node, gast.Dict):
      for e in node.keys:
        self._check_inner_children_have_context(e)
      for e in node.values:
        self._check_inner_children_have_context(e)
    elif isinstance(node, gast.Subscript):
      self._check_inner_children_have_context(node.value)
      self._check_inner_children_have_context(node.slice)
    elif isinstance(node, gast.Slice):
      self._check_inner_children_have_context(node.lower)
      if node.upper:
        self._check_inner_children_have_context(node.upper)
      if node.step:
        self._check_inner_children_have_context(node.step)
    elif isinstance(node, gast.Name):
      self._check_has_context(node)
    elif isinstance(node, (gast.Str, gast.Num)):
      pass
    else:
      raise ValueError('unexpected node type "%s"' % node)

  def _set_inner_child_context(self, node, ctx):
    if isinstance(node, gast.Attribute):
      self._set_inner_child_context(node.value, gast.Load())
      node.ctx = ctx
    elif isinstance(node, gast.Tuple):
      for e in node.elts:
        self._set_inner_child_context(e, ctx)
      node.ctx = ctx
    elif isinstance(node, gast.Name):
      node.ctx = ctx
    elif isinstance(node, gast.Call):
      self._set_inner_child_context(node.func, ctx)
      # We may be able to override these to Load(), but for now it's simpler
      # to just assert that they're set.
      for a in node.args:
        self._check_inner_children_have_context(a)
      for k in node.keywords:
        self._check_inner_children_have_context(k.value)
    elif isinstance(node, gast.Dict):
      # We may be able to override these to Load(), but for now it's simpler
      # to just assert that they're set.
      for e in node.keys:
        self._check_inner_children_have_context(e)
      for e in node.values:
        self._check_inner_children_have_context(e)
    elif isinstance(node, gast.Subscript):
      self._set_inner_child_context(node.value, ctx)
      self._check_inner_children_have_context(node.slice)
    elif isinstance(node, (gast.Str, gast.Num)):
      pass
    else:
      raise ValueError('unexpected node type "%s"' % node)

  def visit_Attribute(self, node):
    node = self.generic_visit(node)
    if node.attr not in self.replacements:
      return node

    repl = self.replacements[node.attr]
    if not isinstance(repl, gast.Name):
      raise ValueError(
          'An attribute can only be replaced by a Name node. Found: %s' % repl)
    node.attr = repl.id
    return node

  def visit_Name(self, node):
    if node.id not in self.replacements:
      return node

    new_nodes = self._prepare_replacement(node, node.id)

    # Preserve the target context.
    for n in new_nodes:
      if isinstance(n, gast.Tuple):
        for e in n.elts:
          self._set_inner_child_context(e, node.ctx)
      if isinstance(n, gast.Attribute):
        # For attributes, the inner Name node receives the context, while the
        # outer ones have it set to Load.
        self._set_inner_child_context(n, node.ctx)
      else:
        n.ctx = node.ctx

    if len(new_nodes) == 1:
      new_nodes, = new_nodes

    return new_nodes


def _convert_to_ast(n):
  """Converts from a known data type to AST."""
  if isinstance(n, str):
    # Note: the node will receive the ctx value from the template, see
    # ReplaceTransformer.visit_Name.
    return gast.Name(id=n, ctx=None, annotation=None)
  if isinstance(n, qual_names.QN):
    return n.ast()
  if isinstance(n, list):
    return [_convert_to_ast(e) for e in n]
  if isinstance(n, tuple):
    return tuple(_convert_to_ast(e) for e in n)
  return n


def replace(template, **replacements):
  """Replaces placeholders in a Python template.

  AST Name and Tuple nodes always receive the context that inferred from
  the template. However, when replacing more complex nodes (that can potentially
  contain Name children), then the caller is responsible for setting the
  appropriate context.

  Args:
    template: A string representing Python code. Any symbol name can be used
        that appears in the template code can be used as placeholder.
    **replacements: A mapping from placeholder names to (lists of) AST nodes
        that these placeholders will be replaced by. String values are also
        supported as a shorthand for AST Name nodes with the respective ID.

  Returns:
    An AST node or list of AST nodes with the replacements made. If the
    template was a function, a list will be returned. If the template was a
    node, the same node will be returned. If the template was a string, an
    AST node will be returned (a `Module` node in the case of a multi-line
    string, an `Expr` node otherwise).

  Raises:
    ValueError: if the arguments are incorrect.
  """
  if not isinstance(template, str):
    raise ValueError('Expected string template, got %s' % type(template))
  tree = parser.parse_str(textwrap.dedent(template))
  for k in replacements:
    replacements[k] = _convert_to_ast(replacements[k])
  results = ReplaceTransformer(replacements).visit(tree).body
  if isinstance(results, list):
    return [qual_names.resolve(r) for r in results]
  return qual_names.resolve(results)


def replace_as_expression(template, **replacements):
  """Variant of replace that generates expressions, instead of code blocks."""
  replacement = replace(template, **replacements)
  if len(replacement) != 1:
    raise ValueError(
        'single expression expected; for more general templates use replace')
  node = replacement[0]
  node = qual_names.resolve(node)

  if isinstance(node, gast.Expr):
    return node.value
  elif isinstance(node, gast.Name):
    return node

  raise ValueError(
      'the template is expected to generate an expression or a name node;'
      ' instead found %s' % node)
