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

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names


class ContextAdjuster(gast.NodeTransformer):
  """Adjusts the ctx field of nodes to ensure consistency.

  This transformer can change the ctx fields of a variable, tuple and other
  AST elements that allow one, based on whether the element is being read or
  written.
  """

  def __init__(self, override_value):
    self._ctx_override = override_value

  def visit(self, node):
    original_override = self._ctx_override
    node = super(ContextAdjuster, self).visit(node)
    if hasattr(node, 'ctx'):
      assert node.ctx is not None, 'node {} has ctx unset'.format(node)
    self._ctx_override = original_override
    return node

  def _apply_override(self, node):
    if self._ctx_override is not None:
      node.ctx = self._ctx_override()

  def visit_Attribute(self, node):
    self._apply_override(node)
    self._ctx_override = gast.Load
    node = self.generic_visit(node)
    return node

  def visit_Tuple(self, node):
    self._apply_override(node)
    return self.generic_visit(node)

  def visit_List(self, node):
    self._apply_override(node)
    return self.generic_visit(node)

  def visit_Name(self, node):
    self._apply_override(node)
    return self.generic_visit(node)

  def visit_Call(self, node):
    self._apply_override(node)
    # We may be able to override these to Load(), but for now it's simpler
    # to just assert that they're set.
    self._ctx_override = None
    return self.generic_visit(node)

  def visit_Dict(self, node):
    # We may be able to override these to Load(), but for now it's simpler
    # to just assert that they're set.
    self._ctx_override = None
    return self.generic_visit(node)

  def visit_Subscript(self, node):
    self._apply_override(node)
    self._ctx_override = gast.Load
    node.value = self.visit(node.value)
    return self.generic_visit(node)

  def visit_comprehension(self, node):
    # We may be able to override some of these, but for now it's simpler
    # to just assert that they're set.
    self._ctx_override = None
    return self.generic_visit(node)

  def visit_Lambda(self, node):
    # We may be able to override some of these, but for now it's simpler
    # to just assert that they're set.
    self._ctx_override = None
    return self.generic_visit(node)


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
        'extra_test',
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
        'non-empty list of keywords. Found: {} for keyword {}'.format(
            repl, node.arg))

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

    if not new_nodes:
      return new_nodes

    # Preserve the target context.
    adjuster = ContextAdjuster(type(node.ctx))
    for n in new_nodes:
      if hasattr(n, 'ctx'):
        adjuster.visit(n)

    if len(new_nodes) == 1:
      new_nodes, = new_nodes

    return new_nodes


def _convert_to_ast(n):
  """Converts from a known data type to AST."""
  # Note: When generating AST nodes from strings/QNs in isolation, ctx is
  # unknown. ctx must be filled in according to the template being used.
  # See ReplaceTransformer.visit_Name.
  if isinstance(n, str):
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
  for k in replacements:
    replacements[k] = _convert_to_ast(replacements[k])
  template_str = parser.STANDARD_PREAMBLE + textwrap.dedent(template)
  nodes = parser.parse_str(
      template_str,
      preamble_len=parser.STANDARD_PREAMBLE_LEN,
      single_node=False)
  results = []
  for node in nodes:
    node = ReplaceTransformer(replacements).visit(node)
    if isinstance(node, (list, tuple)):
      results.extend(node)
    else:
      results.append(node)
  results = [qual_names.resolve(r) for r in results]
  return results


def replace_as_expression(template, **replacements):
  """Variant of replace that generates expressions, instead of code blocks."""
  replacement = replace(template, **replacements)
  if len(replacement) != 1:
    raise ValueError(
        'single expression expected; for more general templates use replace')
  node, = replacement

  if isinstance(node, gast.Expr):
    return node.value
  elif isinstance(node, gast.Name):
    return node

  raise ValueError(
      'the template is expected to generate an expression or a name node;'
      ' instead found %s' % node)
