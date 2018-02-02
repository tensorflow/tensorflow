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
import copy
import textwrap

import gast

from tensorflow.contrib.py2tf.pyct import parser


class ReplaceTransformer(gast.NodeTransformer):
  """Replace AST nodes."""

  def __init__(self, replacements):
    """Create a new ReplaceTransformer.

    Args:
      replacements: A mapping from placeholder names to (lists of) AST nodes
          that these placeholders will be replaced by.
    """
    self.replacements = replacements

  # TODO(mdan): Make a more detailed pass and clean up if needed.

  def visit_Expr(self, node):
    if (isinstance(node.value, gast.Name) and
        node.value.id in self.replacements):
      return self.visit(node.value)
    self.generic_visit(node)
    return node

  def visit_FunctionDef(self, node):
    node = self.generic_visit(node)
    if node.name in self.replacements:
      repl = self.replacements[node.name]
      if not isinstance(repl, (gast.Name, ast.Name)):
        raise ValueError(
            'A function name can only be replaced by a Name node. Found: %s',
            repl)
      node.name = repl.id
    return node

  def visit_Name(self, node):
    if node.id in self.replacements:
      # TODO(mdan): Sanitize the nodes by erasing scope-dependent annotations.
      new_nodes = copy.copy(self.replacements[node.id])
      if isinstance(new_nodes, gast.AST):
        new_nodes = [new_nodes]
      # Preserve the target context.
      for n in new_nodes:
        if isinstance(n, gast.Tuple):
          for e in n.elts:
            e.ctx = node.ctx
        n.ctx = node.ctx
      if len(new_nodes) == 1:
        new_nodes, = new_nodes
      return new_nodes
    else:
      return node


def _strings_to_names(n):
  if isinstance(n, str):
    # Note: the node will receive the ctx value from the template, see
    # ReplaceTransformer.visit_Name.
    return gast.Name(id=n, ctx=None, annotation=None)
  if isinstance(n, list):
    return [_strings_to_names(e) for e in n]
  if isinstance(n, tuple):
    return tuple(_strings_to_names(e) for e in n)
  return n


def replace(template, **replacements):
  """Replace placeholders in a Python template.

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
    replacements[k] = _strings_to_names(replacements[k])
  return ReplaceTransformer(replacements).visit(tree).body
