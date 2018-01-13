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


def replace(template, **replacements):
  """Replace placeholders in a Python template.

  Args:
    template: A function to be used as a template. Any placeholder is expected
        to also be a function argument.
    **replacements: A mapping from placeholder names to (lists of) AST nodes
        that these placeholders will be replaced by.

  Returns:
    body: An AST node or list of AST nodes with the replacements made. If the
        template was a function, a list will be returned. If the template was a
        node, the same node will be returned. If the template was a string, an
        AST node will be returned (a `Module` node in the case of a multi-line
        string, an `Expr` node otherwise).

  Raises:
    ValueError: If a function is used as a template and an incorrect set of
        replacements was passed.
  """
  tree = parser.parse_object(template).body[0]
  placeholders = set(arg.id for arg in tree.args.args)
  tree.args.args = []
  if tree.args.vararg:
    placeholders.add(tree.args.vararg)
    tree.args.vararg = None
  if set(replacements.keys()) != placeholders:
    raise ValueError(
        'too many or few replacements. replacements: %s; placeholders: %s' %
        (replacements.keys(), placeholders))

  # Perform the replacement, stripping the function into which the template was
  # wrapped.
  return ReplaceTransformer(replacements).visit(tree).body
