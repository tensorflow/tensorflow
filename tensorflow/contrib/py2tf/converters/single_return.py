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
"""Canonicalizes functions with multiple returns to use just one."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.contrib.py2tf.pyct import anno
from tensorflow.contrib.py2tf.pyct import ast_util
from tensorflow.contrib.py2tf.pyct import templates
from tensorflow.contrib.py2tf.pyct import transformer
from tensorflow.contrib.py2tf.pyct.static_analysis.annos import NodeAnno


# TODO(mdan): Move this logic into transformer_base.
class BodyVisitor(transformer.Base):
  """Walks breadth- or depth-first the list-of-nodes bodies of AST nodes."""

  def __init__(self, context, depth_first=False):
    self.depth_first = depth_first
    self.changes_made = False
    super(BodyVisitor, self).__init__(context)

  def visit_nodelist(self, nodelist):
    for node in nodelist:
      if isinstance(node, list):
        node = self.visit_nodelist(node)
      else:
        node = self.generic_visit(node)
    return nodelist

  def visit_If(self, node):
    if self.depth_first:
      node = self.generic_visit(node)
    node.body = self.visit_nodelist(node.body)
    node.orelse = self.visit_nodelist(node.orelse)
    if not self.depth_first:
      node = self.generic_visit(node)
    return node

  def visit_For(self, node):
    if self.depth_first:
      node = self.generic_visit(node)
    node.body = self.visit_nodelist(node.body)
    node.orelse = self.visit_nodelist(node.orelse)
    if not self.depth_first:
      node = self.generic_visit(node)
    return node

  def visit_While(self, node):
    if self.depth_first:
      node = self.generic_visit(node)
    node.body = self.visit_nodelist(node.body)
    node.orelse = self.visit_nodelist(node.orelse)
    if not self.depth_first:
      node = self.generic_visit(node)
    return node

  def visit_Try(self, node):
    if self.depth_first:
      node = self.generic_visit(node)
    node.body = self.visit_nodelist(node.body)
    node.orelse = self.visit_nodelist(node.orelse)
    node.finalbody = self.visit_nodelist(node.finalbody)
    for i in range(len(node.handlers)):
      node.handlers[i].body = self.visit_nodelist(node.handlers[i].body)
    if not self.depth_first:
      node = self.generic_visit(node)
    return node

  def visit_With(self, node):
    if self.depth_first:
      node = self.generic_visit(node)
    node.body = self.visit_nodelist(node.body)
    if not self.depth_first:
      node = self.generic_visit(node)
    return node

  def visit_FunctionDef(self, node):
    if self.depth_first:
      node = self.generic_visit(node)
    node.body = self.visit_nodelist(node.body)
    self.generic_visit(node)
    if not self.depth_first:
      node = self.generic_visit(node)
    return node


class FoldElse(BodyVisitor):

  def visit_nodelist(self, nodelist):
    for i in range(len(nodelist)):
      node = nodelist[i]
      if isinstance(node, gast.If):
        true_branch_returns = isinstance(node.body[-1], gast.Return)
        false_branch_returns = len(node.orelse) and isinstance(
            node.orelse[-1], gast.Return)
        # If the last node in the if body is a return,
        # then every line after this if statement effectively
        # belongs in the else.
        if true_branch_returns and not false_branch_returns:
          for j in range(i + 1, len(nodelist)):
            nodelist[i].orelse.append(ast_util.copy_clean(nodelist[j]))
          if nodelist[i + 1:]:
            self.changes_made = True
          return nodelist[:i + 1]
        elif not true_branch_returns and false_branch_returns:
          for j in range(i + 1, len(nodelist)):
            nodelist[i].body.append(ast_util.copy_clean(nodelist[j]))
          if nodelist[i + 1:]:
            self.changes_made = True
          return nodelist[:i + 1]
        elif true_branch_returns and false_branch_returns:
          if nodelist[i + 1:]:
            raise ValueError(
                'Unreachable code after conditional where both branches return.'
            )
          return nodelist
      elif isinstance(node, gast.Return) and nodelist[i + 1:]:
        raise ValueError(
            'Cannot have statements after a return in the same basic block')
    return nodelist


def contains_return(node):
  for n in gast.walk(node):
    if isinstance(n, gast.Return):
      return True
  return False


class LiftReturn(transformer.Base):
  """Move return statements out of If and With blocks."""

  def __init__(self, context):
    self.changes_made = False
    self.common_return_name = None
    super(LiftReturn, self).__init__(context)

  def visit_If(self, node):
    # Depth-first traversal of if statements
    node = self.generic_visit(node)

    # We check if both branches return, and if so, lift the return out of the
    # conditional. We don't enforce that the true and false branches either
    # both return or both do not, because FoldElse might move a return
    # into a branch after this transform completes. FoldElse and LiftReturn
    # are alternately run until the code reaches a fixed point.
    true_branch_returns = isinstance(node.body[-1], gast.Return)
    false_branch_returns = len(node.orelse) and isinstance(
        node.orelse[-1], gast.Return)
    if true_branch_returns and false_branch_returns:
      node.body[-1] = templates.replace(
          'a = b', a=self.common_return_name, b=node.body[-1].value)[0]
      node.orelse[-1] = templates.replace(
          'a = b', a=self.common_return_name, b=node.orelse[-1].value)[0]
      return_node = templates.replace('return a', a=self.common_return_name)[0]
      self.changes_made = True
      return [node, return_node]
    else:
      return node

  def visit_With(self, node):
    # Depth-first traversal of syntax
    node = self.generic_visit(node)

    # If the with statement returns, lift the return
    if isinstance(node.body[-1], gast.Return):
      node.body[-1] = templates.replace(
          'a = b', a=self.common_return_name, b=node.body[-1].value)[0]
      return_node = templates.replace('return a', a=self.common_return_name)[0]
      node = self.generic_visit(node)
      self.changes_made = True
      return [node, return_node]
    else:
      return node

  def visit_FunctionDef(self, node):
    # Ensure we're doing depth-first traversal
    last_return_name = self.common_return_name
    body_scope = anno.getanno(node, NodeAnno.BODY_SCOPE)
    referenced_names = body_scope.referenced
    self.common_return_name = self.context.namer.new_symbol(
        'return_', referenced_names)
    node = self.generic_visit(node)
    self.common_return_name = last_return_name
    return node


class DetectReturnInUnsupportedControlFlow(gast.NodeVisitor):
  """Throws an error if code returns inside loops or try/except."""

  # First, throw an error if we detect a return statement in a loop.
  # TODO(alexbw): we need to learn to handle returns inside a loop,
  # but don't currently have the TF constructs to do so (need something
  # that looks vaguely like a goto).

  def __init__(self):
    self.cant_return = False
    super(DetectReturnInUnsupportedControlFlow, self).__init__()

  def visit_While(self, node):
    self.cant_return = True
    self.generic_visit(node)
    self.cant_return = False

  def visit_For(self, node):
    self.cant_return = True
    self.generic_visit(node)
    self.cant_return = False

  def visit_Try(self, node):
    self.cant_return = True
    self.generic_visit(node)
    self.cant_return = False

  def visit_Return(self, node):
    if self.cant_return:
      raise ValueError(
          'Pyflow currently does not support `return` statements in loops. '
          'Try assigning to a variable in the while loop, and returning '
          'outside of the loop')


class DetectReturnInConditional(gast.NodeVisitor):
  """Assert that no return statements are present in conditionals."""

  def __init__(self):
    self.cant_return = False
    super(DetectReturnInConditional, self).__init__()

  def visit_If(self, node):
    self.cant_return = True
    self.generic_visit(node)
    self.cant_return = False

  def visit_Return(self, node):
    if self.cant_return:
      raise ValueError(
          'After transforms, a conditional contained a `return `statement, '
          'which is not allowed. This is a bug, and should not happen.')


class DetectReturnInFunctionDef(gast.NodeVisitor):

  def visit_FunctionDef(self, node):
    self.generic_visit(node)
    if not contains_return(node):
      raise ValueError(
          'Each function definition should contain at least one return.')


def transform(node, context):
  """Ensure a function has only a single return.

  This transforms an AST node with multiple returns successively into containing
  only a single return node.
  There are a few restrictions on what we can handle:
   - An AST being transformed must contain at least one return.
   - No returns allowed in loops. We have to know the type of the return value,
   and we currently don't have either a type inference system to discover it,
   nor do we have a mechanism for late type binding in TensorFlow.
   - After all transformations are finished, a Return node is not allowed inside
   control flow. If we were unable to move a return outside of control flow,
   this is an error.

  Args:
     node: an AST node to transform
     context: a context object

  Returns:
     new_node: an AST with a single return value

  Raises:
    ValueError: if the AST is structured so that we can't perform the
   transform.
  """
  # Make sure that the function has at least one return statement
  # TODO(alexbw): turning off this assertion for now --
  # we need to not require this in e.g. class constructors.
  # DetectReturnInFunctionDef().visit(node)

  # Make sure there's no returns in unsupported locations (loops, try/except)
  DetectReturnInUnsupportedControlFlow().visit(node)

  while True:

    # Try to lift all returns out of if statements and with blocks
    lr = LiftReturn(context)
    node = lr.visit(node)
    changes_made = lr.changes_made
    fe = FoldElse(context)
    node = fe.visit(node)
    changes_made = changes_made or fe.changes_made

    if not changes_made:
      break

  # Make sure we've scrubbed all returns from conditionals
  DetectReturnInConditional().visit(node)

  return node
