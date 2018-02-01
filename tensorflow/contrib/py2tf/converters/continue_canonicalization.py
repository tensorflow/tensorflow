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
"""Canonicalizes continue statements by de-sugaring into a control boolean."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.contrib.py2tf.pyct import anno
from tensorflow.contrib.py2tf.pyct import templates


class ContinueCanonicalizationTransformer(gast.NodeTransformer):
  """Canonicalizes continue statements into additional conditionals."""

  def __init__(self, namer):
    self.namer = namer
    # This is a stack structure, to correctly process nested loops.
    self.continuation_uses = []

  def _create_continuation_check(self):
    template = """
      if not var_name:
        pass
    """
    cond, = templates.replace(template, var_name=self.continuation_uses[-1][1])
    cond.body = []
    return cond

  def _create_continuation_trigger(self):
    template = """
      var_name = True
    """
    assign, = templates.replace(
        template, var_name=self.continuation_uses[-1][1])
    return assign

  def _create_continuation_init(self):
    template = """
      var_name = False
    """
    assign, = templates.replace(
        template, var_name=self.continuation_uses[-1][1])
    return assign

  def _visit_and_reindent_if_necessary(self, nodes):
    reorganized_nodes = []
    current_dest = reorganized_nodes
    continue_used_in_block = False
    for i, n in enumerate(nodes):
      # TODO(mdan): This could be optimized if control structures are simple.
      self.continuation_uses[-1][0] = False
      n = self.visit(n)
      current_dest.append(n)
      if self.continuation_uses[-1][0]:
        continue_used_in_block = True
        if i < len(nodes) - 1:  # Last statement in block needs no protection.
          cond = self._create_continuation_check()
          current_dest.append(cond)
          current_dest = cond.body
    self.continuation_uses[-1][0] = continue_used_in_block
    return reorganized_nodes

  def _process_loop_block(self, block, scope):
    cont_var = self.namer.new_symbol('cont_requested', scope.referenced)
    self.continuation_uses.append([False, cont_var])
    block = self._visit_and_reindent_if_necessary(block)
    if self.continuation_uses[-1][0]:
      block.insert(0, self._create_continuation_init())
    self.continuation_uses.pop()
    return block

  def visit_While(self, node):
    self.generic_visit(node.test)
    node.body = self._process_loop_block(node.body,
                                         anno.getanno(node, 'body_scope'))
    for n in node.orelse:
      self.generic_visit(n)
    return node

  def visit_For(self, node):
    self.generic_visit(node.target)
    self.generic_visit(node.iter)
    node.body = self._process_loop_block(node.body,
                                         anno.getanno(node, 'body_scope'))
    for n in node.orelse:
      self.generic_visit(n)
    return node

  def visit_If(self, node):
    if self.continuation_uses:
      self.generic_visit(node.test)
      node.body = self._visit_and_reindent_if_necessary(node.body)
      continue_used_in_body = self.continuation_uses[-1][0]
      node.orelse = self._visit_and_reindent_if_necessary(node.orelse)
      self.continuation_uses[-1][0] = (
          continue_used_in_body or self.continuation_uses[-1][0])
    else:
      node = self.generic_visit(node)
    return node

  def visit_Continue(self, node):
    self.continuation_uses[-1][0] = True
    return self._create_continuation_trigger()

  def visit_Break(self, node):
    assert False, 'break statement should be desugared at this point'


def transform(node, namer):
  transformer = ContinueCanonicalizationTransformer(namer)
  node = transformer.visit(node)
  return node
