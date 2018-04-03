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
"""Canonicalizes break statements by de-sugaring into a control boolean."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.contrib.autograph.pyct import anno
from tensorflow.contrib.autograph.pyct import templates
from tensorflow.contrib.autograph.pyct import transformer
from tensorflow.contrib.autograph.pyct.static_analysis.annos import NodeAnno


class BreakCanonicalizationTransformer(transformer.Base):
  """Canonicalizes break statements into additional conditionals."""

  def __init__(self, context):
    super(BreakCanonicalizationTransformer, self).__init__(context)
    # This is a stack structure, to correctly process nested loops.
    self.break_uses = []

  def _create_break_check(self):
    template = """
      (not var_name)
    """
    expr, = templates.replace(template, var_name=self.break_uses[-1][1])
    return expr.value

  def _create_break_trigger(self):
    template = """
      var_name = True
    """
    block = templates.replace(template, var_name=self.break_uses[-1][1])
    block.append(gast.Continue())
    return block

  def _create_break_init(self):
    template = """
      var_name = False
    """
    assign, = templates.replace(template, var_name=self.break_uses[-1][1])
    return assign

  # TODO(mdan): Surely the transformer supports this better?
  def _manual_visit_list(self, block):
    new_block = []
    for n in block:
      new_n = self.visit(n)
      if isinstance(new_n, list):
        new_block.extend(new_n)
      else:
        new_block.append(new_n)
    return new_block

  def visit_While(self, node):
    self.generic_visit(node.test)
    scope = anno.getanno(node, NodeAnno.BODY_SCOPE)

    break_var = self.context.namer.new_symbol('break_requested',
                                              scope.referenced)
    self.break_uses.append([False, break_var])
    node.body = self._manual_visit_list(node.body)
    if self.break_uses[-1][0]:
      node.test = gast.BoolOp(gast.And(), [
          node.test,
          gast.UnaryOp(gast.Not(), gast.Name(break_var, gast.Load(), None))
      ])
      final_nodes = [self._create_break_init(), node]
    else:
      final_nodes = node
    self.break_uses.pop()

    for n in node.orelse:
      self.generic_visit(n)
    return final_nodes

  def visit_For(self, node):
    self.generic_visit(node.target)
    self.generic_visit(node.iter)
    scope = anno.getanno(node, NodeAnno.BODY_SCOPE)

    break_var = self.context.namer.new_symbol('break_requested',
                                              scope.referenced)
    self.break_uses.append([False, break_var])
    node.body = self._manual_visit_list(node.body)
    if self.break_uses[-1][0]:
      anno.setanno(node, 'extra_cond',
                   gast.UnaryOp(gast.Not(),
                                gast.Name(break_var, gast.Load(), None)))
      final_nodes = [self._create_break_init(), node]
    else:
      final_nodes = node
    self.break_uses.pop()

    for n in node.orelse:
      self.generic_visit(n)
    return final_nodes

  def visit_Break(self, node):
    self.break_uses[-1][0] = True
    return self._create_break_trigger()


def transform(node, context):
  return BreakCanonicalizationTransformer(context).visit(node)
