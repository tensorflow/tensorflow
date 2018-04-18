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

from tensorflow.contrib.autograph.pyct import anno
from tensorflow.contrib.autograph.pyct import templates
from tensorflow.contrib.autograph.pyct import transformer
from tensorflow.contrib.autograph.pyct.static_analysis.annos import NodeAnno


class BreakCanonicalizationTransformer(transformer.Base):
  """Canonicalizes break statements into additional conditionals."""

  def __init__(self, context):
    super(BreakCanonicalizationTransformer, self).__init__(context)
    # This is a stack structure, to correctly process nested loops.
    # Each item is a list [break_used, break_variable_name]
    self.break_uses = []

  def visit_Break(self, node):
    self.break_uses[-1][0] = True
    template = """
      var_name = True
      continue
    """
    return templates.replace(template, var_name=self.break_uses[-1][1])

  def visit_While(self, node):
    scope = anno.getanno(node, NodeAnno.BODY_SCOPE)
    break_var = self.context.namer.new_symbol('break_requested',
                                              scope.referenced)

    self.break_uses.append([False, break_var])
    node = self.generic_visit(node)
    if self.break_uses[-1][0]:
      template = """
        var_name = False
        while original_test and not var_name:
          original_body
        else:
          original_orelse
      """
      node = templates.replace(
          template,
          var_name=break_var,
          original_test=node.test,
          original_body=node.body,
          original_orelse=node.orelse)
    self.break_uses.pop()

    return node

  def visit_For(self, node):
    scope = anno.getanno(node, NodeAnno.BODY_SCOPE)
    break_var = self.context.namer.new_symbol('break_requested',
                                              scope.referenced)

    self.break_uses.append([False, break_var])
    node = self.generic_visit(node)
    if self.break_uses[-1][0]:
      template = """
        var_name = False
        original_for
      """
      node = templates.replace(
          template,
          var_name=break_var,
          original_for=node)
      extra_cond = templates.replace_as_expression(
          'not var_name', var_name=break_var)
      new_for_node = node[1]
      anno.setanno(new_for_node, 'extra_cond', extra_cond)
    self.break_uses.pop()

    return node


def transform(node, context):
  return BreakCanonicalizationTransformer(context).visit(node)
