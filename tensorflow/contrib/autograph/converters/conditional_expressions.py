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
"""Converts the ternary conditional operator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.autograph.core import converter
from tensorflow.contrib.autograph.pyct import anno
from tensorflow.contrib.autograph.pyct import templates
from tensorflow.contrib.autograph.pyct.static_analysis.annos import NodeAnno


class _FunctionDefs(object):

  def __init__(self):
    self.nodes = []


class _Statement(object):

  def __init__(self):
    self.scope = None


class ConditionalExpressionTransformer(converter.Base):
  """Converts contitional expressions to functional form."""

  def _postprocess_statement(self, node):
    """Inserts any separate functions that node may use."""
    replacements = []
    for def_node in self.state[_FunctionDefs].nodes:
      replacements.extend(def_node)
    replacements.append(node)
    node = replacements
    # The corresponding enter is called by self.visit_block (see _process_block)
    self.state[_FunctionDefs].exit()
    return node, None

  def _create_branch(self, expr, name_stem):
    scope = self.state[_Statement].scope
    name = self.ctx.namer.new_symbol(name_stem, scope.referenced)
    template = """
      def name():
        return expr,
    """
    node = templates.replace(template, name=name, expr=expr)
    self.state[_FunctionDefs].nodes.append(node)
    return name

  def visit_IfExp(self, node):
    if anno.hasanno(node.test, anno.Basic.QN):
      name_root = anno.getanno(node.test, anno.Basic.QN).ssf()
    else:
      name_root = 'ifexp'

    true_fn_name = self._create_branch(node.body, '%s_true' % name_root)
    false_fn_name = self._create_branch(node.orelse, '%s_false' % name_root)

    return templates.replace_as_expression(
        'ag__.utils.run_cond(test, true_fn_name, false_fn_name)',
        test=node.test,
        true_fn_name=true_fn_name,
        false_fn_name=false_fn_name)

  def _process_block(self, scope, block):
    self.state[_Statement].enter()
    self.state[_Statement].scope = scope
    block = self.visit_block(
        block,
        before_visit=self.state[_FunctionDefs].enter,
        after_visit=self._postprocess_statement)
    self.state[_Statement].exit()
    return block

  def visit_FunctionDef(self, node):
    node.args = self.generic_visit(node.args)
    node.decorator_list = self.visit_block(node.decorator_list)
    node.body = self._process_block(
        anno.getanno(node, anno.Static.SCOPE), node.body)
    return node

  def visit_For(self, node):
    node.target = self.visit(node.target)
    node.body = self._process_block(
        anno.getanno(node, NodeAnno.BODY_SCOPE), node.body)
    node.orelse = self._process_block(
        anno.getanno(node, NodeAnno.ORELSE_SCOPE), node.orelse)
    return node

  def visit_While(self, node):
    node.test = self.visit(node.test)
    node.body = self._process_block(
        anno.getanno(node, NodeAnno.BODY_SCOPE), node.body)
    node.orelse = self._process_block(
        anno.getanno(node, NodeAnno.ORELSE_SCOPE), node.orelse)
    return node

  def visit_If(self, node):
    node.test = self.visit(node.test)
    node.body = self._process_block(
        anno.getanno(node, NodeAnno.BODY_SCOPE), node.body)
    node.orelse = self._process_block(
        anno.getanno(node, NodeAnno.ORELSE_SCOPE), node.orelse)
    return node

  def visit_With(self, node):
    node.items = self.visit_block(node.items)
    node.body = self._process_block(
        anno.getanno(node, NodeAnno.BODY_SCOPE), node.body)
    return node


def transform(node, ctx):
  node = ConditionalExpressionTransformer(ctx).visit(node)
  return node
