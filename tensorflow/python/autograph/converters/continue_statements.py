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

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno


class _Continue(object):

  def __init__(self):
    self.used = False
    self.control_var_name = None
    self.create_guard = False
    self.guard_created = False

  def __repr__(self):
    return 'used: %s, var: %s' % (self.used, self.control_var_name)


class ContinueCanonicalizationTransformer(converter.Base):
  """Canonicalizes continue statements into additional conditionals."""

  def visit_Continue(self, node):
    self.state[_Continue].used = True
    template = """
      var_name = True
    """
    return templates.replace(
        template, var_name=self.state[_Continue].control_var_name)

  def _postprocess_statement(self, node):
    # Example of how the state machine below works:
    #
    #   1| stmt           # State: Continue_.used = False
    #    |                # Action: none
    #   2| if cond:
    #   3|   continue     # State: Continue_.used = True,
    #    |                #        Continue_.guard_created = False,
    #    |                #        Continue_.create_guard = False
    #    |                # Action: Continue_.create_guard = True
    #   4| stmt           # State: Continue_.used = True,
    #    |                #        Continue_.guard_created = False,
    #    |                #        Continue_.create_guard = True
    #    |                # Action: create `if not continue_used`,
    #    |                #         set Continue_.guard_created = True
    #   5| stmt           # State: Continue_.used = True,
    #    |                #        Continue_.guard_created = True
    #    |                # Action: none (will be wrapped under previously
    #    |                #         created if node)

    if self.state[_Continue].used:
      if self.state[_Continue].guard_created:
        return node, None

      elif not self.state[_Continue].create_guard:
        self.state[_Continue].create_guard = True
        return node, None

      else:
        self.state[_Continue].guard_created = True
        template = """
          if not var_name:
            original_node
        """
        cond, = templates.replace(
            template,
            var_name=self.state[_Continue].control_var_name,
            original_node=node)
        return cond, cond.body
    return node, None

  def _visit_loop_body(self, node, nodes):
    self.state[_Continue].enter()
    scope = anno.getanno(node, NodeAnno.BODY_SCOPE)
    continue_var = self.ctx.namer.new_symbol('continue_', scope.referenced)
    self.state[_Continue].control_var_name = continue_var

    nodes = self.visit_block(nodes, after_visit=self._postprocess_statement)

    if self.state[_Continue].used:
      template = """
        var_name = False
      """
      control_var_init = templates.replace(template, var_name=continue_var)
      nodes = control_var_init + nodes

    self.state[_Continue].exit()
    return nodes

  def visit_While(self, node):
    node.test = self.visit(node.test)
    node.body = self._visit_loop_body(node, node.body)
    # A continue in the else clause applies to the containing scope.
    node.orelse = self.visit_block(node.orelse)
    return node

  def visit_For(self, node):
    node.target = self.generic_visit(node.target)
    node.iter = self.generic_visit(node.iter)
    node.body = self._visit_loop_body(node, node.body)
    # A continue in the else clause applies to the containing scope.
    node.orelse = self.visit_block(node.orelse)
    return node


def transform(node, ctx):
  transformer = ContinueCanonicalizationTransformer(ctx)
  node = transformer.visit(node)
  return node
