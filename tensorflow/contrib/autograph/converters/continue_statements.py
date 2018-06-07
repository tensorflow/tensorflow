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

from tensorflow.contrib.autograph.pyct import anno
from tensorflow.contrib.autograph.pyct import templates
from tensorflow.contrib.autograph.pyct import transformer
from tensorflow.contrib.autograph.pyct.static_analysis.annos import NodeAnno


# Tags for local state.
CONTROL_VAR_NAME = 'control_var_name'
CONTINUE_USED = 'continue_used'
GUARD_CREATED = 'guard_created'
CREATE_GUARD_NEXT = 'create_guard_next'


class ContinueCanonicalizationTransformer(transformer.Base):
  """Canonicalizes continue statements into additional conditionals."""

  def visit_Continue(self, node):
    self.set_local(CONTINUE_USED, True)
    template = """
      var_name = True
    """
    return templates.replace(
        template, var_name=self.get_local(CONTROL_VAR_NAME))

  def _postprocess_statement(self, node):
    # Example of how the state machine below works:
    #
    #   1| stmt           # State: CONTINUE_USED = False
    #    |                # Action: none
    #   2| if cond:
    #   3|   continue     # State: CONTINUE_USED = True,
    #    |                #        GUARD_CREATED = False,
    #    |                #        CREATE_GUARD_NEXT = False
    #    |                # Action: set CREATE_GUARD_NEXT = True
    #   4| stmt           # State: CONTINUE_USED = True,
    #    |                #        GUARD_CREATED = False,
    #    |                #        CREATE_GUARD_NEXT = True
    #    |                # Action: create `if not continue_used`,
    #    |                #         set GUARD_CREATED = True
    #   5| stmt           # State: CONTINUE_USED = True, GUARD_CREATED = True
    #    |                # Action: none (will be wrapped under previously
    #    |                #         created if node)

    if self.get_local(CONTINUE_USED, False):
      if self.get_local(GUARD_CREATED, False):
        return node, None

      elif not self.get_local(CREATE_GUARD_NEXT, False):
        self.set_local(CREATE_GUARD_NEXT, True)
        return node, None

      else:
        self.set_local(GUARD_CREATED, True)
        template = """
          if not var_name:
            original_node
        """
        cond, = templates.replace(
            template,
            var_name=self.get_local(CONTROL_VAR_NAME),
            original_node=node)
        return cond, cond.body
    return node, None

  def _visit_loop_body(self, node, nodes):
    self.enter_local_scope()
    scope = anno.getanno(node, NodeAnno.BODY_SCOPE)
    continue_var = self.context.namer.new_symbol('continue_', scope.referenced)
    self.set_local(CONTROL_VAR_NAME, continue_var)

    nodes = self.visit_block(nodes, after_visit=self._postprocess_statement)

    if self.get_local(CONTINUE_USED, False):
      template = """
        var_name = False
      """
      control_var_init = templates.replace(template, var_name=continue_var)
      nodes = control_var_init + nodes

    self.exit_local_scope()
    return nodes

  def _visit_non_loop_body(self, nodes):
    self.enter_local_scope(inherit=(CONTROL_VAR_NAME,))
    nodes = self.visit_block(nodes, after_visit=self._postprocess_statement)
    continue_used = self.get_local(CONTINUE_USED, False)
    self.exit_local_scope(keep=(CONTINUE_USED,))
    return nodes, continue_used

  def visit_While(self, node):
    node.test = self.visit(node.test)
    node.body = self._visit_loop_body(node, node.body)
    # A continue in the else clause applies to the containing scope.
    node.orelse, _ = self._visit_non_loop_body(node.orelse)
    return node

  def visit_For(self, node):
    node.target = self.generic_visit(node.target)
    node.iter = self.generic_visit(node.iter)
    node.body = self._visit_loop_body(node, node.body)
    # A continue in the else clause applies to the containing scope.
    node.orelse, _ = self._visit_non_loop_body(node.orelse)
    return node

  def visit_If(self, node):
    node.test = self.generic_visit(node.test)
    node.body, continue_used_body = self._visit_non_loop_body(node.body)
    node.orelse, continue_used_orelse = self._visit_non_loop_body(node.orelse)
    self.set_local(CONTINUE_USED, continue_used_body or continue_used_orelse)
    return node

  def visit_With(self, node):
    node.items = self.visit_block(node.items)
    node.body, _ = self._visit_non_loop_body(node.body)
    return node


def transform(node, namer):
  return ContinueCanonicalizationTransformer(namer).visit(node)
