# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Handles control flow statements: while, for, if."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.contrib.autograph.core import converter
from tensorflow.contrib.autograph.pyct import anno
from tensorflow.contrib.autograph.pyct import ast_util
from tensorflow.contrib.autograph.pyct import parser
from tensorflow.contrib.autograph.pyct import templates
from tensorflow.contrib.autograph.pyct.static_analysis import cfg
from tensorflow.contrib.autograph.pyct.static_analysis.annos import NodeAnno


class SymbolNamer(object):
  """Describes the interface for ControlFlowTransformer's namer."""

  def new_symbol(self, name_root, reserved_locals):
    """Generate a new unique symbol.

    Args:
      name_root: String, used as stem in the new name.
      reserved_locals: Set(string), additional local symbols that are reserved
          and which should not be used.
    Returns:
      String.
    """
    raise NotImplementedError()


class ControlFlowTransformer(converter.Base):
  """Transforms control flow structures like loops an conditionals."""

  def _create_cond_branch(self, body_name, aliased_orig_names,
                          aliased_new_names, body, returns):
    if aliased_orig_names:
      template = """
        def body_name():
          aliased_new_names, = aliased_orig_names,
          body
          return (returns,)
      """
      return templates.replace(
          template,
          body_name=body_name,
          body=body,
          aliased_orig_names=aliased_orig_names,
          aliased_new_names=aliased_new_names,
          returns=returns)
    else:
      template = """
        def body_name():
          body
          return (returns,)
      """
      return templates.replace(
          template, body_name=body_name, body=body, returns=returns)

  def _create_cond_expr(self, results, test, body_name, orelse_name):
    if results is not None:
      template = """
        results = ag__.utils.run_cond(test, body_name, orelse_name)
      """
      return templates.replace(
          template,
          test=test,
          results=results,
          body_name=body_name,
          orelse_name=orelse_name)
    else:
      template = """
        ag__.utils.run_cond(test, body_name, orelse_name)
      """
      return templates.replace(
          template, test=test, body_name=body_name, orelse_name=orelse_name)

  def visit_If(self, node):
    self.generic_visit(node)

    body_scope = anno.getanno(node, NodeAnno.BODY_SCOPE)
    orelse_scope = anno.getanno(node, NodeAnno.ORELSE_SCOPE)
    body_defs = body_scope.created | body_scope.modified
    orelse_defs = orelse_scope.created | orelse_scope.modified
    live = anno.getanno(node, 'live_out')

    # We'll need to check if we're closing over variables that are defined
    # elsewhere in the function
    # NOTE: we can only detect syntactic closure in the scope
    # of the code passed in. If the AutoGraph'd function itself closes
    # over other variables, this analysis won't take that into account.
    defined = anno.getanno(node, 'defined_in')

    # We only need to return variables that are
    # - modified by one or both branches
    # - live (or has a live parent) at the end of the conditional
    modified = []
    for def_ in body_defs | orelse_defs:
      def_with_parents = set((def_,)) | def_.support_set
      if live & def_with_parents:
        modified.append(def_)

    # We need to check if live created variables are balanced
    # in both branches
    created = live & (body_scope.created | orelse_scope.created)

    # The if statement is illegal if there are variables that are created,
    # that are also live, but both branches don't create them.
    if created:
      if created != (body_scope.created & live):
        raise ValueError(
            'The main branch does not create all live symbols that the else '
            'branch does.')
      if created != (orelse_scope.created & live):
        raise ValueError(
            'The else branch does not create all live symbols that the main '
            'branch does.')

    # Alias the closure variables inside the conditional functions
    # to avoid errors caused by the local variables created in the branch
    # functions.
    # We will alias variables independently for body and orelse scope,
    # because different branches might write different variables.
    aliased_body_orig_names = tuple(body_scope.modified - body_scope.created)
    aliased_orelse_orig_names = tuple(orelse_scope.modified -
                                      orelse_scope.created)
    aliased_body_new_names = tuple(
        self.ctx.namer.new_symbol(s.ssf(), body_scope.referenced)
        for s in aliased_body_orig_names)
    aliased_orelse_new_names = tuple(
        self.ctx.namer.new_symbol(s.ssf(), orelse_scope.referenced)
        for s in aliased_orelse_orig_names)

    alias_body_map = dict(zip(aliased_body_orig_names, aliased_body_new_names))
    alias_orelse_map = dict(
        zip(aliased_orelse_orig_names, aliased_orelse_new_names))

    node_body = ast_util.rename_symbols(node.body, alias_body_map)
    node_orelse = ast_util.rename_symbols(node.orelse, alias_orelse_map)

    if not modified:
      # When the cond would return no value, we leave the cond called without
      # results. That in turn should trigger the side effect guards. The
      # branch functions will return a dummy value that ensures cond
      # actually has some return value as well.
      results = None
    elif len(modified) == 1:
      results = modified[0]
    else:
      results = gast.Tuple([s.ast() for s in modified], None)

    body_name = self.ctx.namer.new_symbol('if_true', body_scope.referenced)
    orelse_name = self.ctx.namer.new_symbol('if_false', orelse_scope.referenced)
    if modified:

      def build_returns(aliased_names, alias_map, scope):
        """Builds list of return variables for a branch of a conditional."""
        returns = []
        for s in modified:
          if s in aliased_names:
            returns.append(alias_map[s])
          else:
            if s not in scope.created | defined:
              raise ValueError(
                  'Attempting to return variable "%s" from the true branch of '
                  'a conditional, but it was not closed over, or created in '
                  'this branch.' % str(s))
            else:
              returns.append(s)
        return tuple(returns)

      body_returns = build_returns(aliased_body_orig_names, alias_body_map,
                                   body_scope)
      orelse_returns = build_returns(aliased_orelse_orig_names,
                                     alias_orelse_map, orelse_scope)

    else:
      body_returns = orelse_returns = templates.replace('tf.ones(())')[0].value

    body_def = self._create_cond_branch(
        body_name,
        aliased_orig_names=tuple(aliased_body_orig_names),
        aliased_new_names=tuple(aliased_body_new_names),
        body=node_body,
        returns=body_returns)
    orelse_def = self._create_cond_branch(
        orelse_name,
        aliased_orig_names=tuple(aliased_orelse_orig_names),
        aliased_new_names=tuple(aliased_orelse_new_names),
        body=node_orelse,
        returns=orelse_returns)
    cond_expr = self._create_cond_expr(results, node.test, body_name,
                                       orelse_name)

    return body_def + orelse_def + cond_expr

  def visit_While(self, node):
    self.generic_visit(node)

    body_scope = anno.getanno(node, NodeAnno.BODY_SCOPE)
    body_closure = body_scope.modified - body_scope.created
    all_referenced = body_scope.referenced

    cond_scope = anno.getanno(node, NodeAnno.COND_SCOPE)
    cond_closure = set()
    for s in cond_scope.referenced:
      for root in s.support_set:
        if root not in body_scope.created:
          cond_closure.add(root)

    state = list(body_closure)
    if not state:
      # TODO(mdan): Implement this properly.
      # To complete this statement, we need to check whether any variable
      # created inside the body scope is used before being modified outside the
      # scope. This should be done during activity analysis, and in general
      # should cover the case where variables may not be initialized.
      raise ValueError('cannot convert while loop: no outputs')

    state_ssf = [
        self.ctx.namer.new_symbol(s.ssf(), all_referenced) for s in state
    ]
    ssf_map = {
        name: ssf
        for name, ssf in zip(state, state_ssf)
        if str(name) != ssf
    }

    if len(state) == 1:
      state = state[0]
      state_ssf = state_ssf[0]
      state_ast_tuple = state
    else:
      state_ast_tuple = gast.Tuple([n.ast() for n in state], None)

    node_body = ast_util.rename_symbols(node.body, ssf_map)
    test = ast_util.rename_symbols(node.test, ssf_map)

    template = """
      def test_name(state_ssf):
        return test
      def body_name(state_ssf):
        body
        return state_ssf,
      state_ast_tuple = ag__.while_stmt(
          test_name, body_name, (state,), (extra_deps,))
    """
    node = templates.replace(
        template,
        state=state,
        state_ssf=state_ssf,
        state_ast_tuple=state_ast_tuple,
        test_name=self.ctx.namer.new_symbol('loop_test', body_scope.referenced),
        test=test,
        body_name=self.ctx.namer.new_symbol('loop_body', body_scope.referenced),
        body=node_body,
        extra_deps=tuple(s.ast() for s in cond_closure),
    )

    return node

  def visit_For(self, node):
    self.generic_visit(node)

    body_scope = anno.getanno(node, NodeAnno.BODY_SCOPE)
    body_closure = body_scope.modified - body_scope.created
    all_referenced = body_scope.referenced

    state = list(body_closure)

    state_ssf = [
        self.ctx.namer.new_symbol(s.ssf(), all_referenced) for s in state
    ]
    ssf_map = {
        name: ssf
        for name, ssf in zip(state, state_ssf)
        if str(name) != ssf
    }

    if len(state) == 1:
      state = state[0]
      state_ssf = state_ssf[0]
      state_ast_tuple = state
    else:
      state_ast_tuple = gast.Tuple([n.ast() for n in state], None)

    node_body = ast_util.rename_symbols(node.body, ssf_map)
    if anno.hasanno(node, 'extra_test'):
      extra_test = anno.getanno(node, 'extra_test')
      extra_test = ast_util.rename_symbols(extra_test, ssf_map)
    else:
      extra_test = parser.parse_expression('True')

    template = """
      def extra_test_name(state_ssf):
        return extra_test_expr
      def body_name(iterate, state_ssf):
        body
        return state_ssf,
      state_ast_tuple = ag__.for_stmt(
          iter_, extra_test_name, body_name, (state,))
    """
    node = templates.replace(
        template,
        state=state,
        state_ssf=state_ssf,
        state_ast_tuple=state_ast_tuple,
        iter_=node.iter,
        iterate=node.target,
        extra_test_name=self.ctx.namer.new_symbol('extra_test', all_referenced),
        extra_test_expr=extra_test,
        body_name=self.ctx.namer.new_symbol('loop_body', all_referenced),
        body=node_body)

    return node


def transform(node, ctx):
  cfg.run_analyses(node, cfg.Liveness(ctx.info))
  cfg.run_analyses(node, cfg.Defined(ctx.info))
  node = ControlFlowTransformer(ctx).visit(node)
  return node
