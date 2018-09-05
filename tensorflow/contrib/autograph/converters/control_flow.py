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
from tensorflow.contrib.autograph.pyct.static_analysis import annos


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

  def _fmt_symbol_list(self, symbol_set):
    if not symbol_set:
      return 'no variables'
    return ', '.join(map(str, symbol_set))

  def _validate_no_live_vars_created(self, node):
    body_scope = anno.getanno(node, annos.NodeAnno.BODY_SCOPE)
    live_vars_out = anno.getanno(node, anno.Static.LIVE_VARS_OUT)
    live_vars_created_in_body = live_vars_out & body_scope.created
    if live_vars_created_in_body:
      raise ValueError(
          'The following variables are created inside the loop and used later:'
          '\n%s\n'
          'Variables must be declared outside loops because loops may not'
          ' necessarily execute.' % self._fmt_symbol_list(
              live_vars_created_in_body))

  def visit_If(self, node):
    node = self.generic_visit(node)

    body_scope = anno.getanno(node, annos.NodeAnno.BODY_SCOPE)
    orelse_scope = anno.getanno(node, annos.NodeAnno.ORELSE_SCOPE)
    defined_in = anno.getanno(node, anno.Static.DEFINED_VARS_IN)
    live_out = anno.getanno(node, anno.Static.LIVE_VARS_OUT)

    modified_in_cond = body_scope.modified | orelse_scope.modified
    returned_from_cond = set()
    for s in modified_in_cond:
      if s in live_out:
        returned_from_cond.add(s)
      elif s.is_composite():
        # Special treatment for compound objects: if any of their owner entities
        # are live, then they are outputs as well.
        if any(owner in live_out for owner in s.owner_set):
          returned_from_cond.add(s)

    need_alias_in_body = body_scope.modified & defined_in
    need_alias_in_orelse = orelse_scope.modified & defined_in

    created_in_body = body_scope.modified & returned_from_cond - defined_in
    created_in_orelse = orelse_scope.modified & returned_from_cond - defined_in

    if created_in_body != created_in_orelse:
      raise ValueError(
          'if statement may not initialize all variables: the true branch'
          ' creates %s, while the false branch creates %s. Make sure all'
          ' these variables are initialized either in both'
          ' branches or before the if statement.' %
          (self._fmt_symbol_list(created_in_body),
           self._fmt_symbol_list(created_in_orelse)))

    # Alias the closure variables inside the conditional functions, to allow
    # the functions access to the respective variables.
    # We will alias variables independently for body and orelse scope,
    # because different branches might write different variables.
    aliased_body_orig_names = tuple(need_alias_in_body)
    aliased_orelse_orig_names = tuple(need_alias_in_orelse)
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

    returned_from_cond = tuple(returned_from_cond)
    if returned_from_cond:
      if len(returned_from_cond) == 1:
        # TODO(mdan): Move this quirk into the operator implementation.
        cond_results = returned_from_cond[0]
      else:
        cond_results = gast.Tuple([s.ast() for s in returned_from_cond], None)

      returned_from_body = tuple(
          alias_body_map[s] if s in need_alias_in_body else s
          for s in returned_from_cond)
      returned_from_orelse = tuple(
          alias_orelse_map[s] if s in need_alias_in_orelse else s
          for s in returned_from_cond)

    else:
      # When the cond would return no value, we leave the cond called without
      # results. That in turn should trigger the side effect guards. The
      # branch functions will return a dummy value that ensures cond
      # actually has some return value as well.
      cond_results = None
      # TODO(mdan): This doesn't belong here; it's specific to the operator.
      returned_from_body = templates.replace_as_expression('tf.constant(1)')
      returned_from_orelse = templates.replace_as_expression('tf.constant(1)')

    body_name = self.ctx.namer.new_symbol('if_true', body_scope.referenced)
    orelse_name = self.ctx.namer.new_symbol('if_false', orelse_scope.referenced)

    body_def = self._create_cond_branch(
        body_name,
        aliased_orig_names=aliased_body_orig_names,
        aliased_new_names=aliased_body_new_names,
        body=node_body,
        returns=returned_from_body)
    orelse_def = self._create_cond_branch(
        orelse_name,
        aliased_orig_names=aliased_orelse_orig_names,
        aliased_new_names=aliased_orelse_new_names,
        body=node_orelse,
        returns=returned_from_orelse)
    cond_expr = self._create_cond_expr(cond_results, node.test, body_name,
                                       orelse_name)

    return body_def + orelse_def + cond_expr

  def visit_While(self, node):
    self.generic_visit(node)

    self._validate_no_live_vars_created(node)

    body_scope = anno.getanno(node, annos.NodeAnno.BODY_SCOPE)
    body_closure = body_scope.modified - body_scope.created
    all_referenced = body_scope.referenced

    cond_scope = anno.getanno(node, annos.NodeAnno.COND_SCOPE)
    cond_closure = set()
    for s in cond_scope.used:
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

    # TODO(b/113118541) investigate the need-for and correctness-of extra_deps
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

    self._validate_no_live_vars_created(node)

    body_scope = anno.getanno(node, annos.NodeAnno.BODY_SCOPE)
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
      def body_name(loop_vars, state_ssf):
        # Workaround for PEP-3113
        iterate = loop_vars
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
  node = ControlFlowTransformer(ctx).visit(node)
  return node
