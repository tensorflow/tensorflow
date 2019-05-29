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

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct.static_analysis import annos


class ControlFlowTransformer(converter.Base):
  """Transforms control flow structures like loops an conditionals."""

  def _create_cond_branch(self, body_name, aliased_orig_names,
                          aliased_new_names, body, returns):
    if not returns:
      # TODO(b/110167197): Replace with a plain return.
      template = """
        return 1
      """
      return_stmt = templates.replace(template)
    elif len(returns) == 1:
      template = """
        return retval
      """
      return_stmt = templates.replace(template, retval=returns[0])
    else:
      template = """
        return (retvals,)
      """
      return_stmt = templates.replace(template, retvals=returns)

    if aliased_orig_names:
      template = """
        def body_name():
          aliased_new_names, = aliased_orig_names,
          body
          return_stmt
      """
      return templates.replace(
          template,
          body_name=body_name,
          body=body,
          aliased_orig_names=aliased_orig_names,
          aliased_new_names=aliased_new_names,
          return_stmt=return_stmt)
    else:
      template = """
        def body_name():
          body
          return_stmt
      """
      return templates.replace(
          template, body_name=body_name, body=body, return_stmt=return_stmt)

  def _create_cond_expr(self, results, test, body_name, orelse_name,
                        state_getter_name,
                        state_setter_name):
    if results is not None:
      template = """
        results = ag__.if_stmt(test, body_name, orelse_name,
                               state_getter_name, state_setter_name)
      """
      return templates.replace(
          template,
          test=test,
          results=results,
          body_name=body_name,
          orelse_name=orelse_name,
          state_getter_name=state_getter_name,
          state_setter_name=state_setter_name)
    else:
      template = """
        ag__.if_stmt(test, body_name, orelse_name, getter_name, setter_name)
      """
      return templates.replace(
          template,
          test=test,
          body_name=body_name,
          orelse_name=orelse_name,
          getter_name=state_getter_name,
          setter_name=state_setter_name)

  def _fmt_symbols(self, symbol_set):
    if not symbol_set:
      return 'no variables'
    return ', '.join(map(str, symbol_set))

  def _determine_aliased_symbols(self, scope, node_defined_in, block):
    if block:
      block_live_in = set(anno.getanno(block[0], anno.Static.LIVE_VARS_IN))
    else:
      block_live_in = set()

    modified_live = scope.modified & node_defined_in & block_live_in
    # Composite symbols are handled elsewhere see _create_state_functions
    return {s for s in modified_live if not s.is_composite()}

  def _create_state_functions(self, composites,
                              state_getter_name, state_setter_name):
    if composites:
      composite_tuple = tuple(composites)
      template = """
        def state_getter_name():
          return composite_tuple,
        def state_setter_name(vals):
          composite_tuple, = vals
      """
      node = templates.replace(
          template,
          state_getter_name=state_getter_name,
          state_setter_name=state_setter_name,
          composite_tuple=composite_tuple)
    else:
      template = """
        def state_getter_name():
          return ()
        def state_setter_name(_):
          pass
        """
      node = templates.replace(
          template,
          state_getter_name=state_getter_name,
          state_setter_name=state_setter_name)

    return node

  def _create_undefined_assigns(self, undefined_symbols):
    assignments = []
    for s in undefined_symbols:
      template = '''
        var = ag__.Undefined(symbol_name)
      '''
      assignments += templates.replace(
          template,
          var=s,
          symbol_name=gast.Str(s.ssf()))
    return assignments

  def visit_If(self, node):
    body_scope = anno.getanno(node, annos.NodeAnno.BODY_SCOPE)
    orelse_scope = anno.getanno(node, annos.NodeAnno.ORELSE_SCOPE)
    defined_in = anno.getanno(node, anno.Static.DEFINED_VARS_IN)
    live_out = anno.getanno(node, anno.Static.LIVE_VARS_OUT)

    # Note: this information needs to be extracted before the body conversion
    # that happens in the call to generic_visit below, because the conversion
    # generates nodes that lack static analysis annotations.
    need_alias_in_body = self._determine_aliased_symbols(
        body_scope, defined_in, node.body)
    need_alias_in_orelse = self._determine_aliased_symbols(
        orelse_scope, defined_in, node.orelse)

    node = self.generic_visit(node)

    modified_in_cond = body_scope.modified | orelse_scope.modified
    returned_from_cond = set()
    composites = set()
    for s in modified_in_cond:
      if s in live_out and not s.is_composite():
        returned_from_cond.add(s)
      if s.is_composite():
        # Special treatment for compound objects, always return them.
        # This allows special handling within the if_stmt itself.
        # For example, in TensorFlow we need to restore the state of composite
        # symbols to ensure that only effects from the executed branch are seen.
        composites.add(s)

    created_in_body = body_scope.modified & returned_from_cond - defined_in
    created_in_orelse = orelse_scope.modified & returned_from_cond - defined_in

    basic_created_in_body = tuple(
        s for s in created_in_body if not s.is_composite())
    basic_created_in_orelse = tuple(
        s for s in created_in_orelse if not s.is_composite())

    # These variables are defined only in a single branch. This is fine in
    # Python so we pass them through. Another backend, e.g. Tensorflow, may need
    # to handle these cases specially or throw an Error.
    possibly_undefined = (set(basic_created_in_body) ^
                          set(basic_created_in_orelse))

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

    cond_var_name = self.ctx.namer.new_symbol('cond', body_scope.referenced)
    body_name = self.ctx.namer.new_symbol('if_true', body_scope.referenced)
    orelse_name = self.ctx.namer.new_symbol('if_false', orelse_scope.referenced)
    all_referenced = body_scope.referenced | orelse_scope.referenced
    state_getter_name = self.ctx.namer.new_symbol('get_state', all_referenced)
    state_setter_name = self.ctx.namer.new_symbol('set_state', all_referenced)

    returned_from_cond = tuple(returned_from_cond)
    if returned_from_cond:
      if len(returned_from_cond) == 1:
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
      # TODO(mdan): Replace with None once side_effect_guards is retired.
      returned_from_body = (templates.replace_as_expression(
          'ag__.match_staging_level(1, cond_var_name)',
          cond_var_name=cond_var_name),)
      returned_from_orelse = (templates.replace_as_expression(
          'ag__.match_staging_level(1, cond_var_name)',
          cond_var_name=cond_var_name),)

    cond_assign = self.create_assignment(cond_var_name, node.test)
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
    undefined_assigns = self._create_undefined_assigns(possibly_undefined)
    composite_defs = self._create_state_functions(
        composites, state_getter_name, state_setter_name)

    cond_expr = self._create_cond_expr(cond_results, cond_var_name, body_name,
                                       orelse_name, state_getter_name,
                                       state_setter_name)

    if_ast = (undefined_assigns + cond_assign + composite_defs + body_def +
              orelse_def + cond_expr)
    return if_ast

  def _get_loop_state(self, node, modified_symbols):
    body_scope = anno.getanno(node, annos.NodeAnno.BODY_SCOPE)
    defined_in = anno.getanno(node, anno.Static.DEFINED_VARS_IN)
    live_in = anno.getanno(node, anno.Static.LIVE_VARS_IN)
    live_out = anno.getanno(node, anno.Static.LIVE_VARS_OUT)
    reserved_symbols = body_scope.referenced
    loop_state = []
    for s in modified_symbols:

      # Variables not live into or out of the loop are considered local to the
      # loop.
      if s not in live_in and s not in live_out:
        continue

      # Mutations made to objects created inside the loop will appear as writes
      # to composite symbols. Because these mutations appear as modifications
      # made to composite symbols, we check whether the composite's parent is
      # actually live into the loop.
      # Example:
      #   while cond:
      #     x = Foo()
      #     x.foo = 2 * x.foo  # x.foo is live into the loop, but x is not.
      if s.is_composite() and not all(p in live_in for p in s.support_set):
        continue

      loop_state.append(s)
    loop_state = frozenset(loop_state)

    # Variable that are used or defined inside the loop, but not defined
    # before entering the loop
    undefined_lives = loop_state - defined_in

    # Only simple variables must be defined. The composite ones will be
    # implicitly checked at runtime.
    possibly_undefs = {v for v in undefined_lives if v.is_simple()}

    return loop_state, reserved_symbols, possibly_undefs

  def _state_constructs(self, loop_state, reserved_symbols):
    loop_state = tuple(loop_state)
    state_ssf = [
        self.ctx.namer.new_symbol(s.ssf(), reserved_symbols) for s in loop_state
    ]
    ssf_map = {
        name: ssf
        for name, ssf in zip(loop_state, state_ssf)
        if str(name) != ssf
    }

    state_ast_tuple = gast.Tuple([n.ast() for n in loop_state], None)

    if len(loop_state) == 1:
      loop_state = loop_state[0]
      state_ssf = state_ssf[0]

    return loop_state, state_ssf, state_ast_tuple, ssf_map

  def visit_While(self, node):
    self.generic_visit(node)

    loop_state, reserved_symbols, possibly_undefs = self._get_loop_state(
        node, anno.getanno(node, annos.NodeAnno.BODY_SCOPE).modified)
    loop_state, state_ssf, state_ast_tuple, ssf_map = self._state_constructs(
        loop_state, reserved_symbols)
    node_body = ast_util.rename_symbols(node.body, ssf_map)
    test = ast_util.rename_symbols(node.test, ssf_map)

    if loop_state:
      template = """
        def test_name(state_ssf):
          return test
        def body_name(state_ssf):
          body
          return state_ssf,
        state_ast_tuple = ag__.while_stmt(test_name, body_name, (state,))
      """
      node = templates.replace(
          template,
          state=loop_state,
          state_ssf=state_ssf,
          state_ast_tuple=state_ast_tuple,
          test_name=self.ctx.namer.new_symbol('loop_test', reserved_symbols),
          test=test,
          body_name=self.ctx.namer.new_symbol('loop_body', reserved_symbols),
          body=node_body)
    else:
      template = """
        def test_name():
          return test
        def body_name():
          body
          return ()
        ag__.while_stmt(test_name, body_name, ())
      """
      node = templates.replace(
          template,
          test_name=self.ctx.namer.new_symbol('loop_test', reserved_symbols),
          test=test,
          body_name=self.ctx.namer.new_symbol('loop_body', reserved_symbols),
          body=node_body)

    undefined_assigns = self._create_undefined_assigns(possibly_undefs)
    return undefined_assigns + node

  def _for_loop_with_extra_test(self, loop_state, state_ssf, state_ast_tuple,
                                original_node, extra_test_name, extra_test,
                                body_name, loop_body, ssf_map):
    target_nodes = ast_util.rename_symbols(original_node.target, ssf_map)
    template = """
      def extra_test_name(state_ssf):
        return extra_test_expr
      def body_name(loop_vars, state_ssf):
        # Workaround for PEP-3113
        target = loop_vars
        body
        return state_ssf,
      state_ast_tuple = ag__.for_stmt(
          iter_, extra_test_name, body_name, (state,))
    """
    return templates.replace(
        template,
        state=loop_state,
        state_ssf=state_ssf,
        state_ast_tuple=state_ast_tuple,
        iter_=original_node.iter,
        target=target_nodes,
        extra_test_name=extra_test_name,
        extra_test_expr=extra_test,
        body_name=body_name,
        body=loop_body)

  def _for_loop_with_state(self, loop_state, state_ssf, state_ast_tuple,
                           original_node, body_name, loop_body, ssf_map):
    target_nodes = ast_util.rename_symbols(original_node.target, ssf_map)
    template = """
      def body_name(loop_vars, state_ssf):
        # Workaround for PEP-3113
        target = loop_vars
        body
        return state_ssf,
      state_ast_tuple = ag__.for_stmt(
          iter_, None, body_name, (state,))
    """
    return templates.replace(
        template,
        state=loop_state,
        state_ssf=state_ssf,
        state_ast_tuple=state_ast_tuple,
        iter_=original_node.iter,
        target=target_nodes,
        body_name=body_name,
        body=loop_body)

  def _for_loop_without_state(self, original_node, body_name, loop_body):
    template = """
      def body_name(loop_vars):
        # Workaround for PEP-3113
        iterate = loop_vars
        body
        return ()
      ag__.for_stmt(iter_, None, body_name, ())
    """
    return templates.replace(
        template,
        iter_=original_node.iter,
        iterate=original_node.target,
        body_name=body_name,
        body=loop_body)

  def visit_For(self, node):
    self.generic_visit(node)

    loop_state, reserved_symbols, possibly_undefs = self._get_loop_state(
        node,
        (anno.getanno(node, annos.NodeAnno.BODY_SCOPE).modified |
         anno.getanno(node, annos.NodeAnno.ITERATE_SCOPE).modified))
    loop_state, state_ssf, state_ast_tuple, ssf_map = self._state_constructs(
        loop_state, reserved_symbols)
    node_body = ast_util.rename_symbols(node.body, ssf_map)
    body_name = self.ctx.namer.new_symbol('loop_body', reserved_symbols)

    has_extra_test = anno.hasanno(node, 'extra_test')
    if loop_state:
      if has_extra_test:
        # Loop with early stopping (e.g. break or return)
        extra_test = anno.getanno(node, 'extra_test')
        extra_test = ast_util.rename_symbols(extra_test, ssf_map)
        extra_test_name = self.ctx.namer.new_symbol('extra_test',
                                                    reserved_symbols)
        loop_nodes = self._for_loop_with_extra_test(
            loop_state, state_ssf, state_ast_tuple, node, extra_test_name,
            extra_test, body_name, node_body, ssf_map)
      else:
        # Loop with loop-carried state and no early stopping
        loop_nodes = self._for_loop_with_state(
            loop_state, state_ssf, state_ast_tuple, node, body_name, node_body,
            ssf_map)
    else:
      # Loop with no loop-carried state and no early stopping
      assert not has_extra_test, ('Early stopping (e.g. break and/or return) '
                                  'should create state variables.')
      loop_nodes = self._for_loop_without_state(node, body_name, node_body)

    undefined_assigns = self._create_undefined_assigns(possibly_undefs)
    return undefined_assigns + loop_nodes


def transform(node, ctx):
  node = ControlFlowTransformer(ctx).visit(node)
  return node
