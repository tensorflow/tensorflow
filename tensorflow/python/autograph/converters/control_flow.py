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
from tensorflow.python.autograph.lang import directives
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import annos
from tensorflow.python.autograph.pyct.static_analysis import liveness
from tensorflow.python.autograph.pyct.static_analysis import reaching_definitions
from tensorflow.python.autograph.pyct.static_analysis import reaching_fndefs
from tensorflow.python.autograph.utils import compat_util


# TODO(mdan): Refactor functions to make them smaller.


class _Function(object):

  scope = None


class ControlFlowTransformer(converter.Base):
  """Transforms control flow structures like loops an conditionals."""

  def visit_Lambda(self, node):
    with self.state[_Function] as fn:
      fn.scope = anno.getanno(node, anno.Static.SCOPE)
      return self.generic_visit(node)

  def visit_FunctionDef(self, node):
    with self.state[_Function] as fn:
      fn.scope = anno.getanno(node, annos.NodeAnno.BODY_SCOPE)
      return self.generic_visit(node)

  def _create_cond_branch(self, body_name, aliased_orig_names,
                          aliased_new_names, body, returns):
    if len(returns) == 1:
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
      alias_declarations = []
      for new_name, old_name in zip(aliased_new_names, aliased_orig_names):
        template = """
          try:
            aliased_new_name = aliased_orig_name
          except NameError:
            aliased_new_name = ag__.Undefined(symbol_name)
        """

        alias_declarations.extend(
            templates.replace(
                template,
                aliased_new_name=new_name,
                aliased_orig_name=old_name,
                symbol_name=gast.Constant(str(old_name), kind=None)))

      template = """
        def body_name():
          alias_declarations
          body
          return_stmt
      """
      return templates.replace(
          template,
          alias_declarations=alias_declarations,
          body_name=body_name,
          body=body,
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
                        state_getter_name, state_setter_name,
                        basic_symbol_names, composite_symbol_names):
    if results is not None:
      template = """
        results = ag__.if_stmt(test, body_name, orelse_name,
                               state_getter_name, state_setter_name,
                               (basic_symbol_names,),
                               (composite_symbol_names,))
      """
      return templates.replace(
          template,
          test=test,
          results=results,
          body_name=body_name,
          orelse_name=orelse_name,
          state_getter_name=state_getter_name,
          state_setter_name=state_setter_name,
          basic_symbol_names=basic_symbol_names,
          composite_symbol_names=composite_symbol_names)
    else:
      template = """
        ag__.if_stmt(test, body_name, orelse_name, getter_name, setter_name,
                     (basic_symbol_names,), (composite_symbol_names,))
      """
      return templates.replace(
          template,
          test=test,
          body_name=body_name,
          orelse_name=orelse_name,
          getter_name=state_getter_name,
          setter_name=state_setter_name,
          basic_symbol_names=basic_symbol_names,
          composite_symbol_names=composite_symbol_names)

  def _fmt_symbols(self, symbol_set):
    if not symbol_set:
      return 'no variables'
    return ', '.join(map(str, symbol_set))

  def _determine_aliased_symbols(self, scope, node_defined_in):
    modified_live = scope.modified & node_defined_in
    # Composite symbols are handled elsewhere, see _create_state_functions
    return {
        s for s in modified_live
        if not s.is_composite() and s not in self.state[_Function].scope.globals
    }

  def _create_nonlocal_declarations(self, loop_vars):
    results = []
    global_vars = self.state[_Function].scope.globals

    if global_vars:
      results.append(gast.Global([str(v) for v in global_vars]))

    nonlocal_vars = [
        v for v in loop_vars if not v.is_composite() and v not in global_vars]
    if nonlocal_vars:
      results.append(gast.Nonlocal([str(v) for v in nonlocal_vars]))

    return results

  def _create_state_functions(
      self, loop_vars, nonlocal_declarations, getter_name, setter_name):
    if loop_vars:
      template = """
        def getter_name():
          return state_vars,
        def setter_name(loop_vars):
          nonlocal_declarations
          state_vars, = loop_vars
      """
      return templates.replace(
          template,
          nonlocal_declarations=nonlocal_declarations,
          getter_name=getter_name,
          setter_name=setter_name,
          state_vars=tuple(loop_vars))
    else:
      template = """
        def getter_name():
          return ()
        def setter_name(loop_vars):
          pass
      """
      return templates.replace(
          template, getter_name=getter_name, setter_name=setter_name)

  def _create_loop_options(self, node):
    if not anno.hasanno(node, anno.Basic.DIRECTIVES):
      return gast.Dict([], [])

    loop_directives = anno.getanno(node, anno.Basic.DIRECTIVES)
    if directives.set_loop_options not in loop_directives:
      return gast.Dict([], [])

    opts_dict = loop_directives[directives.set_loop_options]
    str_keys, values = zip(*opts_dict.items())
    keys = [gast.Constant(s, kind=None) for s in str_keys]
    values = list(values)  # ast and gast don't play well with tuples.
    return gast.Dict(keys, values)

  def _create_undefined_assigns(self, undefined_symbols):
    assignments = []
    for s in undefined_symbols:
      template = '''
        var = ag__.Undefined(symbol_name)
      '''
      assignments += templates.replace(
          template,
          var=s,
          symbol_name=gast.Constant(s.ssf(), kind=None))
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
        body_scope, defined_in)
    need_alias_in_orelse = self._determine_aliased_symbols(
        orelse_scope, defined_in)

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
    composites = tuple(composites)

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
        composites, [], state_getter_name, state_setter_name)

    basic_symbol_names = tuple(
        gast.Constant(str(symbol), kind=None) for symbol in returned_from_cond)
    composite_symbol_names = tuple(
        gast.Constant(str(symbol), kind=None) for symbol in composites)

    cond_expr = self._create_cond_expr(cond_results, cond_var_name, body_name,
                                       orelse_name, state_getter_name,
                                       state_setter_name, basic_symbol_names,
                                       composite_symbol_names)

    if_ast = (
        undefined_assigns + composite_defs + body_def + orelse_def +
        cond_assign + cond_expr)
    return if_ast

  def _get_basic_loop_vars(self, modified, live_in, live_out):
    # The loop variables corresponding to simple symbols (e.g. `x`).
    basic_loop_vars = []
    for s in modified:
      if s.is_composite():
        # TODO(mdan): Raise an error when this happens for a TF loop.
        continue
      # Variables not live into or out of the loop are considered local to the
      # loop.
      if s not in live_in and s not in live_out:
        continue
      basic_loop_vars.append(s)
    return frozenset(basic_loop_vars)

  def _get_composite_loop_vars(self, modified, live_in):
    # The loop variables corresponding to composite symbols (e.g. `self.x`).
    composite_loop_vars = []
    for s in modified:
      if not s.is_composite():
        continue
      # Mutations made to objects created inside the loop will appear as writes
      # to composite symbols. Because these mutations appear as modifications
      # made to composite symbols, we check whether the composite's parent is
      # actually live into the loop.
      # Example:
      #   while cond:
      #     x = Foo()
      #     x.foo = 2 * x.foo  # x.foo is live into the loop, but x is not.
      #
      # Note that some parents might not be symbols - for example, in x['foo'],
      # 'foo' is a parent, but it's a literal, not a symbol. We don't check the
      # liveness of literals.
      support_set_symbols = tuple(
          sss for sss in s.support_set if sss.is_symbol())
      if not all(sss in live_in for sss in support_set_symbols):
        continue
      composite_loop_vars.append(s)
    return frozenset(composite_loop_vars)

  def _get_loop_vars(self, node, modified):
    body_scope = anno.getanno(node, annos.NodeAnno.BODY_SCOPE)
    defined_in = anno.getanno(node, anno.Static.DEFINED_VARS_IN)
    live_in = anno.getanno(node, anno.Static.LIVE_VARS_IN)
    live_out = anno.getanno(node, anno.Static.LIVE_VARS_OUT)
    reserved_symbols = body_scope.referenced

    basic_loop_vars = self._get_basic_loop_vars(modified, live_in, live_out)
    composite_loop_vars = self._get_composite_loop_vars(modified, live_in)
    loop_vars = tuple(basic_loop_vars | composite_loop_vars)

    # Variable that are used or defined inside the loop, but not defined
    # before entering the loop. Only simple variables must be defined. The
    # composite ones will be implicitly checked at runtime.
    undefined_lives = basic_loop_vars - defined_in

    return loop_vars, reserved_symbols, undefined_lives

  def visit_While(self, node):
    node = self.generic_visit(node)
    body_scope = anno.getanno(node, annos.NodeAnno.BODY_SCOPE)

    loop_vars, reserved_symbols, possibly_undefs = self._get_loop_vars(
        node, body_scope.modified)

    undefined_assigns = self._create_undefined_assigns(possibly_undefs)

    nonlocal_declarations = self._create_nonlocal_declarations(loop_vars)

    state_getter_name = self.ctx.namer.new_symbol('get_state', reserved_symbols)
    state_setter_name = self.ctx.namer.new_symbol('set_state', reserved_symbols)
    state_functions = self._create_state_functions(
        loop_vars, nonlocal_declarations, state_getter_name, state_setter_name)

    opts = self._create_loop_options(node)

    template = """
      state_functions
      def body_name():
        nonlocal_declarations
        body
      def test_name():
        return test
      undefined_assigns
      ag__.while_stmt(
          test_name,
          body_name,
          state_getter_name,
          state_setter_name,
          (symbol_names,),
          opts)
    """
    return templates.replace(
        template,
        body=node.body,
        body_name=self.ctx.namer.new_symbol('loop_body', reserved_symbols),
        nonlocal_declarations=nonlocal_declarations,
        opts=opts,
        state_functions=state_functions,
        state_getter_name=state_getter_name,
        state_setter_name=state_setter_name,
        symbol_names=tuple(gast.Constant(str(s), kind=None) for s in loop_vars),
        test=node.test,
        test_name=self.ctx.namer.new_symbol('loop_test', reserved_symbols),
        undefined_assigns=undefined_assigns)

  def visit_For(self, node):
    node = self.generic_visit(node)
    body_scope = anno.getanno(node, annos.NodeAnno.BODY_SCOPE)
    iter_scope = anno.getanno(node, annos.NodeAnno.ITERATE_SCOPE)

    loop_vars, reserved_symbols, possibly_undefs = self._get_loop_vars(
        node, body_scope.modified | iter_scope.modified)

    undefined_assigns = self._create_undefined_assigns(possibly_undefs)

    nonlocal_declarations = self._create_nonlocal_declarations(loop_vars)

    state_getter_name = self.ctx.namer.new_symbol('get_state', reserved_symbols)
    state_setter_name = self.ctx.namer.new_symbol('set_state', reserved_symbols)
    state_functions = self._create_state_functions(
        loop_vars, nonlocal_declarations, state_getter_name, state_setter_name)

    opts = self._create_loop_options(node)
    opts.keys.append(gast.Constant('iterate_names', kind=None))
    opts.values.append(gast.Constant(
        parser.unparse(node.target, include_encoding_marker=False), kind=None))

    if anno.hasanno(node, anno.Basic.EXTRA_LOOP_TEST):
      extra_test = anno.getanno(node, anno.Basic.EXTRA_LOOP_TEST)
      extra_test_name = self.ctx.namer.new_symbol(
          'extra_test', reserved_symbols)
      template = """
        def extra_test_name():
          nonlocal_declarations
          return extra_test_expr
      """
      extra_test_function = templates.replace(
          template,
          extra_test_expr=extra_test,
          extra_test_name=extra_test_name,
          loop_vars=loop_vars,
          nonlocal_declarations=nonlocal_declarations)
    else:
      extra_test_name = parser.parse_expression('None')
      extra_test_function = []

    # iterate_arg_name holds a single arg with the iterates, which may be a
    # tuple.
    iterate_arg_name = self.ctx.namer.new_symbol('itr', reserved_symbols)
    template = """
      iterates = iterate_arg_name
    """
    iterate_expansion = templates.replace(
        template, iterate_arg_name=iterate_arg_name, iterates=node.target)

    template = """
      state_functions
      def body_name(iterate_arg_name):
        nonlocal_declarations
        iterate_expansion
        body
      extra_test_function
      undefined_assigns
      ag__.for_stmt(
          iterated,
          extra_test_name,
          body_name,
          state_getter_name,
          state_setter_name,
          (symbol_names,),
          opts)
    """
    return templates.replace(
        template,
        body=node.body,
        body_name=self.ctx.namer.new_symbol('loop_body', reserved_symbols),
        extra_test_function=extra_test_function,
        extra_test_name=extra_test_name,
        iterate_arg_name=iterate_arg_name,
        iterate_expansion=iterate_expansion,
        iterated=node.iter,
        nonlocal_declarations=nonlocal_declarations,
        opts=opts,
        symbol_names=tuple(gast.Constant(str(s), kind=None) for s in loop_vars),
        state_functions=state_functions,
        state_getter_name=state_getter_name,
        state_setter_name=state_setter_name,
        undefined_assigns=undefined_assigns)


class AnnotatedDef(reaching_definitions.Definition):

  def __init__(self):
    super(AnnotatedDef, self).__init__()
    self.directives = {}


def transform(node, ctx):
  graphs = cfg.build(node)
  node = qual_names.resolve(node)
  node = activity.resolve(node, ctx, None)
  node = reaching_definitions.resolve(node, ctx, graphs)
  node = reaching_fndefs.resolve(node, ctx, graphs)
  node = liveness.resolve(node, ctx, graphs)

  node = ControlFlowTransformer(ctx).visit(node)
  return node


compat_util.deprecated_py2_support(__name__)
