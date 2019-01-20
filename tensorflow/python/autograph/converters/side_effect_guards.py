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
"""Adds guards against function calls with side effects.

Only standalone calls are guarded.

WARNING: This mechanism is incomplete. Particularly, it only guards the
arguments passed to functions, and does not account for indirectly modified
state.

Example:
  y = tf.layers.dense(x)       # Creates TF variable 'foo'
  loss = loss(y)
  opt.minimize(loss)           # indirectly affects 'foo'
  z = tf.get_variable('foo')   # Indirectly affects `loss` and 'foo'
  # Here, `loss` can be guarded. But `z` cannot.

# TODO(mdan): We should probably define a safe mode where we guard everything.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno


class SymbolNamer(object):
  """Describes the interface for SideEffectGuardTransformer's namer."""

  def new_symbol(self, name_root, reserved_locals):
    """Generate a new unique function_name.

    Args:
      name_root: String, used as stem in the new name.
      reserved_locals: Set(string), additional local symbols that are reserved.
    Returns:
      String.
    """
    raise NotImplementedError()


class SideEffectGuardTransformer(converter.Base):
  """Adds control dependencies to functions with side effects."""

  def _visit_and_reindent(self, nodes):
    new_nodes = []
    current_dest = new_nodes
    alias_map = {}
    reindent_requested = False
    for n in nodes:
      n = self.visit(n)
      # NOTE: the order in which these statements execute is important; in
      # particular, watch out for ending up with cycles in the AST.
      if alias_map:
        n = ast_util.rename_symbols(n, alias_map)
      if isinstance(n, (list, tuple)):
        current_dest.extend(n)
      else:
        current_dest.append(n)
      if anno.hasanno(n, anno.Basic.INDENT_BLOCK_REMAINDER):
        reindent_requested = True
        new_dest, new_alias_map = anno.getanno(
            n, anno.Basic.INDENT_BLOCK_REMAINDER)
        anno.delanno(n, anno.Basic.INDENT_BLOCK_REMAINDER)
        new_alias_map.update(alias_map)
        alias_map = new_alias_map
        current_dest = new_dest

    if reindent_requested:
      no_controls_to_gate = False
      if not current_dest:
        no_controls_to_gate = True
      if len(current_dest) == 1:
        if ast_util.matches(current_dest[0], 'return'):
          no_controls_to_gate = True
        if ast_util.matches(current_dest[0], 'return ()'):
          no_controls_to_gate = True
        if ast_util.matches(current_dest[0], 'return []'):
          no_controls_to_gate = True
        if ast_util.matches(current_dest[0], 'return {}'):
          no_controls_to_gate = True
      if no_controls_to_gate:
        # TODO(mdan): There may still be something that could be done.
        raise ValueError(
            'Unable to insert statement into the computation flow: it is not'
            ' followed by any computation which the statement could gate.')

    return new_nodes

  def visit_FunctionDef(self, node):
    node.body = self._visit_and_reindent(node.body)
    return node

  def visit_With(self, node):
    node.body = self._visit_and_reindent(node.body)
    return node

  def visit_If(self, node):
    node.body = self._visit_and_reindent(node.body)
    node.orelse = self._visit_and_reindent(node.orelse)
    return node

  def visit_While(self, node):
    node.body = self._visit_and_reindent(node.body)
    node.orelse = self._visit_and_reindent(node.orelse)
    return node

  def visit_Expr(self, node):
    self.generic_visit(node)
    if isinstance(node.value, gast.Call):
      # Patterns of single function calls, like:
      #   opt.minimize(loss)
      # or:
      #   tf.py_func(...)

      # First, attempt to gate future evaluation of args. If that's not
      # possible, gate all remaining statements (and that may fail too, see
      # _visit_and_reindent.
      args_scope = anno.getanno(node.value, NodeAnno.ARGS_SCOPE)
      live_out = anno.getanno(node, anno.Static.LIVE_VARS_OUT)
      # NOTE: We can't guard object attributes because they may not be writable.
      # In addition, avoid renaming well-known names.
      # TODO(mdan): Move these names into config.
      unguarded_names = (qual_names.QN('self'), qual_names.QN('ag__'))
      guarded_args = tuple(s for s in live_out
                           if not s.is_composite() and s not in unguarded_names)

      # TODO(mdan): Include all arguments which depended on guarded_args too.
      # For example, the following will still cause a race:
      #   tf.assign(a, a + 1)
      #   b = a + 1
      #   tf.assign(a, a + 1)  # Control deps here should include `b`
      #   c = b + 1
      # Or maybe we should just raise an "unsafe assign" error?

      if guarded_args:
        # The aliases may need new names to avoid incorrectly making them local.
        # TODO(mdan): This is brutal. It will even rename modules - any fix?
        need_alias = tuple(
            s for s in guarded_args if s not in args_scope.parent.modified)
        aliased_new_names = tuple(
            qual_names.QN(
                self.ctx.namer.new_symbol(
                    s.ssf(), args_scope.parent.referenced)) for s in need_alias)
        alias_map = dict(zip(need_alias, aliased_new_names))
        if len(guarded_args) == 1:
          s, = guarded_args
          aliased_guarded_args = alias_map.get(s, s)
        else:
          aliased_guarded_args = gast.Tuple(
              [alias_map.get(s, s).ast() for s in guarded_args], None)

        template = """
          with ag__.utils.control_dependency_on_returns(call):
            aliased_guarded_args = ag__.utils.alias_tensors(guarded_args)
        """
        control_deps_guard = templates.replace(
            template,
            call=node.value,
            aliased_guarded_args=aliased_guarded_args,
            guarded_args=guarded_args)[-1]
      else:
        alias_map = {}

        template = """
          with ag__.utils.control_dependency_on_returns(call):
            pass
        """
        control_deps_guard = templates.replace(template, call=node.value)[-1]
        control_deps_guard.body = []

      node = control_deps_guard
      anno.setanno(node, anno.Basic.INDENT_BLOCK_REMAINDER,
                   (node.body, alias_map))
    return node


def transform(node, ctx):
  return SideEffectGuardTransformer(ctx).visit(node)
