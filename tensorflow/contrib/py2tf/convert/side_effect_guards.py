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

from tensorflow.contrib.py2tf.pyct import anno
from tensorflow.contrib.py2tf.pyct import templates


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


class SideEffectGuardTransformer(gast.NodeTransformer):
  """Adds control dependencies to functions with side effects."""

  def __init__(self, namer):
    self.namer = namer
    self.indent_next = False
    self.next_indent_owner = None

  # pylint:disable=invalid-name

  def _visit_and_reindent(self, nodes):
    new_nodes = []
    current_dest = new_nodes
    for n in nodes:
      n = self.visit(n)
      if isinstance(n, (list, tuple)):
        current_dest.extend(n)
      else:
        current_dest.append(n)
      if self.indent_next:
        assert self.next_indent_owner is not None
        current_dest.append(self.next_indent_owner)
        current_dest = self.next_indent_owner.body
        self.next_indent_owner = None
        self.indent_next = False
    if not current_dest:
      # TODO(mdan): There may still be something that could be done.
      raise ValueError('Unable to insert statement into the computation flow: '
                       'it is not followed by any computation that can we can '
                       'condition on the statement.')
    return new_nodes

  def visit_FunctionDef(self, node):
    if anno.hasanno(node, 'skip_processing'):
      return node
    node.body = self._visit_and_reindent(node.body)
    return node

  def _gate_symbols(self, guard_statement, guarded_args):

    def template(args):  # pylint:disable=unused-argument
      (args,) = (tf.identity(a) for a in (args,))  # pylint:disable=undefined-variable

    guards = templates.replace(
        template, args=tuple(gast.Name(a, None, None) for a in guarded_args))
    guard_statement.body.extend(guards)
    return guard_statement

  def visit_Expr(self, node):
    self.generic_visit(node)
    if isinstance(node.value, gast.Call):
      # Patterns of single function calls, like:
      #   opt.minimize(loss)
      # or:
      #   tf.py_func(...)

      args_scope = anno.getanno(node.value, 'args_scope')
      temp_name = self.namer.new_symbol('temp', args_scope.parent.referenced)
      # TODO(mdan): Unsafe reference modification!
      args_scope.mark_write(temp_name)

      def template(call, temp_result):
        temp_result = call
        if temp_result is not None:
          if not isinstance(temp_result, (list, tuple)):
            temp_result = (temp_result,)
          ctx = tf.control_dependencies(temp_result)  # pylint:disable=undefined-variable
        else:
          ctx = contextmanager(lambda: (yield))()  # pylint:disable=undefined-variable
        with ctx:
          # TODO(mdan): Also insert ops to re-fetch if variables are involved.
          pass  # Will be removed below.

      # TODO(mdan): This is brittle. Reorganize this mechanism.
      statements = templates.replace(
          template,
          call=node.value,
          temp_result=gast.Name(temp_name, None, None))
      control_deps_guard = statements[-1]
      control_deps_guard.body = []

      # First, attempt to gate future evaluation of args. If that's not
      # possible, gate all remaining statements (and that may fail too, see
      # _visit_and_reindent.
      guarded_args = tuple(
          n for n in args_scope.used if n in args_scope.parent.modified)
      if guarded_args:
        node = tuple(statements[:-1]) + (
            self._gate_symbols(control_deps_guard, guarded_args),)
      else:
        node = tuple(statements[:-1])
        # The mechanism will insert the guard statement later.
        self.indent_next = True
        self.next_indent_owner = control_deps_guard
    return node

  # pylint:enable=invalid-name


def transform(node, namer):
  transformer = SideEffectGuardTransformer(namer)
  return transformer.visit(node)
