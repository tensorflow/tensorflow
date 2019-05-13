# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Handles directives.

This converter removes the directive functions from the code and moves the
information they specify into AST annotations. It is a specialized form of
static analysis, one that is specific to AutoGraph.

Note that this requires that the actual directive functions are static - that
is, they do not change at runtime. So if you do something like this:

  tf.autograph.set_loop_options = <new function>

Then the directive will may no longer be recognized. Furthermore, if the
converted function is cached, such an action action may be irreversible.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.lang import directives
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.util import tf_inspect

ENCLOSING_LOOP = 'enclosing_loop'
STATIC_VALUE = 'static_value'
"""Used for AST annotations, see visit_Name."""


def _map_args(call_node, function):
  """Maps AST call nodes to the actual function's arguments.

  Args:
    call_node: ast.Call
    function: Callable[..., Any], the actual function matching call_node
  Returns:
    Dict[Text, ast.AST], mapping each of the function's argument names to
    the respective AST node.
  Raises:
      ValueError: if the default arguments are not correctly set
  """
  args = call_node.args
  kwds = {kwd.arg: kwd.value for kwd in call_node.keywords}
  call_args = tf_inspect.getcallargs(function, *args, **kwds)

  # Keyword arguments not specified in kwds will be mapped to their defaults,
  # which are Python values. Since we don't currently have a way to transform
  # those into AST references, we simply remove them. By convention, directives
  # use UNSPECIFIED as default value for for optional arguments. No other
  # defaults should be present.
  unexpected_defaults = []
  for k in call_args:
    if (k not in kwds
        and call_args[k] not in args
        and call_args[k] is not directives.UNSPECIFIED):
      unexpected_defaults.append(k)
  if unexpected_defaults:
    raise ValueError('Unexpected keyword argument values, %s, for function %s'
                     % (zip(unexpected_defaults,
                            [call_args[k] for k in unexpected_defaults]),
                        function))
  return {k: v for k, v in call_args.items() if v is not directives.UNSPECIFIED}


class DirectivesTransformer(converter.Base):
  """Parses compiler directives and converts them into AST annotations."""

  def _process_symbol_directive(self, call_node, directive):
    if len(call_node.args) < 1:
      raise ValueError('"%s" requires a positional first argument'
                       ' as the target' % directive.__name__)
    target = call_node.args[0]
    defs = anno.getanno(target, anno.Static.ORIG_DEFINITIONS)
    for def_ in defs:
      def_.directives[directive] = _map_args(call_node, directive)
    return call_node

  def _process_statement_directive(self, call_node, directive):
    if self.local_scope_level < 2:
      raise ValueError(
          '"%s" must be used inside a statement' % directive.__name__)
    target = self.get_local(ENCLOSING_LOOP)
    node_anno = anno.getanno(target, converter.AgAnno.DIRECTIVES, {})
    node_anno[directive] = _map_args(call_node, directive)
    anno.setanno(target, converter.AgAnno.DIRECTIVES, node_anno)
    return call_node

  def visit_Name(self, node):
    node = self.generic_visit(node)
    if isinstance(node.ctx, gast.Load):
      defs = anno.getanno(node, anno.Static.DEFINITIONS, ())
      is_defined = bool(defs)
      if not is_defined and node.id in self.ctx.info.namespace:
        anno.setanno(node, STATIC_VALUE, self.ctx.info.namespace[node.id])
    return node

  def visit_Attribute(self, node):
    node = self.generic_visit(node)
    parent_val = anno.getanno(node.value, STATIC_VALUE, default=None)
    if parent_val is not None and tf_inspect.ismodule(parent_val):
      if hasattr(parent_val, node.attr):
        anno.setanno(node, STATIC_VALUE, getattr(parent_val, node.attr))
    return node

  def visit_Expr(self, node):
    node = self.generic_visit(node)
    if isinstance(node.value, gast.Call):
      call_node = node.value
      static_val = anno.getanno(call_node.func, STATIC_VALUE, default=None)
      if static_val is not None:
        # Note: directive calls are not output in the generated code, hence
        # the removal from the code by returning None.

        if static_val is directives.set_element_type:
          self._process_symbol_directive(call_node, static_val)
          return None
        elif static_val is directives.set_loop_options:
          self._process_statement_directive(call_node, static_val)
          return None
    return node

  # TODO(mdan): This will be insufficient for other control flow.
  # That means that if we ever have a directive that affects things other than
  # loops, we'll need support for parallel scopes, or have multiple converters.
  def _track_and_visit_loop(self, node):
    self.enter_local_scope()
    self.set_local(ENCLOSING_LOOP, node)
    node = self.generic_visit(node)
    self.exit_local_scope()
    return node

  def visit_While(self, node):
    return self._track_and_visit_loop(node)

  def visit_For(self, node):
    return self._track_and_visit_loop(node)


def transform(node, ctx):
  return DirectivesTransformer(ctx).visit(node)
