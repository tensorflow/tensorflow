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
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.contrib.autograph.core import converter
from tensorflow.contrib.autograph.lang import directives
from tensorflow.contrib.autograph.pyct import anno
from tensorflow.python.util import tf_inspect

ENCLOSING_LOOP = 'enclosing_loop'


def _map_args(call_node, function):
  """Maps AST call nodes to the actual function's arguments.

  Args:
    call_node: ast.Call
    function: Callable[..., Any], the actual function matching call_node
  Returns:
    Dict[Text, ast.AST], mapping each of the function's argument names to
    the respective AST node.
  """
  args = call_node.args
  kwds = {kwd.arg: kwd.value for kwd in call_node.keywords}
  return tf_inspect.getcallargs(function, *args, **kwds)


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
    if self.local_scope_level < 1:
      raise ValueError(
          '"%s" must be used inside a statement' % directive.__name__)
    target = self.get_local(ENCLOSING_LOOP)
    node_anno = anno.getanno(target, converter.AgAnno.DIRECTIVES, {})
    node_anno[directive] = _map_args(call_node, directive)
    anno.setanno(target, converter.AgAnno.DIRECTIVES, node_anno)
    return call_node

  def visit_Expr(self, node):
    if isinstance(node.value, gast.Call):
      call_node = node.value
      if anno.hasanno(call_node.func, 'live_val'):
        live_val = anno.getanno(call_node.func, 'live_val')

        if live_val is directives.set_element_type:
          call_node = self._process_symbol_directive(call_node, live_val)
        elif live_val is directives.set_loop_options:
          call_node = self._process_statement_directive(call_node, live_val)
        else:
          return self.generic_visit(node)

        return None  # Directive calls are not output in the generated code.
    return self.generic_visit(node)

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
