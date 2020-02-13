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
"""Wraps the body of a converted function with auxiliary constructs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct.static_analysis import annos


class _Function(object):

  def __init__(self):
    self.context_name = None


class FunctionBodyTransformer(converter.Base):
  """Wraps function bodies around autograph-specific boilerplate."""

  def visit_Return(self, node):
    if node.value is None:
      return node
    return templates.replace(
        'return function_context_name.mark_return_value(value)',
        function_context_name=self.state[_Function].context_name,
        value=node.value)

  def _function_scope_options(self):
    """Returns the options with which to create function scopes."""
    # Top-level function receive the options that were directly requested.
    # All others receive the options corresponding to a recursive conversion.
    # Note: this mainly controls the user_requested flag, which is important
    # primarily because the FunctionScope context also creates a
    # ControlStatusCtx(autograph=ENABLED) when user_requested is True. See
    # function_wrappers.py.
    if self.state[_Function].level == 2:
      return self.ctx.program.options
    return self.ctx.program.options.call_options()

  def visit_Lambda(self, node):
    self.state[_Function].enter()
    node = self.generic_visit(node)

    # Only wrap the top-level function. Theoretically, we can and should wrap
    # everything, but that can lead to excessive boilerplate when lambdas are
    # nested.
    # TODO(mdan): Looks more closely for use cases that actually require this.
    if self.state[_Function].level > 2:
      self.state[_Function].exit()
      return node

    scope = anno.getanno(node, anno.Static.SCOPE)
    function_context_name = self.ctx.namer.new_symbol('lscope',
                                                      scope.referenced)
    self.state[_Function].context_name = function_context_name
    anno.setanno(node, 'function_context_name', function_context_name)

    template = """
      ag__.with_function_scope(
          lambda function_context: body, function_context_name, options)
    """
    node.body = templates.replace_as_expression(
        template,
        options=self._function_scope_options().to_ast(),
        function_context=function_context_name,
        function_context_name=gast.Constant(function_context_name, kind=None),
        body=node.body)

    self.state[_Function].exit()
    return node

  def visit_FunctionDef(self, node):
    self.state[_Function].enter()
    scope = anno.getanno(node, annos.NodeAnno.BODY_SCOPE)

    function_context_name = self.ctx.namer.new_symbol('fscope',
                                                      scope.referenced)
    self.state[_Function].context_name = function_context_name
    anno.setanno(node, 'function_context_name', function_context_name)

    node = self.generic_visit(node)

    docstring_node = None
    if node.body:
      first_statement = node.body[0]
      if (isinstance(first_statement, gast.Expr) and
          isinstance(first_statement.value, gast.Constant)):
        docstring_node = first_statement
        node.body = node.body[1:]

    template = """
      with ag__.FunctionScope(
          function_name, context_name, options) as function_context:
        body
    """
    wrapped_body = templates.replace(
        template,
        function_name=gast.Constant(node.name, kind=None),
        context_name=gast.Constant(function_context_name, kind=None),
        options=self._function_scope_options().to_ast(),
        function_context=function_context_name,
        body=node.body)

    if docstring_node is not None:
      wrapped_body = [docstring_node] + wrapped_body

    node.body = wrapped_body

    self.state[_Function].exit()
    return node


def transform(node, ctx):
  return FunctionBodyTransformer(ctx).visit(node)
