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
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct.static_analysis import annos


class _Function(object):

  def __init__(self):
    self.context_name = None


class FunctionBodyTransformer(converter.Base):
  """Wraps function bodies around autograph-specific boilerplate."""

  def _sanitize(self, name):
    """See https://www.tensorflow.org/api_docs/python/tf/Graph#name_scope."""
    # TensorFlow doesn't like leading underscores at the top level.
    while name[0] == '_':
      name = name[1:]
    return name

  def visit_Return(self, node):
    if node.value is None:
      return node
    return templates.replace(
        'return function_context_name.mark_return_value(value)',
        function_context_name=self.state[_Function].context_name,
        value=node.value)

  def visit_FunctionDef(self, node):
    self.state[_Function].enter()
    scope = anno.getanno(node, annos.NodeAnno.BODY_SCOPE)

    function_context_name = self.ctx.namer.new_symbol(
        'fn_context', scope.referenced)
    self.state[_Function].context_name = function_context_name

    node = self.generic_visit(node)

    docstring_node = None
    if node.body:
      first_statement = node.body[0]
      if (isinstance(first_statement, gast.Expr) and
          isinstance(first_statement.value, gast.Str)):
        docstring_node = first_statement
        node.body = node.body[1:]

    if self.ctx.program.options.uses(converter.Feature.NAME_SCOPES):
      use_name_scopes = parser.parse_expression('True')
      scope_name = gast.Str(self._sanitize(node.name))
    else:
      use_name_scopes = parser.parse_expression('False')
      scope_name = parser.parse_expression('None')

    use_auto_deps = parser.parse_expression(str(
        self.ctx.program.options.uses(converter.Feature.AUTO_CONTROL_DEPS)))

    template = """
      with ag__.FunctionScope(
          use_name_scopes, scope_name, use_auto_deps) as function_context_name:
        body
    """
    wrapped_body = templates.replace(
        template,
        use_name_scopes=use_name_scopes,
        scope_name=scope_name,
        use_auto_deps=use_auto_deps,
        function_context_name=function_context_name,
        body=node.body)

    if docstring_node is not None:
      wrapped_body = [docstring_node] + wrapped_body

    node.body = wrapped_body

    self.state[_Function].exit()
    return node


def transform(node, ctx):
  return FunctionBodyTransformer(ctx).visit(node)
