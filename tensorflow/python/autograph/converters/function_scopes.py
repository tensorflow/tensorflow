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
from tensorflow.python.autograph.pyct import templates


class FunctionBodyTransformer(converter.Base):
  """Wraps function bodies around autograph-specific boilerplate."""

  def _name_for_current_scope(self):
    innermost = self.enclosing_entities[-1]
    if len(self.enclosing_entities) > 1:
      parent = self.enclosing_entities[-2]
      if isinstance(parent, gast.ClassDef):
        # Methods also take the name of their class.
        name = '%s/%s' % (parent.name, innermost.name)
      else:
        name = innermost.name
    else:
      name = innermost.name

    # Sanitize the name.
    # See https://www.tensorflow.org/api_docs/python/tf/Graph#name_scope
    # TensorFlow doesn't like leading underscores at the top level.
    while name[0] == '_':
      name = name[1:]
    return name

  def visit_FunctionDef(self, node):
    node = self.generic_visit(node)

    final_body = []
    indented_body = node.body
    if node.body:
      first_statement = node.body[0]
      # Skip the docstring, if any.
      if (isinstance(first_statement, gast.Expr) and
          isinstance(first_statement.value, gast.Str)):
        indented_body = indented_body[1:]
        final_body.append(first_statement)

    template = """
      with ag__.function_scope(scope_name):
        body
    """
    scoped_body = templates.replace(
        template,
        scope_name=gast.Str(self._name_for_current_scope()),
        body=indented_body)
    final_body.extend(scoped_body)
    node.body = final_body
    return node


def transform(node, ctx):
  return FunctionBodyTransformer(ctx).visit(node)
