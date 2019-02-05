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
"""Wraps function bodies with a try/except to rewrite error tracebacks.

Only adds try/except wrappers to functions that have the anno.Basic.ORIGIN
annotation because these are the functions originally written by the user.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import templates


class ErrorRewritingTransformer(converter.Base):
  """Possibly wraps the body of a function in a try/except.

  Only wraps functions that were originally defined by the user, detected by
  checking for the anno.Basic.ORIGIN annotation.
  """

  def visit_FunctionDef(self, node):
    node = self.generic_visit(node)

    if (anno.hasanno(node, anno.Basic.ORIGIN) and
        len(self.enclosing_entities) <= 1):
      template = """
        try:
          body
        except:
          ag__.rewrite_graph_construction_error(ag_source_map__)
      """
      node.body = templates.replace(template, body=node.body)
    return node


def transform(node, ctx):
  return ErrorRewritingTransformer(ctx).visit(node)
