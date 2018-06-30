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
"""Handles builtins and other special functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.contrib.autograph.core import converter
from tensorflow.contrib.autograph.pyct import templates


class BuiltinFunctionTransformer(converter.Base):
  """Handles builtin functions.

  This transformer only covers functions that are translated into a
  TF equivalent, like `len`.
  """

  def _convert_builtin(self, node):
    template = """
      ag__.utils.dynamic_builtin(func, args)
    """
    return templates.replace(template, func=node.func, args=node.args)[0].value

  def _convert_print(self, node):
    template = """
      ag__.utils.dynamic_print(args)
    """
    return templates.replace(template, args=node.args)[0].value

  def visit_Call(self, node):
    self.generic_visit(node)
    # TODO(mdan): This won't work if the function was hidden.
    # TODO(mdan): Rely on the live_val and use inspect_utils.is_builtin instead.
    if (isinstance(node.func, gast.Name) and
        node.func.id in ('len', 'range', 'xrange', 'float', 'int')):
      return self._convert_builtin(node)
    # Print needs to be handled separately because it can be read as statement.
    if isinstance(node.func, gast.Name) and node.func.id == 'print':
      return self._convert_print(node)
    return node

  def visit_Print(self, node):
    self.generic_visit(node)
    args = node.values
    # Following is the case when calling print(a, b)
    if len(args) == 1 and isinstance(args[0], gast.Tuple):
      args = args[0].elts
    template = """
      fname(args)
    """
    function_call = templates.replace(template, fname='print', args=args)[0]
    return self.visit(function_call)


def transform(node, ctx):
  return BuiltinFunctionTransformer(ctx).visit(node)
