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

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import templates


class BuiltinFunctionTransformer(converter.Base):
  """Handles builtin functions.

  This transformer only covers functions that are translated into a
  TF equivalent, like `len`.
  """

  def _convert_builtin(self, f, args, as_expression):
    template = """
      ag__.func(args)
    """
    if as_expression:
      return templates.replace_as_expression(
          template, func=py_builtins.overload_of(f).__name__, args=args)
    else:
      return templates.replace(
          template, func=py_builtins.overload_of(f).__name__, args=args)

  def visit_Call(self, node):
    node = self.generic_visit(node)
    if anno.hasanno(node.func, 'live_val'):
      live_val = anno.getanno(node.func, 'live_val')
      try:
        if live_val in py_builtins.SUPPORTED_BUILTINS:
          node = self._convert_builtin(live_val, node.args, as_expression=True)
      except TypeError:
        # Not everything in Python is hashable. If it isn't then it's definitely
        # not a supported built-in.
        return node
    return node

  def visit_Print(self, node):
    node = self.generic_visit(node)
    args = node.values
    # Following is the case when calling print(a, b)
    if len(args) == 1 and isinstance(args[0], gast.Tuple):
      args = args[0].elts
    return self._convert_builtin(print, args, as_expression=False)


def transform(node, ctx):
  return BuiltinFunctionTransformer(ctx).visit(node)
