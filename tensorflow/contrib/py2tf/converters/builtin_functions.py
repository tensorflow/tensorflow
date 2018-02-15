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

from tensorflow.contrib.py2tf.pyct import templates
from tensorflow.contrib.py2tf.pyct import transformer


class BuiltinFunctionTransformer(transformer.Base):
  """Handles builtin functions.

  This transformer only covers functions that are translated into a
  TF equivalent, like `len`.
  """

  def __init__(self, context):
    super(BuiltinFunctionTransformer, self).__init__(context)

  # pylint:disable=invalid-name

  def _convert_len(self, node):
    template = """
      tf.shape(args)[0]
    """
    return templates.replace(template, args=node.args)[0].value

  def _convert_print(self, node):
    template = """
      py2tf_utils.call_print(args)
    """
    return templates.replace(template, args=node.args)[0].value

  def visit_Call(self, node):
    self.generic_visit(node)
    # TODO(mdan): This won't work if the function was hidden.
    if isinstance(node.func, gast.Name) and node.func.id == 'len':
      return self._convert_len(node)
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

  # pylint:enable=invalid-name


def transform(node, context):
  return BuiltinFunctionTransformer(context).visit(node)
