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
"""Converter for logical expressions, e.g. `a and b -> tf.logical_and(a, b)`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import templates

# TODO(mdan): Properly extract boolean ops according to lazy eval rules.
# Note that this isn't completely safe either, because tensors may have control
# dependencies.
# Note that for loops that should be done after the loop was converted to
# tf.while_loop so that the expanded conditionals are properly scoped.

# Used to signal that an operand is safe for non-lazy evaluation.
SAFE_BOOLEAN_OPERAND = 'SAFE_BOOLEAN_OPERAND'


LOGICAL_OPERATORS = {
    gast.And: 'ag__.and_',
    gast.Not: 'ag__.not_',
    gast.Or: 'ag__.or_',
}

EQUALITY_OPERATORS = {
    gast.Eq: 'ag__.eq',
    gast.NotEq: 'ag__.not_eq',
}


class LogicalExpressionTransformer(converter.Base):
  """Converts logical expressions to corresponding TF calls."""

  def _overload_of(self, operator):
    op_type = type(operator)
    if op_type in LOGICAL_OPERATORS:
      return LOGICAL_OPERATORS[op_type]
    if self.ctx.user.options.uses(converter.Feature.EQUALITY_OPERATORS):
      if op_type in EQUALITY_OPERATORS:
        return EQUALITY_OPERATORS[op_type]
    return None

  def _as_lambda(self, expr):
    return templates.replace_as_expression('lambda: expr', expr=expr)

  def _as_binary_function(self, func_name, arg1, arg2):
    return templates.replace_as_expression(
        'func_name(arg1, arg2)',
        func_name=parser.parse_expression(func_name),
        arg1=arg1,
        arg2=arg2)

  def _as_binary_operation(self, op, arg1, arg2):
    template = templates.replace_as_expression(
        'arg1 is arg2',
        arg1=arg1,
        arg2=arg2)
    template.ops[0] = op
    return template

  def _as_unary_function(self, func_name, arg):
    return templates.replace_as_expression(
        'func_name(arg)', func_name=parser.parse_expression(func_name), arg=arg)

  def visit_Compare(self, node):
    node = self.generic_visit(node)

    if (not self.ctx.user.options.uses(
        converter.Feature.EQUALITY_OPERATORS)):
      return node

    ops_and_comps = list(zip(node.ops, node.comparators))
    left = node.left

    # Repeated comparisons are converted to conjunctions:
    #   a < b < c   ->   a < b and b < c
    op_tree = None
    while ops_and_comps:
      op, right = ops_and_comps.pop(0)
      overload = self._overload_of(op)
      if overload is not None:
        binary_comparison = self._as_binary_function(overload, left, right)
      else:
        binary_comparison = self._as_binary_operation(op, left, right)
      if op_tree is not None:
        op_tree = self._as_binary_function('ag__.and_',
                                           self._as_lambda(op_tree),
                                           self._as_lambda(binary_comparison))
      else:
        op_tree = binary_comparison
      left = right

    assert op_tree is not None
    return op_tree

  def visit_UnaryOp(self, node):
    node = self.generic_visit(node)

    overload = self._overload_of(node.op)
    if overload is None:
      return node

    return self._as_unary_function(overload, node.operand)

  def visit_BoolOp(self, node):
    node = self.generic_visit(node)
    node_values = node.values
    right = node.values.pop()
    while node_values:
      left = node_values.pop()
      right = self._as_binary_function(
          self._overload_of(node.op), self._as_lambda(left),
          self._as_lambda(right))
    return right


def transform(node, ctx):
  transformer = LogicalExpressionTransformer(ctx)
  return transformer.visit(node)
