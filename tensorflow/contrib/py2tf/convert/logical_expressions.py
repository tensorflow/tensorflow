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
"""Converter for logical expressions.

e.g. `a and b -> tf.logical_and(a, b)`. This is not done automatically in TF.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.contrib.py2tf.pyct import parser


class LogicalExpressionTransformer(gast.NodeTransformer):
  """Converts logical expressions to corresponding TF calls."""

  def __init__(self):
    # TODO(mdan): Look into replacing with bitwise operators instead.
    self.op_mapping = {
        gast.And: 'tf.logical_and',
        gast.Or: 'tf.logical_or',
        gast.Not: 'tf.logical_not',
    }

  def visit_UnaryOp(self, node):
    if isinstance(node.op, gast.Not):
      tf_function = parser.parse_str(self.op_mapping[type(
          node.op)]).body[0].value
      node = gast.Call(func=tf_function, args=[node.operand], keywords=[])
    return node

  def visit_BoolOp(self, node):
    # TODO(mdan): A normalizer may be useful here. Use ANF?
    tf_function = parser.parse_str(self.op_mapping[type(node.op)]).body[0].value
    left = node.values[0]
    for i in range(1, len(node.values)):
      left = gast.Call(
          func=tf_function, args=[left, node.values[i]], keywords=[])
    return left


def transform(node):
  transformer = LogicalExpressionTransformer()
  node = transformer.visit(node)
  return node
