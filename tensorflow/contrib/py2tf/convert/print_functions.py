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
"""Compatibility support. Converts Print nodes to function calls."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.contrib.py2tf.pyct import anno


class PrintFunctionTransformer(gast.NodeTransformer):
  """Transforms Print nodes to Call so they can be handled as functions."""

  # pylint:disable=invalid-name

  def visit_Print(self, node):
    self.generic_visit(node)
    for n in node.values:
      n.ctx = gast.Param()
    call_node = gast.Call(
        func=gast.Name('print', gast.Load(), None),
        args=node.values,
        keywords=[])
    anno.setanno(call_node.func, 'live_val', print)
    anno.setanno(call_node.func, 'fqn', 'print')
    anno.setanno(call_node, 'args_scope', anno.getanno(node, 'args_scope'))
    node = gast.Expr(call_node)
    return node

  # pylint:enable=invalid-name


def transform(node):
  transformer = PrintFunctionTransformer()
  node = transformer.visit(node)
  return node
