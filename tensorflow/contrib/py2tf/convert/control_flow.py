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
"""Identity converter. Useful for testing and diagnostic."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.contrib.py2tf.pyct import anno
from tensorflow.contrib.py2tf.pyct import templates


class SymbolNamer(object):
  """Describes the interface for ControlFlowTransformer's namer."""

  def new_symbol(self, name_root, reserved_locals):
    """Generate a new unique symbol.

    Args:
      name_root: String, used as stem in the new name.
      reserved_locals: Set(string), additional local symbols that are reserved
          and which should not be used.
    Returns:
      String.
    """
    raise NotImplementedError()


class ControlFlowTransformer(gast.NodeTransformer):
  """Transforms control flow structures like loops an conditionals."""

  def __init__(self, namer):
    self.namer = namer

  # pylint:disable=invalid-name

  def _tuple_or_item(self, elts):
    elts = tuple(elts)
    if len(elts) == 1:
      return elts[0]
    return elts

  def _ast_tuple_or_item(self, elts, ctx):
    elts = list(elts)
    if len(elts) == 1:
      return elts[0]
    return gast.Tuple(elts, ctx)

  def visit_If(self, node):
    raise NotImplementedError()

  def visit_While(self, node):
    self.generic_visit(node)
    # Scrape out the data flow analysis
    body_scope = anno.getanno(node, 'body_scope')
    parent_scope_values = anno.getanno(node, 'parent_scope_values')
    body_closure = tuple(body_scope.modified - body_scope.created)

    def template(
        state_args,  # pylint:disable=unused-argument
        state_locals,
        state_results,  # pylint:disable=unused-argument
        test_name,
        test,  # pylint:disable=unused-argument
        body_name,
        body,
        state_init):

      def test_name(state_args):  # pylint:disable=function-redefined,unused-argument
        return test

      def body_name(state_args):  # pylint:disable=function-redefined,unused-argument
        body  # pylint:disable=pointless-statement
        return state_locals

      state_results = tf.while_loop(test_name, body_name, [state_init])  # pylint:disable=undefined-variable

    test_name = self.namer.new_symbol('loop_test', body_scope.used)
    body_name = self.namer.new_symbol('loop_body', body_scope.used)
    node = templates.replace(
        template,
        state_args=self._tuple_or_item(
            gast.Name(n, gast.Param(), None) for n in body_closure),
        state_locals=self._ast_tuple_or_item(
            (gast.Name(n, gast.Load(), None) for n in body_closure),
            gast.Load()),
        state_results=self._ast_tuple_or_item(
            (gast.Name(n, gast.Store(), None) for n in body_closure),
            gast.Store()),
        test_name=gast.Name(test_name, gast.Load(), None),
        test=node.test,
        body_name=gast.Name(body_name, gast.Load(), None),
        body=node.body,
        state_init=[parent_scope_values.getval(n) for n in body_closure])

    return node

  # pylint:enable=invalid-name


def transform(node, namer):
  transformer = ControlFlowTransformer(namer)
  node = transformer.visit(node)
  return node
