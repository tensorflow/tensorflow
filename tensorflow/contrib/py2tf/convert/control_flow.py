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
"""Handles control flow statements: while, if."""

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

  def visit_For(self, node):
    assert False, 'for statement should have been canonicalized at this point'

  def visit_If(self, node):
    raise NotImplementedError()

  def visit_While(self, node):
    self.generic_visit(node)
    # Scrape out the data flow analysis
    body_scope = anno.getanno(node, 'body_scope')
    body_closure = tuple(body_scope.modified - body_scope.created)

    def template(
        state,  # pylint:disable=unused-argument
        state_ast_tuple,  # pylint:disable=unused-argument
        test_name,
        test,  # pylint:disable=unused-argument
        body_name,
        body):

      def test_name(state):  # pylint:disable=function-redefined,unused-argument
        return test

      def body_name(state):  # pylint:disable=function-redefined,unused-argument
        body  # pylint:disable=pointless-statement
        return state,

      state_ast_tuple = tf.while_loop(test_name, body_name, [state])  # pylint:disable=undefined-variable

    test_name = self.namer.new_symbol('loop_test', body_scope.used)
    body_name = self.namer.new_symbol('loop_body', body_scope.used)
    if len(body_closure) == 1:
      state = gast.Name(body_closure[0], None, None)
      state_ast_tuple = state
    else:
      state = tuple(gast.Name(n, None, None) for n in body_closure)
      state_ast_tuple = gast.Tuple(state, None)
    node = templates.replace(
        template,
        state=state,
        state_ast_tuple=state_ast_tuple,
        test_name=gast.Name(test_name, gast.Load(), None),
        test=node.test,
        body_name=gast.Name(body_name, gast.Load(), None),
        body=node.body)

    return node

  # pylint:enable=invalid-name


def transform(node, namer):
  transformer = ControlFlowTransformer(namer)
  node = transformer.visit(node)
  return node
