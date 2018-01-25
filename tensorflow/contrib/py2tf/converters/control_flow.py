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


class SymbolRenamer(gast.NodeTransformer):

  def __init__(self, name_map):
    self.name_map = name_map

  def visit_Name(self, node):
    if node.id in self.name_map:
      node.id = self.name_map[node.id]
    return node


class ControlFlowTransformer(gast.NodeTransformer):
  """Transforms control flow structures like loops an conditionals."""

  def __init__(self, namer):
    self.namer = namer

  # pylint:disable=invalid-name

  def visit_For(self, node):
    assert False, 'for statement should have been canonicalized at this point'

  def visit_If(self, node):
    self.generic_visit(node)

    body_scope = anno.getanno(node, 'body_scope')
    orelse_scope = anno.getanno(node, 'orelse_scope')

    if body_scope.created - orelse_scope.created:
      raise ValueError(
          'The if branch creates new symbols that the else branch does not.')
    if orelse_scope.created - body_scope.created:
      raise ValueError(
          'The else branch creates new symbols that the if branch does not.')

    def template(  # pylint:disable=missing-docstring
        test,
        body_name,
        body,
        orelse_name,
        orelse,
        aliased,
        aliases,  # pylint:disable=unused-argument
        aliased_results,
        results):  # pylint:disable=unused-argument

      def body_name():  # pylint:disable=function-redefined
        aliases, = aliased,  # pylint:disable=unused-variable
        body  # pylint:disable=pointless-statement
        return (aliased_results,)

      def orelse_name():  # pylint:disable=function-redefined
        aliases, = aliased,  # pylint:disable=unused-variable
        orelse  # pylint:disable=pointless-statement
        return (aliased_results,)

      results = tf.cond(test, body_name, orelse_name)  # pylint:disable=undefined-variable

    all_modified = tuple(body_scope.modified | orelse_scope.modified)
    all_referenced = body_scope.referenced | orelse_scope.referenced

    # Alias the closure variables inside the conditional functions
    # to avoid errors caused by the local variables created in the branch
    # functions.
    need_alias = (
        (body_scope.modified | orelse_scope.modified) -
        (body_scope.created | orelse_scope.created))
    aliased = tuple(need_alias)
    aliases = tuple(
        self.namer.new_symbol(s, all_referenced) for s in aliased)
    alias_map = dict(zip(aliased, aliases))
    node_body = node.body
    node_body = [SymbolRenamer(alias_map).visit(n) for n in node_body]
    node_orelse = node.orelse
    node_orelse = [SymbolRenamer(alias_map).visit(n) for n in node_orelse]

    if len(all_modified) == 1:
      results = gast.Name(all_modified[0], None, None)
    else:
      results = gast.Tuple(
          tuple(gast.Name(s, None, None) for s in all_modified), None)

    return templates.replace(
        template,
        test=node.test,
        body_name=gast.Name(
            self.namer.new_symbol('if_true', all_referenced), None, None),
        body=node_body,
        orelse_name=gast.Name(
            self.namer.new_symbol('if_false', all_referenced), None, None),
        orelse=node_orelse,
        aliased=tuple(gast.Name(s, None, None) for s in aliased),
        aliases=tuple(gast.Name(s, None, None) for s in aliases),
        aliased_results=tuple(
            gast.Name(alias_map[s] if s in aliased else s, None, None)
            for s in all_modified),
        results=results)

  def visit_While(self, node):
    self.generic_visit(node)

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

    test_name = self.namer.new_symbol('loop_test', body_scope.referenced)
    body_name = self.namer.new_symbol('loop_body', body_scope.referenced)
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
