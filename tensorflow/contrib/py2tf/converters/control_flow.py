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
from tensorflow.contrib.py2tf.pyct import ast_util
from tensorflow.contrib.py2tf.pyct import templates
from tensorflow.contrib.py2tf.pyct import transformer
from tensorflow.contrib.py2tf.pyct.static_analysis.annos import NodeAnno


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


class ControlFlowTransformer(transformer.Base):
  """Transforms control flow structures like loops an conditionals."""

  def __init__(self, context):
    super(ControlFlowTransformer, self).__init__(context)

  # pylint:disable=invalid-name

  def visit_For(self, node):
    assert False, 'for statement should have been canonicalized at this point'

  def _create_cond_branch(self, body_name, aliased_orig_names,
                          aliased_new_names, body, returns):
    if aliased_orig_names:
      template = """
        def body_name():
          aliased_new_names, = aliased_orig_names,
          body
          return (returns,)
      """
      return templates.replace(
          template,
          body_name=body_name,
          body=body,
          aliased_orig_names=aliased_orig_names,
          aliased_new_names=aliased_new_names,
          returns=returns)
    else:
      template = """
        def body_name():
          body
          return (returns,)
      """
      return templates.replace(
          template, body_name=body_name, body=body, returns=returns)

  def _create_cond_expr(self, results, test, body_name, orelse_name):
    if results is not None:
      template = """
        results = py2tf_utils.run_cond(test, body_name, orelse_name)
      """
      return templates.replace(
          template,
          test=test,
          results=results,
          body_name=body_name,
          orelse_name=orelse_name)
    else:
      template = """
        py2tf_utils.run_cond(test, body_name, orelse_name)
      """
      return templates.replace(
          template, test=test, body_name=body_name, orelse_name=orelse_name)

  def visit_If(self, node):
    self.generic_visit(node)

    body_scope = anno.getanno(node, NodeAnno.BODY_SCOPE)
    orelse_scope = anno.getanno(node, NodeAnno.ORELSE_SCOPE)

    if body_scope.created - orelse_scope.created:
      raise ValueError(
          'The if branch creates new symbols that the else branch does not.')
    if orelse_scope.created - body_scope.created:
      raise ValueError(
          'The else branch creates new symbols that the if branch does not.')

    modified = tuple(body_scope.modified | orelse_scope.modified)
    all_referenced = body_scope.referenced | orelse_scope.referenced

    # Alias the closure variables inside the conditional functions
    # to avoid errors caused by the local variables created in the branch
    # functions.
    need_alias = (
        (body_scope.modified | orelse_scope.modified) -
        (body_scope.created | orelse_scope.created))
    aliased_orig_names = tuple(need_alias)
    aliased_new_names = tuple(
        self.context.namer.new_symbol(s.ssf(), all_referenced)
        for s in aliased_orig_names)
    alias_map = dict(zip(aliased_orig_names, aliased_new_names))
    node_body = ast_util.rename_symbols(node.body, alias_map)
    node_orelse = ast_util.rename_symbols(node.orelse, alias_map)

    if not modified:
      # When the cond would return no value, we leave the cond called without
      # results. That in turn should trigger the side effect guards. The
      # branch functions will return a dummy value that ensures cond
      # actually has some return value as well.
      results = None
    elif len(modified) == 1:
      results = modified[0]
    else:
      results = gast.Tuple([s.ast() for s in modified], None)

    body_name = self.context.namer.new_symbol('if_true', all_referenced)
    orelse_name = self.context.namer.new_symbol('if_false', all_referenced)
    if modified:
      body_returns = tuple(
          alias_map[s] if s in aliased_orig_names else s for s in modified)
    else:
      body_returns = templates.replace('tf.ones(())')[0].value

    body_def = self._create_cond_branch(
        body_name,
        aliased_orig_names=tuple(aliased_orig_names),
        aliased_new_names=tuple(aliased_new_names),
        body=node_body,
        returns=body_returns)
    orelse_def = self._create_cond_branch(
        orelse_name,
        aliased_orig_names=tuple(aliased_orig_names),
        aliased_new_names=tuple(aliased_new_names),
        body=node_orelse,
        returns=body_returns)
    cond_expr = self._create_cond_expr(results, node.test, body_name,
                                       orelse_name)

    return body_def + orelse_def + cond_expr

  def visit_While(self, node):
    self.generic_visit(node)

    body_scope = anno.getanno(node, NodeAnno.BODY_SCOPE)
    body_closure = body_scope.modified - body_scope.created
    all_referenced = body_scope.referenced

    state = list(body_closure)
    state_ssf = [
        self.context.namer.new_symbol(s.ssf(), all_referenced) for s in state
    ]
    ssf_map = {
        name: ssf
        for name, ssf in zip(state, state_ssf)
        if str(name) != ssf
    }

    if len(state) == 1:
      state = state[0]
      state_ssf = state_ssf[0]
      state_ast_tuple = state
    else:
      state_ast_tuple = gast.Tuple([n.ast() for n in state], None)

    node_body = ast_util.rename_symbols(node.body, ssf_map)
    test = ast_util.rename_symbols(node.test, ssf_map)

    template = """
      def test_name(state_ssf):
        return test
      def body_name(state_ssf):
        body
        return state_ssf,
      state_ast_tuple = py2tf_utils.run_while(test_name, body_name, [state])
    """
    node = templates.replace(
        template,
        state=state,
        state_ssf=state_ssf,
        state_ast_tuple=state_ast_tuple,
        test_name=self.context.namer.new_symbol('loop_test',
                                                body_scope.referenced),
        test=test,
        body_name=self.context.namer.new_symbol('loop_body',
                                                body_scope.referenced),
        body=node_body)

    return node

  # pylint:enable=invalid-name


def transform(node, context):
  t = ControlFlowTransformer(context)
  node = t.visit(node)
  return node
