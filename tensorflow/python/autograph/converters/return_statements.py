# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Canonicalizes functions with multiple returns to use just one."""

import gast

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno


BODY_DEFINITELY_RETURNS = 'BODY_DEFINITELY_RETURNS'
ORELSE_DEFINITELY_RETURNS = 'ORELSE_DEFINITELY_RETURNS'
STMT_DEFINITELY_RETURNS = 'STMT_DEFINITELY_RETURNS'


class _RewriteBlock(object):

  def __init__(self):
    self.definitely_returns = False


class ConditionalReturnRewriter(converter.Base):
  """Rewrites a pattern where it's unobvious that all paths return a value.

  This rewrite allows avoiding intermediate None return values.

  The following pattern:

      if cond:
        <block 1>
        return
      else:
        <block 2>
      <block 3>

  is converted to:

      if cond:
        <block 1>
        return
      else:
        <block 2>
        <block 3>

  and vice-versa (if the else returns, subsequent statements are moved under the
  if branch).
  """

  def visit_Return(self, node):
    self.state[_RewriteBlock].definitely_returns = True
    return node

  def _postprocess_statement(self, node):
    # If the node definitely returns (e.g. it's a with statement with a
    # return statement in it), then the current block also definitely returns.
    if anno.getanno(node, STMT_DEFINITELY_RETURNS, default=False):
      self.state[_RewriteBlock].definitely_returns = True

    # The special case: collapse a typical conditional return pattern into
    # a single conditional with possibly returns on both branches. This
    # reduces the use of None return values, which don't work with TF
    # conditionals.
    if (isinstance(node, gast.If)
        and anno.getanno(node, BODY_DEFINITELY_RETURNS, default=False)):
      return node, node.orelse
    elif (isinstance(node, gast.If)
          and anno.getanno(node, ORELSE_DEFINITELY_RETURNS, default=False)):
      return node, node.body

    return node, None

  def _visit_statement_block(self, node, nodes):
    self.state[_RewriteBlock].enter()
    new_nodes = self.visit_block(nodes, after_visit=self._postprocess_statement)
    block_definitely_returns = self.state[_RewriteBlock].definitely_returns
    self.state[_RewriteBlock].exit()
    return new_nodes, block_definitely_returns

  def visit_While(self, node):
    node.test = self.visit(node.test)
    node.body, _ = self._visit_statement_block(node, node.body)
    node.orelse, _ = self._visit_statement_block(node, node.orelse)
    return node

  def visit_For(self, node):
    node.iter = self.visit(node.iter)
    node.target = self.visit(node.target)
    node.body, _ = self._visit_statement_block(node, node.body)
    node.orelse, _ = self._visit_statement_block(node, node.orelse)
    return node

  def visit_With(self, node):
    node.items = self.visit_block(node.items)
    node.body, definitely_returns = self._visit_statement_block(node, node.body)
    if definitely_returns:
      anno.setanno(node, STMT_DEFINITELY_RETURNS, True)
    return node

  def visit_Try(self, node):
    # We could decide whether a 'try' DEFINITELY_RETURNS based on its components
    # It is not clear whether we want to do anything with this given
    # a 'try' is likely to throw an exception in some circumstances.
    node.body, _ = self._visit_statement_block(node, node.body)
    node.orelse, _ = self._visit_statement_block(node, node.orelse)
    node.finalbody, _ = self._visit_statement_block(node, node.finalbody)
    node.handlers = self.visit_block(node.handlers)
    return node

  def visit_ExceptHandler(self, node):
    # To determine whether `try` DEFINITELY_RETURNS we need to revisit this.
    node.body, _ = self._visit_statement_block(node, node.body)
    return node

  def visit_If(self, node):
    node.test = self.visit(node.test)

    node.body, body_definitely_returns = self._visit_statement_block(
        node, node.body)
    if body_definitely_returns:
      anno.setanno(node, BODY_DEFINITELY_RETURNS, True)

    node.orelse, orelse_definitely_returns = self._visit_statement_block(
        node, node.orelse)
    if orelse_definitely_returns:
      anno.setanno(node, ORELSE_DEFINITELY_RETURNS, True)

    if body_definitely_returns and orelse_definitely_returns:
      self.state[_RewriteBlock].definitely_returns = True

    return node

  def visit_FunctionDef(self, node):
    node.args = self.visit(node.args)
    node.body, _ = self._visit_statement_block(node, node.body)
    return node


class _Block(object):

  def __init__(self):
    self.is_function = False
    self.return_used = False
    self.create_guard_next = False
    self.create_guard_now = False

  def __repr__(self):
    return 'used: {}'.format(
        self.return_used)


class _Function(object):

  def __init__(self):
    self.do_return_var_name = None
    self.retval_var_name = None

  def __repr__(self):
    return 'return control: {}, return value: {}'.format(
        self.do_return_var_name, self.retval_var_name)


class ReturnStatementsTransformer(converter.Base):
  """Lowers return statements into variables and conditionals.

  Specifically, the following pattern:

      <block 1>
      return val
      <block 2>

  is converted to:

      do_return = False
      retval = None

      <block 1>

      do_return = True
      retval = val

      if not do_return:
        <block 2>

      return retval

  The conversion adjusts loops as well:

      <block 1>
      while cond:
        <block 2>
        return retval

  is converted to:

      <block 1>
      while not do_return and cond:
        <block 2>
        do_return = True
        retval = val
  """

  def __init__(self, ctx, allow_missing_return):
    super(ReturnStatementsTransformer, self).__init__(ctx)
    self.allow_missing_return = allow_missing_return

  def visit_Return(self, node):
    for block in reversed(self.state[_Block].stack):
      block.return_used = True
      block.create_guard_next = True
      if block.is_function:
        break

    retval = node.value if node.value else parser.parse_expression('None')

    # Note: If `return <expr> raises, then the return is aborted.
    # The try-catch below ensures the variables remain consistent in that case.
    template = """
      try:
        do_return_var_name = True
        retval_var_name = retval
      except:
        do_return_var_name = False
        raise
    """
    node = templates.replace(
        template,
        do_return_var_name=self.state[_Function].do_return_var_name,
        retval_var_name=self.state[_Function].retval_var_name,
        retval=retval)

    return node

  def _postprocess_statement(self, node):
    if not self.state[_Block].return_used:
      return node, None

    state = self.state[_Block]
    if state.create_guard_now:
      template = """
        if not do_return_var_name:
          original_node
      """
      cond, = templates.replace(
          template,
          do_return_var_name=self.state[_Function].do_return_var_name,
          original_node=node)
      node, block = cond, cond.body
    else:
      node, block = node, None

    state.create_guard_now = state.create_guard_next
    state.create_guard_next = False

    return node, block

  def _visit_statement_block(self, node, nodes):
    self.state[_Block].enter()
    nodes = self.visit_block(nodes, after_visit=self._postprocess_statement)
    self.state[_Block].exit()
    return nodes

  def visit_While(self, node):
    node.test = self.visit(node.test)

    # Add the check for return to the loop condition.
    node.body = self._visit_statement_block(node, node.body)
    if self.state[_Block].return_used:
      node.test = templates.replace_as_expression(
          'not control_var and test',
          test=node.test,
          control_var=self.state[_Function].do_return_var_name)

    node.orelse = self._visit_statement_block(node, node.orelse)
    return node

  def visit_For(self, node):
    node.iter = self.visit(node.iter)
    node.target = self.visit(node.target)

    # Add the check for return to the loop condition.
    node.body = self._visit_statement_block(node, node.body)
    if self.state[_Block].return_used:
      extra_test = anno.getanno(node, anno.Basic.EXTRA_LOOP_TEST, default=None)
      if extra_test is not None:
        extra_test = templates.replace_as_expression(
            'not control_var and extra_test',
            extra_test=extra_test,
            control_var=self.state[_Function].do_return_var_name)
      else:
        extra_test = templates.replace_as_expression(
            'not control_var',
            control_var=self.state[_Function].do_return_var_name)
      anno.setanno(node, anno.Basic.EXTRA_LOOP_TEST, extra_test)

    node.orelse = self._visit_statement_block(node, node.orelse)
    return node

  def visit_With(self, node):
    node.items = self.visit_block(node.items)
    node.body = self._visit_statement_block(node, node.body)
    return node

  def visit_Try(self, node):
    node.body = self._visit_statement_block(node, node.body)
    node.orelse = self._visit_statement_block(node, node.orelse)
    node.finalbody = self._visit_statement_block(node, node.finalbody)
    node.handlers = self.visit_block(node.handlers)
    return node

  def visit_ExceptHandler(self, node):
    node.body = self._visit_statement_block(node, node.body)
    return node

  def visit_If(self, node):
    node.test = self.visit(node.test)
    node.body = self._visit_statement_block(node, node.body)
    node.orelse = self._visit_statement_block(node, node.orelse)
    return node

  def visit_FunctionDef(self, node):
    with self.state[_Function] as fn:
      with self.state[_Block] as block:
        block.is_function = True

        scope = anno.getanno(node, NodeAnno.BODY_SCOPE)
        do_return_var_name = self.ctx.namer.new_symbol('do_return',
                                                       scope.referenced)
        retval_var_name = self.ctx.namer.new_symbol('retval_', scope.referenced)
        fn.do_return_var_name = do_return_var_name
        fn.retval_var_name = retval_var_name

        node.body = self._visit_statement_block(node, node.body)

        if block.return_used:

          if self.allow_missing_return:
            # The function would have a single `with` node that wraps the
            # entire body. If the function had a docstring, the body has two
            # nodes, with the `with` as the second node.
            wrapper_node = node.body[-1]
            assert isinstance(wrapper_node, gast.With), (
                'This transformer requires the functions converter.')

            template = """
              do_return_var_name = False
              retval_var_name = ag__.UndefinedReturnValue()
              body
              return function_context.ret(retval_var_name, do_return_var_name)
            """

            wrapper_node.body = templates.replace(
                template,
                body=wrapper_node.body,
                do_return_var_name=do_return_var_name,
                function_context=anno.getanno(node, 'function_context_name'),
                retval_var_name=retval_var_name)
          else:
            template = """
              body
              return retval_var_name
            """
            node.body = templates.replace(
                template,
                body=node.body,
                do_return_var_name=do_return_var_name,
                retval_var_name=retval_var_name)

    return node


def transform(node, ctx, default_to_null_return=True):
  """Ensure a function has only a single return, at the end."""
  node = qual_names.resolve(node)
  node = activity.resolve(node, ctx, None)

  # Note: Technically, these two could be merged into a single walk, but
  # keeping them separate helps with readability.
  node = ConditionalReturnRewriter(ctx).visit(node)

  node = qual_names.resolve(node)
  node = activity.resolve(node, ctx, None)
  transformer = ReturnStatementsTransformer(
      ctx, allow_missing_return=default_to_null_return)
  node = transformer.visit(node)
  return node
