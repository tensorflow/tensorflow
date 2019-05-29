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
"""Converting code to AST.

Adapted from Tangent.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import textwrap

import gast

from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.util import tf_inspect


STANDARD_PREAMBLE = textwrap.dedent("""
    from __future__ import division
    from __future__ import print_function
""")
STANDARD_PREAMBLE_LEN = 2


def parse_entity(entity, future_features):
  """Returns the AST and source code of given entity.

  Args:
    entity: Any, Python function/method/class
    future_features: Iterable[Text], future features to use (e.g.
      'print_statement'). See
      https://docs.python.org/2/reference/simple_stmts.html#future

  Returns:
    gast.AST, Text: the parsed AST node; the source code that was parsed to
    generate the AST (including any prefixes that this function may have added).
  """
  try:
    source = inspect_utils.getimmediatesource(entity)
  except (IOError, OSError) as e:
    raise ValueError(
        'Unable to locate the source code of {}. Note that functions defined'
        ' in certain environments, like the interactive Python shell do not'
        ' expose their source code. If that is the case, you should to define'
        ' them in a .py source file. If you are certain the code is'
        ' graph-compatible, wrap the call using'
        ' @tf.autograph.do_not_convert. Original error: {}'.format(entity, e))

  def raise_parse_failure(comment):
    raise ValueError(
        'Failed to parse source code of {}, which Python reported as:\n{}\n'
        '{}'.format(entity, source, comment))

  # Comments and multiline strings can appear at arbitrary indentation levels,
  # causing textwrap.dedent to not correctly dedent source code.
  # TODO(b/115884650): Automatic handling of comments/multiline strings.
  source = textwrap.dedent(source)

  future_statements = tuple(
      'from __future__ import {}'.format(name) for name in future_features)
  source = '\n'.join(future_statements + (source,))

  try:
    return parse_str(source, preamble_len=len(future_features)), source

  except IndentationError:
    # The text below lists the causes of this error known to us. There may
    # be more.
    raise_parse_failure(
        'This may be caused by multiline strings or comments not indented at'
        ' the same level as the code.')

  except SyntaxError as e:
    if not tf_inspect.isfunction(entity) or entity.__name__ != '<lambda>':
      raise

    # Certain entities, like lambdas, only hold the raw code lines which defined
    # them, which may include surrounding tokens and may be syntactically
    # invalid out of context. For example:
    #
    #     l = (
    #         lambda x: x,)[0]
    #
    # will have the dedented source "lambda x: x,)[0]"
    # Here we make an attempt to stip away the garbage by looking at the
    # information in the syntax error.
    lines = source.split('\n')
    lineno, offset = e.lineno, e.offset  # 1-based

    # Give up if there's nothing we can chip away.
    if len(lines) == lineno and len(lines[-1]) == offset:
      raise_parse_failure(
          'If this is a lambda function, the error may be avoided by creating'
          ' the lambda in a standalone statement.')

    # Drop all lines following the error location
    # TODO(mdan): What's with the pylint errors?
    lines = lines[:lineno]  # pylint:disable=invalid-slice-index
    # Drop all characters following the error location
    lines[-1] = lines[-1][:offset - 1]  # pylint:disable=invalid-slice-index
    source = '\n'.join(lines)

    try:
      return parse_str(source, preamble_len=len(future_features)), source
    except SyntaxError as e:
      raise_parse_failure(
          'If this is a lambda function, the error may be avoided by creating'
          ' the lambda in a standalone statement. Tried to strip down the'
          ' source to:\n{}\nBut that did not work.'.format(source))


# TODO(mdan): This should take futures as input instead.
def parse_str(src, preamble_len=0, single_node=True):
  """Returns the AST of given piece of code.

  Args:
    src: Text
    preamble_len: Int, indicates leading nodes in the parsed AST which should be
      dropped.
    single_node: Bool, whether `src` is assumed to be represented by exactly one
      AST node.

  Returns:
    ast.AST
  """
  module_node = gast.parse(src)
  nodes = module_node.body
  if preamble_len:
    nodes = nodes[preamble_len:]
  if single_node:
    if len(nodes) != 1:
      raise ValueError('expected exactly one node node, found {}'.format(nodes))
    return nodes[0]
  return nodes


def parse_expression(src):
  """Returns the AST of given identifier.

  Args:
    src: A piece of code that represents a single Python expression
  Returns:
    A gast.AST object.
  Raises:
    ValueError: if src does not consist of a single Expression.
  """
  src = STANDARD_PREAMBLE + src.strip()
  node = parse_str(src, preamble_len=STANDARD_PREAMBLE_LEN, single_node=True)
  if __debug__:
    if not isinstance(node, gast.Expr):
      raise ValueError(
          'expected a single expression, found instead {}'.format(node))
  return node.value
