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
import six

from tensorflow.python.util import tf_inspect


def parse_entity(entity):
  """Returns the AST of given entity."""
  source = tf_inspect.getsource(entity)

  def fail(comment):
    raise ValueError(
        'Failed to parse source code of {}, which Python reported as:\n{}\n'
        '{}'.format(entity, source, comment))

  # Comments and multiline strings can appear at arbitrary indentation levels,
  # causing textwrap.dedent to not correctly dedent source code.
  # TODO(b/115884650): Automatic handling of comments/multiline strings.
  source = textwrap.dedent(source)

  try:
    return parse_str(source), source

  except IndentationError:
    # The text below lists the causes of this error known to us. There may
    # be more.
    fail('This may be caused by multiline strings or comments not indented at'
         'the same level as the code.')

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
      fail('If this is a lambda function, the error may be avoided by creating'
           ' the lambda in a standalone statement.')

    # Drop all lines following the error location
    # TODO(mdan): What's with the pylint errors?
    lines = lines[:lineno]  # pylint:disable=invalid-slice-index
    # Drop all characters following the error location
    lines[-1] = lines[-1][:offset - 1]  # pylint:disable=invalid-slice-index
    new_source = '\n'.join(lines)

    try:
      return parse_str(new_source), new_source
    except SyntaxError as e:
      fail('If this is a lambda function, the error may be avoided by creating'
           ' the lambda in a standalone statement. Tried to strip down the'
           ' source to:\n{}\nBut that did not work.'.format(new_source))


def parse_str(src):
  """Returns the AST of given piece of code."""
  # TODO(mdan): This should exclude the module things are autowrapped in.

  if six.PY2 and '.print(' in src:
    # This special treatment is required because gast.parse is not aware of
    # whether print_function was present in the original context.
    src = 'from __future__ import print_function\n' + src
    parsed_module = gast.parse(src)
    parsed_module.body = parsed_module.body[1:]
  else:
    parsed_module = gast.parse(src)

  return parsed_module


def parse_expression(src):
  """Returns the AST of given identifier.

  Args:
    src: A piece of code that represents a single Python expression
  Returns:
    A gast.AST object.
  Raises:
    ValueError: if src does not consist of a single Expression.
  """
  node = parse_str(src)
  assert isinstance(node, gast.Module)
  if len(node.body) != 1 or not isinstance(node.body[0], gast.Expr):
    raise ValueError(
        'Expected a single expression, found instead %s' % node.body)
  return node.body[0].value
