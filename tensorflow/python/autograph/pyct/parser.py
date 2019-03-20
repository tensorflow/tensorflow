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

import itertools
import textwrap
import threading

import gast

from tensorflow.python.util import tf_inspect


_parse_lock = threading.Lock()  # Prevents linecache concurrency errors.


def parse_entity(entity, future_imports):
  """Returns the AST and source code of given entity.

  Args:
    entity: A python function/method/class
    future_imports: An iterable of future imports to use when parsing AST. (e.g.
        ('print_statement', 'division', 'unicode_literals'))

  Returns:
    gast.AST, List[gast.AST], str: a tuple of the AST node corresponding
    exactly to the entity; a list of future import AST nodes, and the string
    that was parsed to generate the AST.
  """
  try:
    with _parse_lock:
      source = tf_inspect.getsource_no_unwrap(entity)
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
  future_import_strings = ('from __future__ import {}'.format(name)
                           for name in future_imports)
  source = '\n'.join(itertools.chain(future_import_strings, [source]))

  try:
    module_node = parse_str(source)
    return _select_entity_node(module_node, source, future_imports)

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
    new_source = '\n'.join(lines)

    try:
      module_node = parse_str(new_source)
      return _select_entity_node(module_node, new_source, future_imports)
    except SyntaxError as e:
      raise_parse_failure(
          'If this is a lambda function, the error may be avoided by creating'
          ' the lambda in a standalone statement. Tried to strip down the'
          ' source to:\n{}\nBut that did not work.'.format(new_source))


def parse_str(src):
  """Returns the AST of given piece of code."""
  return gast.parse(src)


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


def _select_entity_node(module_node, source, future_imports):
  assert len(module_node.body) == 1 + len(future_imports)
  return module_node.body[-1], module_node.body[:-1], source

