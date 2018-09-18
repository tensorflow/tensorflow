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

from tensorflow.python.util import tf_inspect


def parse_entity(entity):
  """Returns the AST of given entity."""
  source = tf_inspect.getsource(entity)
  # Comments and multiline strings can appear at arbitrary indentation levels,
  # causing textwrap.dedent to not correctly dedent source code.
  # TODO(b/115884650): Automatic handling of comments/multiline strings.
  source = textwrap.dedent(source)
  try:
    return parse_str(source), source
  except IndentationError:
    # Because we are parsing the source code of entities that have already
    # successfully parsed once, any IndentationErrors are guaranteed to be
    # caused by insufficient dedenting.
    raise ValueError(
        'Failed to dedent prior to parsing source code. If you have comments '
        'or multiline strings in your code, try indenting them. '
        'Multiline strings can be rewritten using textwrap.dedent.\n'
        'Offending source code: \n %s' % source)


def parse_str(src):
  """Returns the AST of given piece of code."""
  # TODO(mdan): This should exclude the module things are autowrapped in.
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
  if len(node.body) != 1 and not isinstance(node.body[0], gast.Expr):
    raise ValueError(
        'Expected a single expression, found instead %s' % node.body)
  return node.body[0].value
