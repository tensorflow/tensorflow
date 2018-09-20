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
  source = textwrap.dedent(source)
  return parse_str(source), source


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
