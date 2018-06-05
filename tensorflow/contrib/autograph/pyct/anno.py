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
"""Handling annotations on AST nodes.

Adapted from Tangent.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from enum import Enum


class NoValue(Enum):

  def __repr__(self):
    return self.name


class Basic(NoValue):
  """Container for annotation keys.

  The enum values are used strictly for documentation purposes.
  """

  QN = 'Qualified name, as it appeared in the code.'
  SKIP_PROCESSING = (
      'This node should be preserved as is and not processed any further.')
  INDENT_BLOCK_REMAINDER = (
      'When a node is annotated with this, the remainder of the block should '
      'be indented below it. The annotation contains a tuple '
      '(new_body, name_map), where `new_body` is the new indented block and '
      '`name_map` allows renaming symbols.')


FAIL = object()


def getanno(node, key, default=FAIL, field_name='___pyct_anno'):
  if (default is FAIL or
      (hasattr(node, field_name) and (key in getattr(node, field_name)))):
    return getattr(node, field_name)[key]
  else:
    return default


def hasanno(node, key, field_name='___pyct_anno'):
  return hasattr(node, field_name) and key in getattr(node, field_name)


def setanno(node, key, value, field_name='___pyct_anno'):
  annotations = getattr(node, field_name, {})
  setattr(node, field_name, annotations)
  annotations[key] = value

  # So that the annotations survive gast_to_ast() and ast_to_gast()
  if field_name not in node._fields:
    node._fields += (field_name,)


def delanno(node, key, field_name='___pyct_anno'):
  annotations = getattr(node, field_name)
  del annotations[key]
  if not annotations:
    delattr(node, field_name)
    node._fields = tuple(f for f in node._fields if f != field_name)


def copyanno(from_node, to_node, key, field_name='___pyct_anno'):
  if hasanno(from_node, key, field_name=field_name):
    setanno(
        to_node,
        key,
        getanno(from_node, key, field_name=field_name),
        field_name=field_name)
