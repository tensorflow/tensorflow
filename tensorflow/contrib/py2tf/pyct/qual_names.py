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
"""Utilities for manipulating qualified names.

A qualified name is a uniform way to refer to simple (e.g. 'foo') and composite
(e.g. 'foo.bar') syntactic symbols.

This is *not* related to the __qualname__ attribute used by inspect, which
refers to scopes.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.contrib.py2tf.pyct import anno


class QN(object):
  """Represents a qualified name.

  """

  def __init__(self, base, attr=None):
    if attr:
      if not isinstance(base, QN):
        raise ValueError('For attribute QNs, base must be a QN.')
      self._parent = base
      self.qn = base.qn + (attr,)
    else:
      self._parent = None
      self.qn = tuple(base.split('.'))

  def is_composite(self):
    return len(self.qn) > 1

  @property
  def parent(self):
    if self._parent is None:
      raise ValueError('Cannot get parent of simple name "%s".' % self.qn[0])
    return self._parent

  def __hash__(self):
    return hash(self.qn)

  def __eq__(self, other):
    return self.qn == other.qn

  def __str__(self):
    return '.'.join(self.qn)

  def __repr__(self):
    return str(self)

  def ssf(self):
    """Simple symbol form."""
    return '_'.join(self.qn)

  def ast(self):
    # The caller must adjust the context appropriately.
    if self.is_composite():
      return gast.Attribute(self.parent.ast(), self.qn[-1], None)
    return gast.Name(self.qn[0], None, None)


class QnResolver(gast.NodeTransformer):
  """Annotates nodes with QN information.

  Note: Not using NodeAnnos to avoid circular dependencies.
  """

  def visit_Name(self, node):
    self.generic_visit(node)
    anno.setanno(node, anno.Basic.QN, QN(node.id))
    return node

  def visit_Attribute(self, node):
    self.generic_visit(node)
    anno.setanno(node, anno.Basic.QN,
                 QN(anno.getanno(node.value, anno.Basic.QN), node.attr))
    return node


def resolve(node):
  return QnResolver().visit(node)
