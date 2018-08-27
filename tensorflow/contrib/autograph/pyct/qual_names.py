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

import collections

import gast

from tensorflow.contrib.autograph.pyct import anno
from tensorflow.contrib.autograph.pyct import parser


class Symbol(collections.namedtuple('Symbol', ['name'])):
  """Represents a Python symbol."""


class StringLiteral(collections.namedtuple('StringLiteral', ['value'])):
  """Represents a Python string literal."""

  def __str__(self):
    return '\'%s\'' % self.value

  def __repr__(self):
    return str(self)


class NumberLiteral(collections.namedtuple('NumberLiteral', ['value'])):
  """Represents a Python numeric literal."""

  def __str__(self):
    return '%s' % self.value

  def __repr__(self):
    return str(self)


# TODO(mdan): Use subclasses to remove the has_attr has_subscript booleans.
class QN(object):
  """Represents a qualified name."""

  def __init__(self, base, attr=None, subscript=None):
    if attr is not None and subscript is not None:
      raise ValueError('A QN can only be either an attr or a subscript, not '
                       'both: attr={}, subscript={}.'.format(attr, subscript))
    self._has_attr = False
    self._has_subscript = False

    if attr is not None:
      if not isinstance(base, QN):
        raise ValueError(
            'for attribute QNs, base must be a QN; got instead "%s"' % base)
      if not isinstance(attr, str):
        raise ValueError('attr may only be a string; got instead "%s"' % attr)
      self._parent = base
      # TODO(mdan): Get rid of the tuple - it can only have 1 or 2 elements now.
      self.qn = (base, attr)
      self._has_attr = True

    elif subscript is not None:
      if not isinstance(base, QN):
        raise ValueError('For subscript QNs, base must be a QN.')
      self._parent = base
      self.qn = (base, subscript)
      self._has_subscript = True

    else:
      if not isinstance(base, (str, StringLiteral, NumberLiteral)):
        # TODO(mdan): Require Symbol instead of string.
        raise ValueError(
            'for simple QNs, base must be a string or a Literal object;'
            ' got instead "%s"' % type(base))
      assert '.' not in base and '[' not in base and ']' not in base
      self._parent = None
      self.qn = (base,)

  def is_symbol(self):
    return isinstance(self.qn[0], str)

  def is_composite(self):
    return len(self.qn) > 1

  def has_subscript(self):
    return self._has_subscript

  def has_attr(self):
    return self._has_attr

  @property
  def parent(self):
    if self._parent is None:
      raise ValueError('Cannot get parent of simple name "%s".' % self.qn[0])
    return self._parent

  @property
  def owner_set(self):
    """Returns all the symbols (simple or composite) that own this QN.

    In other words, if this symbol was modified, the symbols in the owner set
    may also be affected.

    Examples:
      'a.b[c.d]' has two owners, 'a' and 'a.b'
    """
    owners = set()
    if self.has_attr() or self.has_subscript():
      owners.add(self.parent)
      owners.update(self.parent.owner_set)
    return owners

  @property
  def support_set(self):
    """Returns the set of simple symbols that this QN relies on.

    This would be the smallest set of symbols necessary for the QN to
    statically resolve (assuming properties and index ranges are verified
    at runtime).

    Examples:
      'a.b' has only one support symbol, 'a'
      'a[i]' has two support symbols, 'a' and 'i'
    """
    # TODO(mdan): This might be the set of Name nodes in the AST. Track those?
    roots = set()
    if self.has_attr():
      roots.update(self.parent.support_set)
    elif self.has_subscript():
      roots.update(self.parent.support_set)
      roots.update(self.qn[1].support_set)
    else:
      roots.add(self)
    return roots

  def __hash__(self):
    return hash(self.qn + (self._has_attr, self._has_subscript))

  def __eq__(self, other):
    return (isinstance(other, QN) and self.qn == other.qn and
            self.has_subscript() == other.has_subscript() and
            self.has_attr() == other.has_attr())

  def __str__(self):
    if self.has_subscript():
      return str(self.qn[0]) + '[' + str(self.qn[1]) + ']'
    if self.has_attr():
      return '.'.join(map(str, self.qn))
    else:
      return str(self.qn[0])

  def __repr__(self):
    return str(self)

  def ssf(self):
    """Simple symbol form."""
    ssfs = [n.ssf() if isinstance(n, QN) else n for n in self.qn]
    ssf_string = ''
    for i in range(0, len(self.qn) - 1):
      if self.has_subscript():
        delimiter = '_sub_'
      else:
        delimiter = '_'
      ssf_string += ssfs[i] + delimiter
    return ssf_string + ssfs[-1]

  def ast(self):
    # The caller must adjust the context appropriately.
    if self.has_subscript():
      return gast.Subscript(self.parent.ast(), gast.Index(self.qn[-1].ast()),
                            None)
    if self.has_attr():
      return gast.Attribute(self.parent.ast(), self.qn[-1], None)

    base = self.qn[0]
    if isinstance(base, str):
      return gast.Name(base, None, None)
    elif isinstance(base, StringLiteral):
      return gast.Str(base.value)
    elif isinstance(base, NumberLiteral):
      return gast.Num(base.value)
    else:
      assert False, ('the constructor should prevent types other than '
                     'str, StringLiteral and NumberLiteral')


class QnResolver(gast.NodeTransformer):
  """Annotates nodes with QN information.

  Note: Not using NodeAnnos to avoid circular dependencies.
  """

  def visit_Name(self, node):
    node = self.generic_visit(node)
    anno.setanno(node, anno.Basic.QN, QN(node.id))
    return node

  def visit_Attribute(self, node):
    node = self.generic_visit(node)
    if anno.hasanno(node.value, anno.Basic.QN):
      anno.setanno(node, anno.Basic.QN,
                   QN(anno.getanno(node.value, anno.Basic.QN), attr=node.attr))
    return node

  def visit_Subscript(self, node):
    # TODO(mdan): This may no longer apply if we overload getitem.
    node = self.generic_visit(node)
    s = node.slice
    if not isinstance(s, gast.Index):
      # TODO(mdan): Support range and multi-dimensional indices.
      # Continuing silently because some demos use these.
      return node
    if isinstance(s.value, gast.Num):
      subscript = QN(NumberLiteral(s.value.n))
    elif isinstance(s.value, gast.Str):
      subscript = QN(StringLiteral(s.value.s))
    else:
      # The index may be an expression, case in which a name doesn't make sense.
      if anno.hasanno(node.slice.value, anno.Basic.QN):
        subscript = anno.getanno(node.slice.value, anno.Basic.QN)
      else:
        return node
    if anno.hasanno(node.value, anno.Basic.QN):
      anno.setanno(node, anno.Basic.QN,
                   QN(anno.getanno(node.value, anno.Basic.QN),
                      subscript=subscript))
    return node


def resolve(node):
  return QnResolver().visit(node)


def from_str(qn_str):
  node = parser.parse_expression(qn_str)
  node = resolve(node)
  return anno.getanno(node, anno.Basic.QN)
