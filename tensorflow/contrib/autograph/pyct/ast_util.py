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
"""Copy an AST tree, discarding annotations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast

import gast

from tensorflow.contrib.autograph.pyct import anno
from tensorflow.contrib.autograph.pyct import parser


class CleanCopier(gast.NodeVisitor):
  """Copies AST nodes.

  The copied nodes will ignore almost all fields that are prefixed by '__'.
  Exceptions make some annotations.
  """

  # TODO(mdan): Parametrize which annotations get carried over.

  def generic_visit(self, node):
    new_fields = {}
    for f in node._fields:
      if f.startswith('__'):
        continue
      if not hasattr(node, f):
        continue
      v = getattr(node, f)
      if isinstance(v, list):
        v = [self.generic_visit(n) for n in v]
      elif isinstance(v, tuple):
        v = tuple(self.generic_visit(n) for n in v)
      elif isinstance(v, (gast.AST, ast.AST)):
        v = self.generic_visit(v)
      else:
        # Assume everything else is a value type.
        pass
      new_fields[f] = v
    new_node = type(node)(**new_fields)
    if anno.hasanno(node, anno.Basic.SKIP_PROCESSING):
      anno.setanno(new_node, anno.Basic.SKIP_PROCESSING, True)
    return new_node


def copy_clean(node):
  copier = CleanCopier()
  if isinstance(node, list):
    return [copier.visit(n) for n in node]
  elif isinstance(node, tuple):
    return tuple(copier.visit(n) for n in node)
  else:
    return copier.visit(node)


class SymbolRenamer(gast.NodeTransformer):
  """Transformer that can rename symbols to a simple names."""

  def __init__(self, name_map):
    self.name_map = name_map

  def _process(self, node):
    qn = anno.getanno(node, anno.Basic.QN)
    if qn in self.name_map:
      return gast.Name(str(self.name_map[qn]), node.ctx, None)
    return self.generic_visit(node)

  def visit_Name(self, node):
    return self._process(node)

  def visit_Attribute(self, node):
    if anno.hasanno(node, anno.Basic.QN):
      return self._process(node)
    # Attributes of dynamic objects will not have a QN.
    return self.generic_visit(node)


def rename_symbols(node, name_map):
  renamer = SymbolRenamer(name_map)
  if isinstance(node, list):
    return [renamer.visit(n) for n in node]
  elif isinstance(node, tuple):
    return tuple(renamer.visit(n) for n in node)
  return renamer.visit(node)


def keywords_to_dict(keywords):
  keys = []
  values = []
  for kw in keywords:
    keys.append(gast.Str(kw.arg))
    values.append(kw.value)
  return gast.Dict(keys=keys, values=values)


class PatternMatcher(gast.NodeVisitor):
  """Matches a node against a pattern represented by a node.

  The pattern may contain wildcards represented by the symbol '_'.
  """

  def __init__(self, pattern):
    self.pattern = pattern
    self.pattern_stack = []
    self.matches = True

  def compare_and_visit(self, node, pattern):
    self.pattern_stack.append(self.pattern)
    self.pattern = pattern
    self.generic_visit(node)
    self.pattern = self.pattern_stack.pop()

  def no_match(self):
    self.matches = False
    return False

  def is_wildcard(self, p):
    if isinstance(p, (list, tuple)) and len(p) == 1:
      p, = p
    if isinstance(p, gast.Name) and p.id == '_':
      return True
    if p == '_':
      return True
    return False

  def generic_visit(self, node):
    if not self.matches:
      return

    pattern = self.pattern
    for f in node._fields:
      if f.startswith('__'):
        continue

      if not hasattr(node, f):
        if hasattr(pattern, f) and getattr(pattern, f):
          return self.no_match()
        else:
          continue
      if not hasattr(pattern, f):
        return self.no_match()

      v = getattr(node, f)
      p = getattr(pattern, f)

      if self.is_wildcard(p):
        continue
      if isinstance(v, (list, tuple)):
        if not isinstance(p, (list, tuple)) or len(v) != len(p):
          return self.no_match()
        for v_item, p_item in zip(v, p):
          self.compare_and_visit(v_item, p_item)
      elif isinstance(v, (gast.AST, ast.AST)):
        if not isinstance(v, type(p)) and not isinstance(p, type(v)):
          return self.no_match()
        self.compare_and_visit(v, p)
      else:
        # Assume everything else is a value type.
        if v != p:
          return self.no_match()


def matches(node, pattern):
  if isinstance(pattern, str):
    pattern = parser.parse_expression(pattern)
  matcher = PatternMatcher(pattern)
  matcher.visit(node)
  return matcher.matches

