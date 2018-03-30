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


class CleanCopier(gast.NodeVisitor):
  """Copy AST nodes.

  The copied nodes will ignore almost all fields that prefixed by '__'.
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
