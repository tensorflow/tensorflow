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

from tensorflow.contrib.py2tf.pyct import anno


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
