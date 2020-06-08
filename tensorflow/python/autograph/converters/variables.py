# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Overloads all variable read operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import templates


class VariableAccessTransformer(converter.Base):
  """Rewrites basic symbol reads.

  This transformer rewrites variable reads with a "read" operator which allows
  tracking activity.

  Example:

  For a basic statement:

      a = b + c

  This is translated to:

      a = ld(b) + ld(c)

  Augmented assignment operations also introduce a `ld` operator:

      a += b

  The assignment target also receives an operator to properly represent the
  read:

      a = ld(a)
      a += ld(b)
  """

  def visit_Name(self, node):
    # Only the loads which existed in the original code are overloaded.
    if not anno.hasanno(node, anno.Static.ORIG_DEFINITIONS):
      return node
    if isinstance(node.ctx, gast.Load):
      node = templates.replace_as_expression('ag__.ld(var_)', var_=node)
    return node

  def visit_Delete(self, node):
    node = self.generic_visit(node)

    rewrite_targets = []
    for tgt in node.targets:
      # Don't rewrite composites like `del a[0]`.
      if isinstance(tgt, gast.Name):
        rewrite_targets.append(tgt)

    if not rewrite_targets:
      return node

    results = []
    for tgt in rewrite_targets:
      template = """
        var_ = ag__.Undefined(var_name)
      """
      results.extend(templates.replace(
          template, var_=tgt, var_name=gast.Constant(tgt.id, kind=None)))
    remaining_targets = [n for n in node.targets if n not in rewrite_targets]
    if remaining_targets:
      results.append(gast.Delete(targets=remaining_targets))

    return results

  def visit_AugAssign(self, node):
    if isinstance(node.target, gast.Name):
      template = """
        var_ = ag__.ld(var_)
        original
      """
      node = templates.replace(template, var_=node.target, original=node)
    else:
      node = self.generic_visit(node)
    return node


def transform(node, ctx):
  return VariableAccessTransformer(ctx).visit(node)
