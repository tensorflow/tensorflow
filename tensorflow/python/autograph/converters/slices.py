# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Converter for slice operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.lang import directives
from tensorflow.python.autograph.pyct import templates


class SliceTransformer(converter.Base):
  """Converts slicing operations to their TF counterpart.

  Currently, relying on the default slice operator that Tensor uses is
  insufficient, because TensorArray and tensor lists use dedicated index read
  and write functions.
  """

  def _process_single_assignment(self, target, value):
    if not isinstance(target, gast.Subscript):
      return None
    if not isinstance(target.slice, gast.Index):
      return None

    template = """
      target = ag__.set_item(target, key, item)
    """
    return templates.replace(
        template, target=target.value, key=target.slice.value, item=value)

  def visit_Assign(self, node):
    node = self.generic_visit(node)
    # TODO(mdan): Support unpackings and multiple assignments.
    if len(node.targets) != 1:
      raise NotImplementedError('multiple assignment')
    replacement = self._process_single_assignment(node.targets[0], node.value)
    if replacement is not None:
      return replacement
    return node

  def visit_Subscript(self, node):
    node = self.generic_visit(node)
    if not isinstance(node.slice, gast.Index):
      return node

    if not isinstance(node.ctx, gast.Load):
      # Index writes are handled at a higher level, one at which the rvalue is
      # also available.
      return node

    dtype = self.get_definition_directive(
        node.value,
        directives.set_element_type,
        'dtype',
        default=templates.replace_as_expression('None'))

    template = """
      ag__.get_item(
          target,
          key,
          opts=ag__.GetItemOpts(element_dtype=dtype))
    """
    return templates.replace_as_expression(
        template, target=node.value, key=node.slice.value, dtype=dtype)


def transform(node, ctx):
  return SliceTransformer(ctx).visit(node)
