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
"""Lowers list comprehensions into for and if statements.

Example:

  result = [x * x for x in xs]

becomes

  result = []
  for x in xs:
    elt = x * x
    result.append(elt)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.contrib.autograph.core import converter
from tensorflow.contrib.autograph.pyct import templates


# TODO(mdan): This should covert directly to operator calls.


class ListCompTransformer(converter.Base):
  """Lowers list comprehensions into standard control flow."""

  def visit_Assign(self, node):
    if not isinstance(node.value, gast.ListComp):
      return self.generic_visit(node)
    if len(node.targets) > 1:
      raise NotImplementedError('multiple assignments')

    target, = node.targets
    list_comp_node = node.value

    template = """
      target = []
    """
    initialization = templates.replace(template, target=target)

    template = """
      target.append(elt)
    """
    body = templates.replace(template, target=target, elt=list_comp_node.elt)

    for gen in reversed(list_comp_node.generators):
      for gen_if in reversed(gen.ifs):
        template = """
          if test:
            body
        """
        body = templates.replace(template, test=gen_if, body=body)
      template = """
        for target in iter_:
          body
      """
      body = templates.replace(
          template, iter_=gen.iter, target=gen.target, body=body)

    return initialization + body


def transform(node, ctx):
  return ListCompTransformer(ctx).visit(node)
