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
"""Canonicalizing list comprehensions into for and if statements.

e.g.
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

from tensorflow.contrib.autograph.pyct import parser
from tensorflow.contrib.autograph.pyct import templates
from tensorflow.contrib.autograph.pyct import transformer


class ListCompCanonicalizationTransformer(transformer.Base):
  """NodeTransformer to canonicalize list comprehensions."""

  def __init__(self, context):
    super(ListCompCanonicalizationTransformer, self).__init__(context)

  def make_update_list_node(self, list_, elt):
    return templates.replace('list_.append(elt)', list_=list_, elt=elt)[0]

  def instantiate_list_node(self):
    return parser.parse_str('[]').body[0].value

  def visit_Assign(self, node):
    if not isinstance(node.value, gast.ListComp):
      return node
    if len(node.targets) > 1:
      raise ValueError('Only support single assignment.')
    return self.canonicalize_listcomp(node.targets[0], node.value)

  def canonicalize_listcomp(self, result_node, list_comp_node):

    make_list = templates.replace(
        'list_ = create_list',
        list_=result_node,
        create_list=self.instantiate_list_node())
    loop_body = self.make_update_list_node(result_node, list_comp_node.elt)

    for gen in reversed(list_comp_node.generators):
      for gen_if in reversed(gen.ifs):
        loop_body = templates.replace(
            'if test: loop_body', test=gen_if, loop_body=loop_body)
      loop_body = templates.replace(
          'for target in iter_: loop_body',
          iter_=gen.iter,
          target=gen.target,
          loop_body=loop_body)

    return make_list + loop_body


def transform(node, context):
  return ListCompCanonicalizationTransformer(context).visit(node)
