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
"""Wraps a function body with a `name_scope` of the function name.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.contrib.autograph.pyct import templates
from tensorflow.contrib.autograph.pyct import transformer


class FunctionNameScopeTransformer(transformer.Base):
  """Wrap a function body with a `name_scope` of the function name."""

  def __init__(self, context):
    super(FunctionNameScopeTransformer, self).__init__(context)
    self._function_level = 0

  def visit_FunctionDef(self, node):
    self._function_level += 1
    try:
      self.generic_visit(node)
    finally:
      self._function_level -= 1
    scope_name = node.name
    if self._function_level == 0 and self.context.owner_type is not None:
      scope_name = '{}/{}'.format(self.context.owner_type.__name__, scope_name)
    node.body = templates.replace(
        'with tf.name_scope(scope_name): body',
        scope_name=gast.Str(scope_name),
        body=node.body)
    return node


def transform(node, context):
  return FunctionNameScopeTransformer(context).visit(node)
