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
"""Handles builtins and other special functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.contrib.py2tf.pyct import templates


class BuiltinFunctionTransformer(gast.NodeTransformer):
  """Transforms Print nodes to Call so they can be handled as functions."""

  # TODO(mdan): Bring print_functions in here.

  def _convert_len(self, node):
    template = """
      tf.shape(args)[0]
    """
    new_call = templates.replace(template, args=node.args)[0].value
    return new_call

  # pylint:disable=invalid-name

  def visit_Call(self, node):
    self.generic_visit(node)
    # TODO(mdan): This won't work if the function was hidden.
    if isinstance(node.func, gast.Name) and node.func.id == 'len':
      return self._convert_len(node)
    return node

  # pylint:enable=invalid-name


def transform(node):
  transformer = BuiltinFunctionTransformer()
  node = transformer.visit(node)
  return node
