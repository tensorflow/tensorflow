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
"""Converts assert statements to their corresponding TF calls."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import templates


class AssertTransformer(converter.Base):
  """Transforms Assert nodes to Call so they can be handled as functions."""

  def visit_Assert(self, node):
    self.generic_visit(node)

    # Note: The lone tf.Assert call will be wrapped with control_dependencies
    # by side_effect_guards.
    template = """
      tf.Assert(test, (msg,))
    """

    if node.msg is None:
      return templates.replace(
          template, test=node.test, msg=gast.Str('Assertion error'))
    elif isinstance(node.msg, gast.Str):
      return templates.replace(template, test=node.test, msg=node.msg)
    else:
      raise NotImplementedError('can only convert string messages for now.')


def transform(node, ctx):
  return AssertTransformer(ctx).visit(node)
