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
"""Modifies the signature to allow resolving the value of default arguments.

Normally, function symbols are captured either in a function's globals or
closure. This is not true for default arguments, which are evaluated when the
function is defined:

    b = 1
    c = 2
    def f(a=b + 1):
      return a + c

In the above example, the namespace of the function would include `c = 2` but
not `b`.

If we were to naively generate a new function:

    def new_f(a=b + 1):
      return a + c

The generated code would fail to load unless we exposed a symbol `b`. Capturing
the closure of such an expression is difficult. However, we can capture the
default value of argument `a` with relative ease.

This converter replaces all default argument expressions with a constant so
that they don't cause loading to fail. This requires that the default values
are reset after loading the transformed function:

    def new_f(a=None):
      return a + c

    # ... later, after new_f was loaded ...
    new_f.__defaults__ = f.__defaults__

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import parser


class ArgDefaultsTransformer(converter.Base):
  """Transforms top level argument defaults.

  This transformer modifies self.ctx.arg_defaults directly.
  """

  def visit_arguments(self, node):
    for i in range(len(node.defaults)):
      node.defaults[i] = parser.parse_expression('None')

    for i, d in enumerate(node.kw_defaults):
      if d is not None:
        node.kw_defaults[i] = parser.parse_expression('None')

    # Only the top level function is modified - no need to visit the children.
    return node


def transform(node, ctx):
  """Transform function call to the compiled counterparts.

  Args:
    node: AST
    ctx: EntityContext
  Returns:
    A tuple (node, new_names):
        node: The transformed AST
        new_names: set(string), containing any newly-generated names
  """
  return ArgDefaultsTransformer(ctx).visit(node)
