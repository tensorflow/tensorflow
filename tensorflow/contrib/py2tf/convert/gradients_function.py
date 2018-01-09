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
"""Allows converting Eager-style gradients to graph versions."""
# TODO(mdan): This is not needed. Remove once the static analysis works.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.contrib.py2tf.pyct import templates


class GradientsFunctionTransformer(gast.NodeTransformer):
  """Hack: transforms eager-style gradients to TF compatible calls.

  Requires an expression of exactly this form:
      ... = tfe.value_and_gradients_function(...)(...)
  """

  # pylint:disable=invalid-name

  def visit_Assign(self, node):
    self.generic_visit(node)

    val = node.value
    if isinstance(val, gast.Call):
      if isinstance(val.func, gast.Call):
        if isinstance(val.func.func, gast.Attribute):
          if isinstance(val.func.func.value, gast.Name):
            if (val.func.func.value.id == 'tfe' and
                val.func.func.attr == 'value_and_gradients_function'):

              # pylint:disable=unused-argument,undefined-variable

              def template(loss_var, loss_fn, args, d_vars, wrt_vars):
                loss_var = loss_fn(args)
                d_vars = tf.gradients(loss_var, [wrt_vars])

              # pylint:enable=unused-argument,undefined-variable

              # How to get these values? Print out the node.
              loss_var = gast.Name(node.targets[0].elts[0].id, gast.Store(),
                                   None)
              loss_fn = gast.Name(val.func.args[0].id, gast.Load(), None)
              args = tuple(
                  gast.Name(a.id, gast.Param(), None) for a in val.args)
              d_vars = node.targets[0].elts[1]
              wrt_vars = [val.args[e.n] for e in val.func.args[1].elts]

              node = templates.replace(
                  template,
                  loss_var=loss_var,
                  loss_fn=loss_fn,
                  args=args,
                  d_vars=d_vars,
                  wrt_vars=wrt_vars)

    return node

  # pylint:enable=invalid-name


def transform(node):
  transformer = GradientsFunctionTransformer()
  node = transformer.visit(node)
  return node
