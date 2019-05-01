# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Gradients for Popnn operators."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.compiler.plugin.poplar.ops import gen_poprand_ops
"""
    These gradient function should *never* be called directly.
"""


@ops.RegisterGradient("IpuDropout")
def _poputil_dropout_layer_backward(op, grads, seed_grads):
  """Gradients for the IpuDropout op."""
  seed = op.outputs[1]
  rate = op.get_attr("rate")
  scale = op.get_attr("scale")
  seed_modifier = op.get_attr("seed_modifier")

  return [
      gen_poprand_ops.ipu_dropout(
          grads,
          seed=seed,
          user_seed=1,
          rate=rate,
          scale=scale,
          name=op.name + "_grad",
          is_using_user_seed=True,
          seed_modifier=seed_modifier)[0],

      # The seed is an input so needs a gradient as well
      None
  ]
