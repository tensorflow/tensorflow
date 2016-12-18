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
"""Sparsemax op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.util import loader
from tensorflow.python.platform import resource_loader
from tensorflow.python.framework import ops, common_shapes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops

_sparsemax = loader.load_op_library(
    resource_loader.get_path_to_datafile("_sparsemax.so"))

# The C++ op fully defines the op
sparsemax = _sparsemax.sparsemax

# Bind the python shape to C++ shape op
ops.RegisterShape("Sparsemax")(common_shapes.call_cpp_shape_fn)


@ops.RegisterGradient("Sparsemax")
def _sparsemax_grad(op, grad):
    """The gradients for the Sparsemax op.

    Args:
    op: The `Sparsemax` operation that we are differentiating, which we
      can use to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `Sparsemax` op.

    Returns:
    Gradients with respect to the input of `Sparsemax`.
    """
    # Construct S(z)
    sparsemax = op.outputs[0]
    support = math_ops.cast(sparsemax > 0, sparsemax.dtype)

    # Calculate \hat{v}, which will be a vector (scalar for each z)
    v_hat = math_ops.reduce_sum(math_ops.mul(grad, support), 1) \
        / math_ops.reduce_sum(support, 1)

    # Calculates J(z) * v
    return [support * (grad - v_hat[:, array_ops.newaxis])]
