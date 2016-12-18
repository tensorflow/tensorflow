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
"""Sparsemax Loss op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.util import loader
from tensorflow.python.platform import resource_loader
from tensorflow.python.framework import ops, common_shapes
from tensorflow.python.ops import array_ops

_sparsemax_loss = loader.load_op_library(
    resource_loader.get_path_to_datafile("_sparsemax_loss.so"))

# The C++ op fully defines the op
sparsemax_loss = _sparsemax_loss.sparsemax_loss

# Bind the python shape to C++ shape op
ops.RegisterShape("SparsemaxLoss")(common_shapes.call_cpp_shape_fn)


@ops.RegisterGradient("SparsemaxLoss")
def _sparsemax_loss_grad(op, grad):
    """The gradients for the SparsemaxLoss op.

    Args:
    op: The `SparsemaxLoss` operation that we are differentiating, which we
      can use to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `SparsemaxLoss` op.

    Returns:
    Gradients with respect to the input of `SparsemaxLoss`.
    """
    # Get parameters in correct shape
    sparsemax = op.inputs[1]
    labels = op.inputs[2]
    grad = array_ops.expand_dims(grad, 1)

    return [
        grad * (-labels + sparsemax),
        None,
        None
    ]
