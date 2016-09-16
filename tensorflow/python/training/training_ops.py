# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Python wrappers for training ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.training import gen_training_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.training.gen_training_ops import *
# pylint: enable=wildcard-import


# Shape functions for fused training ops
# --------------------------------------
#
# The fused training ops all have the same basic structure: they take
# one or more variables with the same shape, and emit a reference to
# the original variable (which has the same shape as the first
# input). In addition, they take one or more scalar tensors containing
# hyperparameters.
#
# The sparse ops take the gradients as a Python IndexedSlices, which
# means that the indices are a vector of length N, and the gradient
# values are a tensor whose size is the same as the original variable,
# except for the 0th dimension, which has size N.


def _AssertInputIsScalar(op, index):
  """Raises ValueError if `op.inputs[index]` is not scalar."""
  op.inputs[index].get_shape().assert_is_compatible_with(tensor_shape.scalar())


ops.RegisterShape("ApplyAdadelta")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("ApplyAdagrad")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("ApplyProximalAdagrad")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("ApplyFtrl")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("ApplyAdagradDA")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("ApplyAdam")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("ApplyMomentum")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("ApplyRMSProp")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("ApplyGradientDescent")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("ApplyProximalGradientDescent")(
    common_shapes.call_cpp_shape_fn)
ops.RegisterShape("SparseApplyProximalGradientDescent")(
    common_shapes.call_cpp_shape_fn)
ops.RegisterShape("SparseApplyRMSProp")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("SparseApplyAdadelta")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("SparseApplyAdagrad")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("SparseApplyProximalAdagrad")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("SparseApplyFtrl")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("SparseApplyAdagradDA")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("SparseApplyMomentum")(common_shapes.call_cpp_shape_fn)
