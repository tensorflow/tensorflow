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
"""Wrappers for efficient GPU layer_norm_fused operations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.util import loader
from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader

import tensorflow as tf
import logging

_layer_norm_fused_op = loader.load_op_library(
    resource_loader.get_path_to_datafile("_layer_norm_fused_op.so"))


@ops.RegisterGradient("LayerNormCustom")
def _LayerNormCustomGrad(op, grad):
    return [_layer_norm_fused_op.layer_norm_backprop_custom(
        op.inputs[0], grad, op.get_attr("epsilon"))]


@ops.RegisterGradient("LayerNormBiasAddCustom")
def _LayerNormBiasAddCustomGrad(op, grad):
    in_back, beta_back = _layer_norm_fused_op.layer_norm_bias_add_backprop_custom(
        op.inputs[0], grad, op.inputs[1],
        op.get_attr("epsilon"))
    return [in_back, beta_back]


@ops.RegisterGradient("LayerNormFusedCustom")
def _LayerNormFusedCustomGrad(op, grad):
    in_back, gamma_back, beta_back = _layer_norm_fused_op.layer_norm_fused_backprop_custom(
        op.inputs[0], grad, op.inputs[1],
        op.get_attr("epsilon"))
    return [in_back, gamma_back, beta_back]


def layer_norm_fused_op(input_tensor, gamma=None, beta=None,
                        epsilon=1e-12, name=None):
    """Fast and efficient layer normalization along the last dimension

    See layer_norm_fused_op.cc for more details.

    Args:
      input_tensor: A `Tensor` which will be normalized.
      gamma: 1-D tensor for scaling after normalization.
             Must be the same size as the last dimension of input_tensor.
             Will be omitted if is None.
      beta: 1-D tensor for centering after normalization.
            Must be the same size as the last dimension of input_tensor.
             Will be omitted if is None.
      epsilon: small number added before variance calculation
              to avoid division by zero.

    Returns:
      A normalized `Tensor` with same dtype and shape as the input_tensor.
    """
    if epsilon <= 0:
        logging.warn("epsilon is %f <= ...", epsilon)
    if gamma is not None and beta is not None:
        return _layer_norm_fused_op.layer_norm_fused_custom(
            input_tensor, gamma, beta, epsilon=epsilon, name=name)
    elif gamma is not None:
        dtype = input_tensor.dtype.base_dtype
        beta = tf.zeros(input_tensor.get_shape().as_list()[-1], dtype=dtype,
                        name="dummy_beta")
        return _layer_norm_fused_op.layer_norm_fused_custom(
            input_tensor, gamma, beta, epsilon=epsilon, name=name)
    elif beta is not None:
        return _layer_norm_fused_op.layer_norm_bias_add_custom(
            input_tensor, beta, epsilon=epsilon, name=name)
    else:
        return _layer_norm_fused_op.layer_norm_custom(input_tensor,
                                                      epsilon=epsilon,
                                                      name=name)
