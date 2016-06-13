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

"""TensorFlow ops for Batch Normalization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops as array_ops_
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.training import moving_averages


def batch_normalize(tensor_in,
                    epsilon=1e-5,
                    convnet=False,
                    decay=0.9,
                    scale_after_normalization=True):
  """Batch normalization.

  Args:
    tensor_in: input `Tensor`, 4D shape: [batch, in_height, in_width, in_depth].
    epsilon : A float number to avoid being divided by 0.
    convnet: Whether this is for convolutional net use. If `True`, moments
        will sum across axis `[0, 1, 2]`. Otherwise, only `[0]`.
    decay: Decay rate for exponential moving average.
    scale_after_normalization: Whether to scale after normalization.

  Returns:
    A batch-normalized `Tensor`.
  """
  shape = tensor_in.get_shape().as_list()

  with vs.variable_scope("batch_norm"):
    gamma = vs.get_variable(
        "gamma", [shape[-1]],
        initializer=init_ops.random_normal_initializer(1., 0.02))
    beta = vs.get_variable("beta", [shape[-1]],
                           initializer=init_ops.constant_initializer(0.))
    moving_mean = vs.get_variable(
        'moving_mean',
        shape=[shape[-1]],
        initializer=init_ops.zeros_initializer,
        trainable=False)
    moving_var = vs.get_variable(
        'moving_var',
        shape=[shape[-1]],
        initializer=init_ops.ones_initializer,
        trainable=False)

    def _update_mean_var():
      """Internal function that updates mean and variance during training."""
      axis = [0, 1, 2] if convnet else [0]
      mean, var = nn.moments(tensor_in, axis)
      update_moving_mean = moving_averages.assign_moving_average(
          moving_mean, mean, decay)
      update_moving_var = moving_averages.assign_moving_average(
          moving_var, var, decay)
      with ops.control_dependencies([update_moving_mean, update_moving_var]):
        return array_ops_.identity(mean), array_ops_.identity(var)

    is_training = array_ops_.squeeze(ops.get_collection("IS_TRAINING"))
    mean, variance = control_flow_ops.cond(is_training, _update_mean_var,
                                           lambda: (moving_mean, moving_var))
    return nn.batch_norm_with_global_normalization(
        tensor_in,
        mean,
        variance,
        beta,
        gamma,
        epsilon,
        scale_after_normalization=scale_after_normalization)
