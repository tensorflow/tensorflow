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
"""A simple network to use in tests and examples."""

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.layers import core
from tensorflow.python.layers import normalization
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


def minimize_loss_example(optimizer, use_bias=False, use_callable_loss=True):
  """Example of non-distribution-aware legacy code."""

  def dataset_fn():
    dataset = dataset_ops.Dataset.from_tensors([[1.]]).repeat()
    # TODO(isaprykin): batch with drop_remainder causes shapes to be
    # fully defined for TPU.  Remove this when XLA supports dynamic shapes.
    return dataset.batch(1, drop_remainder=True)

  layer = core.Dense(1, use_bias=use_bias)

  def model_fn(x):
    """A very simple model written by the user."""

    def loss_fn():
      y = array_ops.reshape(layer(x), []) - constant_op.constant(1.)
      return y * y

    if isinstance(optimizer, optimizer_v2.OptimizerV2):
      return optimizer.minimize(loss_fn, lambda: layer.trainable_variables)
    elif use_callable_loss:
      return optimizer.minimize(loss_fn)
    else:
      return optimizer.minimize(loss_fn())

  return model_fn, dataset_fn, layer


def batchnorm_example(optimizer_fn,
                      batch_per_epoch=1,
                      momentum=0.9,
                      renorm=False,
                      update_ops_in_replica_mode=False):
  """Example of non-distribution-aware legacy code with batch normalization."""

  def dataset_fn():
    # input shape is [16, 8], input values are increasing in both dimensions.
    return dataset_ops.Dataset.from_tensor_slices(
        [[[float(x * 8 + y + z * 100)
           for y in range(8)]
          for x in range(16)]
         for z in range(batch_per_epoch)]).repeat()

  optimizer = optimizer_fn()
  batchnorm = normalization.BatchNormalization(
      renorm=renorm, momentum=momentum, fused=False)
  layer = core.Dense(1, use_bias=False)

  def model_fn(x):
    """A model that uses batchnorm."""

    def loss_fn():
      y = batchnorm(x, training=True)
      with ops.control_dependencies(
          ops.get_collection(ops.GraphKeys.UPDATE_OPS)
          if update_ops_in_replica_mode else []):
        loss = math_ops.reduce_mean(
            math_ops.reduce_sum(layer(y)) - constant_op.constant(1.))
      # `x` and `y` will be fetched by the gradient computation, but not `loss`.
      return loss

    if isinstance(optimizer, optimizer_v2.OptimizerV2):
      return optimizer.minimize(loss_fn, lambda: layer.trainable_variables)

    # Callable loss.
    return optimizer.minimize(loss_fn)

  return model_fn, dataset_fn, batchnorm
