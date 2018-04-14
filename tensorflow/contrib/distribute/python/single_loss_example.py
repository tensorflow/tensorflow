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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distribute.python import step_fn
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.layers import core
from tensorflow.python.layers import normalization
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


def single_loss_example(optimizer_fn, distribution, use_bias=False):
  """Build a very simple network to use in tests and examples."""
  dataset = dataset_ops.Dataset.from_tensors([[1.]]).repeat()
  optimizer = optimizer_fn()
  layer = core.Dense(1, use_bias=use_bias)

  def loss_fn(x):
    y = array_ops.reshape(layer(x), []) - constant_op.constant(1.)
    return y * y

  single_loss_step = step_fn.StandardSingleLossStep(dataset, loss_fn, optimizer,
                                                    distribution)

  # Layer is returned for inspecting the kernels in tests.
  return single_loss_step, layer


def minimize_loss_example(optimizer_fn,
                          use_bias=False,
                          use_callable_loss=True,
                          create_optimizer_inside_model_fn=False):
  """Example of non-distribution-aware legacy code."""
  dataset = dataset_ops.Dataset.from_tensors([[1.]]).repeat()
  # An Optimizer instance is created either outside or inside model_fn.
  outer_optimizer = None
  if not create_optimizer_inside_model_fn:
    outer_optimizer = optimizer_fn()

  layer = core.Dense(1, use_bias=use_bias)

  def model_fn(x):
    """A very simple model written by the user."""

    def loss_fn():
      y = array_ops.reshape(layer(x), []) - constant_op.constant(1.)
      return y * y

    optimizer = outer_optimizer or optimizer_fn()

    if use_callable_loss:
      return optimizer.minimize(loss_fn)
    else:
      return optimizer.minimize(loss_fn())

  return model_fn, dataset, layer


def batchnorm_example(optimizer_fn,
                      batch_per_epoch=1,
                      momentum=0.9,
                      renorm=False):
  """Example of non-distribution-aware legacy code with batch normalization."""
  # input shape is [16, 8], input values are increasing in both dimensions.
  dataset = dataset_ops.Dataset.from_tensor_slices(
      [[[float(x * 8 + y + z * 100)
         for y in range(8)]
        for x in range(16)]
       for z in range(batch_per_epoch)]).repeat()
  optimizer = optimizer_fn()
  batchnorm = normalization.BatchNormalization(
      renorm=renorm, momentum=momentum, fused=False)

  def model_fn(x):

    def loss_fn():
      y = math_ops.reduce_sum(batchnorm(x, training=True), axis=1)
      loss = math_ops.reduce_mean(y - constant_op.constant(1.))
      return loss

    # Callable loss.
    return optimizer.minimize(loss_fn)

  return model_fn, dataset, batchnorm
