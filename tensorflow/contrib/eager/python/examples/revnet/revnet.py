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
"""Reversible residual network compatible with eager execution.

Code for main model.

Reference [The Reversible Residual Network: Backpropagation
Without Storing Activations](https://arxiv.org/pdf/1707.04585.pdf)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import operator

import six
import tensorflow as tf
from tensorflow.contrib.eager.python.examples.revnet import blocks


class RevNet(tf.keras.Model):
  """RevNet that depends on all the blocks."""

  def __init__(self, config):
    """Initialize RevNet with building blocks.

    Args:
      config: tf.contrib.training.HParams object; specifies hyperparameters
    """
    super(RevNet, self).__init__()
    self.axis = 1 if config.data_format == "channels_first" else 3
    self.config = config

    self._init_block = self._construct_init_block()
    self._block_list = self._construct_intermediate_blocks()
    self._final_block = self._construct_final_block()

  def _construct_init_block(self):
    init_block = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                filters=self.config.init_filters,
                kernel_size=self.config.init_kernel,
                strides=(self.config.init_stride, self.config.init_stride),
                data_format=self.config.data_format,
                use_bias=False,
                padding="SAME",
                input_shape=self.config.input_shape,
                dtype=self.config.dtype),
            tf.keras.layers.BatchNormalization(
                axis=self.axis,
                fused=self.config.fused,
                dtype=self.config.dtype),
            tf.keras.layers.Activation("relu"),
        ],
        name="init")
    if self.config.init_max_pool:
      init_block.add(
          tf.keras.layers.MaxPooling2D(
              pool_size=(3, 3),
              strides=(2, 2),
              padding="SAME",
              data_format=self.config.data_format,
              dtype=self.config.dtype))
    return init_block

  def _construct_final_block(self):
    f = self.config.filters[-1]  # Number of filters
    r = functools.reduce(operator.mul, self.config.strides, 1)  # Reduce ratio
    r *= self.config.init_stride
    if self.config.init_max_pool:
      r *= 2

    if self.config.data_format == "channels_first":
      w, h = self.config.input_shape[1], self.config.input_shape[2]
      input_shape = (f, w // r, h // r)
    elif self.config.data_format == "channels_last":
      w, h = self.config.input_shape[0], self.config.input_shape[1]
      input_shape = (w // r, h // r, f)
    else:
      raise ValueError("Data format should be either `channels_first`"
                       " or `channels_last`")

    final_block = tf.keras.Sequential(
        [
            tf.keras.layers.BatchNormalization(
                axis=self.axis,
                input_shape=input_shape,
                fused=self.config.fused,
                dtype=self.config.dtype),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.GlobalAveragePooling2D(
                data_format=self.config.data_format, dtype=self.config.dtype),
            tf.keras.layers.Dense(
                self.config.n_classes, dtype=self.config.dtype)
        ],
        name="final")
    return final_block

  def _construct_intermediate_blocks(self):
    # Precompute input shape after initial block
    stride = self.config.init_stride
    if self.config.init_max_pool:
      stride *= 2
    if self.config.data_format == "channels_first":
      w, h = self.config.input_shape[1], self.config.input_shape[2]
      input_shape = (self.config.init_filters, w // stride, h // stride)
    else:
      w, h = self.config.input_shape[0], self.config.input_shape[1]
      input_shape = (w // stride, h // stride, self.config.init_filters)

    # Aggregate intermediate blocks
    block_list = tf.contrib.checkpoint.List()
    for i in range(self.config.n_rev_blocks):
      # RevBlock configurations
      n_res = self.config.n_res[i]
      filters = self.config.filters[i]
      if filters % 2 != 0:
        raise ValueError("Number of output filters must be even to ensure"
                         "correct partitioning of channels")
      stride = self.config.strides[i]
      strides = (self.config.strides[i], self.config.strides[i])

      # Add block
      rev_block = blocks.RevBlock(
          n_res,
          filters,
          strides,
          input_shape,
          batch_norm_first=(i != 0),  # Only skip on first block
          data_format=self.config.data_format,
          bottleneck=self.config.bottleneck,
          fused=self.config.fused,
          dtype=self.config.dtype)
      block_list.append(rev_block)

      # Precompute input shape for the next block
      if self.config.data_format == "channels_first":
        w, h = input_shape[1], input_shape[2]
        input_shape = (filters, w // stride, h // stride)
      else:
        w, h = input_shape[0], input_shape[1]
        input_shape = (w // stride, h // stride, filters)

    return block_list

  def call(self, inputs, training=True):
    """Forward pass."""

    if training:
      saved_hidden = [inputs]

    h = self._init_block(inputs, training=training)
    if training:
      saved_hidden.append(h)

    for block in self._block_list:
      h = block(h, training=training)
      if training:
        saved_hidden.append(h)

    logits = self._final_block(h, training=training)

    return (logits, saved_hidden) if training else (logits, None)

  def compute_loss(self, logits, labels):
    """Compute cross entropy loss."""

    if self.config.dtype == tf.float32 or self.config.dtype == tf.float16:
      cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=labels)
    else:
      # `sparse_softmax_cross_entropy_with_logits` does not have a GPU kernel
      # for float64, int32 pairs
      labels = tf.one_hot(
          labels, depth=self.config.n_classes, axis=1, dtype=self.config.dtype)
      cross_ent = tf.nn.softmax_cross_entropy_with_logits(
          logits=logits, labels=labels)

    return tf.reduce_mean(cross_ent)

  def compute_gradients(self, inputs, labels, training=True, l2_reg=True):
    """Manually computes gradients.

    When eager execution is enabled, this method also SILENTLY updates the
    running averages of batch normalization when `training` is set to True.

    Args:
      inputs: Image tensor, either NHWC or NCHW, conforming to `data_format`
      labels: One-hot labels for classification
      training: Use the mini-batch stats in batch norm if set to True
      l2_reg: Apply l2 regularization

    Returns:
      list of tuples each being (grad, var) for optimizer to use
    """

    # Run forward pass to record hidden states; avoid updating running averages
    vars_and_vals = self.get_moving_stats()
    _, saved_hidden = self.call(inputs, training=training)
    self.restore_moving_stats(vars_and_vals)

    grads_all = []
    vars_all = []

    # Manually backprop through last block
    x = saved_hidden[-1]
    with tf.GradientTape() as tape:
      x = tf.identity(x)
      tape.watch(x)
      # Running stats updated below
      logits = self._final_block(x, training=training)
      loss = self.compute_loss(logits, labels)

    grads_combined = tape.gradient(loss,
                                   [x] + self._final_block.trainable_variables)
    dy, grads_ = grads_combined[0], grads_combined[1:]
    grads_all += grads_
    vars_all += self._final_block.trainable_variables

    # Manually backprop through intermediate blocks
    for block in reversed(self._block_list):
      y = saved_hidden.pop()
      x = saved_hidden[-1]
      dy, grads, vars_ = block.backward_grads_and_vars(
          x, y, dy, training=training)
      grads_all += grads
      vars_all += vars_

    # Manually backprop through first block
    saved_hidden.pop()
    x = saved_hidden.pop()
    assert not saved_hidden  # Cleared after backprop

    with tf.GradientTape() as tape:
      x = tf.identity(x)
      # Running stats updated below
      y = self._init_block(x, training=training)

    grads_all += tape.gradient(
        y, self._init_block.trainable_variables, output_gradients=dy)
    vars_all += self._init_block.trainable_variables

    # Apply weight decay
    if l2_reg:
      grads_all = self._apply_weight_decay(grads_all, vars_all)

    return grads_all, vars_all, loss

  def _apply_weight_decay(self, grads, vars_):
    """Update gradients to reflect weight decay."""
    # Don't decay bias
    return [
        g + self.config.weight_decay * v if v.name.endswith("kernel:0") else g
        for g, v in zip(grads, vars_)
    ]

  def get_moving_stats(self):
    """Get moving averages of batch normalization.

    This is needed to avoid updating the running average twice in one iteration.

    Returns:
      A dictionary mapping variables for batch normalization moving averages
      to their current values.
    """
    vars_and_vals = {}

    def _is_moving_var(v):
      n = v.name
      return n.endswith("moving_mean:0") or n.endswith("moving_variance:0")

    for v in filter(_is_moving_var, self.variables):
      vars_and_vals[v] = v.read_value()

    return vars_and_vals

  def restore_moving_stats(self, vars_and_vals):
    """Restore moving averages of batch normalization.

    This is needed to avoid updating the running average twice in one iteration.

    Args:
      vars_and_vals: The dictionary mapping variables to their previous values.
    """
    for var_, val in six.iteritems(vars_and_vals):
      var_.assign(val)
