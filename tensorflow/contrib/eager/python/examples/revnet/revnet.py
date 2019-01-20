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

    self._init_block = blocks.InitBlock(config=self.config)
    self._final_block = blocks.FinalBlock(config=self.config)
    self._block_list = self._construct_intermediate_blocks()
    self._moving_average_variables = []

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

    saved_hidden = None
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

  def compute_gradients(self, saved_hidden, labels, training=True, l2_reg=True):
    """Manually computes gradients.

    This method silently updates the running averages of batch normalization.

    Args:
      saved_hidden: List of hidden states Tensors
      labels: One-hot labels for classification
      training: Use the mini-batch stats in batch norm if set to True
      l2_reg: Apply l2 regularization

    Returns:
      A tuple with the first entry being a list of all gradients and the second
      being the loss
    """

    def _defunable_pop(l):
      """Functional style list pop that works with `tfe.defun`."""
      t, l = l[-1], l[:-1]
      return t, l

    # Backprop through last block
    x = saved_hidden[-1]
    with tf.GradientTape() as tape:
      tape.watch(x)
      logits = self._final_block(x, training=training)
      loss = self.compute_loss(logits, labels)
    grads_combined = tape.gradient(loss,
                                   [x] + self._final_block.trainable_variables)
    dy, final_grads = grads_combined[0], grads_combined[1:]

    # Backprop through intermediate blocks
    intermediate_grads = []
    for block in reversed(self._block_list):
      y, saved_hidden = _defunable_pop(saved_hidden)
      x = saved_hidden[-1]
      dy, grads = block.backward_grads(x, y, dy, training=training)
      intermediate_grads = grads + intermediate_grads

    # Backprop through first block
    _, saved_hidden = _defunable_pop(saved_hidden)
    x, saved_hidden = _defunable_pop(saved_hidden)
    assert not saved_hidden
    with tf.GradientTape() as tape:
      y = self._init_block(x, training=training)
    init_grads = tape.gradient(
        y, self._init_block.trainable_variables, output_gradients=dy)

    # Ordering match up with `model.trainable_variables`
    grads_all = init_grads + final_grads + intermediate_grads
    if l2_reg:
      grads_all = self._apply_weight_decay(grads_all)

    return grads_all, loss

  def _apply_weight_decay(self, grads):
    """Update gradients to reflect weight decay."""
    return [
        g + self.config.weight_decay * v if v.name.endswith("kernel:0") else g
        for g, v in zip(grads, self.trainable_variables)
    ]

  def get_moving_stats(self):
    """Get moving averages of batch normalization."""
    device = "/gpu:0" if tf.test.is_gpu_available() else "/cpu:0"
    with tf.device(device):
      return [v.read_value() for v in self.moving_average_variables]

  def restore_moving_stats(self, values):
    """Restore moving averages of batch normalization."""
    device = "/gpu:0" if tf.test.is_gpu_available() else "/cpu:0"
    with tf.device(device):
      for var_, val in zip(self.moving_average_variables, values):
        var_.assign(val)

  @property
  def moving_average_variables(self):
    """Get all variables that are batch norm moving averages."""

    def _is_moving_avg(v):
      n = v.name
      return n.endswith("moving_mean:0") or n.endswith("moving_variance:0")

    if not self._moving_average_variables:
      self._moving_average_variables = filter(_is_moving_avg, self.variables)

    return self._moving_average_variables
