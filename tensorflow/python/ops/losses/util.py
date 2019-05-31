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
"""Utilities for manipulating the loss collections."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["losses.add_loss"])
def add_loss(loss, loss_collection=ops.GraphKeys.LOSSES):
  """Adds a externally defined loss to the collection of losses.

  Args:
    loss: A loss `Tensor`.
    loss_collection: Optional collection to add the loss to.
  """
  # Since we have no way of figuring out when a training iteration starts or
  # ends, holding on to a loss when executing eagerly is indistingishable from
  # leaking memory. We instead leave the collection empty.
  if loss_collection and not context.executing_eagerly():
    ops.add_to_collection(loss_collection, loss)


@tf_export(v1=["losses.get_losses"])
def get_losses(scope=None, loss_collection=ops.GraphKeys.LOSSES):
  """Gets the list of losses from the loss_collection.

  Args:
    scope: An optional scope name for filtering the losses to return.
    loss_collection: Optional losses collection.

  Returns:
    a list of loss tensors.
  """
  return ops.get_collection(loss_collection, scope)


@tf_export(v1=["losses.get_regularization_losses"])
def get_regularization_losses(scope=None):
  """Gets the list of regularization losses.

  Args:
    scope: An optional scope name for filtering the losses to return.

  Returns:
    A list of regularization losses as Tensors.
  """
  return ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES, scope)


@tf_export(v1=["losses.get_regularization_loss"])
def get_regularization_loss(scope=None, name="total_regularization_loss"):
  """Gets the total regularization loss.

  Args:
    scope: An optional scope name for filtering the losses to return.
    name: The name of the returned tensor.

  Returns:
    A scalar regularization loss.
  """
  losses = get_regularization_losses(scope)
  if losses:
    return math_ops.add_n(losses, name=name)
  else:
    return constant_op.constant(0.0)


@tf_export(v1=["losses.get_total_loss"])
def get_total_loss(add_regularization_losses=True, name="total_loss"):
  """Returns a tensor whose value represents the total loss.

  In particular, this adds any losses you have added with `tf.add_loss()` to
  any regularization losses that have been added by regularization parameters
  on layers constructors e.g. `tf.layers`. Be very sure to use this if you
  are constructing a loss_op manually. Otherwise regularization arguments
  on `tf.layers` methods will not function.

  Args:
    add_regularization_losses: A boolean indicating whether or not to use the
      regularization losses in the sum.
    name: The name of the returned tensor.

  Returns:
    A `Tensor` whose value represents the total loss.

  Raises:
    ValueError: if `losses` is not iterable.
  """
  losses = get_losses()
  if add_regularization_losses:
    losses += get_regularization_losses()
  return math_ops.add_n(losses, name=name)
