# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Losses that are useful for training GANs.

The losses belong to two main groups, but there are others that do not:
1) xxxxx_generator_loss
2) xxxxx_discriminator_loss

Example:
1) wasserstein_generator_loss
2) wasserstein_discriminator_loss

Other example:
wasserstein_gradient_penalty

All losses must be able to accept 1D or 2D Tensors, so as to be compatible with
patchGAN style losses (https://arxiv.org/abs/1611.07004).

To make these losses usable in the TFGAN framework, please create a tuple
version of the losses with `losses_utils.py`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.framework.python.ops import variables as contrib_variables_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.distributions import distribution as ds
from tensorflow.python.ops.losses import losses
from tensorflow.python.ops.losses import util
from tensorflow.python.summary import summary


__all__ = [
    'acgan_discriminator_loss',
    'acgan_generator_loss',
    'least_squares_discriminator_loss',
    'least_squares_generator_loss',
    'modified_discriminator_loss',
    'modified_generator_loss',
    'minimax_discriminator_loss',
    'minimax_generator_loss',
    'wasserstein_discriminator_loss',
    'wasserstein_generator_loss',
    'wasserstein_gradient_penalty',
    'mutual_information_penalty',
    'combine_adversarial_loss',
    'cycle_consistency_loss',
]


# Wasserstein losses from `Wasserstein GAN` (https://arxiv.org/abs/1701.07875).
def wasserstein_generator_loss(
    discriminator_gen_outputs,
    weights=1.0,
    scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
  """Wasserstein generator loss for GANs.

  See `Wasserstein GAN` (https://arxiv.org/abs/1701.07875) for more details.

  Args:
    discriminator_gen_outputs: Discriminator output on generated data. Expected
      to be in the range of (-inf, inf).
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `discriminator_gen_outputs`, and must be broadcastable to
      `discriminator_gen_outputs` (i.e., all dimensions must be either `1`, or
      the same as the corresponding dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: A `tf.losses.Reduction` to apply to loss.
    add_summaries: Whether or not to add detailed summaries for the loss.

  Returns:
    A loss Tensor. The shape depends on `reduction`.
  """
  with ops.name_scope(scope, 'generator_wasserstein_loss', (
      discriminator_gen_outputs, weights)) as scope:
    discriminator_gen_outputs = math_ops.to_float(discriminator_gen_outputs)

    loss = - discriminator_gen_outputs
    loss = losses.compute_weighted_loss(
        loss, weights, scope, loss_collection, reduction)

    if add_summaries:
      summary.scalar('generator_wass_loss', loss)

  return loss


def wasserstein_discriminator_loss(
    discriminator_real_outputs,
    discriminator_gen_outputs,
    real_weights=1.0,
    generated_weights=1.0,
    scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
  """Wasserstein discriminator loss for GANs.

  See `Wasserstein GAN` (https://arxiv.org/abs/1701.07875) for more details.

  Args:
    discriminator_real_outputs: Discriminator output on real data.
    discriminator_gen_outputs: Discriminator output on generated data. Expected
      to be in the range of (-inf, inf).
    real_weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `discriminator_real_outputs`, and must be broadcastable to
      `discriminator_real_outputs` (i.e., all dimensions must be either `1`, or
      the same as the corresponding dimension).
    generated_weights: Same as `real_weights`, but for
      `discriminator_gen_outputs`.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: A `tf.losses.Reduction` to apply to loss.
    add_summaries: Whether or not to add summaries for the loss.

  Returns:
    A loss Tensor. The shape depends on `reduction`.
  """
  with ops.name_scope(scope, 'discriminator_wasserstein_loss', (
      discriminator_real_outputs, discriminator_gen_outputs, real_weights,
      generated_weights)) as scope:
    discriminator_real_outputs = math_ops.to_float(discriminator_real_outputs)
    discriminator_gen_outputs = math_ops.to_float(discriminator_gen_outputs)
    discriminator_real_outputs.shape.assert_is_compatible_with(
        discriminator_gen_outputs.shape)

    loss_on_generated = losses.compute_weighted_loss(
        discriminator_gen_outputs, generated_weights, scope,
        loss_collection=None, reduction=reduction)
    loss_on_real = losses.compute_weighted_loss(
        discriminator_real_outputs, real_weights, scope, loss_collection=None,
        reduction=reduction)
    loss = loss_on_generated - loss_on_real
    util.add_loss(loss, loss_collection)

    if add_summaries:
      summary.scalar('discriminator_gen_wass_loss', loss_on_generated)
      summary.scalar('discriminator_real_wass_loss', loss_on_real)
      summary.scalar('discriminator_wass_loss', loss)

  return loss


# ACGAN losses from `Conditional Image Synthesis With Auxiliary Classifier GANs`
# (https://arxiv.org/abs/1610.09585).
def acgan_discriminator_loss(
    discriminator_real_classification_logits,
    discriminator_gen_classification_logits,
    one_hot_labels,
    label_smoothing=0.0,
    real_weights=1.0,
    generated_weights=1.0,
    scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
  """ACGAN loss for the discriminator.

  The ACGAN loss adds a classification loss to the conditional discriminator.
  Therefore, the discriminator must output a tuple consisting of
    (1) the real/fake prediction and
    (2) the logits for the classification (usually the last conv layer,
        flattened).

  For more details:
    ACGAN: https://arxiv.org/abs/1610.09585

  Args:
    discriminator_real_classification_logits: Classification logits for real
      data.
    discriminator_gen_classification_logits: Classification logits for generated
      data.
    one_hot_labels: A Tensor holding one-hot labels for the batch.
    label_smoothing: A float in [0, 1]. If greater than 0, smooth the labels for
      "discriminator on real data" as suggested in
      https://arxiv.org/pdf/1701.00160
    real_weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `discriminator_real_outputs`, and must be broadcastable to
      `discriminator_real_outputs` (i.e., all dimensions must be either `1`, or
      the same as the corresponding dimension).
    generated_weights: Same as `real_weights`, but for
      `discriminator_gen_classification_logits`.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: A `tf.losses.Reduction` to apply to loss.
    add_summaries: Whether or not to add summaries for the loss.

  Returns:
    A loss Tensor. Shape depends on `reduction`.

  Raises:
    TypeError: If the discriminator does not output a tuple.
  """
  with ops.name_scope(
      scope, 'acgan_discriminator_loss',
      (discriminator_real_classification_logits,
       discriminator_gen_classification_logits, one_hot_labels)) as scope:
    loss_on_generated = losses.softmax_cross_entropy(
        one_hot_labels, discriminator_gen_classification_logits,
        weights=generated_weights, scope=scope, loss_collection=None,
        reduction=reduction)
    loss_on_real = losses.softmax_cross_entropy(
        one_hot_labels, discriminator_real_classification_logits,
        weights=real_weights, label_smoothing=label_smoothing, scope=scope,
        loss_collection=None, reduction=reduction)
    loss = loss_on_generated + loss_on_real
    util.add_loss(loss, loss_collection)

    if add_summaries:
      summary.scalar('discriminator_gen_ac_loss', loss_on_generated)
      summary.scalar('discriminator_real_ac_loss', loss_on_real)
      summary.scalar('discriminator_ac_loss', loss)

  return loss


def acgan_generator_loss(
    discriminator_gen_classification_logits,
    one_hot_labels,
    weights=1.0,
    scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
  """ACGAN loss for the generator.

  The ACGAN loss adds a classification loss to the conditional discriminator.
  Therefore, the discriminator must output a tuple consisting of
    (1) the real/fake prediction and
    (2) the logits for the classification (usually the last conv layer,
        flattened).

  For more details:
    ACGAN: https://arxiv.org/abs/1610.09585

  Args:
    discriminator_gen_classification_logits: Classification logits for generated
      data.
    one_hot_labels: A Tensor holding one-hot labels for the batch.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `discriminator_gen_classification_logits`, and must be broadcastable to
      `discriminator_gen_classification_logits` (i.e., all dimensions must be
      either `1`, or the same as the corresponding dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: A `tf.losses.Reduction` to apply to loss.
    add_summaries: Whether or not to add summaries for the loss.

  Returns:
    A loss Tensor. Shape depends on `reduction`.

  Raises:
    ValueError: if arg module not either `generator` or `discriminator`
    TypeError: if the discriminator does not output a tuple.
  """
  with ops.name_scope(
      scope, 'acgan_generator_loss',
      (discriminator_gen_classification_logits, one_hot_labels)) as scope:
    loss = losses.softmax_cross_entropy(
        one_hot_labels, discriminator_gen_classification_logits,
        weights=weights, scope=scope, loss_collection=loss_collection,
        reduction=reduction)

    if add_summaries:
      summary.scalar('generator_ac_loss', loss)

  return loss


# Wasserstein Gradient Penalty losses from `Improved Training of Wasserstein
# GANs` (https://arxiv.org/abs/1704.00028).


def wasserstein_gradient_penalty(
    real_data,
    generated_data,
    generator_inputs,
    discriminator_fn,
    discriminator_scope,
    epsilon=1e-10,
    target=1.0,
    one_sided=False,
    weights=1.0,
    scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
  """The gradient penalty for the Wasserstein discriminator loss.

  See `Improved Training of Wasserstein GANs`
  (https://arxiv.org/abs/1704.00028) for more details.

  Args:
    real_data: Real data.
    generated_data: Output of the generator.
    generator_inputs: Exact argument to pass to the generator, which is used
      as optional conditioning to the discriminator.
    discriminator_fn: A discriminator function that conforms to TFGAN API.
    discriminator_scope: If not `None`, reuse discriminators from this scope.
    epsilon: A small positive number added for numerical stability when
      computing the gradient norm.
    target: Optional Python number or `Tensor` indicating the target value of
      gradient norm. Defaults to 1.0.
    one_sided: If `True`, penalty proposed in https://arxiv.org/abs/1709.08894
      is used. Defaults to `False`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `real_data` and `generated_data`, and must be broadcastable to
      them (i.e., all dimensions must be either `1`, or the same as the
      corresponding dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: A `tf.losses.Reduction` to apply to loss.
    add_summaries: Whether or not to add summaries for the loss.

  Returns:
    A loss Tensor. The shape depends on `reduction`.

  Raises:
    ValueError: If the rank of data Tensors is unknown.
  """
  with ops.name_scope(scope, 'wasserstein_gradient_penalty',
                      (real_data, generated_data)) as scope:
    real_data = ops.convert_to_tensor(real_data)
    generated_data = ops.convert_to_tensor(generated_data)
    if real_data.shape.ndims is None:
      raise ValueError('`real_data` can\'t have unknown rank.')
    if generated_data.shape.ndims is None:
      raise ValueError('`generated_data` can\'t have unknown rank.')

    differences = generated_data - real_data
    batch_size = differences.shape.dims[0].value or array_ops.shape(
        differences)[0]
    alpha_shape = [batch_size] + [1] * (differences.shape.ndims - 1)
    alpha = random_ops.random_uniform(shape=alpha_shape)
    interpolates = real_data + (alpha * differences)

    with ops.name_scope(None):  # Clear scope so update ops are added properly.
      # Reuse variables if variables already exists.
      with variable_scope.variable_scope(discriminator_scope, 'gpenalty_dscope',
                                         reuse=variable_scope.AUTO_REUSE):
        disc_interpolates = discriminator_fn(interpolates, generator_inputs)

    if isinstance(disc_interpolates, tuple):
      # ACGAN case: disc outputs more than one tensor
      disc_interpolates = disc_interpolates[0]

    gradients = gradients_impl.gradients(disc_interpolates, interpolates)[0]
    gradient_squares = math_ops.reduce_sum(
        math_ops.square(gradients), axis=list(range(1, gradients.shape.ndims)))
    # Propagate shape information, if possible.
    if isinstance(batch_size, int):
      gradient_squares.set_shape([
          batch_size] + gradient_squares.shape.as_list()[1:])
    # For numerical stability, add epsilon to the sum before taking the square
    # root. Note tf.norm does not add epsilon.
    slopes = math_ops.sqrt(gradient_squares + epsilon)
    penalties = slopes / target - 1.0
    if one_sided:
      penalties = math_ops.maximum(0., penalties)
    penalties_squared = math_ops.square(penalties)
    penalty = losses.compute_weighted_loss(
        penalties_squared, weights, scope=scope,
        loss_collection=loss_collection, reduction=reduction)

    if add_summaries:
      summary.scalar('gradient_penalty_loss', penalty)

    return penalty


# Original losses from `Generative Adversarial Nets`
# (https://arxiv.org/abs/1406.2661).


def minimax_discriminator_loss(
    discriminator_real_outputs,
    discriminator_gen_outputs,
    label_smoothing=0.25,
    real_weights=1.0,
    generated_weights=1.0,
    scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
  """Original minimax discriminator loss for GANs, with label smoothing.

  Note that the authors don't recommend using this loss. A more practically
  useful loss is `modified_discriminator_loss`.

  L = - real_weights * log(sigmoid(D(x)))
      - generated_weights * log(1 - sigmoid(D(G(z))))

  See `Generative Adversarial Nets` (https://arxiv.org/abs/1406.2661) for more
  details.

  Args:
    discriminator_real_outputs: Discriminator output on real data.
    discriminator_gen_outputs: Discriminator output on generated data. Expected
      to be in the range of (-inf, inf).
    label_smoothing: The amount of smoothing for positive labels. This technique
      is taken from `Improved Techniques for Training GANs`
      (https://arxiv.org/abs/1606.03498). `0.0` means no smoothing.
    real_weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `real_data`, and must be broadcastable to `real_data` (i.e., all
      dimensions must be either `1`, or the same as the corresponding
      dimension).
    generated_weights: Same as `real_weights`, but for `generated_data`.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: A `tf.losses.Reduction` to apply to loss.
    add_summaries: Whether or not to add summaries for the loss.

  Returns:
    A loss Tensor. The shape depends on `reduction`.
  """
  with ops.name_scope(scope, 'discriminator_minimax_loss', (
      discriminator_real_outputs, discriminator_gen_outputs, real_weights,
      generated_weights, label_smoothing)) as scope:

    # -log((1 - label_smoothing) - sigmoid(D(x)))
    loss_on_real = losses.sigmoid_cross_entropy(
        array_ops.ones_like(discriminator_real_outputs),
        discriminator_real_outputs, real_weights, label_smoothing, scope,
        loss_collection=None, reduction=reduction)
    # -log(- sigmoid(D(G(x))))
    loss_on_generated = losses.sigmoid_cross_entropy(
        array_ops.zeros_like(discriminator_gen_outputs),
        discriminator_gen_outputs, generated_weights, scope=scope,
        loss_collection=None, reduction=reduction)

    loss = loss_on_real + loss_on_generated
    util.add_loss(loss, loss_collection)

    if add_summaries:
      summary.scalar('discriminator_gen_minimax_loss', loss_on_generated)
      summary.scalar('discriminator_real_minimax_loss', loss_on_real)
      summary.scalar('discriminator_minimax_loss', loss)

  return loss


def minimax_generator_loss(
    discriminator_gen_outputs,
    label_smoothing=0.0,
    weights=1.0,
    scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
  """Original minimax generator loss for GANs.

  Note that the authors don't recommend using this loss. A more practically
  useful loss is `modified_generator_loss`.

  L = log(sigmoid(D(x))) + log(1 - sigmoid(D(G(z))))

  See `Generative Adversarial Nets` (https://arxiv.org/abs/1406.2661) for more
  details.

  Args:
    discriminator_gen_outputs: Discriminator output on generated data. Expected
      to be in the range of (-inf, inf).
    label_smoothing: The amount of smoothing for positive labels. This technique
      is taken from `Improved Techniques for Training GANs`
      (https://arxiv.org/abs/1606.03498). `0.0` means no smoothing.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `discriminator_gen_outputs`, and must be broadcastable to
      `discriminator_gen_outputs` (i.e., all dimensions must be either `1`, or
      the same as the corresponding dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: A `tf.losses.Reduction` to apply to loss.
    add_summaries: Whether or not to add summaries for the loss.

  Returns:
    A loss Tensor. The shape depends on `reduction`.
  """
  with ops.name_scope(scope, 'generator_minimax_loss') as scope:
    loss = - minimax_discriminator_loss(
        array_ops.ones_like(discriminator_gen_outputs),
        discriminator_gen_outputs, label_smoothing, weights, weights, scope,
        loss_collection, reduction, add_summaries=False)

  if add_summaries:
    summary.scalar('generator_minimax_loss', loss)

  return loss


def modified_discriminator_loss(
    discriminator_real_outputs,
    discriminator_gen_outputs,
    label_smoothing=0.25,
    real_weights=1.0,
    generated_weights=1.0,
    scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
  """Same as minimax discriminator loss.

  See `Generative Adversarial Nets` (https://arxiv.org/abs/1406.2661) for more
  details.

  Args:
    discriminator_real_outputs: Discriminator output on real data.
    discriminator_gen_outputs: Discriminator output on generated data. Expected
      to be in the range of (-inf, inf).
    label_smoothing: The amount of smoothing for positive labels. This technique
      is taken from `Improved Techniques for Training GANs`
      (https://arxiv.org/abs/1606.03498). `0.0` means no smoothing.
    real_weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `discriminator_gen_outputs`, and must be broadcastable to
      `discriminator_gen_outputs` (i.e., all dimensions must be either `1`, or
      the same as the corresponding dimension).
    generated_weights: Same as `real_weights`, but for
      `discriminator_gen_outputs`.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: A `tf.losses.Reduction` to apply to loss.
    add_summaries: Whether or not to add summaries for the loss.

  Returns:
    A loss Tensor. The shape depends on `reduction`.
  """
  return minimax_discriminator_loss(
      discriminator_real_outputs,
      discriminator_gen_outputs,
      label_smoothing,
      real_weights,
      generated_weights,
      scope or 'discriminator_modified_loss',
      loss_collection,
      reduction,
      add_summaries)


def modified_generator_loss(
    discriminator_gen_outputs,
    label_smoothing=0.0,
    weights=1.0,
    scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
  """Modified generator loss for GANs.

  L = -log(sigmoid(D(G(z))))

  This is the trick used in the original paper to avoid vanishing gradients
  early in training. See `Generative Adversarial Nets`
  (https://arxiv.org/abs/1406.2661) for more details.

  Args:
    discriminator_gen_outputs: Discriminator output on generated data. Expected
      to be in the range of (-inf, inf).
    label_smoothing: The amount of smoothing for positive labels. This technique
      is taken from `Improved Techniques for Training GANs`
      (https://arxiv.org/abs/1606.03498). `0.0` means no smoothing.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `discriminator_gen_outputs`, and must be broadcastable to `labels` (i.e.,
      all dimensions must be either `1`, or the same as the corresponding
      dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: A `tf.losses.Reduction` to apply to loss.
    add_summaries: Whether or not to add summaries for the loss.

  Returns:
    A loss Tensor. The shape depends on `reduction`.
  """
  with ops.name_scope(scope, 'generator_modified_loss',
                      [discriminator_gen_outputs]) as scope:
    loss = losses.sigmoid_cross_entropy(
        array_ops.ones_like(discriminator_gen_outputs),
        discriminator_gen_outputs, weights, label_smoothing, scope,
        loss_collection, reduction)

    if add_summaries:
      summary.scalar('generator_modified_loss', loss)

  return loss


# Least Squares loss from `Least Squares Generative Adversarial Networks`
# (https://arxiv.org/abs/1611.04076).


def least_squares_generator_loss(
    discriminator_gen_outputs,
    real_label=1,
    weights=1.0,
    scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
  """Least squares generator loss.

  This loss comes from `Least Squares Generative Adversarial Networks`
  (https://arxiv.org/abs/1611.04076).

  L = 1/2 * (D(G(z)) - `real_label`) ** 2

  where D(y) are discriminator logits.

  Args:
    discriminator_gen_outputs: Discriminator output on generated data. Expected
      to be in the range of (-inf, inf).
    real_label: The value that the generator is trying to get the discriminator
      to output on generated data.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `discriminator_gen_outputs`, and must be broadcastable to
      `discriminator_gen_outputs` (i.e., all dimensions must be either `1`, or
      the same as the corresponding dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: A `tf.losses.Reduction` to apply to loss.
    add_summaries: Whether or not to add summaries for the loss.

  Returns:
    A loss Tensor. The shape depends on `reduction`.
  """
  with ops.name_scope(scope, 'lsq_generator_loss',
                      (discriminator_gen_outputs, real_label)) as scope:
    discriminator_gen_outputs = math_ops.to_float(discriminator_gen_outputs)
    loss = math_ops.squared_difference(
        discriminator_gen_outputs, real_label) / 2.0
    loss = losses.compute_weighted_loss(
        loss, weights, scope, loss_collection, reduction)

  if add_summaries:
    summary.scalar('generator_lsq_loss', loss)

  return loss


def least_squares_discriminator_loss(
    discriminator_real_outputs,
    discriminator_gen_outputs,
    real_label=1,
    fake_label=0,
    real_weights=1.0,
    generated_weights=1.0,
    scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
  """Least squares discriminator loss.

  This loss comes from `Least Squares Generative Adversarial Networks`
  (https://arxiv.org/abs/1611.04076).

  L = 1/2 * (D(x) - `real`) ** 2 +
      1/2 * (D(G(z)) - `fake_label`) ** 2

  where D(y) are discriminator logits.

  Args:
    discriminator_real_outputs: Discriminator output on real data.
    discriminator_gen_outputs: Discriminator output on generated data. Expected
      to be in the range of (-inf, inf).
    real_label: The value that the discriminator tries to output for real data.
    fake_label: The value that the discriminator tries to output for fake data.
    real_weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `discriminator_real_outputs`, and must be broadcastable to
      `discriminator_real_outputs` (i.e., all dimensions must be either `1`, or
      the same as the corresponding dimension).
    generated_weights: Same as `real_weights`, but for
      `discriminator_gen_outputs`.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: A `tf.losses.Reduction` to apply to loss.
    add_summaries: Whether or not to add summaries for the loss.

  Returns:
    A loss Tensor. The shape depends on `reduction`.
  """
  with ops.name_scope(scope, 'lsq_discriminator_loss',
                      (discriminator_gen_outputs, real_label)) as scope:
    discriminator_real_outputs = math_ops.to_float(discriminator_real_outputs)
    discriminator_gen_outputs = math_ops.to_float(discriminator_gen_outputs)
    discriminator_real_outputs.shape.assert_is_compatible_with(
        discriminator_gen_outputs.shape)

    real_losses = math_ops.squared_difference(
        discriminator_real_outputs, real_label) / 2.0
    fake_losses = math_ops.squared_difference(
        discriminator_gen_outputs, fake_label) / 2.0

    loss_on_real = losses.compute_weighted_loss(
        real_losses, real_weights, scope, loss_collection=None,
        reduction=reduction)
    loss_on_generated = losses.compute_weighted_loss(
        fake_losses, generated_weights, scope, loss_collection=None,
        reduction=reduction)

    loss = loss_on_real + loss_on_generated
    util.add_loss(loss, loss_collection)

  if add_summaries:
    summary.scalar('discriminator_gen_lsq_loss', loss_on_generated)
    summary.scalar('discriminator_real_lsq_loss', loss_on_real)
    summary.scalar('discriminator_lsq_loss', loss)

  return loss


# InfoGAN loss from `InfoGAN: Interpretable Representation Learning by
# `Information Maximizing Generative Adversarial Nets`
# https://arxiv.org/abs/1606.03657


def _validate_distributions(distributions):
  if not isinstance(distributions, (list, tuple)):
    raise ValueError('`distributions` must be a list or tuple. Instead, '
                     'found %s.', type(distributions))
  for x in distributions:
    if not isinstance(x, ds.Distribution):
      raise ValueError('`distributions` must be a list of `Distributions`. '
                       'Instead, found %s.', type(x))


def _validate_information_penalty_inputs(
    structured_generator_inputs, predicted_distributions):
  """Validate input to `mutual_information_penalty`."""
  _validate_distributions(predicted_distributions)
  if len(structured_generator_inputs) != len(predicted_distributions):
    raise ValueError('`structured_generator_inputs` length %i must be the same '
                     'as `predicted_distributions` length %i.' % (
                         len(structured_generator_inputs),
                         len(predicted_distributions)))


def mutual_information_penalty(
    structured_generator_inputs,
    predicted_distributions,
    weights=1.0,
    scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
  """Returns a penalty on the mutual information in an InfoGAN model.

  This loss comes from an InfoGAN paper https://arxiv.org/abs/1606.03657.

  Args:
    structured_generator_inputs: A list of Tensors representing the random noise
      that must  have high mutual information with the generator output. List
      length should match `predicted_distributions`.
    predicted_distributions: A list of `tfp.distributions.Distribution`s.
      Predicted by the recognizer, and used to evaluate the likelihood of the
      structured noise. List length should match `structured_generator_inputs`.
    weights: Optional `Tensor` whose rank is either 0, or the same dimensions as
      `structured_generator_inputs`.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: A `tf.losses.Reduction` to apply to loss.
    add_summaries: Whether or not to add summaries for the loss.

  Returns:
    A scalar Tensor representing the mutual information loss.
  """
  _validate_information_penalty_inputs(
      structured_generator_inputs, predicted_distributions)

  with ops.name_scope(scope, 'mutual_information_loss') as scope:
    # Calculate the negative log-likelihood of the reconstructed noise.
    log_probs = [math_ops.reduce_mean(dist.log_prob(noise)) for dist, noise in
                 zip(predicted_distributions, structured_generator_inputs)]
    loss = -1 * losses.compute_weighted_loss(
        log_probs, weights, scope, loss_collection=loss_collection,
        reduction=reduction)

    if add_summaries:
      summary.scalar('mutual_information_penalty', loss)

  return loss


def _numerically_stable_global_norm(tensor_list):
  """Compute the global norm of a list of Tensors, with improved stability.

  The global norm computation sometimes overflows due to the intermediate L2
  step. To avoid this, we divide by a cheap-to-compute max over the
  matrix elements.

  Args:
    tensor_list: A list of tensors, or `None`.

  Returns:
    A scalar tensor with the global norm.
  """
  if np.all([x is None for x in tensor_list]):
    return 0.0

  list_max = math_ops.reduce_max([math_ops.reduce_max(math_ops.abs(x)) for x in
                                  tensor_list if x is not None])
  return list_max * clip_ops.global_norm([x / list_max for x in tensor_list
                                          if x is not None])


def _used_weight(weights_list):
  for weight in weights_list:
    if weight is not None:
      return tensor_util.constant_value(ops.convert_to_tensor(weight))


def _validate_args(losses_list, weight_factor, gradient_ratio):
  for loss in losses_list:
    loss.shape.assert_is_compatible_with([])
  if weight_factor is None and gradient_ratio is None:
    raise ValueError(
        '`weight_factor` and `gradient_ratio` cannot both be `None.`')
  if weight_factor is not None and gradient_ratio is not None:
    raise ValueError(
        '`weight_factor` and `gradient_ratio` cannot both be specified.')


# TODO(joelshor): Add ability to pass in gradients, to avoid recomputing.
def combine_adversarial_loss(main_loss,
                             adversarial_loss,
                             weight_factor=None,
                             gradient_ratio=None,
                             gradient_ratio_epsilon=1e-6,
                             variables=None,
                             scalar_summaries=True,
                             gradient_summaries=True,
                             scope=None):
  """Utility to combine main and adversarial losses.

  This utility combines the main and adversarial losses in one of two ways.
  1) Fixed coefficient on adversarial loss. Use `weight_factor` in this case.
  2) Fixed ratio of gradients. Use `gradient_ratio` in this case. This is often
    used to make sure both losses affect weights roughly equally, as in
    https://arxiv.org/pdf/1705.05823.

  One can optionally also visualize the scalar and gradient behavior of the
  losses.

  Args:
    main_loss: A floating scalar Tensor indicating the main loss.
    adversarial_loss: A floating scalar Tensor indication the adversarial loss.
    weight_factor: If not `None`, the coefficient by which to multiply the
      adversarial loss. Exactly one of this and `gradient_ratio` must be
      non-None.
    gradient_ratio: If not `None`, the ratio of the magnitude of the gradients.
      Specifically,
        gradient_ratio = grad_mag(main_loss) / grad_mag(adversarial_loss)
      Exactly one of this and `weight_factor` must be non-None.
    gradient_ratio_epsilon: An epsilon to add to the adversarial loss
      coefficient denominator, to avoid division-by-zero.
    variables: List of variables to calculate gradients with respect to. If not
      present, defaults to all trainable variables.
    scalar_summaries: Create scalar summaries of losses.
    gradient_summaries: Create gradient summaries of losses.
    scope: Optional name scope.

  Returns:
    A floating scalar Tensor indicating the desired combined loss.

  Raises:
    ValueError: Malformed input.
  """
  _validate_args([main_loss, adversarial_loss], weight_factor, gradient_ratio)
  if variables is None:
    variables = contrib_variables_lib.get_trainable_variables()

  with ops.name_scope(scope, 'adversarial_loss',
                      values=[main_loss, adversarial_loss]):
    # Compute gradients if we will need them.
    if gradient_summaries or gradient_ratio is not None:
      main_loss_grad_mag = _numerically_stable_global_norm(
          gradients_impl.gradients(main_loss, variables))
      adv_loss_grad_mag = _numerically_stable_global_norm(
          gradients_impl.gradients(adversarial_loss, variables))

    # Add summaries, if applicable.
    if scalar_summaries:
      summary.scalar('main_loss', main_loss)
      summary.scalar('adversarial_loss', adversarial_loss)
    if gradient_summaries:
      summary.scalar('main_loss_gradients', main_loss_grad_mag)
      summary.scalar('adversarial_loss_gradients', adv_loss_grad_mag)

    # Combine losses in the appropriate way.
    # If `weight_factor` is always `0`, avoid computing the adversarial loss
    # tensor entirely.
    if _used_weight((weight_factor, gradient_ratio)) == 0:
      final_loss = main_loss
    elif weight_factor is not None:
      final_loss = (main_loss +
                    array_ops.stop_gradient(weight_factor) * adversarial_loss)
    elif gradient_ratio is not None:
      grad_mag_ratio = main_loss_grad_mag / (
          adv_loss_grad_mag + gradient_ratio_epsilon)
      adv_coeff = grad_mag_ratio / gradient_ratio
      summary.scalar('adversarial_coefficient', adv_coeff)
      final_loss = (main_loss +
                    array_ops.stop_gradient(adv_coeff) * adversarial_loss)

  return final_loss


def cycle_consistency_loss(data_x,
                           reconstructed_data_x,
                           data_y,
                           reconstructed_data_y,
                           scope=None,
                           add_summaries=False):
  """Defines the cycle consistency loss.

  The cyclegan model has two partial models where `model_x2y` generator F maps
  data set X to Y, `model_y2x` generator G maps data set Y to X. For a `data_x`
  in data set X, we could reconstruct it by
  * reconstructed_data_x = G(F(data_x))
  Similarly
  * reconstructed_data_y = F(G(data_y))

  The cycle consistency loss is about the difference between data and
  reconstructed data, namely
  * loss_x2x = |data_x - G(F(data_x))| (L1-norm)
  * loss_y2y = |data_y - F(G(data_y))| (L1-norm)
  * loss = (loss_x2x + loss_y2y) / 2
  where `loss` is the final result.

  For the L1-norm, we follow the original implementation:
  https://github.com/junyanz/CycleGAN/blob/master/models/cycle_gan_model.lua
  we use L1-norm of pixel-wise error normalized by data size such that
  `cycle_loss_weight` can be specified independent of image size.

  See https://arxiv.org/abs/1703.10593 for more details.

  Args:
    data_x: A `Tensor` of data X.
    reconstructed_data_x: A `Tensor` of reconstructed data X.
    data_y: A `Tensor` of data Y.
    reconstructed_data_y: A `Tensor` of reconstructed data Y.
    scope: The scope for the operations performed in computing the loss.
      Defaults to None.
    add_summaries: Whether or not to add detailed summaries for the loss.
      Defaults to False.

  Returns:
    A scalar `Tensor` of cycle consistency loss.
  """

  with ops.name_scope(
      scope,
      'cycle_consistency_loss',
      values=[data_x, reconstructed_data_x, data_y, reconstructed_data_y]):
    loss_x2x = losses.absolute_difference(data_x, reconstructed_data_x)
    loss_y2y = losses.absolute_difference(data_y, reconstructed_data_y)
    loss = (loss_x2x + loss_y2y) / 2.0
    if add_summaries:
      summary.scalar('cycle_consistency_loss_x2x', loss_x2x)
      summary.scalar('cycle_consistency_loss_y2y', loss_y2y)
      summary.scalar('cycle_consistency_loss', loss)

  return loss
