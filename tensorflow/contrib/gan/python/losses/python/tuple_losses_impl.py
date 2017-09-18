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
"""TFGAN utilities for loss functions that accept GANModel namedtuples.

Example:
  ```python
  # `tfgan.losses.args` losses take individual arguments.
  w_loss = tfgan.losses.args.wasserstein_discriminator_loss(
    discriminator_real_outputs,
    discriminator_gen_outputs)

  # `tfgan.losses` losses take GANModel namedtuples.
  w_loss2 = tfgan.losses.wasserstein_discriminator_loss(gan_model)
  ```
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.gan.python.losses.python import losses_impl
from tensorflow.python.util import tf_inspect


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
]


def _args_to_gan_model(loss_fn):
  """Converts a loss taking individual args to one taking a GANModel namedtuple.

  The new function has the same name as the original one.

  Args:
    loss_fn: A python function taking a `GANModel` object and returning a loss
      Tensor calculated from that object. The shape of the loss depends on
      `reduction`.

  Returns:
    A new function that takes a GANModel namedtuples and returns the same loss.
  """
  # Match arguments in `loss_fn` to elements of `namedtuple`.
  # TODO(joelshor): Properly handle `varargs` and `keywords`.
  argspec = tf_inspect.getargspec(loss_fn)
  defaults = argspec.defaults or []

  required_args = set(argspec.args[:-len(defaults)])
  args_with_defaults = argspec.args[-len(defaults):]
  default_args_dict = dict(zip(args_with_defaults, defaults))

  def new_loss_fn(gan_model, **kwargs):  # pylint:disable=missing-docstring
    def _asdict(namedtuple):
      """Returns a namedtuple as a dictionary.

      This is required because `_asdict()` in Python 3.x.x is broken in classes
      that inherit from `collections.namedtuple`. See
      https://bugs.python.org/issue24931 for more details.

      Args:
        namedtuple: An object that inherits from `collections.namedtuple`.

      Returns:
        A dictionary version of the tuple.
      """
      return {k: getattr(namedtuple, k) for k in namedtuple._fields}
    gan_model_dict = _asdict(gan_model)

    # Make sure non-tuple required args are supplied.
    args_from_tuple = set(argspec.args).intersection(set(gan_model._fields))
    required_args_not_from_tuple = required_args - args_from_tuple
    for arg in required_args_not_from_tuple:
      if arg not in kwargs:
        raise ValueError('`%s` must be supplied to %s loss function.' % (
            arg, loss_fn.__name__))

    # Make sure tuple args aren't also supplied as keyword args.
    ambiguous_args = set(gan_model._fields).intersection(set(kwargs.keys()))
    if ambiguous_args:
      raise ValueError(
          'The following args are present in both the tuple and keyword args '
          'for %s: %s' % (loss_fn.__name__, ambiguous_args))

    # Add required args to arg dictionary.
    required_args_from_tuple = required_args.intersection(args_from_tuple)
    for arg in required_args_from_tuple:
      assert arg not in kwargs
      kwargs[arg] = gan_model_dict[arg]

    # Add arguments that have defaults.
    for arg in default_args_dict:
      val_from_tuple = gan_model_dict[arg] if arg in gan_model_dict else None
      val_from_kwargs = kwargs[arg] if arg in kwargs else None
      assert not (val_from_tuple is not None and val_from_kwargs is not None)
      kwargs[arg] = (val_from_tuple if val_from_tuple is not None else
                     val_from_kwargs if val_from_kwargs is not None else
                     default_args_dict[arg])

    return loss_fn(**kwargs)

  new_docstring = """The gan_model version of %s.""" % loss_fn.__name__
  new_loss_fn.__docstring__ = new_docstring
  new_loss_fn.__name__ = loss_fn.__name__
  new_loss_fn.__module__ = loss_fn.__module__
  return new_loss_fn


# Wasserstein losses from `Wasserstein GAN` (https://arxiv.org/abs/1701.07875).
wasserstein_generator_loss = _args_to_gan_model(
    losses_impl.wasserstein_generator_loss)
wasserstein_discriminator_loss = _args_to_gan_model(
    losses_impl.wasserstein_discriminator_loss)
wasserstein_gradient_penalty = _args_to_gan_model(
    losses_impl.wasserstein_gradient_penalty)

# ACGAN losses from `Conditional Image Synthesis With Auxiliary Classifier GANs`
# (https://arxiv.org/abs/1610.09585).
acgan_discriminator_loss = _args_to_gan_model(
    losses_impl.acgan_discriminator_loss)
acgan_generator_loss = _args_to_gan_model(
    losses_impl.acgan_generator_loss)


# Original losses from `Generative Adversarial Nets`
# (https://arxiv.org/abs/1406.2661).
minimax_discriminator_loss = _args_to_gan_model(
    losses_impl.minimax_discriminator_loss)
minimax_generator_loss = _args_to_gan_model(
    losses_impl.minimax_generator_loss)
modified_discriminator_loss = _args_to_gan_model(
    losses_impl.modified_discriminator_loss)
modified_generator_loss = _args_to_gan_model(
    losses_impl.modified_generator_loss)


# Least Squares loss from `Least Squares Generative Adversarial Networks`
# (https://arxiv.org/abs/1611.04076).
least_squares_generator_loss = _args_to_gan_model(
    losses_impl.least_squares_generator_loss)
least_squares_discriminator_loss = _args_to_gan_model(
    losses_impl.least_squares_discriminator_loss)


# InfoGAN loss from `InfoGAN: Interpretable Representation Learning by
# `Information Maximizing Generative Adversarial Nets`
# https://arxiv.org/abs/1606.03657
mutual_information_penalty = _args_to_gan_model(
    losses_impl.mutual_information_penalty)


def combine_adversarial_loss(gan_loss,
                             gan_model,
                             non_adversarial_loss,
                             weight_factor=None,
                             gradient_ratio=None,
                             gradient_ratio_epsilon=1e-6,
                             scalar_summaries=True,
                             gradient_summaries=True):
  """Combine adversarial loss and main loss.

  Uses `combine_adversarial_loss` to combine the losses, and returns
  a modified GANLoss namedtuple.

  Args:
    gan_loss: A GANLoss namedtuple. Assume the GANLoss.generator_loss is the
      adversarial loss.
    gan_model: A GANModel namedtuple. Used to access the generator's variables.
    non_adversarial_loss: Same as `main_loss` from
      `combine_adversarial_loss`.
    weight_factor: Same as `weight_factor` from
      `combine_adversarial_loss`.
    gradient_ratio: Same as `gradient_ratio` from
      `combine_adversarial_loss`.
    gradient_ratio_epsilon: Same as `gradient_ratio_epsilon` from
      `combine_adversarial_loss`.
    scalar_summaries: Same as `scalar_summaries` from
      `combine_adversarial_loss`.
    gradient_summaries: Same as `gradient_summaries` from
      `combine_adversarial_loss`.

  Returns:
    A modified GANLoss namedtuple, with `non_adversarial_loss` included
    appropriately.
  """
  combined_loss = losses_impl.combine_adversarial_loss(
      non_adversarial_loss,
      gan_loss.generator_loss,
      weight_factor,
      gradient_ratio,
      gradient_ratio_epsilon,
      gan_model.generator_variables,
      scalar_summaries,
      gradient_summaries)
  return gan_loss._replace(generator_loss=combined_loss)
