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
"""A TFGAN-backed GAN Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum

from tensorflow.contrib.framework.python.ops import variables as variable_lib
from tensorflow.contrib.gan.python import namedtuples as tfgan_tuples
from tensorflow.contrib.gan.python import train as tfgan_train
from tensorflow.contrib.gan.python.estimator.python import head as head_lib
from tensorflow.contrib.gan.python.eval.python import summaries as tfgan_summaries
from tensorflow.python.estimator import estimator
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope


__all__ = [
    'GANEstimator',
    'SummaryType'
]


class SummaryType(enum.IntEnum):
  NONE = 0
  VARIABLES = 1
  IMAGES = 2
  IMAGE_COMPARISON = 3


_summary_type_map = {
    SummaryType.VARIABLES: tfgan_summaries.add_gan_model_summaries,
    SummaryType.IMAGES: tfgan_summaries.add_gan_model_image_summaries,
    SummaryType.IMAGE_COMPARISON: tfgan_summaries.add_image_comparison_summaries,  # pylint:disable=line-too-long
}


# TODO(joelshor): For now, this only supports 1:1 generator:discriminator
# training sequentially. Find a nice way to expose options to the user without
# exposing internals.
class GANEstimator(estimator.Estimator):
  """An estimator for Generative Adversarial Networks (GANs).

  This Estimator is backed by TFGAN.

  Example:

  ```python
      import tensorflow as tf
      tfgan = tf.contrib.gan

      # See TFGAN's `train.py` for a description of the generator and
      # discriminator API.
      def generator_fn(generator_inputs):
        ...
        return generated_data

      def discriminator_fn(data, conditioning):
        ...
        return logits

      # Create GAN estimator.
      gan_estimator = estimator.GANEstimator(
          model_dir,
          generator_fn=generator_fn,
          discriminator_fn=discriminator_fn,
          generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
          discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
          generator_optimizer=tf.train.AdamOptimizier(0.1, 0.5),
          discriminator_optimizer=tf.train.AdamOptimizier(0.1, 0.5))

      # Train estimator.
      gan_estimator.train(train_input_fn, steps)

      # Evaluate resulting estimator.
      gan_estimator.evaluate(eval_input_fn)

      # Generate samples from generator.
      predictions = np.array([
          x for x in gan_estimator.predict(predict_input_fn)])
    ```
  """

  def __init__(self,
               model_dir=None,
               generator_fn=None,
               discriminator_fn=None,
               generator_loss_fn=None,
               discriminator_loss_fn=None,
               generator_optimizer=None,
               discriminator_optimizer=None,
               add_summaries=None,
               use_loss_summaries=True,
               config=None):
    """Initializes a GANEstimator instance.

    Args:
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator
        to continue training a previously saved model.
      generator_fn: A python function that takes a Tensor, Tensor list, or
        Tensor dictionary as inputs and returns the outputs of the GAN
        generator. See `TFGAN` for more details and examples.
      discriminator_fn: A python function that takes the output of
        `generator_fn` or real data in the GAN setup, and `generator_inputs`.
        Outputs a Tensor in the range [-inf, inf]. See `TFGAN` for more details
        and examples.
      generator_loss_fn: The loss function on the generator. Takes a `GANModel`
        tuple.
      discriminator_loss_fn: The loss function on the discriminator. Takes a
        `GANModel` tuple.
      generator_optimizer: The optimizer for generator updates, or a function
        that takes no arguments and returns an optimizer. This function will
        be called when the default graph is the `GANEstimator`'s graph, so
        utilities like `tf.contrib.framework.get_or_create_global_step` will
        work.
      discriminator_optimizer: Same as `generator_optimizer`, but for the
        discriminator updates.
      add_summaries: `None`, a single `SummaryType`, or a list of `SummaryType`.
      use_loss_summaries: If `True`, add loss summaries. If `False`, does not.
        If `None`, uses defaults.
      config: `RunConfig` object to configure the runtime settings.
    """
    # TODO(joelshor): Explicitly validate inputs.

    def _model_fn(features, labels, mode):
      gopt = (generator_optimizer() if callable(generator_optimizer) else
              generator_optimizer)
      dopt = (discriminator_optimizer() if callable(discriminator_optimizer)
              else discriminator_optimizer)
      gan_head = head_lib.gan_head(
          generator_loss_fn, discriminator_loss_fn, gopt, dopt,
          use_loss_summaries)
      return _gan_model_fn(
          features, labels, mode, generator_fn, discriminator_fn, gan_head,
          add_summaries)

    super(GANEstimator, self).__init__(
        model_fn=_model_fn, model_dir=model_dir, config=config)


def _use_check_shapes(real_data):
  """Determines whether TFGAN should check Tensor shapes."""
  return isinstance(real_data, ops.Tensor)


def _gan_model_fn(
    features,
    labels,
    mode,
    generator_fn,
    discriminator_fn,
    head,
    add_summaries=None,
    generator_scope_name='Generator'):
  """The `model_fn` for the GAN estimator.

  We make the following convention:
    features -> TFGAN's `generator_inputs`
    labels -> TFGAN's `real_data`

  Args:
    features: A dictionary to feed to generator. In the unconditional case,
      this might be just `noise`. In the conditional GAN case, this
      might be the generator's conditioning. The `generator_fn` determines
      what the required keys are.
    labels: Real data. Can be any structure, as long as `discriminator_fn`
      can accept it for the first argument.
    mode: Defines whether this is training, evaluation or prediction.
      See `ModeKeys`.
    generator_fn: A python lambda that takes `generator_inputs` as inputs and
      returns the outputs of the GAN generator.
    discriminator_fn: A python lambda that takes `real_data`/`generated data`
      and `generator_inputs`. Outputs a Tensor in the range [-inf, inf].
    head: A `Head` instance suitable for GANs.
    add_summaries: `None`, a single `SummaryType`, or a list of `SummaryType`.
    generator_scope_name: The name of the generator scope. We need this to be
      the same for GANModels produced by TFGAN's `train.gan_model` and the
      manually constructed ones for predictions.

  Returns:
    `ModelFnOps`

  Raises:
    ValueError: If `labels` isn't `None` during prediction.
  """
  real_data = labels
  generator_inputs = features

  if mode == model_fn_lib.ModeKeys.TRAIN:
    gan_model = _make_train_gan_model(
        generator_fn, discriminator_fn, real_data, generator_inputs,
        generator_scope_name, add_summaries)
  elif mode == model_fn_lib.ModeKeys.EVAL:
    gan_model = _make_eval_gan_model(
        generator_fn, discriminator_fn, real_data, generator_inputs,
        generator_scope_name, add_summaries)
  else:
    if real_data is not None:
      raise ValueError('`labels` must be `None` when mode is `predict`. '
                       'Instead, found %s' % real_data)
    gan_model = _make_prediction_gan_model(
        generator_inputs, generator_fn, generator_scope_name)

  return head.create_estimator_spec(
      features=None,
      mode=mode,
      logits=gan_model,
      labels=None)


def _make_train_gan_model(generator_fn, discriminator_fn, real_data,
                          generator_inputs, generator_scope, add_summaries):
  """Make a `GANModel` for training."""
  gan_model = tfgan_train.gan_model(
      generator_fn,
      discriminator_fn,
      real_data,
      generator_inputs,
      generator_scope=generator_scope,
      check_shapes=_use_check_shapes(real_data))
  if add_summaries:
    if not isinstance(add_summaries, (tuple, list)):
      add_summaries = [add_summaries]
    with ops.name_scope(None):
      for summary_type in add_summaries:
        _summary_type_map[summary_type](gan_model)

  return gan_model


def _make_eval_gan_model(generator_fn, discriminator_fn, real_data,
                         generator_inputs, generator_scope, add_summaries):
  """Make a `GANModel` for evaluation."""
  return _make_train_gan_model(generator_fn, discriminator_fn, real_data,
                               generator_inputs, generator_scope, add_summaries)


def _make_prediction_gan_model(generator_inputs, generator_fn, generator_scope):
  """Make a `GANModel` from just the generator."""
  with variable_scope.variable_scope(generator_scope) as gen_scope:
    generator_inputs = tfgan_train._convert_tensor_or_l_or_d(generator_inputs)  # pylint:disable=protected-access
    generated_data = generator_fn(generator_inputs)
  generator_variables = variable_lib.get_trainable_variables(gen_scope)

  return tfgan_tuples.GANModel(
      generator_inputs,
      generated_data,
      generator_variables,
      gen_scope,
      generator_fn,
      real_data=None,
      discriminator_real_outputs=None,
      discriminator_gen_outputs=None,
      discriminator_variables=None,
      discriminator_scope=None,
      discriminator_fn=None)
