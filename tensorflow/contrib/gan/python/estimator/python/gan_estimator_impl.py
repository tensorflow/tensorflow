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

import functools
import enum

from tensorflow.contrib.framework.python.ops import variables as variable_lib
from tensorflow.contrib.gan.python import namedtuples as tfgan_tuples
from tensorflow.contrib.gan.python import train as tfgan_train
from tensorflow.contrib.gan.python.eval.python import summaries as tfgan_summaries
from tensorflow.python.estimator import estimator
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import metrics as metrics_lib
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import tf_inspect as inspect


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


class GANEstimator(estimator.Estimator):
  """An estimator for Generative Adversarial Networks (GANs).

  This Estimator is backed by TFGAN. The network functions follow the TFGAN API
  except for one exception: if either `generator_fn` or `discriminator_fn` have
  an argument called `mode`, then the tf.Estimator mode is passed in for that
  argument. This helps with operations like batch normalization, which have
  different train and evaluation behavior.

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
      gan_estimator = tfgan.estimator.GANEstimator(
          model_dir,
          generator_fn=generator_fn,
          discriminator_fn=discriminator_fn,
          generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
          discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
          generator_optimizer=tf.train.AdamOptimizer(0.1, 0.5),
          discriminator_optimizer=tf.train.AdamOptimizer(0.1, 0.5))

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
               get_hooks_fn=None,
               get_eval_metric_ops_fn=None,
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
        generator. See `TFGAN` for more details and examples. Additionally, if
        it has an argument called `mode`, the Estimator's `mode` will be passed
        in (ex TRAIN, EVAL, PREDICT). This is useful for things like batch
        normalization.
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
      get_hooks_fn: A function that takes a `GANTrainOps` tuple and returns a
        list of hooks. These hooks are run on the generator and discriminator
        train ops, and can be used to implement the GAN training scheme.
        Defaults to `train.get_sequential_train_hooks()`.
      get_eval_metric_ops_fn: A function that takes a `GANModel`, and returns a
        dict of metric results keyed by name. The output of this function is
        passed into `tf.estimator.EstimatorSpec` during evaluation.
      add_summaries: `None`, a single `SummaryType`, or a list of `SummaryType`.
      use_loss_summaries: If `True`, add loss summaries. If `False`, does not.
        If `None`, uses defaults.
      config: `RunConfig` object to configure the runtime settings.

    Raises:
      ValueError: If loss functions aren't callable.
      ValueError: If `use_loss_summaries` isn't boolean or `None`.
      ValueError: If `get_hooks_fn` isn't callable or `None`.
    """
    if not callable(generator_loss_fn):
      raise ValueError('generator_loss_fn must be callable.')
    if not callable(discriminator_loss_fn):
      raise ValueError('discriminator_loss_fn must be callable.')
    if use_loss_summaries not in [True, False, None]:
      raise ValueError('use_loss_summaries must be True, False or None.')
    if get_hooks_fn is not None and not callable(get_hooks_fn):
      raise TypeError('get_hooks_fn must be callable.')

    def _model_fn(features, labels, mode):
      """GANEstimator model function."""
      if mode not in [model_fn_lib.ModeKeys.TRAIN, model_fn_lib.ModeKeys.EVAL,
                      model_fn_lib.ModeKeys.PREDICT]:
        raise ValueError('Mode not recognized: %s' % mode)
      real_data = labels  # rename inputs for clarity
      generator_inputs = features  # rename inputs for clarity

      # Make GANModel, which encapsulates the GAN model architectures.
      gan_model = _get_gan_model(
          mode, generator_fn, discriminator_fn, real_data, generator_inputs,
          add_summaries)

      # Make the EstimatorSpec, which incorporates the GANModel, losses, eval
      # metrics, and optimizers (if required).
      return _get_estimator_spec(
          mode, gan_model, generator_loss_fn, discriminator_loss_fn,
          get_eval_metric_ops_fn, generator_optimizer, discriminator_optimizer,
          get_hooks_fn)

    super(GANEstimator, self).__init__(
        model_fn=_model_fn, model_dir=model_dir, config=config)


def _get_gan_model(
    mode, generator_fn, discriminator_fn, real_data, generator_inputs,
    add_summaries, generator_scope='Generator'):
  """Makes the GANModel tuple, which encapsulates the GAN model architecture."""
  if mode == model_fn_lib.ModeKeys.PREDICT:
    if real_data is not None:
      raise ValueError('`labels` must be `None` when mode is `predict`. '
                       'Instead, found %s' % real_data)
    gan_model = _make_prediction_gan_model(
        generator_inputs, generator_fn, generator_scope)
  else:  # model_fn_lib.ModeKeys.TRAIN or model_fn_lib.ModeKeys.EVAL
    gan_model = _make_gan_model(
        generator_fn, discriminator_fn, real_data, generator_inputs,
        generator_scope, add_summaries, mode)

  return gan_model


def _get_estimator_spec(
    mode, gan_model, generator_loss_fn, discriminator_loss_fn,
    get_eval_metric_ops_fn, generator_optimizer, discriminator_optimizer,
    get_hooks_fn=None):
  """Get the EstimatorSpec for the current mode."""
  if mode == model_fn_lib.ModeKeys.PREDICT:
    estimator_spec = model_fn_lib.EstimatorSpec(
        mode=mode, predictions=gan_model.generated_data)
  else:
    gan_loss = tfgan_tuples.GANLoss(
        generator_loss=generator_loss_fn(gan_model),
        discriminator_loss=discriminator_loss_fn(gan_model))
    if mode == model_fn_lib.ModeKeys.EVAL:
      estimator_spec = _get_eval_estimator_spec(
          gan_model, gan_loss, get_eval_metric_ops_fn)
    else:  # model_fn_lib.ModeKeys.TRAIN:
      gopt = (generator_optimizer() if callable(generator_optimizer) else
              generator_optimizer)
      dopt = (discriminator_optimizer() if callable(discriminator_optimizer)
              else discriminator_optimizer)
      get_hooks_fn = get_hooks_fn or tfgan_train.get_sequential_train_hooks()
      estimator_spec = _get_train_estimator_spec(
          gan_model, gan_loss, gopt, dopt, get_hooks_fn)

  return estimator_spec


def _make_gan_model(generator_fn, discriminator_fn, real_data,
                    generator_inputs, generator_scope, add_summaries, mode):
  """Construct a `GANModel`, and optionally pass in `mode`."""
  # If network functions have an argument `mode`, pass mode to it.
  if 'mode' in inspect.getargspec(generator_fn).args:
    generator_fn = functools.partial(generator_fn, mode=mode)
  if 'mode' in inspect.getargspec(discriminator_fn).args:
    discriminator_fn = functools.partial(discriminator_fn, mode=mode)
  gan_model = tfgan_train.gan_model(
      generator_fn,
      discriminator_fn,
      real_data,
      generator_inputs,
      generator_scope=generator_scope,
      check_shapes=False)
  if add_summaries:
    if not isinstance(add_summaries, (tuple, list)):
      add_summaries = [add_summaries]
    with ops.name_scope(None):
      for summary_type in add_summaries:
        _summary_type_map[summary_type](gan_model)

  return gan_model


def _make_prediction_gan_model(generator_inputs, generator_fn, generator_scope):
  """Make a `GANModel` from just the generator."""
  # If `generator_fn` has an argument `mode`, pass mode to it.
  if 'mode' in inspect.getargspec(generator_fn).args:
    generator_fn = functools.partial(generator_fn,
                                     mode=model_fn_lib.ModeKeys.PREDICT)
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


def _get_eval_estimator_spec(gan_model, gan_loss, get_eval_metric_ops_fn=None,
                             name=None):
  """Return an EstimatorSpec for the eval case."""
  scalar_loss = gan_loss.generator_loss + gan_loss.discriminator_loss
  with ops.name_scope(None, 'metrics',
                      [gan_loss.generator_loss,
                       gan_loss.discriminator_loss]):
    def _summary_key(head_name, val):
      return '%s/%s' % (val, head_name) if head_name else val
    eval_metric_ops = {
        _summary_key(name, 'generator_loss'):
            metrics_lib.mean(gan_loss.generator_loss),
        _summary_key(name, 'discriminator_loss'):
            metrics_lib.mean(gan_loss.discriminator_loss)
    }
    if get_eval_metric_ops_fn is not None:
      custom_eval_metric_ops = get_eval_metric_ops_fn(gan_model)
      if not isinstance(custom_eval_metric_ops, dict):
        raise TypeError('get_eval_metric_ops_fn must return a dict, '
                        'received: {}'.format(custom_eval_metric_ops))
      eval_metric_ops.update(custom_eval_metric_ops)
  return model_fn_lib.EstimatorSpec(
      mode=model_fn_lib.ModeKeys.EVAL,
      predictions=gan_model.generated_data,
      loss=scalar_loss,
      eval_metric_ops=eval_metric_ops)


def _get_train_estimator_spec(
    gan_model, gan_loss, generator_optimizer, discriminator_optimizer,
    get_hooks_fn, train_op_fn=tfgan_train.gan_train_ops):
  """Return an EstimatorSpec for the train case."""
  scalar_loss = gan_loss.generator_loss + gan_loss.discriminator_loss
  train_ops = train_op_fn(gan_model, gan_loss, generator_optimizer,
                          discriminator_optimizer)
  training_hooks = get_hooks_fn(train_ops)
  return model_fn_lib.EstimatorSpec(
      loss=scalar_loss,
      mode=model_fn_lib.ModeKeys.TRAIN,
      train_op=train_ops.global_step_inc_op,
      training_hooks=training_hooks)
