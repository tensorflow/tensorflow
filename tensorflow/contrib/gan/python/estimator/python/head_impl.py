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

from tensorflow.contrib.gan.python import namedtuples as tfgan_tuples
from tensorflow.contrib.gan.python import train as tfgan_train
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator.canned import head
from tensorflow.python.estimator.export import export_output
from tensorflow.python.framework import ops
from tensorflow.python.ops import metrics as metrics_lib

__all__ = [
    'GANHead',
    'gan_head',
]

def _summary_key(head_name, val):
  return '%s/%s' % (val, head_name) if head_name else val


def gan_head(generator_loss_fn, discriminator_loss_fn, generator_optimizer,
             discriminator_optimizer, use_loss_summaries=True,
             get_hooks_fn=tfgan_train.get_sequential_train_hooks(),
             get_eval_metric_ops_fn=None, name=None):
  """Creates a `GANHead`.

  Args:
    generator_loss_fn: A TFGAN loss function for the generator. Takes a
      `GANModel` and returns a scalar.
    discriminator_loss_fn: Same as `generator_loss_fn`, but for the
      discriminator.
    generator_optimizer: The optimizer for generator updates.
    discriminator_optimizer: Same as `generator_optimizer`, but for the
      discriminator updates.
    use_loss_summaries: If `True`, add loss summaries. If `False`, does not.
      If `None`, uses defaults.
    get_hooks_fn: A function that takes a `GANTrainOps` tuple and returns a
      list of hooks.
    get_eval_metric_ops_fn: A function that takes a `GANModel`, and returns a
      dict of metric results keyed by name. The output of this function is
      passed into `tf.estimator.EstimatorSpec` during evaluation.
    name: name of the head. If provided, summary and metrics keys will be
      suffixed by `"/" + name`.

  Returns:
    An instance of `GANHead`.
  """
  return GANHead(generator_loss_fn=generator_loss_fn,
                 discriminator_loss_fn=discriminator_loss_fn,
                 generator_optimizer=generator_optimizer,
                 discriminator_optimizer=discriminator_optimizer,
                 use_loss_summaries=use_loss_summaries,
                 get_hooks_fn=get_hooks_fn,
                 get_eval_metric_ops_fn=get_eval_metric_ops_fn,
                 name=name)


class GANHead(head._Head):  # pylint: disable=protected-access
  """`Head` for a GAN."""

  def __init__(self, generator_loss_fn, discriminator_loss_fn,
               generator_optimizer, discriminator_optimizer,
               use_loss_summaries=True,
               get_hooks_fn=None,
               get_eval_metric_ops_fn=None,
               name=None):
    """`Head` for GAN training.

    Args:
      generator_loss_fn: A TFGAN loss function for the generator. Takes a
        `GANModel` and returns a scalar.
      discriminator_loss_fn: Same as `generator_loss_fn`, but for the
      discriminator.
      generator_optimizer: The optimizer for generator updates.
      discriminator_optimizer: Same as `generator_optimizer`, but for the
        discriminator updates.
      use_loss_summaries: If `True`, add loss summaries. If `False`, does not.
        If `None`, uses defaults.
      get_hooks_fn: A function that takes a `GANTrainOps` tuple and returns a
        list of hooks. Defaults to `train.get_sequential_train_hooks()`
      get_eval_metric_ops_fn: A function that takes a `GANModel`, and returns a
        dict of metric results keyed by name. The output of this function is
        passed into `tf.estimator.EstimatorSpec` during evaluation.
      name: name of the head. If provided, summary and metrics keys will be
        suffixed by `"/" + name`.
    """
    if get_hooks_fn is None:
      get_hooks_fn = tfgan_train.get_sequential_train_hooks()
    # TODO(joelshor): Validate inputs.

    if use_loss_summaries in [True, False]:
      generator_loss_fn = functools.partial(
          generator_loss_fn, add_summaries=use_loss_summaries)
      discriminator_loss_fn = functools.partial(
          discriminator_loss_fn, add_summaries=use_loss_summaries)
    self._generator_loss_fn = generator_loss_fn
    self._discriminator_loss_fn = discriminator_loss_fn
    self._generator_optimizer = generator_optimizer
    self._discriminator_optimizer = discriminator_optimizer
    self._get_hooks_fn = get_hooks_fn
    self._get_eval_metric_ops_fn = get_eval_metric_ops_fn
    self._name = name

  @property
  def name(self):
    return self._name

  @property
  def logits_dimension(self):
    return None

  def create_loss(self, features, mode, logits, labels):
    """Returns a GANLoss tuple from the provided GANModel.

    See `Head` for more details.

    Args:
      features: Input `dict` of `Tensor` objects. Unused.
      mode: Estimator's `ModeKeys`.
      logits: A GANModel tuple.
      labels: Must be `None`.

    Returns:
      A GANLoss tuple.

    """
    _validate_logits_and_labels(logits, labels)
    del mode, labels, features  # unused for this head.
    gan_model = logits  # rename variable for clarity
    return tfgan_tuples.GANLoss(
        generator_loss=self._generator_loss_fn(gan_model),
        discriminator_loss=self._discriminator_loss_fn(gan_model))

  def create_estimator_spec(
      self, features, mode, logits, labels=None,
      train_op_fn=tfgan_train.gan_train_ops):
    """Returns `EstimatorSpec` that a model_fn can return.

    See `Head` for more details.

    Args:
      features: Must be `None`.
      mode: Estimator's `ModeKeys`.
      logits: A GANModel tuple.
      labels: Must be `None`.
      train_op_fn: Function that takes a GANModel, GANLoss, generator optimizer,
        and discriminator optimizer, and returns a `GANTrainOps` tuple. For
        example, this function can come from TFGAN's `train.py` library, or can
        be custom.

    Returns:
      `EstimatorSpec`.

    Raises:
      ValueError: If `features` isn't `None`.
      ValueError: If `train_op_fn` isn't provided in train mode.
    """
    _validate_logits_and_labels(logits, labels)
    if features is not None:
      raise ValueError('`features` should be `None`. Instead, found: %s' %
                       features)
    gan_model = logits  # rename variable for clarity
    with ops.name_scope('GANHead'):
      if mode == model_fn_lib.ModeKeys.PREDICT:
        return model_fn_lib.EstimatorSpec(
            mode=model_fn_lib.ModeKeys.PREDICT,
            predictions=gan_model.generated_data,
            export_outputs={
                'predict': export_output.PredictOutput(gan_model.generated_data)
            })
      elif mode == model_fn_lib.ModeKeys.EVAL:
        gan_loss = self.create_loss(
            features=None, mode=mode, logits=gan_model, labels=None)
        scalar_loss = gan_loss.generator_loss + gan_loss.discriminator_loss
        with ops.name_scope(None, 'metrics',
                            [gan_loss.generator_loss,
                             gan_loss.discriminator_loss]):
          eval_metric_ops = {
              _summary_key(self._name, 'generator_loss'):
                  metrics_lib.mean(gan_loss.generator_loss),
              _summary_key(self._name, 'discriminator_loss'):
                  metrics_lib.mean(gan_loss.discriminator_loss)
          }
          if self._get_eval_metric_ops_fn is not None:
            custom_eval_metric_ops = self._get_eval_metric_ops_fn(gan_model)
            if not isinstance(custom_eval_metric_ops, dict):
              raise TypeError('get_eval_metric_ops_fn must return a dict, '
                              'received: {}'.format(custom_eval_metric_ops))
            eval_metric_ops.update(custom_eval_metric_ops)
        return model_fn_lib.EstimatorSpec(
            mode=model_fn_lib.ModeKeys.EVAL,
            predictions=gan_model.generated_data,
            loss=scalar_loss,
            eval_metric_ops=eval_metric_ops)
      elif mode == model_fn_lib.ModeKeys.TRAIN:
        if train_op_fn is None:
          raise ValueError('train_op_fn can not be None.')
        gan_loss = self.create_loss(None, mode, gan_model, None)
        scalar_loss = gan_loss.generator_loss + gan_loss.discriminator_loss
        train_ops = train_op_fn(gan_model, gan_loss, self._generator_optimizer,
                                self._discriminator_optimizer)
        training_hooks = self._get_hooks_fn(train_ops)
        return model_fn_lib.EstimatorSpec(
            loss=scalar_loss,
            mode=model_fn_lib.ModeKeys.TRAIN,
            train_op=train_ops.global_step_inc_op,
            training_hooks=training_hooks)
      else:
        raise ValueError('Mode not recognized: %s' % mode)


def _validate_logits_and_labels(logits, labels):
  if labels is not None:
    raise ValueError('`GANHead`\'s `create_estimator_spec` input `labels` must '
                     'be `None`. Instead, found: %s' % labels)

  if not isinstance(logits, tfgan_tuples.GANModel):
    raise ValueError('`GANHead`\'s `create_estimator_spec` input `logits` must '
                     'be an instnace of a `GANModel`. Instead, found: %s' %
                     logits)
