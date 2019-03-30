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
"""A TF-GAN-backed GAN Estimator that works on TPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.gan.python import namedtuples as tfgan_tuples
from tensorflow.contrib.gan.python import train as tfgan_train
from tensorflow.contrib.gan.python.estimator.python import gan_estimator_impl as gan_estimator_lib
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
from tensorflow.contrib.training.python.training import training
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import metrics as metrics_lib
from tensorflow.python.ops.losses import losses

__all__ = [
    'TPUGANEstimator',
]


class TPUGANEstimator(tpu_estimator.TPUEstimator):
  """An estimator for Generative Adversarial Networks (GANs) on TPU.

  This Estimator is backed by TFGAN. It is similar to `tfgan.GANEstimator`,
  but works on TPU.

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
      config = tpu_config.RunConfig(model_dir='/my/dir')
      gan_estimator = tfgan.estimator.TPUGANEstimator(
          generator_fn=generator_fn,
          discriminator_fn=discriminator_fn,
          generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
          discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
          generator_optimizer=tf.train.AdamOptimizer(0.1, 0.5),
          discriminator_optimizer=tf.train.AdamOptimizer(0.1, 0.5),
          train_batch_size=4,
          config=config)

      # Train estimator.
      gan_estimator.train(train_input_fn, train_steps)

      # Evaluate resulting estimator.
      gan_estimator.evaluate(eval_input_fn, eval_steps)

      # Generate samples from generator.
      predictions = np.array([
          x['generated_data'] for x in gan_estimator.predict(predict_input_fn)])
  ```
  """

  def __init__(self,
               # Arguments to construct the `model_fn`.
               generator_fn=None,
               discriminator_fn=None,
               generator_loss_fn=None,
               discriminator_loss_fn=None,
               generator_optimizer=None,
               discriminator_optimizer=None,
               get_eval_metric_ops_fn=None,
               add_summaries=None,
               joint_train=False,
               gan_train_steps=tfgan_tuples.GANTrainSteps(1, 1),
               # TPUEstimator options.
               model_dir=None,
               config=None,
               params=None,
               use_tpu=True,
               train_batch_size=None,
               eval_batch_size=None,
               predict_batch_size=None,
               batch_axis=None,
               eval_on_tpu=True,
               export_to_tpu=True,
               warm_start_from=None):
    """Initializes a TPUGANEstimator instance.

    Args:
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
      get_eval_metric_ops_fn: A function that takes a list of arguments and
        returns a dict of metric results keyed by name. The output of this
        function is passed into `tf.estimator.EstimatorSpec` during evaluation.
        The arguments must be:
            * generator_inputs
            * generated_data
            * real_data
            * discriminator_real_outputs
            * discriminator_gen_outputs
      add_summaries: `None`, a single `SummaryType`, or a list of `SummaryType`.
        This is ignored for jobs that run on TPU, such as the train job if
        `use_tpu` is `True` or the eval job if `eval_on_tpu` is `True`.
      joint_train: A Python boolean. If `True`, jointly train the generator and
        the discriminator. If `False`, sequentially train them. See `train.py`
        in TFGAN for more details on the differences between the two GAN
        training methods.
      gan_train_steps: A `tfgan.GANTrainSteps` named tuple describing the ratio
        of generator to discriminator steps. For now, only supports 1:1
        training.
      model_dir: Same as `TPUEstimator`: Directory to save model parameters,
        graph and etc. This can also be used to load checkpoints from the
        directory into a estimator to continue training a previously saved
        model. If `None`, the model_dir in `config` will be used if set. If both
        are set, they must be same. If both are `None`, a temporary directory
        will be used.
      config: Same as `TPUEstimator`: An `tpu_config.RunConfig` configuration
        object. Cannot be `None`.
      params: Same as `TPUEstimator`: An optional `dict` of hyper parameters
        that will be passed into `input_fn` and `model_fn`.  Keys are names of
        parameters, values are basic python types. There are reserved keys for
        `TPUEstimator`, including 'batch_size'.
      use_tpu: Same as `TPUEstimator`: A bool indicating whether TPU support is
        enabled. Currently, TPU training and evaluation respect this bit, but
        eval_on_tpu can override execution of eval. See below. Predict still
        happens on CPU.
      train_batch_size: Same as `TPUEstimator`: An int representing the global
        training batch size. TPUEstimator transforms this global batch size to a
        per-shard batch size, as params['batch_size'], when calling `input_fn`
        and `model_fn`. Cannot be `None` if `use_tpu` is `True`. Must be
        divisible by total number of replicas.
      eval_batch_size: Same as `TPUEstimator`: An int representing evaluation
        batch size. Must be divisible by total number of replicas.
      predict_batch_size: Same as `TPUEstimator`: An int representing the
        prediction batch size. Must be divisible by total number of replicas.
      batch_axis: Same as `TPUEstimator`: A python tuple of int values
        describing how each tensor produced by the Estimator `input_fn` should
        be split across the TPU compute shards. For example, if your input_fn
        produced (images, labels) where the images tensor is in `HWCN` format,
        your shard dimensions would be [3, 0], where 3 corresponds to the `N`
        dimension of your images Tensor, and 0 corresponds to the dimension
        along which to split the labels to match up with the corresponding
        images. If None is supplied, and per_host_input_for_training is True,
        batches will be sharded based on the major dimension. If
        tpu_config.per_host_input_for_training is False or `PER_HOST_V2`,
        batch_axis is ignored.
      eval_on_tpu: Same as `TPUEstimator`: If False, evaluation runs on CPU or
        GPU. In this case, the model_fn must return `EstimatorSpec` when called
        with `mode` as `EVAL`.
      export_to_tpu: Same as `TPUEstimator`: If True, `export_savedmodel()`
        exports a metagraph for serving on TPU besides the one on CPU.
      warm_start_from: Same as `TPUEstimator`: Optional string filepath to a
        checkpoint or SavedModel to warm-start from, or a
        `tf.estimator.WarmStartSettings` object to fully configure
        warm-starting.  If the string filepath is provided instead of a
        `WarmStartSettings`, then all variables are warm-started, and it is
        assumed that vocabularies and Tensor names are unchanged.

    Raises:
      ValueError: If loss functions aren't callable.
      ValueError: If `gan_train_steps` isn't a `tfgan_tuples.GANTrainSteps`
        tuple.
      ValueError: If `gan_train_steps` isn't 1:1 training.
    """
    if not callable(generator_loss_fn):
      raise ValueError('generator_loss_fn must be callable.')
    if not callable(discriminator_loss_fn):
      raise ValueError('discriminator_loss_fn must be callable.')
    if not isinstance(gan_train_steps, tfgan_tuples.GANTrainSteps):
      raise ValueError(
          '`gan_train_steps` must be `tfgan_tuples.GANTrainSteps`. Instead, '
          'was type: %s' % type(gan_train_steps))
    if (gan_train_steps.generator_train_steps != 1 or
        gan_train_steps.discriminator_train_steps != 1):
      raise ValueError('Estimator currently only supports 1:1 training.')

    if use_tpu:
      generator_optimizer = _maybe_make_cross_shard_optimizer(
          generator_optimizer)
      discriminator_optimizer = _maybe_make_cross_shard_optimizer(
          discriminator_optimizer)

    def _model_fn(features, labels, mode, params):
      """GANEstimator model function."""
      del params  # unused
      if mode not in [model_fn_lib.ModeKeys.TRAIN, model_fn_lib.ModeKeys.EVAL,
                      model_fn_lib.ModeKeys.PREDICT]:
        raise ValueError('Mode not recognized: %s' % mode)
      real_data = labels  # rename inputs for clarity
      generator_inputs = features  # rename inputs for clarity

      # Make GANModel, which encapsulates the GAN model architectures.
      # TODO(joelshor): Switch TF-GAN over to TPU-compatible summaries, then
      # remove `add_summaries` logic below.
      is_on_tpu = _is_on_tpu(mode, use_tpu, eval_on_tpu)
      gan_model = gan_estimator_lib._get_gan_model(  # pylint:disable=protected-access
          mode, generator_fn, discriminator_fn, real_data, generator_inputs,
          add_summaries=None if is_on_tpu else add_summaries)

      # Make the TPUEstimatorSpec, which incorporates the GANModel, losses, eval
      # metrics, and optimizers (if required).
      estimator_spec = _get_estimator_spec(
          mode, gan_model, generator_loss_fn, discriminator_loss_fn,
          get_eval_metric_ops_fn, generator_optimizer, discriminator_optimizer,
          joint_train, is_on_tpu, gan_train_steps)
      assert isinstance(estimator_spec, tpu_estimator.TPUEstimatorSpec)
      return estimator_spec

    super(TPUGANEstimator, self).__init__(
        model_fn=_model_fn,
        model_dir=model_dir,
        config=config,
        params=params,
        use_tpu=use_tpu,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        predict_batch_size=predict_batch_size,
        batch_axis=batch_axis,
        eval_on_tpu=eval_on_tpu,
        export_to_tpu=export_to_tpu,
        warm_start_from=warm_start_from)


def _is_on_tpu(mode, use_tpu, eval_on_tpu):
  if mode == model_fn_lib.ModeKeys.TRAIN:
    return use_tpu
  elif mode == model_fn_lib.ModeKeys.EVAL:
    return eval_on_tpu
  else:
    return False


def _get_estimator_spec(
    mode, gan_model, generator_loss_fn, discriminator_loss_fn,
    get_eval_metric_ops_fn, generator_optimizer, discriminator_optimizer,
    joint_train, is_on_tpu, gan_train_steps):
  """Get the TPUEstimatorSpec for the current mode."""
  if mode == model_fn_lib.ModeKeys.PREDICT:
    estimator_spec = tpu_estimator.TPUEstimatorSpec(
        mode=mode, predictions={'generated_data': gan_model.generated_data})
  elif mode == model_fn_lib.ModeKeys.EVAL:
    gan_loss = tfgan_tuples.GANLoss(
        generator_loss=generator_loss_fn(
            gan_model, add_summaries=not is_on_tpu),
        discriminator_loss=discriminator_loss_fn(
            gan_model, add_summaries=not is_on_tpu))
    # Eval losses for metrics must preserve batch dimension.
    gan_loss_no_reduction = tfgan_tuples.GANLoss(
        generator_loss=generator_loss_fn(
            gan_model, add_summaries=False, reduction=losses.Reduction.NONE),
        discriminator_loss=discriminator_loss_fn(
            gan_model, add_summaries=False, reduction=losses.Reduction.NONE))
    estimator_spec = _get_eval_estimator_spec(
        gan_model, gan_loss, gan_loss_no_reduction, get_eval_metric_ops_fn)
  else:  # model_fn_lib.ModeKeys.TRAIN:
    gan_loss = tfgan_tuples.GANLoss(
        generator_loss=generator_loss_fn(
            gan_model, add_summaries=not is_on_tpu),
        discriminator_loss=discriminator_loss_fn(
            gan_model, add_summaries=not is_on_tpu))

    # Construct optimizers if arguments were callable. For TPUs, they must be
    # `CrossShardOptimizer`.
    g_callable = callable(generator_optimizer)
    gopt = generator_optimizer() if g_callable  else generator_optimizer
    d_callable = callable(discriminator_optimizer)
    dopt = discriminator_optimizer() if d_callable else discriminator_optimizer

    estimator_spec = _get_train_estimator_spec(
        gan_model, gan_loss, gopt, dopt, joint_train, gan_train_steps)

  return estimator_spec


def _get_eval_estimator_spec(gan_model, gan_loss, gan_loss_no_reduction,
                             get_eval_metric_ops_fn):
  """Return an TPUEstimatorSpec for the eval case."""
  # Make the metric function and tensor names.
  if get_eval_metric_ops_fn is not None:
    def metric_fn(
        generator_inputs, generated_data, real_data, discriminator_real_outputs,
        discriminator_gen_outputs, generator_loss, discriminator_loss):
      """`metric_fn` used in TPUEstimator to calculate metrics."""
      eval_metric_ops = {
          'generator_loss': metrics_lib.mean(generator_loss),
          'discriminator_loss': metrics_lib.mean(discriminator_loss),
      }
      custom_eval_metric_ops = get_eval_metric_ops_fn(
          generator_inputs, generated_data, real_data,
          discriminator_real_outputs, discriminator_gen_outputs)
      if not isinstance(custom_eval_metric_ops, dict):
        raise TypeError('`get_eval_metric_ops_fn` must return a dict, '
                        'received: {}'.format(custom_eval_metric_ops))
      eval_metric_ops.update(custom_eval_metric_ops)
      return eval_metric_ops
    tensors = {
        'generator_loss': gan_loss_no_reduction.generator_loss,
        'discriminator_loss': gan_loss_no_reduction.discriminator_loss,
        'generator_inputs': gan_model.generator_inputs,
        'generated_data': gan_model.generated_data,
        'real_data': gan_model.real_data,
        'discriminator_real_outputs': gan_model.discriminator_real_outputs,
        'discriminator_gen_outputs': gan_model.discriminator_gen_outputs,
    }
  else:
    def metric_fn(generator_loss, discriminator_loss):
      return {
          'generator_loss': metrics_lib.mean(generator_loss),
          'discriminator_loss': metrics_lib.mean(discriminator_loss),
      }
    tensors = {
        'generator_loss': gan_loss_no_reduction.generator_loss,
        'discriminator_loss': gan_loss_no_reduction.discriminator_loss,
    }

  scalar_loss = gan_loss.generator_loss + gan_loss.discriminator_loss
  return tpu_estimator.TPUEstimatorSpec(
      mode=model_fn_lib.ModeKeys.EVAL,
      predictions=gan_model.generated_data,
      loss=scalar_loss,
      eval_metrics=(metric_fn, tensors))


def _get_train_estimator_spec(
    gan_model, gan_loss, generator_optimizer, discriminator_optimizer,
    joint_train, gan_train_steps):
  """Return a TPUEstimatorSpec for the train case."""
  scalar_loss = gan_loss.generator_loss + gan_loss.discriminator_loss

  # Get generator and discriminator update ops. We split them so that update
  # ops aren't accidentally run multiple times. For now, throw an error if
  # there are update ops that aren't associated with either the generator or
  # the discriminator. Might modify the `kwargs` dictionary.
  gen_update_ops, dis_update_ops = tfgan_train._get_update_ops(  # pylint:disable=protected-access
      {}, gan_model.generator_scope.name, gan_model.discriminator_scope.name)

  def gen_train_op():
    with ops.name_scope('generator_train'):
      return training.create_train_op(
          total_loss=gan_loss.generator_loss,
          optimizer=generator_optimizer,
          variables_to_train=gan_model.generator_variables,
          update_ops=gen_update_ops)
  def dis_train_op():
    with ops.name_scope('discriminator_train'):
      return training.create_train_op(
          total_loss=gan_loss.discriminator_loss,
          optimizer=discriminator_optimizer,
          variables_to_train=gan_model.discriminator_variables,
          update_ops=dis_update_ops)

  # Either optimize the generator and discriminator sequentially or jointly.
  tpu_train_op = _combine_train_ops(gen_train_op, dis_train_op, joint_train,
                                    gan_train_steps)

  return tpu_estimator.TPUEstimatorSpec(
      loss=scalar_loss,
      mode=model_fn_lib.ModeKeys.TRAIN,
      train_op=tpu_train_op)


# TODO(joelshor): Add support for multiple D / G steps.
def _combine_train_ops(gen_train_op, dis_train_op, joint_train,
                       gan_train_steps):
  """Combine generator and discriminator train ops into a single op."""
  del gan_train_steps
  if joint_train:
    tpu_train_op = control_flow_ops.group(gen_train_op(), dis_train_op(),
                                          name='joint_train')
  else:
    with ops.control_dependencies([dis_train_op()]):
      tpu_train_op = gen_train_op()

  return tpu_train_op


def _maybe_make_cross_shard_optimizer(opt):
  if callable(opt):
    if not isinstance(opt(), tpu_optimizer.CrossShardOptimizer):
      return lambda: tpu_optimizer.CrossShardOptimizer(opt())
  elif not isinstance(opt, tpu_optimizer.CrossShardOptimizer):
    return tpu_optimizer.CrossShardOptimizer(opt)
  return opt
