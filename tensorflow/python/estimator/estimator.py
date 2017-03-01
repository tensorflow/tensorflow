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

"""Base Estimator class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import inspect
import os
import tempfile

import numpy as np
import six

from tensorflow.core.framework import summary_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator import run_config
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary.writer import writer_cache
from tensorflow.python.training import evaluation
from tensorflow.python.training import saver
from tensorflow.python.training import training

_VALID_MODEL_FN_ARGS = set(
    ['features', 'labels', 'mode', 'params', 'config'])


class Estimator(object):
  """Estimator class to train and evaluate TensorFlow models.

  The Estimator object wraps a model which is specified by a `model_fn`, which,
  given inputs and a number of other parameters, returns the ops necessary to
  perform training, evaluation, or predictions, respectively.

  All outputs (checkpoints, event files, etc.) are written to `model_dir`, or a
  subdirectory thereof. If `model_dir` is not set, a temporary directory is
  used.

  The `config` argument can be passed `RunConfig` object containing information
  about the execution environment. It is passed on to the `model_fn`, if the
  `model_fn` has a parameter named "config" (and input functions in the same
  manner). If the `config` parameter is not passed, it is instantiated by the
  Estimator. Not passing config means that defaults useful for local execution
  are used. Estimator makes config available to the model (for instance, to
  allow specialization based on the number of workers available), and also uses
  some of its fields to control internals, especially regarding checkpointing.

  The `params` argument contains hyperparameters. It is passed to the
  `model_fn`, if the `model_fn` has a parameter named "params", and to the input
  functions in the same manner. Estimator only passes params along, it does not
  inspect it. The structure of params is therefore entirely up to the developer.
  """

  def __init__(self, model_fn=None, model_dir=None, config=None, params=None):
    """Constructs an `Estimator` instance.

    Args:
      model_fn: Model function. Follows the signature:
        * Args:
          * `features`: single `Tensor` or `dict` of `Tensor`s
                 (depending on data passed to `fit`),
          * `labels`: `Tensor` or `dict` of `Tensor`s (for multi-head
                 models). If mode is `ModeKeys.INFER`, `labels=None` will be
                 passed. If the `model_fn`'s signature does not accept
                 `mode`, the `model_fn` must still be able to handle
                 `labels=None`.
          * `mode`: Optional. Specifies if this training, evaluation or
                 prediction. See `ModeKeys`.
          * `params`: Optional `dict` of hyperparameters.  Will receive what
                 is passed to Estimator in `params` parameter. This allows
                 to configure Estimators from hyper parameter tuning.
          * `config`: Optional configuration object. Will receive what is passed
                 to Estimator in `config` parameter, or the default `config`.
                 Allows updating things in your model_fn based on configuration
                 such as `num_ps_replicas`.
          * `model_dir`: Optional directory where model parameters, graph etc
                 are saved. Will receive what is passed to Estimator in
                 `model_dir` parameter, or the default `model_dir`. Allows
                 updating things in your model_fn that expect model_dir, such as
                 training hooks.

        * Returns:
          `EstimatorSpec`
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
      config: Configuration object.
      params: `dict` of hyper parameters that will be passed into `model_fn`.
              Keys are names of parameters, values are basic python types.

    Raises:
      ValueError: parameters of `model_fn` don't match `params`.
    """
    # Model directory.
    self._model_dir = model_dir
    if self._model_dir is None:
      self._model_dir = tempfile.mkdtemp()
      logging.warning('Using temporary folder as model directory: %s',
                      self._model_dir)

    if config is None:
      self._config = run_config.RunConfig()
      logging.info('Using default config.')
    else:
      if not isinstance(config, run_config.RunConfig):
        raise ValueError(
            'config must be an instance of RunConfig, but provided %s.' %
            config)
      self._config = config

    logging.info('Using config: %s', str(vars(self._config)))

    self._device_fn = _get_replica_device_setter(self._config)

    if model_fn is None:
      raise ValueError('model_fn must be provided to Estimator.')
    _verify_model_fn_args(model_fn, params)
    self._model_fn = model_fn
    self._params = params or {}

  @property
  def model_dir(self):
    return self._model_dir

  @property
  def config(self):
    return copy.deepcopy(self._config)

  @property
  def params(self):
    return copy.deepcopy(self._params)

  def fit(self, input_fn, hooks=None, steps=None, max_steps=None):
    """Trains a model given training data input_fn.

    Args:
      input_fn: Input function returning a tuple of:
          features - `Tensor` or dictionary of string feature name to `Tensor`.
          labels - `Tensor` or dictionary of `Tensor` with labels.
      hooks: List of `SessionRunHook` subclass instances. Used for callbacks
        inside the training loop.
      steps: Number of steps for which to train model. If `None`, train forever
        or train until input_fn generates the `OutOfRange` or `StopIteration`
        error. 'steps' works incrementally. If you call two times fit(steps=10)
        then training occurs in total 20 steps. If `OutOfRange` or
        `StopIteration` error occurs in the middle, training stops before 20
        steps. If you don't want to have incremental behaviour please set
        `max_steps` instead. If set, `max_steps` must be `None`.
      max_steps: Number of total steps for which to train model. If `None`,
        train forever or train until input_fn generates the `OutOfRange` or
        `StopIteration` error. If set, `steps` must be `None`. If `OutOfRange`
        or `StopIteration` error occurs in the middle, training stops before
        `max_steps` steps.

        Two calls to `fit(steps=100)` means 200 training
        iterations. On the other hand, two calls to `fit(max_steps=100)` means
        that the second call will not do any iteration since first call did
        all 100 steps.

    Returns:
      `self`, for chaining.

    Raises:
      ValueError: If both `steps` and `max_steps` are not `None`.
      ValueError: If either `steps` or `max_steps` is <= 0.
    """
    if (steps is not None) and (max_steps is not None):
      raise ValueError('Can not provide both steps and max_steps.')
    if steps is not None and steps <= 0:
      raise ValueError('Must specify steps >= 0, given: {}'.format(steps))
    if max_steps is not None and max_steps <= 0:
      raise ValueError(
          'Must specify max_steps >= 0, given: {}'.format(max_steps))

    if max_steps is not None:
      start_step = _load_global_step_from_checkpoint_dir(self._model_dir)
      if max_steps <= start_step:
        logging.info('Skipping training since max_steps has already saved.')
        return self

    hooks = hooks[:] if hooks else []
    if steps is not None or max_steps is not None:
      hooks.append(training.StopAtStepHook(steps, max_steps))

    loss = self._train_model(input_fn=input_fn, hooks=hooks)
    logging.info('Loss for final step: %s.', loss)
    return self

  def evaluate(self, input_fn, steps=None, hooks=None, checkpoint_path=None,
               name=None):
    """Evaluates the model given evaluation data input_fn.

    For each step, calls `input_fn`, which returns one batch of data.
    Evaluates until:
    - `steps` batches are processed, or
    - `input_fn` raises an end-of-input exception (`OutOfRangeError` or
    `StopIteration`).

    Args:
      input_fn: Input function returning a tuple of:
          features - Dictionary of string feature name to `Tensor` or
            `SparseTensor`.
          labels - `Tensor` or dictionary of `Tensor` with labels.
      steps: Number of steps for which to evaluate model. If `None`, evaluates
        until `input_fn` raises an end-of-input exception.
      hooks: List of `SessionRunHook` subclass instances. Used for callbacks
        inside the evaluation call.
      checkpoint_path: Path of a specific checkpoint to evaluate. If `None`, the
        latest checkpoint in `model_dir` is used.
      name: Name of the evaluation if user needs to run multiple evaluations on
        different data sets, such as on training data vs test data. Metrics for
        different evaluations are saved in separate folders, and appear
        separately in tensorboard.

    Returns:
      A dict containing the evaluation metrics specified in `model_fn` keyed by
      name, as well as an entry `global_step` which contains the value of the
      global step for which this evaluation was performed.

    Raises:
      ValueError: If `steps <= 0`.
      ValueError: If no model has been trained, namely `model_dir`, or the
        given `checkpoint_path` is empty.
    """
    # We need to copy the hook array as we modify it, thus [:].
    hooks = hooks[:] if hooks else []
    if steps is not None:
      if steps <= 0:
        raise ValueError('Must specify steps >= 0, given: {}'.format(steps))
      hooks.append(evaluation._StopAfterNEvalsHook(  # pylint: disable=protected-access
          num_evals=steps))

    return self._evaluate_model(
        input_fn=input_fn,
        hooks=hooks,
        checkpoint_path=checkpoint_path,
        name=name)

  def _call_model_fn(self, features, labels, mode):
    """Calls model function.

    Args:
      features: features dict.
      labels: labels dict.
      mode: ModeKeys

    Returns:
      An `EstimatorSpec` object.

    Raises:
      ValueError: if model_fn returns invalid objects.
    """
    model_fn_args = _get_arguments(self._model_fn).args
    kwargs = {}
    if 'mode' in model_fn_args:
      kwargs['mode'] = mode
    if 'params' in model_fn_args:
      kwargs['params'] = self.params
    if 'config' in model_fn_args:
      kwargs['config'] = self.config
    model_fn_results = self._model_fn(
        features=features, labels=labels, **kwargs)

    if not isinstance(model_fn_results, model_fn_lib.EstimatorSpec):
      raise ValueError('model_fn should return an EstimatorSpec.')

    return model_fn_results

  def _train_model(self, input_fn, hooks):
    all_hooks = []
    with ops.Graph().as_default() as g, g.device(self._device_fn):
      random_seed.set_random_seed(self._config.tf_random_seed)
      global_step_tensor = training.create_global_step(g)
      with ops.device('/cpu:0'):
        features, labels = input_fn()
      estimator_spec = self._call_model_fn(features, labels,
                                           model_fn_lib.ModeKeys.FIT)
      ops.add_to_collection(ops.GraphKeys.LOSSES, estimator_spec.loss)
      all_hooks.extend([
          training.NanTensorHook(estimator_spec.loss),
          training.LoggingTensorHook(
              {
                  'loss': estimator_spec.loss,
                  'step': global_step_tensor
              },
              every_n_iter=100)
      ])
      all_hooks.extend(hooks)
      all_hooks.extend(estimator_spec.training_hooks)

      scaffold = estimator_spec.scaffold or training.Scaffold()
      if not (scaffold.saver or ops.get_collection(ops.GraphKeys.SAVERS)):
        ops.add_to_collection(ops.GraphKeys.SAVERS,
                              training.Saver(
                                  sharded=True,
                                  max_to_keep=self._config.keep_checkpoint_max,
                                  defer_build=True))

      chief_hooks = []
      if (self._config.save_checkpoints_secs or
          self._config.save_checkpoints_steps):
        saver_hook_exists = any([
            isinstance(h, training.CheckpointSaverHook)
            for h in (all_hooks + chief_hooks +
                      estimator_spec.training_chief_hooks)
        ])
        if not saver_hook_exists:
          chief_hooks = [
              training.CheckpointSaverHook(
                  self._model_dir,
                  save_secs=self._config.save_checkpoints_secs,
                  save_steps=self._config.save_checkpoints_steps,
                  scaffold=scaffold)
          ]
      with training.MonitoredTrainingSession(
          master=self._config.master,
          is_chief=self._config.is_chief,
          checkpoint_dir=self._model_dir,
          scaffold=scaffold,
          hooks=all_hooks,
          chief_only_hooks=chief_hooks + estimator_spec.training_chief_hooks,
          save_checkpoint_secs=0,  # Saving is handled by a hook.
          save_summaries_steps=self._config.save_summary_steps,
          config=config_pb2.ConfigProto(allow_soft_placement=True)) as mon_sess:
        loss = None
        while not mon_sess.should_stop():
          _, loss = mon_sess.run([estimator_spec.train_op, estimator_spec.loss])
      return loss

  def _evaluate_model(self,
                      input_fn,
                      hooks=None,
                      checkpoint_path=None,
                      name=''):
    """Evaluates the model using the training.evaluation library."""
    # Check that model has been trained (if nothing has been set explicitly).
    if not checkpoint_path:
      latest_path = saver.latest_checkpoint(self._model_dir)
      if not latest_path:
        raise ValueError('Could not find trained model in model_dir: {}.'.
                         format(self._model_dir))
      checkpoint_path = latest_path

    # Setup output directory.
    eval_dir = os.path.join(self._model_dir, 'eval' if not name else
                            'eval_' + name)

    with ops.Graph().as_default() as g:
      random_seed.set_random_seed(self._config.tf_random_seed)
      global_step_tensor = training.create_global_step(g)
      features, labels = input_fn()
      estimator_spec = self._call_model_fn(
          features, labels, model_fn_lib.ModeKeys.EVAL)

      update_op, eval_dict = _extract_metric_update_ops(
          estimator_spec.eval_metric_ops)

      if ops.GraphKeys.GLOBAL_STEP in six.iterkeys(eval_dict):
        raise ValueError(
            'Metric with name `global_step` is not allowed, because Estimator '
            'already defines a default metric with the same name.')
      eval_dict[ops.GraphKeys.GLOBAL_STEP] = global_step_tensor

      eval_results = evaluation._evaluate_once(  # pylint: disable=protected-access
          checkpoint_path=checkpoint_path,
          master=self._config.evaluation_master,
          eval_ops=update_op,
          final_ops=eval_dict,
          hooks=hooks,
          config=config_pb2.ConfigProto(allow_soft_placement=True))

      _write_dict_to_summary(
          output_dir=eval_dir,
          dictionary=eval_results,
          current_global_step=eval_results[ops.GraphKeys.GLOBAL_STEP])

    return eval_results


def _get_replica_device_setter(config):
  """Creates a replica device setter if required as a default device_fn.

  `Estimator` uses ReplicaDeviceSetter as a default device placer. It sets the
  distributed related arguments such as number of ps_replicas based on given
  config.

  Args:
    config: A `RunConfig` instance.

  Returns:
    A replica device setter, or None.
  """
  ps_ops = [
      'Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
      'MutableHashTableOfTensors', 'MutableDenseHashTable'
  ]

  if config.task_type:
    worker_device = '/job:%s/task:%d' % (config.task_type, config.task_id)
  else:
    worker_device = '/job:worker'

  if config.num_ps_replicas > 0:
    return training.replica_device_setter(
        ps_tasks=config.num_ps_replicas,
        worker_device=worker_device,
        merge_devices=True,
        ps_ops=ps_ops,
        cluster=config.cluster_spec)
  else:
    return None


def _get_arguments(func):
  """Returns a spec of given func."""
  if hasattr(func, '__code__'):
    # Regular function.
    return inspect.getargspec(func)
  elif hasattr(func, '__call__'):
    # Callable object.
    return _get_arguments(func.__call__)
  elif hasattr(func, 'func'):
    # Partial function.
    return _get_arguments(func.func)


def _verify_model_fn_args(model_fn, params):
  """Verifies model fn arguments."""
  fn_spec = _get_arguments(model_fn)
  if 'features' not in fn_spec.args:
    raise ValueError('model_fn (%s) must include features argument.' % model_fn)
  if 'labels' not in fn_spec.args:
    raise ValueError('model_fn (%s) must include labels argument.' % model_fn)
  if params is not None and 'params' not in fn_spec.args:
    raise ValueError('model_fn (%s) does not include params argument, '
                     'but params (%s) is passed to Estimator.' % (model_fn,
                                                                  params))
  if params is None and 'params' in fn_spec.args:
    logging.warning('Estimator\'s model_fn (%s) includes params '
                    'argument, but params are not passed to Estimator.',
                    model_fn)
  non_valid_args = list(set(fn_spec.args) - _VALID_MODEL_FN_ARGS)
  if non_valid_args:
    raise ValueError('model_fn (%s) has following not expected args: %s' %
                     (model_fn, non_valid_args))


def _load_global_step_from_checkpoint_dir(checkpoint_dir):
  try:
    checkpoint_reader = training.NewCheckpointReader(
        training.latest_checkpoint(checkpoint_dir))
    return checkpoint_reader.get_tensor(ops.GraphKeys.GLOBAL_STEP)
  except:  # pylint: disable=bare-except
    return 0


def _extract_metric_update_ops(eval_dict):
  """Separate update operations from metric value operations."""
  update_ops = []
  value_ops = {}
  # Sort metrics lexicographically so graph is identical every time.
  for name, metric_ops in sorted(six.iteritems(eval_dict)):
    value_ops[name] = metric_ops[0]
    update_ops.append(metric_ops[1])

  if update_ops:
    update_op = control_flow_ops.group(*update_ops)
  else:
    update_op = None

  return update_op, value_ops


def _dict_to_str(dictionary):
  """Get a `str` representation of a `dict`.

  Args:
    dictionary: The `dict` to be represented as `str`.

  Returns:
    A `str` representing the `dictionary`.
  """
  return ', '.join('%s = %s' % (k, v)
                   for k, v in sorted(six.iteritems(dictionary)))


def _write_dict_to_summary(output_dir,
                           dictionary,
                           current_global_step):
  """Writes a `dict` into summary file in given output directory.

  Args:
    output_dir: `str`, directory to write the summary file in.
    dictionary: the `dict` to be written to summary file.
    current_global_step: `int`, the current global step.
  """
  logging.info('Saving dict for global step %d: %s', current_global_step,
               _dict_to_str(dictionary))
  summary_writer = writer_cache.FileWriterCache.get(output_dir)
  summary_proto = summary_pb2.Summary()
  for key in dictionary:
    if dictionary[key] is None:
      continue
    value = summary_proto.value.add()
    value.tag = key
    if (isinstance(dictionary[key], np.float32) or
        isinstance(dictionary[key], float)):
      value.simple_value = float(dictionary[key])
    else:
      logging.warn('Skipping summary for %s, must be a float or np.float32.',
                   key)
  summary_writer.add_summary(summary_proto, current_global_step)
  summary_writer.flush()
