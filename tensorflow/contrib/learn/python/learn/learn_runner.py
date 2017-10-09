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
"""Runs an Experiment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn.estimators import run_config as run_config_lib
from tensorflow.contrib.learn.python.learn.experiment import Experiment
from tensorflow.contrib.training.python.training import hparam as hparam_lib
from tensorflow.python.platform import tf_logging as logging


# TODO(xiejw): Refactor the learn_runner to make code reusable.
def _execute_schedule(experiment, schedule):
  """Execute the method named `schedule` of `experiment`."""
  if not hasattr(experiment, schedule):
    logging.error('Schedule references non-existent task %s', schedule)
    valid_tasks = [x for x in dir(experiment)
                   if not x.startswith('_')
                   and callable(getattr(experiment, x))]
    logging.error('Allowed values for this experiment are: %s', valid_tasks)
    raise ValueError('Schedule references non-existent task %s' % schedule)

  task = getattr(experiment, schedule)
  if not callable(task):
    logging.error('Schedule references non-callable member %s', schedule)
    valid_tasks = [x for x in dir(experiment)
                   if not x.startswith('_')
                   and callable(getattr(experiment, x))]
    logging.error('Allowed values for this experiment are: %s', valid_tasks)
    raise TypeError('Schedule references non-callable member %s' % schedule)
  return task()


def _wrapped_experiment_fn_with_uid_check(experiment_fn, require_hparams=False):
  """Wraps the `RunConfig` uid check with `experiment_fn`.

  For `experiment_fn` which takes `run_config`, it is expected that the
  `run_config` is passed to the Estimator correctly. Toward that, the wrapped
  `experiment_fn` compares the `uid` of the `RunConfig` instance.

  Args:
    experiment_fn: The original `experiment_fn` which takes `run_config` and
      `hparams`.
    require_hparams: If True, the `hparams` passed to `experiment_fn` cannot be
      `None`.

  Returns:
    A experiment_fn with same signature.
  """
  def wrapped_experiment_fn(run_config, hparams):
    """Calls experiment_fn and checks the uid of `RunConfig`."""
    if not isinstance(run_config, run_config_lib.RunConfig):
      raise ValueError(
          '`run_config` must be `tf.contrib.learn.RunConfig` instance')
    if not run_config.model_dir:
      raise ValueError(
          'Must specify a model directory `model_dir` in `run_config`.')
    if hparams is not None and not isinstance(hparams, hparam_lib.HParams):
      raise ValueError('`hparams` must be `HParams` instance')
    if require_hparams and hparams is None:
      raise ValueError('`hparams` cannot be `None`.')

    expected_uid = run_config.uid()
    experiment = experiment_fn(run_config, hparams)

    if not isinstance(experiment, Experiment):
      raise TypeError('Experiment builder did not return an Experiment '
                      'instance, got %s instead.' % type(experiment))

    config_from_estimator = experiment.estimator.config
    if not hasattr(config_from_estimator, 'uid'):
      raise RuntimeError(
          'Pass `run_config` argument of the `experiment_fn` to the Estimator '
          'in Experiment. It is likely a different `RunConfig` is passed to '
          '`Estimator` or the `config` constructor argument in `Estimator` '
          'is not set.')

    if config_from_estimator.uid() != expected_uid:
      raise RuntimeError(
          '`RunConfig` instance is expected to be used by the `Estimator` '
          'inside the `Experiment`. expected {}, but got {}'.format(
              expected_uid, experiment.estimator.config.uid()))
    return experiment
  return wrapped_experiment_fn


def run(experiment_fn, output_dir=None, schedule=None, run_config=None,
        hparams=None):
  """Make and run an experiment.

  It creates an Experiment by calling `experiment_fn`. Then it calls the
  function named as `schedule` of the Experiment.

  If schedule is not provided, then the default schedule for the current task
  type is used. The defaults are as follows:

   * 'ps' maps to 'serve'
   * 'worker' maps to 'train'
   * 'master' maps to 'local_run'

  If the experiment's config does not include a task type, then an exception
  is raised.

  Example with `run_config` (Recommended):
  ```
    def _create_my_experiment(run_config, hparams):

        # You can change a subset of the run_config properties as
        #   run_config = run_config.replace(save_checkpoints_steps=500)

        return tf.contrib.learn.Experiment(
          estimator=my_estimator(config=run_config, hparams=hparams),
          train_input_fn=my_train_input,
          eval_input_fn=my_eval_input)

    learn_runner.run(
      experiment_fn=_create_my_experiment,
      run_config=run_config_lib.RunConfig(model_dir="some/output/dir"),
      schedule="train_and_evaluate",
      hparams=_create_default_hparams())
  ```
  or simply as
  ```
    learn_runner.run(
      experiment_fn=_create_my_experiment,
      run_config=run_config_lib.RunConfig(model_dir="some/output/dir"))
  ```
  if `hparams` is not used by the `Estimator`. On a single machine, `schedule`
  defaults to `train_and_evaluate`.

  Example with `output_dir` (deprecated):
  ```
    def _create_my_experiment(output_dir):
        return tf.contrib.learn.Experiment(
          estimator=my_estimator(model_dir=output_dir),
          train_input_fn=my_train_input,
          eval_input_fn=my_eval_input)

    learn_runner.run(
      experiment_fn=_create_my_experiment,
      output_dir="some/output/dir",
      schedule="train")
  ```
  Args:
    experiment_fn: A function that creates an `Experiment`. It could be one of
      the two following signatures:
      1) [Deprecated] It accepts an argument `output_dir` which should be used
      to create the `Estimator` (passed as `model_dir` to its constructor). It
      must return an `Experiment`. For this case, `run_config` and `hparams`
      must be None.
      2) It accepts two arguments `run_config` and `hparams`, which should be
      used to create the `Estimator` (`run_config` passed as `config` to its
      constructor; `hparams` used as the hyper-parameters of the model).
      It must return an `Experiment`. For this case, `output_dir` must be None.
    output_dir: Base output directory [Deprecated].
    schedule: The name of the method in the `Experiment` to run.
    run_config: `RunConfig` instance. The `run_config.model_dir` must be
      non-empty. If `run_config` is set, `output_dir` must be None.
    hparams: `HParams` instance. The default hyper-parameters, which will be
      passed to the `experiment_fn` if `run_config` is not None.

  Returns:
    The return value of function `schedule`.

  Raises:
    ValueError: If both `output_dir` and `run_config` are empty or set,
      `schedule` is None but no task type is set in the built experiment's
      config, the task type has no default, `run_config.model_dir` is empty or
      `schedule` doesn't reference a member of `Experiment`.
    TypeError: `schedule` references non-callable member.
  """

  if output_dir is not None and run_config is not None:
    raise ValueError('Cannot provide both `output_dir` and `run_config`')

  if output_dir is None and run_config is None:
    raise ValueError('Must set value for `output_dir` or `run_config`')

  if not callable(experiment_fn):
    raise TypeError('Experiment builder "%s" is not callable.' %
                    experiment_fn)

  experiment = None
  if run_config is not None:
    wrapped_experiment_fn = _wrapped_experiment_fn_with_uid_check(experiment_fn)
    experiment = wrapped_experiment_fn(run_config=run_config, hparams=hparams)
  else:
    if not output_dir:
      raise ValueError('Must specify an output directory')
    if hparams is not None:
      raise ValueError(
          'Must set `hparams` as None for `experiment_fn` with `output_dir`.')
    # Call the builder
    experiment = experiment_fn(output_dir=output_dir)
    if not isinstance(experiment, Experiment):
      raise TypeError('Experiment builder did not return an Experiment '
                      'instance, got %s instead.' % type(experiment))

  # Get the schedule
  run_config = run_config or experiment.estimator.config
  schedule = schedule or _get_default_schedule(run_config)

  return _execute_schedule(experiment, schedule)


def tune(experiment_fn, tuner):
  """Tune an experiment with hyper-parameters.

  It iterates trials by running the Experiment for each trial with the
  corresponding hyper-parameters. For each trial, it retrieves the
  hyper-parameters from `tuner`, creates an Experiment by calling experiment_fn,
  and then reports the measure back to `tuner`.

  Example:
  ```
    def _create_my_experiment(run_config, hparams):
      hidden_units = [hparams.unit_per_layer] * hparams.num_hidden_layers

      return tf.contrib.learn.Experiment(
          estimator=DNNClassifier(config=run_config, hidden_units=hidden_units),
          train_input_fn=my_train_input,
          eval_input_fn=my_eval_input)

    tuner = create_tuner(study_configuration, objective_key)

    learn_runner.tune(experiment_fn=_create_my_experiment, tuner)
  ```
  Args:
    experiment_fn: A function that creates an `Experiment`. It should accept an
      argument `run_config` which should be used to create the `Estimator` (
      passed as `config` to its constructor), and an argument `hparams`, which
      should be used for hyper-parameters tuning. It must return an
      `Experiment`.
    tuner: A `Tuner` instance.
  """
  while tuner.next_trial():
    tuner.run_experiment(
        _wrapped_experiment_fn_with_uid_check(
            experiment_fn, require_hparams=True))


def _is_distributed(config):
  """Returns true if this is a distributed job."""
  if not config.cluster_spec:
    return False

  # This is considered a distributed job if there is more than one task
  # in the cluster spec.
  task_count = 0
  for job in config.cluster_spec.jobs:
    for _ in config.cluster_spec.job_tasks(job):
      task_count += 1

  return task_count > 1


def _get_default_schedule(config):
  """Returns the default schedule for the provided RunConfig."""
  if not config or not _is_distributed(config):
    return 'train_and_evaluate'

  if not config.task_type:
    raise ValueError('Must specify a schedule')

  if config.task_type == run_config_lib.TaskType.MASTER:
    # TODO(rhaertel): handle the case where there is more than one master
    # or explicitly disallow such a case.
    return 'train_and_evaluate'
  elif config.task_type == run_config_lib.TaskType.PS:
    return 'run_std_server'
  elif config.task_type == run_config_lib.TaskType.WORKER:
    return 'train'

  raise ValueError('No default schedule for task type: %s' % (config.task_type))
