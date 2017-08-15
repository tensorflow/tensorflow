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

"""Experiment class collecting information needed for a single training run."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import math
import os
import time

from tensorflow.contrib.framework import deprecated
from tensorflow.contrib.framework import deprecated_args
from tensorflow.contrib.framework.python.framework import experimental
from tensorflow.contrib.learn.python.learn import evaluable
from tensorflow.contrib.learn.python.learn import export_strategy
from tensorflow.contrib.learn.python.learn import monitors
from tensorflow.contrib.learn.python.learn import trainable
from tensorflow.contrib.learn.python.learn.estimators import run_config
from tensorflow.python.estimator import estimator as core_estimator
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import saver
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat


__all__ = ["Experiment"]


class Experiment(object):
  """Experiment is a class containing all information needed to train a model.

  After an experiment is created (by passing an Estimator and inputs for
  training and evaluation), an Experiment instance knows how to invoke training
  and eval loops in a sensible fashion for distributed training.
  """

  # TODO(ispir): remove delay_workers_by_global_step and make global step based
  # waiting as only behavior.
  @deprecated_args(
      "2016-10-23",
      "local_eval_frequency is deprecated as local_run will be renamed to "
      "train_and_evaluate. Use min_eval_frequency and call train_and_evaluate "
      "instead. Note, however, that the default for min_eval_frequency is 1, "
      "meaning models will be evaluated every time a new checkpoint is "
      "available. In contrast, the default for local_eval_frequency is None, "
      "resulting in evaluation occurring only after training has completed. "
      "min_eval_frequency is ignored when calling the deprecated local_run.",
      "local_eval_frequency")
  def __init__(self,
               estimator,
               train_input_fn,
               eval_input_fn,
               eval_metrics=None,
               train_steps=None,
               eval_steps=100,
               train_monitors=None,
               eval_hooks=None,
               local_eval_frequency=None,
               eval_delay_secs=120,
               continuous_eval_throttle_secs=60,
               min_eval_frequency=None,
               delay_workers_by_global_step=False,
               export_strategies=None,
               train_steps_per_iteration=None):
    """Constructor for `Experiment`.

    Creates an Experiment instance. None of the functions passed to this
    constructor are executed at construction time. They are stored and used
    when a method is executed which requires it.

    Args:
      estimator: Object implementing Estimator interface, which could be a
        combination of ${tf.contrib.learn.Trainable} and
        ${tf.contrib.learn.Evaluable} (deprecated), or
        ${tf.estimator.`Estimator}.
      train_input_fn: function, returns features and labels for training.
      eval_input_fn: function, returns features and labels for evaluation. If
        `eval_steps` is `None`, this should be configured only to produce for a
        finite number of batches (generally, 1 epoch over the evaluation data).
      eval_metrics: `dict` of string, metric function. If `None`, default set
        is used. This should be `None` if the `estimator` is
        ${tf.estimator.Estimator}. If metrics are provided they will be
        *appended* to the default set.
      train_steps: Perform this many steps of training. `None`, the default,
        means train forever.
      eval_steps: `evaluate` runs until input is exhausted (or another exception
        is raised), or for `eval_steps` steps, if specified.
      train_monitors: A list of monitors to pass to the `Estimator`'s `fit`
        function.
      eval_hooks: A list of `SessionRunHook` hooks to pass to the
        `Estimator`'s `evaluate` function.
      local_eval_frequency: (applies only to local_run) Frequency of running
        eval in steps. If `None`, runs evaluation only at the end of training.
      eval_delay_secs: Start evaluating after waiting for this many seconds.
      continuous_eval_throttle_secs: Do not re-evaluate unless the last
        evaluation was started at least this many seconds ago for
        continuous_eval().
      min_eval_frequency: (applies only to train_and_evaluate). the minimum
        number of steps between evaluations. Of course, evaluation does not
        occur if no new snapshot is available, hence, this is the minimum.
        If 0, the evaluation will only happen after training.
        If None, defaults to 1, unless model_dir is on GCS, in which case the
        default is 1000.
      delay_workers_by_global_step: if `True` delays training workers
        based on global step instead of time.
      export_strategies: Iterable of `ExportStrategy`s, or a single one, or
        `None`.
      train_steps_per_iteration: (applies only to continuous_train_and_eval).
        Perform this many (integer) number of train steps for each
        training-evaluation iteration. With a small value, the model will be
        evaluated more frequently with more checkpoints saved. If `None`, will
        use a default value (which is smaller than `train_steps` if provided).

    Raises:
      ValueError: if `estimator` does not implement Estimator interface,
        or if export_strategies has the wrong type.
    """
    if isinstance(estimator, core_estimator.Estimator):
      self._core_estimator_used = True
      if eval_metrics is not None:
        raise ValueError(
            "`eval_metrics` must be `None` with `tf.estimator.Estimator`. "
            "Use `eval_metric_ops` in `tf.estimator.EstimatorSpec` instead.")
    else:
      self._core_estimator_used = False
      if not isinstance(estimator, evaluable.Evaluable):
        raise ValueError(
            "`estimator` must implement `tf.contrib.learn.Evaluable` "
            "or `tf.estimator.Estimator`.")
      if not isinstance(estimator, trainable.Trainable):
        raise ValueError(
            "`estimator` must implement `tf.contrib.learn.Trainable`"
            "or `tf.estimator.`Estimator`.")

    super(Experiment, self).__init__()
    # Immutable fields.
    self._estimator = estimator
    self._train_input_fn = train_input_fn
    self._eval_input_fn = eval_input_fn
    self._eval_metrics = eval_metrics
    self._train_steps = train_steps
    self._eval_steps = eval_steps
    self._local_eval_frequency = local_eval_frequency
    self._eval_delay_secs = eval_delay_secs
    self._continuous_eval_throttle_secs = continuous_eval_throttle_secs
    # Using 1 on a non-cached file system requires a lot of overhead to
    # read the checkpoint state file. This is particular bad on GCS, so
    # we use a different default. This is a temporary band-aid, to be
    # fixed holistically later (b/36498507).
    default_min_eval_frequency = 1000 if _is_gcs(estimator.model_dir) else 1
    self._min_eval_frequency = min_eval_frequency if (
        min_eval_frequency is not None) else default_min_eval_frequency
    self._delay_workers_by_global_step = delay_workers_by_global_step
    self._train_monitors = train_monitors[:] if train_monitors else []
    self._eval_hooks = eval_hooks[:] if eval_hooks else []
    self._set_export_strategies(export_strategies)

    self._train_steps_per_iteration = train_steps_per_iteration
    if (self._train_steps_per_iteration is not None and
        not isinstance(self._train_steps_per_iteration, int)):
      raise ValueError(
          "`train_steps_per_iteration` must be an integer.")

  @property
  def estimator(self):
    return self._estimator

  @property
  def eval_metrics(self):
    return self._eval_metrics

  @property
  def train_steps(self):
    return self._train_steps

  @property
  def eval_steps(self):
    return self._eval_steps

  def _set_export_strategies(self, values):  # pylint: disable=missing-docstring
    export_strategies = []
    if values:
      if isinstance(values, export_strategy.ExportStrategy):
        export_strategies.append(values)
      else:
        for value in values:
          if not isinstance(value, export_strategy.ExportStrategy):
            raise ValueError("`export_strategies` must be an ExportStrategy,"
                             " an iterable of ExportStrategy, or `None`,"
                             " found %s." % value)
          export_strategies.append(value)
    self._export_strategies = tuple(export_strategies)

  def extend_train_hooks(self, additional_hooks):
    """Extends the hooks for training."""
    self._train_monitors.extend(additional_hooks)

  def reset_export_strategies(self, new_export_strategies=None):
    """Resets the export strategies with the `new_export_strategies`.

    Args:
      new_export_strategies: A new list of `ExportStrategy`s, or a single one,
        or None.

    Returns:
      The old export strategies.
    """
    old_export_strategies = self._export_strategies
    self._set_export_strategies(new_export_strategies)
    return old_export_strategies

  def train(self, delay_secs=None):
    """Fit the estimator using the training data.

    Train the estimator for `self._train_steps` steps, after waiting for
    `delay_secs` seconds. If `self._train_steps` is `None`, train forever.

    Args:
      delay_secs: Start training after this many seconds.

    Returns:
      The trained estimator.
    """
    start = time.time()

    # Start the server, if needed. It's important to start the server before
    # we (optionally) sleep for the case where no device_filters are set.
    # Otherwise, the servers will wait to connect to each other before starting
    # to train. We might as well start as soon as we can.
    config = self._estimator.config
    if (config.cluster_spec and config.master and
        config.environment == run_config.Environment.LOCAL):
      logging.warn("ClusterSpec and master are provided, but environment is "
                   "set to 'local'. Set environment to 'cloud' if you intend "
                   "to use the distributed runtime.")
    if (config.environment != run_config.Environment.LOCAL and
        config.environment != run_config.Environment.GOOGLE and
        config.cluster_spec and config.master):
      self._start_server()

    extra_hooks = []
    if delay_secs is None:
      task_id = self._estimator.config.task_id or 0
      if self._delay_workers_by_global_step:
        # Wait 5500 global steps for the second worker. Each worker waits more
        # then previous one but with a diminishing number of steps.
        extra_hooks.append(
            basic_session_run_hooks.GlobalStepWaiterHook(
                int(8000.0 * math.log(task_id + 1))))
        delay_secs = 0
      else:
        # Wait 5 secs more for each new worker up to 60 secs.
        delay_secs = min(60, task_id * 5)

    if delay_secs > 0:
      elapsed_secs = time.time() - start
      remaining = delay_secs - elapsed_secs
      logging.info("Waiting %d secs before starting training.", remaining)
      time.sleep(delay_secs)

    return self._call_train(input_fn=self._train_input_fn,
                            max_steps=self._train_steps,
                            hooks=self._train_monitors + extra_hooks)

  def evaluate(self, delay_secs=None, name=None):
    """Evaluate on the evaluation data.

    Runs evaluation on the evaluation data and returns the result. Runs for
    `self._eval_steps` steps, or if it's `None`, then run until input is
    exhausted or another exception is raised. Start the evaluation after
    `delay_secs` seconds, or if it's `None`, defaults to using
    `self._eval_delay_secs` seconds.

    Args:
      delay_secs: Start evaluating after this many seconds. If `None`, defaults
        to using `self._eval_delays_secs`.
      name: Gives the name to the evauation for the case multiple evaluation is
        run for the same experiment.

    Returns:
      The result of the `evaluate` call to the `Estimator`.
    """
    if delay_secs is None:
      delay_secs = self._eval_delay_secs

    if delay_secs:
      logging.info("Waiting %d secs before starting eval.", delay_secs)
      time.sleep(delay_secs)

    return self._call_evaluate(input_fn=self._eval_input_fn,
                               steps=self._eval_steps,
                               metrics=self._eval_metrics,
                               name=(name or "one_pass"),
                               hooks=self._eval_hooks)

  @deprecated(
      "2016-10-23",
      "local_run will be renamed to train_and_evaluate and the new default "
      "behavior will be to run evaluation every time there is a new "
      "checkpoint.")
  def local_run(self):
    with _new_attr_context(self, "_min_eval_frequency"):
      self._min_eval_frequency = self._local_eval_frequency
      return self.train_and_evaluate()

  # TODO(xiejw): Allow continuous_eval_predicate_fn to be passed via constructor
  # once stopping all jobs is implemented.
  def _continuous_eval(self,
                       input_fn,
                       name,
                       delay_secs,
                       throttle_delay_secs,
                       evaluate_checkpoint_only_once=True,
                       continuous_eval_predicate_fn=None):
    """Run continuous eval.

    Runs infinite eval on the evaluation data set. This function starts
    evaluating after `delay_secs` seconds and then runs no more than one
    evaluation (with `self._eval_steps` steps each time) per
    `throttle_delay_secs`. If `train_steps` is not None, will return after
    global_step reaches `train_steps`.

    Args:
      input_fn: The input to use for this eval.
      name: A string appended to the folder name of evaluation results.
      delay_secs: Start evaluating after this many seconds. If None, defaults to
        self._eval_delay_secs.
      throttle_delay_secs: Do not re-evaluate unless the last evaluation was
        started at least this many seconds ago. If None, defaults to
        self._continuous_eval_throttle_secs.
      evaluate_checkpoint_only_once: Whether to skip evaluation of checkpoints
        that have already been evaluated. Default is `True`.
      continuous_eval_predicate_fn: A predicate function determining whether to
        continue eval after each iteration. `predicate_fn` takes the evaluation
        results as arguments. At the beginning of evaluation, the passed eval
        results will be None so it's expected that the predicate function
        handles that gracefully. When `predicate_fn` is not specified,
        continuous eval will run in an infinite loop (if `train_steps` is None)
        or exit once global step reaches `train_steps`.

    Raises:
      ValueError: if `continuous_eval_predicate_fn` is neither None nor
        callable.
    """
    if (continuous_eval_predicate_fn is not None and
        not callable(continuous_eval_predicate_fn)):
      raise ValueError(
          "`continuous_eval_predicate_fn` must be a callable, or None.")

    if delay_secs is None:
      delay_secs = self._eval_delay_secs
    if throttle_delay_secs is None:
      throttle_delay_secs = self._continuous_eval_throttle_secs

    if delay_secs:
      logging.info("Waiting %f secs before starting eval.", delay_secs)
      time.sleep(delay_secs)

    previous_path = None
    eval_result = None
    last_warning_time = 0
    while (not continuous_eval_predicate_fn or
           continuous_eval_predicate_fn(eval_result)):
      # Exit if we have already reached number of steps to train.
      if self._has_training_stopped(eval_result):
        logging.info("Exiting continuous eval, global_step=%s >= "
                     "train_step=%s",
                     eval_result[ops.GraphKeys.GLOBAL_STEP],
                     self._train_steps)
        return

      start = time.time()

      error_msg = None
      latest_path = saver.latest_checkpoint(self._estimator.model_dir)
      if not latest_path:
        error_msg = ("Estimator is not fitted yet. "
                     "Will start an evaluation when a checkpoint is ready.")
      elif evaluate_checkpoint_only_once and latest_path == previous_path:
        error_msg = "No new checkpoint ready for evaluation."

      if error_msg:
        # Print warning message every 10 mins.
        eval_result = {}
        if time.time() - last_warning_time > 600:
          logging.warning(error_msg)
          last_warning_time = time.time()
      else:
        eval_result = self._call_evaluate(input_fn=input_fn,
                                          steps=self._eval_steps,
                                          metrics=self._eval_metrics,
                                          name=name,
                                          checkpoint_path=latest_path,
                                          hooks=self._eval_hooks)
        # Ensure eval result is not None for next round of evaluation.
        if not eval_result:
          eval_result = {}

        self._maybe_export(eval_result, checkpoint_path=latest_path)

        # Clear warning timer and update last evaluated checkpoint
        last_warning_time = 0
        previous_path = latest_path

      duration = time.time() - start
      if duration < throttle_delay_secs:
        difference = throttle_delay_secs - duration
        logging.info("Waiting %f secs before starting next eval run.",
                     difference)
        time.sleep(difference)

  def _has_training_stopped(self, eval_result):
    """Determines whether the training has stopped."""
    if not eval_result:
      return False

    global_step = eval_result.get(ops.GraphKeys.GLOBAL_STEP)
    return global_step and self._train_steps and (
        global_step >= self._train_steps)

  def continuous_eval(self,
                      delay_secs=None,
                      throttle_delay_secs=None,
                      evaluate_checkpoint_only_once=True,
                      continuous_eval_predicate_fn=None):
    self._continuous_eval(
        self._eval_input_fn,
        name="continuous",
        delay_secs=delay_secs,
        throttle_delay_secs=throttle_delay_secs,
        evaluate_checkpoint_only_once=evaluate_checkpoint_only_once,
        continuous_eval_predicate_fn=continuous_eval_predicate_fn)

  def continuous_eval_on_train_data(self,
                                    delay_secs=None,
                                    throttle_delay_secs=None,
                                    continuous_eval_predicate_fn=None):
    self._continuous_eval(
        self._train_input_fn,
        name="continuous_on_train_data",
        delay_secs=delay_secs,
        throttle_delay_secs=throttle_delay_secs,
        continuous_eval_predicate_fn=continuous_eval_predicate_fn)

  def train_and_evaluate(self):
    """Interleaves training and evaluation.

    The frequency of evaluation is controlled by the constructor arg
    `min_eval_frequency`. When this parameter is 0, evaluation happens
    only after training has completed. Note that evaluation cannot happen
    more frequently than checkpoints are taken. If no new snapshots are
    available when evaluation is supposed to occur, then evaluation doesn't
    happen for another `min_eval_frequency` steps (assuming a checkpoint is
    available at that point). Thus, settings `min_eval_frequency` to 1 means
    that the model will be evaluated everytime there is a new checkpoint.

    This is particular useful for a "Master" task in the cloud, whose
    responsibility it is to take checkpoints, evaluate those checkpoints,
    and write out summaries. Participating in training as the supervisor
    allows such a task to accomplish the first and last items, while
    performing evaluation allows for the second.

    Returns:
      The result of the `evaluate` call to the `Estimator` as well as the
      export results using the specified `ExportStrategy`.
    """
    # The directory to which evaluation summaries are written are determined
    # by adding a suffix to 'eval'; that suffix is the 'name' parameter to
    # the various evaluate(...) methods. By setting it to None, we force
    # the directory name to simply be 'eval'.
    eval_dir_suffix = None

    # We set every_n_steps to 1, but evaluation only occurs when a new
    # snapshot is available. If, by the time we finish evaluation
    # there is a new snapshot, then we just evaluate again. Otherwise,
    # we keep training until one becomes available.
    with _new_attr_context(self, "_train_monitors"):
      self._train_monitors = self._train_monitors or []
      if self._min_eval_frequency:
        self._train_monitors += [monitors.ValidationMonitor(
            input_fn=self._eval_input_fn, eval_steps=self._eval_steps,
            metrics=self._eval_metrics, every_n_steps=self._min_eval_frequency,
            name=eval_dir_suffix, hooks=self._eval_hooks
        )]
      self.train(delay_secs=0)

    eval_result = self._call_evaluate(input_fn=self._eval_input_fn,
                                      steps=self._eval_steps,
                                      metrics=self._eval_metrics,
                                      name=eval_dir_suffix,
                                      hooks=self._eval_hooks)
    export_results = self._maybe_export(eval_result)
    return eval_result, export_results

  @experimental
  def continuous_train_and_eval(self,
                                continuous_eval_predicate_fn=None):
    """Interleaves training and evaluation.

    The frequency of evaluation is controlled by the `train_steps_per_iteration`
    (via constructor). The model will be first trained for
    `train_steps_per_iteration`, and then be evaluated in turns.

    This method is intended for single machine usage.

    This differs from `train_and_evaluate` as follows:
      1. The procedure will have train and evaluation in turns. The model
      will be trained for a number of steps (usually smaller than `train_steps`
      if provided) and then be evaluated.  `train_and_evaluate` will train the
      model for `train_steps` (no small training iterations).

      2. Due to the different approach this schedule takes, it leads to two
      differences in resource control. First, the resources (e.g., memory) used
      by training will be released before evaluation (`train_and_evaluate` takes
      double resources). Second, more checkpoints will be saved as a checkpoint
      is generated at the end of each training iteration.

      3. As the estimator.train starts from scratch (new graph, new states for
      input, etc) at each iteration, it is recommended to have the
      `train_steps_per_iteration` larger. It is also recommended to shuffle your
      input.

    Args:
      continuous_eval_predicate_fn: A predicate function determining whether to
        continue after each iteration. `predicate_fn` takes the evaluation
        results as its arguments. At the beginning of evaluation, the passed
        eval results will be None so it's expected that the predicate function
        handles that gracefully. When `predicate_fn` is not specified, this will
        run in an infinite loop or exit when global_step reaches `train_steps`.

    Returns:
      A tuple of the result of the `evaluate` call to the `Estimator` and the
      export results using the specified `ExportStrategy`.

    Raises:
      ValueError: if `continuous_eval_predicate_fn` is neither None nor
        callable.
    """

    if (continuous_eval_predicate_fn is not None and
        not callable(continuous_eval_predicate_fn)):
      raise ValueError(
          "`continuous_eval_predicate_fn` must be a callable, or None.")

    eval_result = None

    # Set the default value for train_steps_per_iteration, which will be
    # overridden by other settings.
    train_steps_per_iteration = 1000
    if self._train_steps_per_iteration is not None:
      train_steps_per_iteration = self._train_steps_per_iteration
    elif self._train_steps is not None:
      train_steps_per_iteration = int(self._train_steps / 10)

    while (not continuous_eval_predicate_fn or
           continuous_eval_predicate_fn(eval_result)):

      if self._has_training_stopped(eval_result):
        # Exits once max steps of training is satisfied.
        logging.info("Stop training model as max steps reached")
        break

      logging.info("Training model for %s steps", train_steps_per_iteration)
      self._call_train(input_fn=self._train_input_fn,
                       steps=train_steps_per_iteration,
                       hooks=self._train_monitors)

      logging.info("Evaluating model now.")
      eval_result = self._call_evaluate(input_fn=self._eval_input_fn,
                                        steps=self._eval_steps,
                                        metrics=self._eval_metrics,
                                        name="one_pass",
                                        hooks=self._eval_hooks)

    return eval_result, self._maybe_export(eval_result)

  def _maybe_export(self, eval_result, checkpoint_path=None):
    """Export the Estimator using export_fn, if defined."""
    export_dir_base = os.path.join(
        compat.as_bytes(self._estimator.model_dir),
        compat.as_bytes("export"))

    export_results = []
    for strategy in self._export_strategies:
      export_results.append(
          strategy.export(
              self._estimator,
              os.path.join(
                  compat.as_bytes(export_dir_base),
                  compat.as_bytes(strategy.name)),
              checkpoint_path=checkpoint_path,
              eval_result=eval_result))

    return export_results

  def run_std_server(self):
    """Starts a TensorFlow server and joins the serving thread.

    Typically used for parameter servers.

    Raises:
      ValueError: if not enough information is available in the estimator's
        config to create a server.
    """
    self._start_server().join()

  def test(self):
    """Tests training, evaluating and exporting the estimator for a single step.

    Returns:
      The result of the `evaluate` call to the `Estimator`.
    """
    self._call_train(input_fn=self._train_input_fn,
                     steps=1,
                     hooks=self._train_monitors)

    eval_result = self._call_evaluate(input_fn=self._eval_input_fn,
                                      steps=1,
                                      metrics=self._eval_metrics,
                                      name="one_pass")
    _ = self._maybe_export(eval_result)

    return eval_result

  def _start_server(self):
    """Creates, starts, and returns a server_lib.Server."""
    config = self._estimator.config
    if (not config.cluster_spec or not config.task_type or not config.master or
        config.task_id is None):
      raise ValueError("Could not start server; be sure to specify "
                       "cluster_spec, task_type, master, and task in "
                       "RunConfig or set the TF_CONFIG environment variable.")
    server = server_lib.Server(
        config.cluster_spec,
        job_name=config.task_type,
        task_index=config.task_id,
        config=config.tf_config,
        start=False)
    server.start()
    return server

  def _call_train(self, _sentinel=None,  # pylint: disable=invalid-name,
                  input_fn=None, steps=None, hooks=None, max_steps=None):
    if _sentinel is not None:
      raise ValueError("_call_train should be called with keyword args only")

    # Estimator in core cannot work with monitors. We need to convert them
    # to hooks. For Estimator in contrib, it is converted internally. So, it is
    # safe to convert for both cases.
    hooks = monitors.replace_monitors_with_hooks(hooks, self._estimator)
    if self._core_estimator_used:
      return self._estimator.train(input_fn=input_fn,
                                   steps=steps,
                                   max_steps=max_steps,
                                   hooks=hooks)
    else:
      return self._estimator.fit(input_fn=input_fn,
                                 steps=steps,
                                 max_steps=max_steps,
                                 monitors=hooks)

  def _call_evaluate(self, _sentinel=None,  # pylint: disable=invalid-name,
                     input_fn=None, steps=None, metrics=None, name=None,
                     checkpoint_path=None, hooks=None):
    if _sentinel is not None:
      raise ValueError("_call_evaluate should be called with keyword args only")

    if self._core_estimator_used:
      if metrics is not None:
        raise ValueError(
            "`eval_metrics` must be `None` with `tf.estimator.Estimator`")
      return self._estimator.evaluate(input_fn=input_fn,
                                      steps=steps,
                                      name=name,
                                      checkpoint_path=checkpoint_path,
                                      hooks=hooks)
    else:
      return self._estimator.evaluate(input_fn=input_fn,
                                      steps=steps,
                                      metrics=metrics,
                                      name=name,
                                      checkpoint_path=checkpoint_path,
                                      hooks=hooks)


@contextlib.contextmanager
def _new_attr_context(obj, attr):
  """Creates a new context in which an object's attribute can be changed.

  This creates a context in which an object's attribute can be changed.
  Once the context is exited, the attribute reverts to its original value.

  Args:
    obj: An object whose attribute to restore at the end of the context.
    attr: An attribute to remember and restore at the end of the context.

  Yields:
    Context.

  Example:
    my_obj.x = 1
    with _new_attr_context(my_obj, "x"):
      my_obj.x = 2
      print(my_obj.x)
    print(my_obj.x)
  """
  saved = getattr(obj, attr)
  try:
    yield
  finally:
    setattr(obj, attr, saved)


def _is_gcs(model_dir):
  return model_dir and model_dir.startswith("gs://")
