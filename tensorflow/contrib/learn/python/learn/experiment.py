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

import time

from tensorflow.contrib.learn.python.learn import evaluable
from tensorflow.contrib.learn.python.learn import monitors
from tensorflow.contrib.learn.python.learn import trainable
from tensorflow.contrib.learn.python.learn.estimators._sklearn import NotFittedError
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib


__all__ = ["Experiment"]


class Experiment(object):
  """Experiment is a class containing all information needed to train a model.

  After an experiment is created (by passing an Estimator and inputs for
  training and evaluation), an Experiment instance knows how to invoke training
  and eval loops in a sensible fashion for distributed training.
  """

  def __init__(self,
               estimator,
               train_input_fn,
               eval_input_fn,
               eval_metrics=None,
               train_steps=None,
               eval_steps=100,
               train_monitors=None,
               local_eval_frequency=None,
               eval_delay_secs=120,
               continuous_eval_throttle_secs=60):
    """Constructor for `Experiment`.

    Creates an Experiment instance. None of the functions passed to this
    constructor are executed at construction time. They are stored and used
    when a method is executed which requires it.

    Args:
      estimator: Object implementing `Trainable` and `Evaluable`.
      train_input_fn: function, returns features and targets for training.
      eval_input_fn: function, returns features and targets for evaluation. If
        `eval_steps` is `None`, this should be configured only to produce for a
        finite number of batches (generally, 1 epoch over the evaluation data).
      eval_metrics: `dict` of string, metric function. If `None`, default set
        is used.
      train_steps: Perform this many steps of training. `None`, the default,
        means train forever.
      eval_steps: `evaluate` runs until input is exhausted (or another exception
        is raised), or for `eval_steps` steps, if specified.
      train_monitors: A list of monitors to pass to the `Estimator`'s `fit`
        function.
      local_eval_frequency: Frequency of running eval in steps,
        when running locally. If `None`, runs evaluation only at the end of
        training.
      eval_delay_secs: Start evaluating after waiting for this many seconds.
      continuous_eval_throttle_secs: Do not re-evaluate unless the last
        evaluation was started at least this many seconds ago for
        continuous_eval().

    Raises:
      ValueError: if `estimator` does not implement `Evaluable` and `Trainable`.
    """
    if not isinstance(estimator, evaluable.Evaluable):
      raise ValueError("`estimator` must implement `Evaluable`.")
    if not isinstance(estimator, trainable.Trainable):
      raise ValueError("`estimator` must implement `Trainable`.")
    super(Experiment, self).__init__()
    self._estimator = estimator
    self._train_input_fn = train_input_fn
    self._eval_input_fn = eval_input_fn
    self._eval_metrics = eval_metrics
    self._train_steps = train_steps
    self._eval_steps = eval_steps
    self._train_monitors = train_monitors
    self._local_eval_frequency = local_eval_frequency
    self._eval_delay_secs = eval_delay_secs
    self._continuous_eval_throttle_secs = continuous_eval_throttle_secs

  @property
  def estimator(self):
    return self._estimator

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
    if delay_secs is None:
      task_id = self._estimator.config.task or 0
      delay_secs = min(60, task_id * 5)

    # Start the server, if needed.
    if self._estimator.config.cluster_spec:
      self._start_server()

    if delay_secs:
      elapsed_secs = time.time() - start
      remaining = delay_secs - elapsed_secs
      logging.info("Waiting %d secs before starting training.", remaining)
      time.sleep(delay_secs)

    return self._estimator.fit(input_fn=self._train_input_fn,
                               max_steps=self._train_steps,
                               monitors=self._train_monitors)

  def evaluate(self, delay_secs=None):
    """Evaluate on the evaluation data.

    Runs evaluation on the evaluation data and returns the result. Runs for
    `self._eval_steps` steps, or if it's `None`, then run until input is
    exhausted or another exception is raised. Start the evaluation after
    `delay_secs` seconds, or if it's `None`, defaults to using
    `self._eval_delay_secs` seconds.

    Args:
      delay_secs: Start evaluating after this many seconds. If `None`, defaults
        to using `self._eval_delays_secs`.

    Returns:
      The result of the `evaluate` call to the `Estimator`.
    """
    if delay_secs is None:
      delay_secs = self._eval_delay_secs

    if delay_secs:
      logging.info("Waiting %d secs before starting eval.", delay_secs)
      time.sleep(delay_secs)

    return self._estimator.evaluate(input_fn=self._eval_input_fn,
                                    steps=self._eval_steps,
                                    metrics=self._eval_metrics,
                                    name="one_pass")

  def local_run(self):
    """Run when called on local machine.

    Returns:
      The result of the `evaluate` call to the `Estimator`.
    """
    self._train_monitors = self._train_monitors or []
    if self._local_eval_frequency:
      self._train_monitors += [monitors.ValidationMonitor(
          input_fn=self._eval_input_fn, eval_steps=self._eval_steps,
          metrics=self._eval_metrics, every_n_steps=self._local_eval_frequency
      )]
    self.train(delay_secs=0)
    return self.evaluate(delay_secs=0)

  def _continuous_eval(self,
                       input_fn,
                       name,
                       delay_secs,
                       throttle_delay_secs):
    """Run continuous eval.

    Runs infinite eval on the evaluation data set. This function starts
    evaluating after `delay_secs` seconds and then runs no more than one
    evaluation (with `self._eval_steps` steps each time) per
    `throttle_delay_secs`. It never returns.

    Args:
      input_fn: The input to use for this eval.
      name: A string appended to the folder name of evaluation results.
      delay_secs: Start evaluating after this many seconds. If None, defaults to
        self._eval_delay_secs.
      throttle_delay_secs: Do not re-evaluate unless the last evaluation was
        started at least this many seconds ago. If None, defaults to
        self._continuous_eval_throttle_secs.
    """
    if delay_secs is None:
      delay_secs = self._eval_delay_secs
    if throttle_delay_secs is None:
      throttle_delay_secs = self._continuous_eval_throttle_secs

    if delay_secs:
      logging.info("Waiting %f secs before starting eval.", delay_secs)
      time.sleep(delay_secs)

    while True:
      start = time.time()
      try:
        self._estimator.evaluate(input_fn=input_fn,
                                 steps=self._eval_steps,
                                 metrics=self._eval_metrics,
                                 name=name)
      except NotFittedError:
        logging.warning("Estimator is not fitted yet, skipping evaluation.")
      duration = time.time() - start
      if duration < throttle_delay_secs:
        difference = throttle_delay_secs - duration
        logging.info("Waiting %f secs before starting next eval run.",
                     difference)
        time.sleep(difference)

  def continuous_eval(self, delay_secs=None, throttle_delay_secs=None):
    self._continuous_eval(self._eval_input_fn,
                          name="continuous",
                          delay_secs=delay_secs,
                          throttle_delay_secs=throttle_delay_secs)

  def continuous_eval_on_train_data(self,
                                    delay_secs=None,
                                    throttle_delay_secs=None):
    self._continuous_eval(self._train_input_fn,
                          name="continuous_on_train_data",
                          delay_secs=delay_secs,
                          throttle_delay_secs=throttle_delay_secs)

  def run_std_server(self):
    """Starts a TensorFlow server and joins the serving thread.

    Typically used for parameter servers.

    Raises:
      ValueError: if not enough information is available in the estimator's
        config to create a server.
    """
    self._start_server().join()

  def test(self):
    """Tests training and evaluating the estimator both for a single step.

    Returns:
      The result of the `evaluate` call to the `Estimator`.
    """
    self._estimator.fit(input_fn=self._train_input_fn,
                        steps=1,
                        monitors=self._train_monitors)

    return self._estimator.evaluate(input_fn=self._eval_input_fn,
                                    steps=1,
                                    metrics=self._eval_metrics,
                                    name="one_pass")

  def _start_server(self):
    """Creates, starts, and returns a server_lib.Server."""
    config = self._estimator.config
    if (not config.cluster_spec or not config.job_name or not config.master or
        config.task is None):
      raise ValueError("Could not start server; be sure to specify "
                       "cluster_spec, job_name, master, and task in "
                       "RunConfig or set the TF_CONFIG environment variable.")
    server = server_lib.Server(
        config.cluster_spec,
        job_name=config.job_name,
        task_index=config.task,
        config=config.tf_config,
        start=False)
    server.start()
    return server
