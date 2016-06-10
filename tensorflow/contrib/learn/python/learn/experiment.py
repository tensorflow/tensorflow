#  Copyright 2016 Google Inc. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Experiment class collecting information needed for a single training run."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from tensorflow.python.platform import tf_logging as logging


class Experiment(object):
  """Experiment is a class containing all information needed to train a model.
  """

  def __init__(self,
               estimator,
               train_input_fn,
               eval_input_fn,
               eval_metrics=None,
               train_steps=None,
               eval_steps=100,
               train_monitors=None,
               local_train_step=1000):
    """Constructor for Experiment.

    Args:
      estimator: `Estimator` object.
      train_input_fn: function, returns features and targets for training.
      eval_input_fn: function, returns features and targets for evaluation.
      eval_metrics: `dict` of string, metric function. If `None`, default set
        is used.
      train_steps: Perform this many steps of training. `None`, the default,
        means train forever.
      eval_steps: Run this many steps of evaluation.
      train_monitors: A list of monitors to pass to the `Estimator`'s `fit`
        function.
      local_train_step: Perform this many training steps when running local.
    """
    super(Experiment, self).__init__()
    self._estimator = estimator
    self._train_input_fn = train_input_fn
    self._eval_input_fn = eval_input_fn
    self._eval_metrics = eval_metrics
    self._train_steps = train_steps
    self._eval_steps = eval_steps
    self._train_monitors = train_monitors
    self._local_train_step = local_train_step

  def train(self, delay_secs=0):
    """Fit the estimator using the training data.

    Train the estimator for `steps` steps, after waiting for `delay_secs`
    seconds. If `steps` is `None`, train forever.

    Args:
      delay_secs: Start training after this many seconds.

    Returns:
      The trained estimator.
    """

    if delay_secs:
      logging.info("Waiting %d secs before starting training.", delay_secs)
      time.sleep(delay_secs)

    return self._estimator.fit(input_fn=self._train_input_fn,
                               steps=self._train_steps,
                               monitors=self._train_monitors)

  def evaluate(self, delay_secs=0):
    """Evaluate on the evaluation data.

    Runs evaluation on the evaluation data and returns the result. If `steps`
    is given, only run for this many steps. Start the evaluation after
    `delay_secs` seconds.

    Args:
      delay_secs: Start evaluating after waiting for this many seconds.

    Returns:
      The result of the `evaluate` call to the `Estimator`.
    """

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
    orig_train_steps = self._train_steps
    self._train_steps = self._local_train_step
    self.train()
    self._train_steps = orig_train_steps
    return self.evaluate()

  def _continuous_eval(self,
                       input_fn,
                       name,
                       delay_secs=0,
                       throttle_delay_secs=60):
    """Run continuous eval.

    Run `steps` steps of evaluation on the evaluation data set. This function
    starts evaluating after `delay_secs` seconds and then runs no more than one
    evaluation per `throttle_delay_secs`. It never returns.

    Args:
      input_fn: The input to use for this eval.
      name: A string appended to the folder name of evaluation results.
      delay_secs: Start evaluating after this many seconds.
      throttle_delay_secs: Do not re-evaluate unless the last evaluation was
        started at least this many seconds ago.
    """
    if delay_secs:
      logging.info("Waiting %f secs before starting eval.", delay_secs)
      time.sleep(delay_secs)

    while True:
      start = time.time()
      self._estimator.evaluate(input_fn=input_fn,
                               steps=self._eval_steps,
                               metrics=self._eval_metrics,
                               name=name)
      duration = time.time() - start
      if duration < throttle_delay_secs:
        difference = throttle_delay_secs - duration
        logging.info("Waiting %f secs before starting next eval run.",
                     difference)
        time.sleep(difference)

  def continuous_eval(self, delay_secs=0, throttle_delay_secs=60):
    self._continuous_eval(self._eval_input_fn,
                          name="continuous",
                          delay_secs=delay_secs,
                          throttle_delay_secs=throttle_delay_secs)

  def continuous_eval_on_train_data(self, delay_secs=0, throttle_delay_secs=60):
    self._continuous_eval(self._train_input_fn,
                          name="continuous_on_train_data",
                          delay_secs=delay_secs,
                          throttle_delay_secs=throttle_delay_secs)
