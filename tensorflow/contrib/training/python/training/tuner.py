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

"""The Tuner interface for hyper-parameters tuning."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from tensorflow.contrib.framework.python.framework import experimental


class Tuner(object):
  """Tuner class is the interface for Experiment hyper-parameters tuning.

  Example:
  ```
    def _create_my_experiment(config, hparams):
      hidden_units = [hparams.unit_per_layer] * hparams.num_hidden_layers

      return tf.contrib.learn.Experiment(
          estimator=DNNClassifier(config=config, hidden_units=hidden_units),
          train_input_fn=my_train_input,
          eval_input_fn=my_eval_input)

    tuner = create_tuner(study_configuration, objective_key)

    learn_runner.tune(experiment_fn=_create_my_experiment, tuner)
  """

  __metaclass__ = abc.ABCMeta

  @experimental
  @abc.abstractmethod
  def next_trial(self):
    """Switch to the next trial.

    Ask the tuning service for a new trial for hyper-parameters tuning.

    Returns:
      A boolean indicating if a trial was assigned to the tuner.

    Raises:
      RuntimeError: If the tuner is initialized correctly.
    """
    raise NotImplementedError("Calling an abstract method.")

  @experimental
  @abc.abstractmethod
  def run_experiment(self, experiment_fn):
    """Creates an Experiment by calling `experiment_fn` and executes it.

    It creates a `RunConfig`, which captures the current execution environment
    configuration and retrieves the hyper-parameters for current trial from the
    tuning service. Both are passed to the `experiment_fn` and used to create
    the Experiment for current trial execution. When finished, the measure will
    be reported to the tuning service.


    If the `RunConfig` does not include a task type, then an exception is
    raised. The task type should be one of the types supported by the tuner. If
    tuner does not support the task type directly, it could delegate the task to
    Experiment, which is usually a function of Experiment. An exception would be
    raised, if neither tuner nor Experiment could support the task type.

    Args:
      experiment_fn: A function that creates an `Experiment`. It should accept
        an argument `config` which should be used to create the `Estimator`
        (passed as `config` to its constructor), and an argument `hparams`,
        which should be used for hyper-parameters tuning. It must return an
        `Experiment`.
    """
    raise NotImplementedError("Calling an abstract method.")
