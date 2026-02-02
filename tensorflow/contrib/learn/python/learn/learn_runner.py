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

from tensorflow.contrib.learn.python.learn import runner_flags  # pylint: disable=unused-import
from tensorflow.contrib.learn.python.learn.experiment import Experiment
from tensorflow.python.platform import flags
from tensorflow.python.platform import tf_logging as logging


FLAGS = flags.FLAGS


def run(experiment_fn):
  """Make and run an experiment.

  It creates an Experiment by calling experiment_fn. It reads a flag `schedule`.
    Then it calls the function named as `schedule` of the Experiment.

  Example:
  ```
    def _create_my_experiment(output_dir):
        return tf.contrib.learn.Experiment(
          estimator=my_estimator(model_dir=output_dir),
          train_input_fn=my_train_input,
          eval_input_fn=my_eval_input)

    learn_runner.run(experiment_fn=_create_my_experiment)
  ```
  Args:
    experiment_fn: A function that creates an `Experiment`. It should accept an
      argument `output_dir` which should be used to create the Estimator (passed
      as `model_dir` to its constructor). It must return an Experiment.

  Returns:
    The return value of function `schedule`.

  Raises:
    RuntimeError: If flags `output_dir` or `schedule` is not specified.
    ValueError: `schedule` doesn't references a member of `Experiment`.
    TypeError: `schedule` references non-callable member.
  """

  if not FLAGS.output_dir:
    raise RuntimeError('Must specify an output directory (use --output_dir).')
  if not FLAGS.schedule:
    raise RuntimeError('Must specify a schedule (use --schedule).')

  if not callable(experiment_fn):
    raise TypeError('Experiment builder "%s" is not callable.' %
                    experiment_fn)

  # Call the builder
  experiment = experiment_fn(output_dir=FLAGS.output_dir)
  if not isinstance(experiment, Experiment):
    raise TypeError('Experiment builder did not return an Experiment '
                    'instance, got %s instead.' % type(experiment))

  # Execute the schedule
  taskname = FLAGS.schedule
  if not hasattr(experiment, taskname):
    logging.error('Schedule references non-existent task %s', taskname)
    valid_tasks = [x for x in experiment.__dict__
                   if callable(getattr(experiment, x))]
    logging.error('Allowed values for this experiment are: %s', valid_tasks)
    raise ValueError('Schedule references non-existent task %s', taskname)

  task = getattr(experiment, taskname)
  if not callable(task):
    logging.error('Schedule references non-callable member %s', taskname)
    valid_tasks = [
        x for x in experiment.__dict__
        if callable(getattr(experiment, x)) and not x.startswith('_')
    ]
    logging.error('Allowed values for this experiment are: %s', valid_tasks)
    raise TypeError('Schedule references non-callable member %s', taskname)

  return task()
