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
"""Some useful session run hooks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.estimator import estimator as estimator_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import training


# pylint: disable=protected-access
class InMemoryEvaluatorHook(training.SessionRunHook):
  """Hook to run evaluation in training without a checkpoint.

  Example:

  ```python
  def train_input_fn():
    ...
    return train_dataset

  def eval_input_fn():
    ...
    return eval_dataset

  estimator = tf.estimator.DNNClassifier(...)

  evaluator = tf.contrib.estimator.InMemoryEvaluatorHook(
      estimator, eval_input_fn)
  estimator.train(train_input_fn, hooks=[evaluator])
  ```

  Current limitations of this approach are:
  * It doesn't support multi-node distributed mode.
  * It doesn't support saveable objects other than variables (such as boosted
    tree support)
  * It doesn't support custom saver logic (such as ExponentialMovingAverage
    support)

  """

  def __init__(self,
               estimator,
               input_fn,
               steps=None,
               hooks=None,
               name=None,
               every_n_iter=100):
    """Initializes a `InMemoryEvaluatorHook`.

    Args:
      estimator: A `tf.estimator.Estimator` instance to call evaluate.
      input_fn:  Equivalent to the `input_fn` arg to `estimator.evaluate`. A
        function that constructs the input data for evaluation.
        See @{$get_started/premade_estimators#create_input_functions} for more
        information. The function should construct and return one of
        the following:

          * A 'tf.data.Dataset' object: Outputs of `Dataset` object must be a
            tuple (features, labels) with same constraints as below.
          * A tuple (features, labels): Where `features` is a `Tensor` or a
            dictionary of string feature name to `Tensor` and `labels` is a
            `Tensor` or a dictionary of string label name to `Tensor`. Both
            `features` and `labels` are consumed by `model_fn`. They should
            satisfy the expectation of `model_fn` from inputs.

      steps: Equivalent to the `steps` arg to `estimator.evaluate`.  Number of
        steps for which to evaluate model. If `None`, evaluates until `input_fn`
        raises an end-of-input exception.
      hooks: Equivalent to the `hooks` arg to `estimator.evaluate`. List of
        `SessionRunHook` subclass instances. Used for callbacks inside the
        evaluation call.
      name:  Equivalent to the `name` arg to `estimator.evaluate`. Name of the
        evaluation if user needs to run multiple evaluations on different data
        sets, such as on training data vs test data. Metrics for different
        evaluations are saved in separate folders, and appear separately in
        tensorboard.
      every_n_iter: `int`, runs the evaluator once every N training iteration.

    Raises:
      ValueError: if `every_n_iter` is non-positive or it's not a single machine
        training
    """
    if every_n_iter is None or every_n_iter <= 0:
      raise ValueError('invalid every_n_iter=%s.' % every_n_iter)
    if (estimator.config.num_ps_replicas > 0 or
        estimator.config.num_worker_replicas > 1):
      raise ValueError(
          'InMemoryEvaluator supports only single machine (aka Local) setting.')
    self._estimator = estimator
    self._input_fn = input_fn
    self._steps = steps
    self._name = name
    self._every_n_iter = every_n_iter
    self._eval_dir = os.path.join(self._estimator.model_dir, 'eval'
                                  if not name else 'eval_' + name)

    self._graph = None
    self._hooks = estimator_lib._check_hooks_type(hooks)
    self._hooks.extend(self._estimator._convert_eval_steps_to_hooks(steps))
    self._timer = training.SecondOrStepTimer(every_steps=every_n_iter)

  def begin(self):
    """Build eval graph and restoring op."""
    self._timer.reset()
    self._iter_count = 0
    self._graph = ops.Graph()
    with self._graph.as_default():
      (self._scaffold, self._update_op, self._eval_dict,
       self._all_hooks) = self._estimator._evaluate_build_graph(
           self._input_fn, self._hooks, checkpoint_path=None)

      if self._scaffold.saver is not None:
        raise ValueError('InMemoryEvaluator does not support custom saver')
      if self._scaffold.init_fn is not None:
        raise ValueError('InMemoryEvaluator does not support custom init_fn')

      self._var_name_to_eval_var = {
          v.name: v for v in ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      }
      self._var_name_to_placeholder = {
          v.name: array_ops.placeholder(v.dtype)
          for v in ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      }

  def after_create_session(self, session, coord):  # pylint: disable=unused-argument
    """Does first run which shows the eval metrics before training."""
    if ops.get_collection(ops.GraphKeys.SAVEABLE_OBJECTS):
      raise ValueError(
          'InMemoryEvaluator does not support saveables other than global '
          'variables.')
    self._var_name_to_train_var = {
        v.name: v for v in ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    }
    var_names_to_transfer = set(self._var_name_to_placeholder.keys()) & set(
        self._var_name_to_train_var.keys())
    # Filter training var names that are not exist in evaluation
    self._var_name_to_train_var = {
        v_name: self._var_name_to_train_var[v_name]
        for v_name in var_names_to_transfer
    }
    # Filter eval var names that are not exist in training
    self._var_name_to_eval_var = {
        v_name: self._var_name_to_eval_var[v_name]
        for v_name in var_names_to_transfer
    }

    with self._graph.as_default():
      self._var_feed_op = control_flow_ops.group([
          state_ops.assign(self._var_name_to_eval_var[v_name],
                           self._var_name_to_placeholder[v_name])
          for v_name in var_names_to_transfer
      ])

    self._evaluate(session)

  def _evaluate(self, train_session):
    var_name_to_value = train_session.run(self._var_name_to_train_var)
    placeholder_to_value = {
        self._var_name_to_placeholder[v_name]: var_name_to_value[v_name]
        for v_name in var_name_to_value
    }

    def feed_variables(scaffold, session):
      del scaffold
      session.run(self._var_feed_op, feed_dict=placeholder_to_value)

    scaffold = training.Scaffold(
        init_fn=feed_variables, copy_from_scaffold=self._scaffold)

    with self._graph.as_default():
      return self._estimator._evaluate_run(
          checkpoint_path=None,
          scaffold=scaffold,
          update_op=self._update_op,
          eval_dict=self._eval_dict,
          all_hooks=self._all_hooks,
          output_dir=self._eval_dir)

    self._timer.update_last_triggered_step(self._iter_count)

  def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
    """Runs evaluator."""
    self._iter_count += 1
    if self._timer.should_trigger_for_step(self._iter_count):
      self._evaluate(run_context.session)

  def end(self, session):  # pylint: disable=unused-argument
    """Runs evaluator for final model."""
    self._evaluate(session)


# pylint: enable=protected-access
