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
"""Class Evaluator holds Metrics for the duration of an evaluation run."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.contrib.eager.python import datasets
from tensorflow.contrib.eager.python import metrics
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops


class Evaluator(object):
  """This holds and updates Metrics for the duration of a single eval run.

  Usage:
    evaluator = my_model.evaluator() # or MyEvaluator(my_model)
    for example_batch in ...:
      evaluator(example_batch)
    results = evaluator.all_metric_results(optional_summary_writer)

  Or, if you are getting your examples from a tf.data.Dataset, you can use
  the evaluate_on_dataset() method.

  Implementers of Evaluators should
  (a) Call `track_metric()` and/or `track_evaluator()` in __init__().
  (b) Override the `call()` method. It will be passed the output of the
      model's `eval_data()` method, and should call its contained metrics
      (treating them as callables) and any child Evaluators (using their
      call() method to avoid calling eval_data() again).

  Args:
    model: A `Model` object with an `eval_data()` method.
  """

  def __init__(self, model):
    self._model = model
    self._metrics = {}
    self._evaluators = {}
    if context.in_graph_mode():
      self.call = function.defun(self.call)

  # ---- API for users ----
  def __call__(self, *args, **kwargs):
    """Update metrics with a minibatch of input examples.

    Args:
      *args:
      **kwargs: Arguments representing an input mini-batch of examples to
        pass to self.model.eval_data().

    Returns:
      The op to execute or None if executing eagerly.
    """
    return self.call(self._model.eval_data(*args, **kwargs))

  def init_variables(self):
    """Return an op for initializing all contained uninitialized variables.

    Only for graph execution. Should be called after variables are created
    in the first execution of __call__().

    Returns:
      An op.

    Raises:
      RuntimeError: if eager execution is enabled.

    @compatibility(eager)
    Only for graph execution.
    @end_compatibility
    """
    if context.in_eager_mode():
      raise RuntimeError("Evaluator.init_variables() not needed when "
                         "eager execution is enabled.")
    return control_flow_ops.group([m.init_variables() for _, m in self.metrics])

  def all_metric_results(self):  # TODO(josh11b): Add optional summary_writer.
    """Returns dict mapping metric name -> value."""
    results = {}
    for name, metric in six.iteritems(self._metrics):
      results[name] = metric.result()
    for prefix, evaluator in six.iteritems(self._evaluators):
      for name, metric in six.iteritems(evaluator._metrics):  # pylint: disable=protected-access
        results[prefix + "/" + name] = metric.result()
    return results

  def evaluate_on_dataset(self, dataset, *args, **kwargs):
    """Convenience method for performing an eval on a Dataset.

    Args:
      dataset: Dataset object with the input data to evaluate on.
      *args:
      **kwargs: Optional additional arguments to __call__().

    Returns:
      @compatibility(eager)
      When eager execution is enabled, this returns the result of performing
      an evaluation as a dictionary. With graph execution, this returns a tuple
      (init_op, call_op, results_op) which may be executed using this code:
      ```python
        sess.run(init_op)
        try:
          while True:
            sess.run(call_op)
        except tf.errors.OutOfRangeError:
          pass
        return sess.run(results_op)  # A dictionary

        # equivalently:
        return evaluator.run_evaluation(init_op, call_op, results_op, sess=sess)
      ```
      @end_compatibility
    """
    # TODO(josh11b): Add optional summary_writer.
    if context.in_graph_mode():
      call_op = self.__call__(dataset.make_one_shot_iterator().get_next(),
                              *args, **kwargs)
      init_op = self.init_variables()
      results_op = self.all_metric_results()
      return (init_op, call_op, results_op)
    # Eager case
    for example in datasets.Iterator(dataset):
      self.__call__(example, *args, **kwargs)
    return self.all_metric_results()

  @staticmethod
  def run_evaluation(init_op, call_op, results_op, sess=None):
    """Convenience method for running the ops returned by evaluate_on_dataset.

    Args:
      init_op: An op that initializes/resets evaluation state.
      call_op: An op that updates evaluation state on a mini-batch of examples.
        Must generate an tf.errors.OutOfRangeError when done.
      results_op: A dictionary of tensors that compute the final evaluation
        results from the evaulation state.
      sess: The Session to run the evaluation in. Defaults to the default
        Session.

    Returns:
      A dictionary of values, parallel to results_op.

    Raises:
      RuntimeError: if eager execution is enabled.

    @compatibility(eager)
    Only for graph execution.
    @end_compatibility
    """
    if context.in_eager_mode():
      raise RuntimeError("Evaluator.run_evaluation() not supported when "
                         "eager execution is enabled.")
    sess = sess or ops.get_default_session()
    sess.run(init_op)
    try:
      while True:
        sess.run(call_op)
    except errors_impl.OutOfRangeError:
      pass
    return sess.run(results_op)

  # ---- To be implemented by descendants ---
  def call(self, eval_data):
    """Update metrics using the output of self.model.

    Note: This function is executed as a graph function in graph mode.
    This means:
    a) Operations on the same resource are executed in textual order.
       This should make it easier to do things like add the updated
       value of a variable to another, for example.
    b) You don't need to worry about collecting the update ops to execute.
       All update ops added to the graph by this function will be executed.
    As a result, code should generally work the same way with graph or
    eager execution.

    Args:
      eval_data: The output of self.model.eval_data() on a mini-batch of
        examples.
    """
    raise NotImplementedError("Evaluators must define a call member function.")

  # ---- For use by descendants ---
  @property
  def model(self):
    return self._model

  def track_metric(self, metric):
    """Add a Metric to be tracked.

    Metrics can only be tracked by one `Evaluator`. Metrics must be
    tracked or they will not appear in `all_metric_results()`.

    Args:
      metric: A `Metric` object.

    Returns:
      The `metric` passed into this function.

    Raises:
      RuntimeError: If called before __init__.
      TypeError: If `metric` is not of the correct type.
      ValueError: If there is a name collision between Metrics or `metric`
        has already been added to another `Evaluator`.
    """
    if not hasattr(self, "_metrics"):
      raise RuntimeError(
          "Need to call Evaluator.__init__ before adding metrics")
    if not isinstance(metric, metrics.Metric):
      raise TypeError(
          "Evaluator.track_metric() passed type %s, not a tfe.metrics.Metric" %
          (type(metric),))
    if metric.name in self._metrics:
      if metric is self._metrics[metric.name]:
        return metric
      raise ValueError(
          "Attempt to add two Metrics with the name '%s' to the same Evaluator "
          "'%s'" % (metric.name, self.name))
    # pylint: disable=protected-access
    if hasattr(metric, "_added_to_an_evaluator"):
      raise ValueError("Metric %s already added to Evaluator %s" %
                       (metric.name, metric._added_to_an_evaluator))
    metric._added_to_an_evaluator = self.__class__.__name__
    # pylint: enable=protected-access
    self._metrics[metric.name] = metric
    return metric

  def track_evaluator(self, prefix, evaluator):
    """Add a contained `Evaluator`.

    This is for delegating to another `Evaluator`, e.g. for when you have a
    model with multiple heads. Users should manually invoke the child
    `Evaluator`'s `call` method from their `call` method.

    Args:
      prefix: A string. Metrics from `evaluator` are exported with this
        prefix and a '/'.
      evaluator: An `Evaluator` object.

    Returns:
      The value of `evaluator` passed into this function.

    Raises:
      RuntimeError: If called before __init__.
      TypeError: If `evaluator` is not of the correct type.
      ValueError: If an `Evaluator` has already been added with that `prefix`.
    """
    if not hasattr(self, "_evaluators"):
      raise RuntimeError(
          "Need to call Evaluator.__init__ before adding evaluators")
    if not isinstance(evaluator, Evaluator):
      raise TypeError(
          "Evaluator.track_evaluator() passed type %s, not a tfe.Evaluator." %
          (type(evaluator),))
    if prefix in self._evaluators:
      if evaluator is self._evaluators[prefix]:
        return evaluator
      raise RuntimeError(
          "Attempt to add two Evaluators with the same prefix '%s'." % prefix)
    self._evaluators[prefix] = evaluator
    return evaluator

  @property
  def metric_variables(self):
    v = []
    for metric in six.itervalues(self._metrics):
      v += metric.variables
    for evaluator in six.itervalues(self._evaluators):
      v += evaluator.metric_variables
    return v

  @property
  def metrics(self):
    """Returns a list of (prefix, metric) pairs."""
    m = []
    for metric in six.itervalues(self._metrics):
      m.append(("", metric))
    for prefix, evaluator in six.iteritems(self._evaluators):
      m += [(prefix + "/" + p, m) for p, m in evaluator.metrics]
    return m


class SparseSoftmaxEvaluator(Evaluator):
  """Evaluator for a sparse softmax model.

  Computes a standard set of metrics for single-label, multi-class
  models.

  Args:
    model: A `SparseSoftmaxModel` object or a `Model` whose `eval_data()`
      method produces a `dict` containing values for the loss, true
      label, predicted class, and optional weights.
    loss_key: Optional key for looking up the value of the loss in the
      `eval_data()` dict. Defaults to "loss".
    label_key: Optional key for looking up the value of the label in the
      `eval_data()` dict. Defaults to "label".
    predicted_class_key: Optional key for looking up the value of the
      predicted class in the `eval_data()` dict. Defaults to "predicted_class".
    weights_key: Optional key for looking up the value of the weights
      in the `eval_data()` dict. Defaults to "weights". Note that weights
      are optional, and default to 1 if not present in `eval_data`.
  """

  def __init__(self, model, loss_key="loss", label_key="label",
               predicted_class_key="predicted_class", weights_key="weights"):
    super(SparseSoftmaxEvaluator, self).__init__(model)
    # TODO(josh11b): Expand this to include everything from the standard
    # SparseSoftmax Head.
    self.avg_loss = self.track_metric(metrics.Mean("Avg Loss"))
    self.accuracy = self.track_metric(metrics.Accuracy())
    self.loss_key = loss_key
    self.label_key = label_key
    self.predicted_class_key = predicted_class_key
    self.weights_key = weights_key

  def call(self, eval_data):
    """Update metrics for `eval_data` dict (described above)."""
    weights = eval_data.get(self.weights_key, None)
    if weights is None:
      self.avg_loss(eval_data[self.loss_key])
      self.accuracy(eval_data[self.label_key],
                    eval_data[self.predicted_class_key])
    else:
      self.avg_loss(eval_data[self.loss_key], weights=weights)
      self.accuracy(eval_data[self.label_key],
                    eval_data[self.predicted_class_key],
                    weights=weights)
