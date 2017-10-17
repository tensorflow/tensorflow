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
  (a) Call `add_metric()` and/or `add_evaluator()` in __init__().
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

  # ---- API for users ----
  def __call__(self, *args, **kwargs):
    """Update metrics with a minibatch of input examples."""
    return self.call(self._model.eval_data(*args, **kwargs))

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
    """Convenience method for performing an eval on a Dataset."""
    for example in datasets.Iterator(dataset):
      self.__call__(example, *args, **kwargs)
    # TODO(josh11b): Add optional summary_writer.
    return self.all_metric_results()

  # ---- To be implemented by descendants ---
  def call(self, eval_data):
    """Update metrics using the output of self.model."""
    raise NotImplementedError("Evaluators must define a call member function.")

  # ---- For use by descendants ---
  @property
  def model(self):
    return self._model

  def add_metric(self, metric):
    """Add a Metric to be tracked.

    Rule: metrics can only be in one `Evaluator`.

    Args:
      metric: A `Metric` object.

    Returns:
      The `metric` passed into this function.

    Raises:
      RuntimeError: If called before __init__.
      TypeError: If `metric` is not of the correct type.
      ValueError: If there is a name collision between Metrics.
    """
    if not hasattr(self, "_metrics"):
      raise RuntimeError(
          "Need to call Evaluator.__init__ before adding metrics")
    if not isinstance(metric, metrics.Metric):
      raise TypeError(
          "Evaluator.add_metric() passed type %s, not a tfe.metrics.Metric" %
          (type(metric),))
    if metric.name in self._metrics:
      if metric is self._metrics[metric.name]:
        return metric
      raise ValueError(
          "Attempt to add two Metrics with the name '%s' to the same Evaluator "
          "'%s'" % (metric.name, self.name))
    self._metrics[metric.name] = metric
    return metric

  def add_evaluator(self, prefix, evaluator):
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
          "Evaluator.add_evaluator() passed type %s, not a tfe.Evaluator." %
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
    m = []
    for metric in six.itervalues(self._metrics):
      m.append(metric)
    for evaluator in six.itervalues(self._evaluators):
      m += evaluator.metrics
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
    self.avg_loss = self.add_metric(metrics.Mean("Avg_Loss"))
    self.accuracy = self.add_metric(metrics.Accuracy())
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
