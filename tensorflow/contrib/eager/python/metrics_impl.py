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
"""Metrics classes for computing the output of an evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope


class Metric(object):
  """A metric holds state for aggregating statistics over an evaluation run.

  Users will use Evaluator.add_metric() to add Metric objects to their
  evaluation, call them in each step, and then use
  Evaluator.all_metric_results() at the end.

  Descendants will implement:
  * call(): Should follow this pattern:
      if not self.built:
        self.var = self.add_variable(...)
      self.add_update(self.var.assign_add(...))
  * aggregate(): Adds in the state from a list of metrics of the same type
    as `self`.  (Default of summing all the variables will be fine for most
    descendants.)
  * result(): Computes and returns a final value for the metric
    from the variables in `self`.
  """

  def __init__(self, name=None):
    self.built = False
    self._vars = []
    self._updates = []
    self._name = name or self.__class__.__name__
    # TODO(josh11b): Need some way to make sure two Metrics in the same
    # Network have distinct names. Maybe we can get a unique name from
    # a name/variable scope?
    # TODO(josh11b): self._in_graph_mode = context.in_graph_mode()

  # ---- API for users ----
  def __call__(self, *args, **kwargs):
    # TODO(josh11b): If self._in_graph_mode is true, make self.call() into a
    # graph callable here, so that variable updates happen without requiring
    # a separate fetch.
    # TODO(josh11b): Do we need a separate build() method to separate
    # initialization from each update? If so, how do we get the arguments
    # to it?  We *could* just pass in *args and **kwargs...
    if not self.built:
      # TODO(ashankar): Set up container isolation so there is no chance
      # distinct metrics objects accidentally share variables.
      # TODO(josh11b): Replace things like spaces in self._name to create
      # a valid scope name.
      with variable_scope.variable_scope(
          self._name, use_resource=True, reuse=False):
        ret = self.call(*args, **kwargs)
      self.built = True
    else:
      ret = self.call(*args, **kwargs)
    return ret

  @property
  def name(self):
    return self._name

  @property
  def variables(self):
    return self._vars

  # ---- To be implemented by descendants ---
  def call(self, *args, **kwargs):
    """Accumulates statistics for the metric."""
    raise NotImplementedError("Metrics must define a call() member function")

  # We can support two different strategies of for doing data-parallel
  # distributed metric computations:
  # * Put metric variables on the first device and rely on small
  #   bandwidth needed to do updates. (Doesn't require any particular
  #   code in Metric implementations.)
  # * Ask each type of metric to define an aggregation method to run
  #   at the end of eval to merge across devices. Note: this is good
  #   for the use case where they want to record the metric's state
  #   for each example and then later decide which examples they want
  #   to aggregate over. (Recommended -- not too much harder and adds
  #   flexibility over previous option.)
  # I'm going with the second strategy since we can define a default
  # implementation of aggregate() that will work for most descendants.
  def aggregate(self, metrics):
    """Adds in the state from a list of metrics.

    Default implementation sums all the metric variables.

    Args:
      metrics: A list of metrics with the same type as `self`.

    Raises:
      ValueError: If metrics contains invalid data.
    """
    for m in metrics:
      if type(self) != type(m):  # pylint: disable=unidiomatic-typecheck
        raise TypeError("All metrics must be the same type, '%s' != '%s'." %
                        (type(self), type(m)))
    # pylint: disable=protected-access
    for i in range(len(self._vars)):
      if any(m._vars[i].name != self._vars[i].name for m in metrics):
        raise ValueError("All metrics must have variables in the same order.")
      self._vars[i].assign_add(math_ops.add_n([m._vars[i] for m in metrics]))
    # pylint: enable=protected-access

  def result(self):  # TODO(josh11b): Add an optional summary_writer parameter.
    """Computes and returns a final value for the metric."""
    raise NotImplementedError("Metrics must define a result() member function")

  # ---- For use by descendants ---
  def add_variable(self, name, shape=None, dtype=None, initializer=None):
    """***Only for use by descendants of Metric***."""
    if self.built:
      raise RuntimeError("Can't call add_variable() after a Metric has been "
                         "built in the first call().")
    v = variable_scope.get_variable(name, shape, dtype, initializer,
                                    trainable=False, use_resource=True)
    self._vars.append(v)
    return v


class Mean(Metric):
  """Computes the (weighted) mean of the given values."""
  # TODO(josh11b): Maybe have a dtype argument that defaults to tf.float64?
  # Or defaults to type of the input if it is tf.float32, else tf.float64?

  def call(self, values, weights=None):
    """Accumulate statistics for computing the mean.

    For example, if values is [1, 3, 5, 7] then the mean is 4.
    If the weights were specified as [1, 1, 0, 0] then the mean would be 2.

    Args:
      values: Tensor with the per-example value.
      weights: Optional weighting of each example. Defaults to 1.
    """
    if not self.built:  # False only in the first call().
      self.numer = self.add_variable(name="numer", shape=(),
                                     dtype=dtypes.float64,
                                     initializer=init_ops.zeros_initializer)
      self.denom = self.add_variable(name="denom", shape=(),
                                     dtype=dtypes.float64,
                                     initializer=init_ops.zeros_initializer)
    if weights is None:
      self.denom.assign_add(
          math_ops.cast(array_ops.size(values), dtypes.float64))
      values = math_ops.reduce_sum(values)
      self.numer.assign_add(math_ops.cast(values, dtypes.float64))
    else:
      weights = math_ops.cast(weights, dtypes.float64)
      self.denom.assign_add(math_ops.reduce_sum(weights))
      values = math_ops.cast(values, dtypes.float64) * weights
      self.numer.assign_add(math_ops.reduce_sum(values))

  def result(self):
    return self.numer / self.denom


class Accuracy(Mean):
  """Calculates how often `predictions` matches `labels`."""

  def call(self, labels, predictions, weights=None):
    """Accumulate accuracy statistics.

    For example, if labels is [1, 2, 3, 4] and predictions is [0, 2, 3, 4]
    then the accuracy is 3/4 or .75.  If the weights were specified as
    [1, 1, 0, 0] then the accuracy would be 1/2 or .5.

    `labels` and `predictions` should have the same shape and type.

    Args:
      labels: Tensor with the true labels for each example.  One example
        per element of the Tensor.
      predictions: Tensor with the predicted label for each example.
      weights: Optional weighting of each example. Defaults to 1.
    """
    matches = math_ops.equal(labels, predictions)
    matches = math_ops.cast(matches, dtypes.float64)
    super(Accuracy, self).call(matches, weights=weights)
