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
"""Classes and helper functions for Stochastic Computation Graphs.

## Stochastic Computation Graph Classes

@@StochasticTensor
@@DistributionTensor

## Stochastic Computation Value Types

@@MeanValue
@@SampleValue
@@SampleAndReshapeValue
@@value_type
@@get_current_value_type

## Stochastic Computation Surrogate Loss Functions

@@score_function
@@get_score_function_with_baseline

## Stochastic Computation Graph Helper Functions

@@surrogate_loss
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import contextlib
import threading

import six

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging

STOCHASTIC_TENSOR_COLLECTION = "_stochastic_tensor_collection_"


@six.add_metaclass(abc.ABCMeta)
class StochasticTensor(object):
  """Base Class for Tensor-like objects that emit stochastic values."""

  def __init__(self):
    # Add self to this graph's Stochsatic Tensor collection for
    # purposes of later performing correct surrogate loss calculation.
    ops.add_to_collection(STOCHASTIC_TENSOR_COLLECTION, self)

  @abc.abstractproperty
  def name(self):
    pass

  @abc.abstractproperty
  def dtype(self):
    pass

  @abc.abstractproperty
  def graph(self):
    pass

  @abc.abstractproperty
  def input_dict(self):
    pass

  @abc.abstractmethod
  def value(self, name=None):
    pass

  @abc.abstractmethod
  def loss(self, sample_losses):
    """Returns the term to add to the surrogate loss.

    This method is called by `surrogate_loss`.  The input `sample_losses`
    should have already had `stop_gradient` applied to them.  This is
    because the surrogate_loss usually provides a monte carlo sample term
    of the form `differentiable_surrogate * sum(sample_losses)` where
    `sample_losses` is considered constant with respect to the input
    for purposes of the gradient.

    Args:
      sample_losses: a list of Tensors, the sample losses downstream of this
        `StochasticTensor`.

    Returns:
      Either `None` or a `Tensor`.
    """
    raise NotImplementedError("surrogate_loss not implemented")

  @staticmethod
  def _tensor_conversion_function(v, dtype=None, name=None, as_ref=False):
    _ = name
    if dtype and not dtype.is_compatible_with(v.dtype):
      raise ValueError(
          "Incompatible type conversion requested to type '%s' for variable "
          "of type '%s'" % (dtype.name, v.dtype.name))
    if as_ref:
      raise ValueError("%s: Ref type is not supported." % v)
    return v.value()


# pylint: disable=protected-access
ops.register_tensor_conversion_function(
    StochasticTensor, StochasticTensor._tensor_conversion_function)
# pylint: enable=protected-access


class _StochasticValueType(object):
  """Interface for the ValueType classes.

  This is the base class for MeanValue, SampleValue, SampleAndReshapeValue,
  and their descendants.
  """

  def pushed_above(self, unused_value_type):
    pass

  def popped_above(self, unused_value_type):
    pass

  def declare_inputs(self, unused_stochastic_tensor, unused_inputs_dict):
    pass

  @abc.abstractproperty
  def stop_gradient(self):
    """Whether the value should be wrapped in stop_gradient.

    StochasticTensors must respect this property.
    """
    pass


class MeanValue(_StochasticValueType):

  def __init__(self, stop_gradient=False):
    self._stop_gradient = stop_gradient

  @property
  def stop_gradient(self):
    return self._stop_gradient


class SampleValue(_StochasticValueType):
  """Draw n samples along a new outer dimension.

  This ValueType draws `n` samples from StochasticTensors run within its
  context, increasing the rank by one along a new outer dimension.

  Example:

  ```python
  mu = tf.zeros((2,3))
  sigma = tf.ones((2, 3))
  with sg.value_type(sg.SampleValue(n=4)):
    dt = sg.DistributionTensor(
      distributions.Normal, mu=mu, sigma=sigma)
  # draws 4 samples each with shape (2, 3) and concatenates
  assertEqual(dt.value().get_shape(), (4, 2, 3))
  ```
  """

  def __init__(self, n=1, stop_gradient=False):
    """Sample `n` times and concatenate along a new outer dimension.

    Args:
      n: A python integer or int32 tensor. The number of samples to take.
      stop_gradient: If `True`, StochasticTensors' values are wrapped in
        `stop_gradient`, to avoid backpropagation through.
    """
    self._n = n
    self._stop_gradient = stop_gradient

  @property
  def n(self):
    return self._n

  @property
  def stop_gradient(self):
    return self._stop_gradient


class SampleAndReshapeValue(_StochasticValueType):
  """Ask the StochasticTensor for n samples and reshape the result.

  Sampling from a StochasticTensor increases the rank of the value by 1
  (because each sample represents a new outer dimension).

  This ValueType requests `n` samples from StochasticTensors run within its
  context that the outer two dimensions are reshaped to intermix the samples
  with the outermost (usually batch) dimension.

  Example:

  ```python
  # mu and sigma are both shaped (2, 3)
  mu = [[0.0, -1.0, 1.0], [0.0, -1.0, 1.0]]
  sigma = tf.constant([[1.1, 1.2, 1.3], [1.1, 1.2, 1.3]])

  with sg.value_type(sg.SampleAndReshapeValue(n=2)):
    dt = sg.DistributionTensor(
        distributions.Normal, mu=mu, sigma=sigma)

  # sample(2) creates a (2, 2, 3) tensor, and the two outermost dimensions
  # are reshaped into one: the final value is a (4, 3) tensor.
  dt_value = dt.value()
  assertEqual(dt_value.get_shape(), (4, 3))

  dt_value_val = sess.run([dt_value])[0]  # or e.g. run([tf.identity(dt)])[0]
  assertEqual(dt_value_val.shape, (4, 3))
  ```
  """

  def __init__(self, n=1, stop_gradient=False):
    """Sample `n` times and reshape the outer 2 axes so rank does not change.

    Args:
      n: A python integer or int32 tensor.  The number of samples to take.
      stop_gradient: If `True`, StochasticTensors' values are wrapped in
        `stop_gradient`, to avoid backpropagation through.
    """
    self._n = n
    self._stop_gradient = stop_gradient

  @property
  def n(self):
    return self._n

  @property
  def stop_gradient(self):
    return self._stop_gradient


# Keeps track of how a StochasticTensor's value should be accessed.
# Used by value_type and get_current_value_type below.
_STOCHASTIC_VALUE_STACK = collections.defaultdict(list)


@contextlib.contextmanager
def value_type(dist_value_type):
  """Creates a value type context for any StochasticTensor created within.

  Typical usage:

  ```
  with sg.value_type(sg.MeanValue(stop_gradients=True)):
    dt = sg.DistributionTensor(distributions.Normal, mu=mu, sigma=sigma)
  ```

  In the example above, `dt.value()` (or equivalently, `tf.identity(dt)`) will
  be the mean value of the Normal distribution, i.e., `mu` (possibly
  broadcasted to the shape of `sigma`).  Furthermore, because the `MeanValue`
  was marked with `stop_gradients=True`, this value will have been wrapped
  in a `stop_gradients` call to disable any possible backpropagation.

  Args:
    dist_value_type: An instance of `MeanValue`, `SampleAndReshapeValue`, or
      any other stochastic value type.

  Yields:
    A context for `StochasticTensor` objects that controls the
    value created when they are initialized.

  Raises:
    TypeError: if `dist_value_type` is not an instance of a stochastic value
      type.
  """
  if not isinstance(dist_value_type, _StochasticValueType):
    raise TypeError("dist_value_type must be a Distribution Value Type")
  thread_id = threading.current_thread().ident
  stack = _STOCHASTIC_VALUE_STACK[thread_id]
  if stack:
    stack[-1].pushed_above(dist_value_type)
  stack.append(dist_value_type)
  yield
  stack.pop()
  if stack:
    stack[-1].popped_above(dist_value_type)


class NoValueTypeSetError(ValueError):
  pass


def get_current_value_type():
  thread_id = threading.current_thread().ident
  if not _STOCHASTIC_VALUE_STACK[thread_id]:
    raise NoValueTypeSetError(
        "No value type currently set for this thread (%s).  Did you forget to "
        "wrap 'with stochastic_graph.value_type(...)'?" % thread_id)
  return _STOCHASTIC_VALUE_STACK[thread_id][-1]


def get_score_function_with_baseline(baseline):

  def score_function_with_baseline(dist_tensor, value, losses):
    advantage = math_ops.add_n(losses) - baseline
    return dist_tensor.distribution.log_prob(value) * advantage

  return score_function_with_baseline


def score_function(dist_tensor, value, losses):
  return dist_tensor.distribution.log_prob(value) * math_ops.add_n(losses)


class DistributionTensor(StochasticTensor):
  """DistributionTensor is a StochasticTensor backed by a distribution."""

  def __init__(self,
               dist_cls,
               name=None,
               dist_value_type=None,
               loss_fn=score_function,
               **dist_args):
    """Construct a `DistributionTensor`.

    `DistributionTensor` will instantiate a distribution from `dist_cls` and
    `dist_args` and its `value` method will return the same value each time
    it is called. What `value` is returned is controlled by the
    `dist_value_type` (defaults to `SampleAndReshapeValue`).

    Some distributions' sample functions are not differentiable (e.g. a sample
    from a discrete distribution like a Bernoulli) and so to differentiate
    wrt parameters upstream of the sample requires a gradient estimator like
    the score function estimator. This is accomplished by passing a
    differentiable `loss_fn` to the `DistributionTensor`, which
    defaults to a function whose derivative is the score function estimator.
    Calling `stochastic_graph.surrogate_loss(final_losses)` will call
    `loss()` on every `DistributionTensor` upstream of final losses.

    `loss()` will return None for `DistributionTensor`s backed by
    reparameterized distributions; it will also return None if the value type is
    `MeanValueType` or if `loss_fn=None`.

    Args:
      dist_cls: a class deriving from `BaseDistribution`.
      name: a name for this `DistributionTensor` and its ops.
      dist_value_type: a `_StochasticValueType`, which will determine what the
          `value` of this `DistributionTensor` will be. If not provided, the
          value type set with the `value_type` context manager will be used.
      loss_fn: callable that takes `(dt, dt.value(), influenced_losses)`, where
          `dt` is this `DistributionTensor`, and returns a `Tensor` loss.
      **dist_args: keyword arguments to be passed through to `dist_cls` on
          construction.
    """
    self._dist_cls = dist_cls
    self._dist_args = dist_args
    if dist_value_type is None:
      try:
        self._value_type = get_current_value_type()
      except NoValueTypeSetError:
        self._value_type = SampleAndReshapeValue()
    else:
      # We want to enforce a value type here, but use the value_type()
      # context manager to enforce some error checking.
      with value_type(dist_value_type):
        self._value_type = get_current_value_type()

    self._value_type.declare_inputs(self, dist_args)

    if loss_fn is not None and not callable(loss_fn):
      raise TypeError("loss_fn must be callable")
    self._loss_fn = loss_fn

    with ops.op_scope(dist_args.values(), name, "DistributionTensor") as scope:
      self._name = scope
      self._dist = dist_cls(**dist_args)
      self._value = self._create_value()

    super(DistributionTensor, self).__init__()

  @property
  def input_dict(self):
    return self._dist_args

  @property
  def value_type(self):
    return self._value_type

  @property
  def distribution(self):
    return self._dist

  def clone(self, name=None, **dist_args):
    return DistributionTensor(self._dist_cls, name=name, **dist_args)

  def _create_value(self):
    """Create the value Tensor based on the value type, store as self._value."""

    if isinstance(self._value_type, MeanValue):
      value_tensor = self._dist.mean()
    elif isinstance(self._value_type, SampleValue):
      value_tensor = self._dist.sample_n(self._value_type.n)
    elif isinstance(self._value_type, SampleAndReshapeValue):
      if self._value_type.n == 1:
        value_tensor = self._dist.sample()
      else:
        samples = self._dist.sample_n(self._value_type.n)
        samples_shape = array_ops.shape(samples)
        samples_static_shape = samples.get_shape()
        new_batch_size = samples_shape[0] * samples_shape[1]
        value_tensor = array_ops.reshape(
            samples, array_ops.concat(0, ([new_batch_size], samples_shape[2:])))
        if samples_static_shape.ndims is not None:
          # Update the static shape for shape inference purposes
          shape_list = samples_static_shape.as_list()
          new_shape = tensor_shape.vector(
              shape_list[0] * shape_list[1]
              if shape_list[0] is not None and shape_list[1] is not None
              else None)
          new_shape = new_shape.concatenate(samples_static_shape[2:])
          value_tensor.set_shape(new_shape)
    else:
      raise TypeError(
          "Unrecognized Distribution Value Type: %s", self._value_type)

    if self._value_type.stop_gradient:
      # stop_gradient is being enforced by the value type
      return array_ops.stop_gradient(value_tensor)

    if isinstance(self._value_type, MeanValue):
      return value_tensor  # Using pathwise-derivative for this one.
    if self._dist.is_continuous and self._dist.is_reparameterized:
      return value_tensor  # Using pathwise-derivative for this one.
    else:
      # Will have to perform some variant of score function
      # estimation.  Call stop_gradient on the sampler just in case we
      # may accidentally leak some gradient from it.
      return array_ops.stop_gradient(value_tensor)

  @property
  def name(self):
    return self._name

  @property
  def graph(self):
    return self._value.graph

  @property
  def dtype(self):
    return self._dist.dtype

  def entropy(self, name="entropy"):
    return self._dist.entropy(name=name)

  def mean(self, name="mean"):
    return self._dist.mean(name=name)

  def value(self, name="value"):
    return self._value

  def loss(self, final_losses, name="Loss"):
    # Return a loss based on final_losses and the distribution. Returns
    # None if pathwise derivatives are supported, if the loss_fn
    # was explicitly set to None, or if the value type is MeanValue.
    if self._loss_fn is None:
      return None

    if (self._dist.is_continuous and self._dist.is_reparameterized and
        not self._value_type.stop_gradient):
      # Can perform pathwise-derivative on this one; no additional loss needed.
      return None

    with ops.op_scope(final_losses, self.name):
      with ops.name_scope(name):
        if (self._value_type.stop_gradient or
            isinstance(self._value_type, SampleAndReshapeValue) or
            isinstance(self._value_type, SampleValue)):
          return self._loss_fn(self, self._value, final_losses)
        elif isinstance(self._value_type, MeanValue):
          return None  # MeanValue generally provides its own gradient
        else:
          raise TypeError("Unrecognized Distribution Value Type: %s",
                          self._value_type)


def _stochastic_dependencies_map(fixed_losses, stochastic_tensors=None):
  """Map stochastic tensors to the fixed losses that depend on them.

  Args:
    fixed_losses: a list of `Tensor`s.
    stochastic_tensors: a list of `StochasticTensor`s to map to fixed losses.
      If `None`, all `StochasticTensor`s in the graph will be used.

  Returns:
    A dict `dependencies` that maps `StochasticTensor` objects to subsets of
    `fixed_losses`.

    If `loss in dependencies[st]`, for some `loss` in `fixed_losses` then there
    is a direct path from `st.value()` to `loss` in the graph.
  """
  stoch_value_collection = stochastic_tensors or ops.get_collection(
      STOCHASTIC_TENSOR_COLLECTION)

  if not stoch_value_collection:
    return {}

  stoch_value_map = dict(
      (node.value(), node) for node in stoch_value_collection)

  # Step backwards through the graph to see which surrogate losses correspond
  # to which fixed_losses.
  stoch_dependencies_map = collections.defaultdict(set)
  for loss in fixed_losses:
    boundary = set([loss])
    while boundary:
      edge = boundary.pop()
      edge_stoch_node = stoch_value_map.get(edge, None)
      if edge_stoch_node:
        stoch_dependencies_map[edge_stoch_node].add(loss)
      boundary.update(edge.op.inputs)

  return stoch_dependencies_map


def surrogate_loss(sample_losses,
                   stochastic_tensors=None,
                   name="SurrogateLoss"):
  """Surrogate loss for stochastic graphs.

  This function will call `loss_fn` on each `StochasticTensor`
  upstream of `sample_losses`, passing the losses that it influenced.

  Note that currently `surrogate_loss` does not work with `StochasticTensor`s
  instantiated in `while_loop`s or other control structures.

  Args:
    sample_losses: a list or tuple of final losses. Each loss should be per
      example in the batch (and possibly per sample); that is, it should have
      dimensionality of 1 or greater. All losses should have the same shape.
    stochastic_tensors: a list of `StochasticTensor`s to add loss terms for.
      If None, defaults to all `StochasticTensor`s in the graph upstream of
      the `Tensor`s in `sample_losses`.
    name: the name with which to prepend created ops.

  Returns:
    `Tensor` loss, which is the sum of `sample_losses` and the
    `loss_fn`s returned by the `StochasticTensor`s.

  Raises:
    TypeError: if `sample_losses` is not a list or tuple, or if its elements
      are not `Tensor`s.
    ValueError: if any loss in `sample_losses` does not have dimensionality 1
      or greater.
  """
  with ops.op_scope(sample_losses, name):
    fixed_losses = []
    if not isinstance(sample_losses, (list, tuple)):
      raise TypeError("sample_losses must be a list or tuple")
    for loss in sample_losses:
      if not isinstance(loss, ops.Tensor):
        raise TypeError("loss is not a Tensor: %s" % loss)
      ndims = loss.get_shape().ndims
      if not (ndims is not None and ndims >= 1):
        raise ValueError("loss must have dimensionality 1 or greater: %s" %
                         loss)
      fixed_losses.append(array_ops.stop_gradient(loss))

    stoch_dependencies_map = _stochastic_dependencies_map(
        fixed_losses, stochastic_tensors=stochastic_tensors)
    if not stoch_dependencies_map:
      logging.warn(
          "No collection of Stochastic Tensors found for current graph.")
      return math_ops.add_n(sample_losses)

    # Iterate through all of the stochastic dependencies, adding
    # surrogate terms where necessary.
    sample_losses = [ops.convert_to_tensor(loss) for loss in sample_losses]
    loss_terms = sample_losses
    for (stoch_node, dependent_losses) in stoch_dependencies_map.items():
      loss_term = stoch_node.loss(list(dependent_losses))
      if loss_term is not None:
        loss_terms.append(loss_term)

    return math_ops.add_n(loss_terms)
