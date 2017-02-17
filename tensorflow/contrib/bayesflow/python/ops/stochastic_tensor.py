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
"""Support for creating Stochastic Tensors.

See the ${@python/contrib.bayesflow.stochastic_tensor} guide.

@@BaseStochasticTensor
@@StochasticTensor
@@MeanValue
@@SampleValue
@@value_type
@@get_current_value_type
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import contextlib
import threading

import six

from tensorflow.contrib.bayesflow.python.ops import stochastic_gradient_estimators as sge
from tensorflow.contrib.distributions.python.ops import distribution
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

STOCHASTIC_TENSOR_COLLECTION = "_stochastic_tensor_collection_"


@six.add_metaclass(abc.ABCMeta)
class BaseStochasticTensor(object):
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

  @abc.abstractmethod
  def value(self, name=None):
    pass

  @abc.abstractmethod
  def loss(self, sample_loss):
    """Returns the term to add to the surrogate loss.

    This method is called by `surrogate_loss`.  The input `sample_loss` should
    have already had `stop_gradient` applied to it.  This is because the
    surrogate_loss usually provides a Monte Carlo sample term of the form
    `differentiable_surrogate * sample_loss` where `sample_loss` is considered
    constant with respect to the input for purposes of the gradient.

    Args:
      sample_loss: `Tensor`, sample loss downstream of this `StochasticTensor`.

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
    BaseStochasticTensor, BaseStochasticTensor._tensor_conversion_function)

# pylint: enable=protected-access


class _StochasticValueType(object):
  """Interface for the ValueType classes.

  This is the base class for MeanValue, SampleValue, and their descendants.
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
  """Draw samples, possibly adding new outer dimensions along the way.

  This ValueType draws samples from StochasticTensors run within its
  context, increasing the rank according to the requested shape.

  Examples:

  ```python
  mu = tf.zeros((2,3))
  sigma = tf.ones((2, 3))
  with sg.value_type(sg.SampleValue()):
    st = sg.StochasticTensor(
      tf.contrib.distributions.Normal, mu=mu, sigma=sigma)
  # draws 1 sample and does not reshape
  assertEqual(st.value().get_shape(), (2, 3))
  ```

  ```python
  mu = tf.zeros((2,3))
  sigma = tf.ones((2, 3))
  with sg.value_type(sg.SampleValue(4)):
    st = sg.StochasticTensor(
      tf.contrib.distributions.Normal, mu=mu, sigma=sigma)
  # draws 4 samples each with shape (2, 3) and concatenates
  assertEqual(st.value().get_shape(), (4, 2, 3))
  ```
  """

  def __init__(self, shape=(), stop_gradient=False):
    """Sample according to shape.

    For the given StochasticTensor `st` using this value type,
    the shape of `st.value()` will match that of
    `st.distribution.sample(shape)`.

    Args:
      shape: A shape tuple or int32 tensor.  The sample shape.
        Default is a scalar: take one sample and do not change the size.
      stop_gradient: If `True`, StochasticTensors' values are wrapped in
        `stop_gradient`, to avoid backpropagation through.
    """
    self._shape = shape
    self._stop_gradient = stop_gradient

  @property
  def shape(self):
    return self._shape

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
    st = sg.StochasticTensor(tf.contrib.distributions.Normal, mu=mu,
                             sigma=sigma)
  ```

  In the example above, `st.value()` (or equivalently, `tf.identity(st)`) will
  be the mean value of the Normal distribution, i.e., `mu` (possibly
  broadcasted to the shape of `sigma`).  Furthermore, because the `MeanValue`
  was marked with `stop_gradients=True`, this value will have been wrapped
  in a `stop_gradients` call to disable any possible backpropagation.

  Args:
    dist_value_type: An instance of `MeanValue`, `SampleValue`, or
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


class StochasticTensor(BaseStochasticTensor):
  """StochasticTensor is a BaseStochasticTensor backed by a distribution."""

  def __init__(self,
               dist,
               name="StochasticTensor",
               dist_value_type=None,
               loss_fn=sge.score_function):
    """Construct a `StochasticTensor`.

    `StochasticTensor` is backed by the `dist` distribution and its `value`
    method will return the same value each time it is called. What `value` is
    returned is controlled by the `dist_value_type` (defaults to
    `SampleValue`).

    Some distributions' sample functions are not differentiable (e.g. a sample
    from a discrete distribution like a Bernoulli) and so to differentiate
    wrt parameters upstream of the sample requires a gradient estimator like
    the score function estimator. This is accomplished by passing a
    differentiable `loss_fn` to the `StochasticTensor`, which
    defaults to a function whose derivative is the score function estimator.
    Calling `stochastic_graph.surrogate_loss(final_losses)` will call
    `loss()` on every `StochasticTensor` upstream of final losses.

    `loss()` will return None for `StochasticTensor`s backed by
    reparameterized distributions; it will also return None if the value type is
    `MeanValueType` or if `loss_fn=None`.

    Args:
      dist: an instance of `Distribution`.
      name: a name for this `StochasticTensor` and its ops.
      dist_value_type: a `_StochasticValueType`, which will determine what the
          `value` of this `StochasticTensor` will be. If not provided, the
          value type set with the `value_type` context manager will be used.
      loss_fn: callable that takes
          `(st, st.value(), influenced_loss)`, where
          `st` is this `StochasticTensor`, and returns a `Tensor` loss. By
          default, `loss_fn` is the `score_function`, or more precisely, the
          integral of the score function, such that when the gradient is taken,
          the score function results. See the `stochastic_gradient_estimators`
          module for additional loss functions and baselines.

    Raises:
      TypeError: if `dist` is not an instance of `Distribution`.
      TypeError: if `loss_fn` is not `callable`.
    """
    if not isinstance(dist, distribution.Distribution):
      raise TypeError("dist must be an instance of Distribution")
    if dist_value_type is None:
      try:
        self._value_type = get_current_value_type()
      except NoValueTypeSetError:
        self._value_type = SampleValue()
    else:
      # We want to enforce a value type here, but use the value_type()
      # context manager to enforce some error checking.
      with value_type(dist_value_type):
        self._value_type = get_current_value_type()

    if loss_fn is not None and not callable(loss_fn):
      raise TypeError("loss_fn must be callable")
    self._loss_fn = loss_fn

    with ops.name_scope(name) as scope:
      self._name = scope
      self._dist = dist
      self._value = self._create_value()

    super(StochasticTensor, self).__init__()

  @property
  def value_type(self):
    return self._value_type

  @property
  def distribution(self):
    return self._dist

  def _create_value(self):
    """Create the value Tensor based on the value type, store as self._value."""

    if isinstance(self._value_type, MeanValue):
      value_tensor = self._dist.mean()
    elif isinstance(self._value_type, SampleValue):
      value_tensor = self._dist.sample(self._value_type.shape)
    else:
      raise TypeError("Unrecognized Distribution Value Type: %s",
                      self._value_type)

    if self._value_type.stop_gradient:
      # stop_gradient is being enforced by the value type
      return array_ops.stop_gradient(value_tensor)

    if isinstance(self._value_type, MeanValue):
      return value_tensor  # Using pathwise-derivative for this one.
    if self._dist.is_continuous and (
        self._dist.reparameterization_type
        is distribution.FULLY_REPARAMETERIZED):
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

  def loss(self, final_loss, name="Loss"):
    # Return a loss based on final_loss and the distribution. Returns
    # None if pathwise derivatives are supported, if the loss_fn
    # was explicitly set to None, or if the value type is MeanValue.
    if self._loss_fn is None:
      return None

    if (self._dist.is_continuous and
        self._dist.reparameterization_type is distribution.FULLY_REPARAMETERIZED
        and not self._value_type.stop_gradient):
      # Can perform pathwise-derivative on this one; no additional loss needed.
      return None

    with ops.name_scope(self.name, values=[final_loss]):
      with ops.name_scope(name):
        if (self._value_type.stop_gradient or
            isinstance(self._value_type, SampleValue)):
          return self._loss_fn(self, self._value, final_loss)
        elif isinstance(self._value_type, MeanValue):
          return None  # MeanValue generally provides its own gradient
        else:
          raise TypeError("Unrecognized Distribution Value Type: %s",
                          self._value_type)


class ObservedStochasticTensor(StochasticTensor):
  """A StochasticTensor with an observed value."""

  # pylint: disable=super-init-not-called
  def __init__(self, dist, value, name=None):
    """Construct an `ObservedStochasticTensor`.

    `ObservedStochasticTensor` is backed by distribution `dist` and uses the
    provided value instead of using the current value type to draw a value from
    the distribution. The provided value argument must be appropriately shaped
    to have come from the distribution.

    Args:
      dist: an instance of `Distribution`.
      value: a Tensor containing the observed value
      name: a name for this `ObservedStochasticTensor` and its ops.

    Raises:
      TypeError: if `dist` is not an instance of `Distribution`.
      ValueError: if `value` is not compatible with the distribution.
    """
    if not isinstance(dist, distribution.Distribution):
      raise TypeError("dist must be an instance of Distribution")
    with ops.name_scope(name, "ObservedStochasticTensor", [value]) as scope:
      self._name = scope
      self._dist = dist
      dist_shape = self._dist.batch_shape.concatenate(
          self._dist.event_shape)
      value = ops.convert_to_tensor(value)
      value_shape = value.get_shape()

      if not value_shape.is_compatible_with(dist_shape):
        if value_shape.ndims < dist_shape.ndims:
          raise ValueError(
              "Rank of observed value (%d) must be >= rank of a sample from the"
              " distribution (%d)." % (value_shape.ndims, dist_shape.ndims))
        sample_shape = value_shape[(value_shape.ndims - dist_shape.ndims):]
        if not sample_shape.is_compatible_with(dist_shape):
          raise ValueError(
              "Shape of observed value %s is incompatible with the shape of a "
              "sample from the distribution %s." % (value_shape, dist_shape))
      if value.dtype != self._dist.dtype:
        raise ValueError("Type of observed value (%s) does not match type of "
                         "distribution (%s)." % (value.dtype, self._dist.dtype))
      self._value = array_ops.identity(value)
    # pylint: disable=non-parent-init-called
    BaseStochasticTensor.__init__(self)

  def loss(self, final_loss, name=None):
    return None


__all__ = [
    "BaseStochasticTensor",
    "StochasticTensor",
    "ObservedStochasticTensor",
    "MeanValue",
    "SampleValue",
    "value_type",
    "get_current_value_type",
]
