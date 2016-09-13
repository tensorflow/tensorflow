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
"""Classes and helper functions for creating Stochastic Tensors.

`StochasticTensor` objects wrap `Distribution` objects.  Their
values may be samples from the underlying distribution, or the distribution
mean (as governed by `value_type`).  These objects provide a `loss`
method for use when sampling from a non-reparameterized distribution.
The `loss`method is used in conjunction with `stochastic_graph.surrogate_loss`
to produce a single differentiable loss in stochastic graphs having
both continuous and discrete stochastic nodes.

## Stochastic Tensor Classes

@@BaseStochasticTensor
@@StochasticTensor

## Stochastic Tensor Value Types

@@MeanValue
@@SampleValue
@@SampleAndReshapeValue

@@value_type
@@get_current_value_type
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import contextlib
import inspect
import threading

import six

from tensorflow.contrib import distributions
from tensorflow.contrib.bayesflow.python.ops import stochastic_gradient_estimators as sge
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
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

  @abc.abstractproperty
  def input_dict(self):
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


class StochasticTensor(BaseStochasticTensor):
  """StochasticTensor is a BaseStochasticTensor backed by a distribution."""

  def __init__(self,
               dist_cls,
               name=None,
               dist_value_type=None,
               loss_fn=sge.score_function,
               **dist_args):
    """Construct a `StochasticTensor`.

    `StochasticTensor` will instantiate a distribution from `dist_cls` and
    `dist_args` and its `value` method will return the same value each time
    it is called. What `value` is returned is controlled by the
    `dist_value_type` (defaults to `SampleAndReshapeValue`).

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
      dist_cls: a `Distribution` class.
      name: a name for this `StochasticTensor` and its ops.
      dist_value_type: a `_StochasticValueType`, which will determine what the
          `value` of this `StochasticTensor` will be. If not provided, the
          value type set with the `value_type` context manager will be used.
      loss_fn: callable that takes `(dt, dt.value(), influenced_loss)`, where
          `dt` is this `StochasticTensor`, and returns a `Tensor` loss. By
          default, `loss_fn` is the `score_function`, or more precisely, the
          integral of the score function, such that when the gradient is taken,
          the score function results. See the `stochastic_gradient_estimators`
          module for additional loss functions and baselines.
      **dist_args: keyword arguments to be passed through to `dist_cls` on
          construction.

    Raises:
      TypeError: if `dist_cls` is not a `Distribution`.
      TypeError: if `loss_fn` is not `callable`.
    """
    if not issubclass(dist_cls, distributions.Distribution):
      raise TypeError("dist_cls must be a subclass of Distribution")
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

    with ops.name_scope(name, "StochasticTensor",
                        dist_args.values()) as scope:
      self._name = scope
      self._dist = dist_cls(**dist_args)
      self._value = self._create_value()

    super(StochasticTensor, self).__init__()

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
    return StochasticTensor(self._dist_cls, name=name, **dist_args)

  def _create_value(self):
    """Create the value Tensor based on the value type, store as self._value."""

    if isinstance(self._value_type, MeanValue):
      value_tensor = self._dist.mean()
    elif isinstance(self._value_type, SampleValue):
      value_tensor = self._dist.sample(self._value_type.n)
    elif isinstance(self._value_type, SampleAndReshapeValue):
      if self._value_type.n == 1:
        value_tensor = self._dist.sample()
      else:
        samples = self._dist.sample(self._value_type.n)
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

  def loss(self, final_loss, name="Loss"):
    # Return a loss based on final_loss and the distribution. Returns
    # None if pathwise derivatives are supported, if the loss_fn
    # was explicitly set to None, or if the value type is MeanValue.
    if self._loss_fn is None:
      return None

    if (self._dist.is_continuous and self._dist.is_reparameterized and
        not self._value_type.stop_gradient):
      # Can perform pathwise-derivative on this one; no additional loss needed.
      return None

    with ops.name_scope(self.name, values=[final_loss]):
      with ops.name_scope(name):
        if (self._value_type.stop_gradient or
            isinstance(self._value_type, SampleAndReshapeValue) or
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
  def __init__(self, dist_cls, value, name=None, **dist_args):
    """Construct an `ObservedStochasticTensor`.

    `ObservedStochasticTensor` will instantiate a distribution from `dist_cls`
    and `dist_args` but use the provided value instead of sampling from the
    distribution. The provided value argument must be appropriately shaped
    to have come from the constructed distribution.

    Args:
      dist_cls: a `Distribution` class.
      value: a Tensor containing the observed value
      name: a name for this `ObservedStochasticTensor` and its ops.
      **dist_args: keyword arguments to be passed through to `dist_cls` on
          construction.

    Raises:
      TypeError: if `dist_cls` is not a `Distribution`.
      ValueError: if `value` is not compatible with the distribution.
    """
    if not issubclass(dist_cls, distributions.Distribution):
      raise TypeError("dist_cls must be a subclass of Distribution")
    self._dist_cls = dist_cls
    self._dist_args = dist_args
    with ops.name_scope(name, "ObservedStochasticTensor",
                        list(dist_args.values()) + [value]) as scope:
      self._name = scope
      self._dist = dist_cls(**dist_args)
      dist_shape = self._dist.get_batch_shape().concatenate(
          self._dist.get_event_shape())
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
                         "distribuiton (%s)." % (value.dtype, self._dist.dtype))
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
    "SampleAndReshapeValue",
    "value_type",
    "get_current_value_type",
]

_globals = globals()
# pylint: disable=redefined-builtin
__doc__ += "\n\n## Automatically Generated StochasticTensors\n\n"
# pylint: enable=redefined-builtin
for _name in sorted(dir(distributions)):
  _candidate = getattr(distributions, _name)
  if (inspect.isclass(_candidate)
      and _candidate != distributions.Distribution
      and issubclass(_candidate, distributions.Distribution)):
    _local_name = "%sTensor" % _name

    class _WrapperTensor(StochasticTensor):
      _my_candidate = _candidate

      def __init__(self, name=None, dist_value_type=None,
                   loss_fn=sge.score_function, **dist_args):
        StochasticTensor.__init__(
            self,
            dist_cls=self._my_candidate,
            name=name,
            dist_value_type=dist_value_type,
            loss_fn=loss_fn, **dist_args)

    _WrapperTensor.__name__ = _local_name
    _WrapperTensor.__doc__ = (
        "`%s` is a `StochasticTensor` backed by the distribution `%s`."""
        % (_local_name, _name))
    _globals[_local_name] = _WrapperTensor
    del _WrapperTensor
    del _candidate

    __all__.append(_local_name)
    __doc__ += "@@%s\n" % _local_name

    del _local_name
