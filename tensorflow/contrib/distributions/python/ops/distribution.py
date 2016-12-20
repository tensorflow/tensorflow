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
"""Base classes for probability distributions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import contextlib
import inspect
import types
import warnings

import numpy as np
import six

from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


_DISTRIBUTION_PUBLIC_METHOD_WRAPPERS = [
    "batch_shape", "get_batch_shape", "event_shape", "get_event_shape",
    "sample_n", "log_prob", "prob", "log_cdf", "cdf", "log_survival_function",
    "survival_function", "entropy", "mean", "variance", "std", "mode"]


@six.add_metaclass(abc.ABCMeta)
class _BaseDistribution(object):
  """Abstract base class needed for resolving subclass hierarchy."""
  pass


def _copy_fn(fn):
  """Create a deep copy of fn.

  Args:
    fn: a callable

  Returns:
    A `FunctionType`: a deep copy of fn.

  Raises:
    TypeError: if `fn` is not a callable.
  """
  if not callable(fn):
    raise TypeError("fn is not callable: %s" % fn)
  # The blessed way to copy a function.  copy.deepcopy fails to create
  # a non-reference copy.  Since:
  #   types.FunctionType == type(lambda: None),
  # and the docstring for the function type states:
  #
  #   function(code, globals[, name[, argdefs[, closure]]])
  #
  #   Create a function object from a code object and a dictionary.
  #   ...
  #
  # Here we can use this to create a new function with the old function's
  # code, globals, closure, etc.
  return types.FunctionType(
      code=fn.__code__, globals=fn.__globals__,
      name=fn.__name__, argdefs=fn.__defaults__,
      closure=fn.__closure__)


def _update_docstring(old_str, append_str):
  """Update old_str by inserting append_str just before the "Args:" section."""
  old_str_lines = old_str.split("\n")

  # Step 0: Prepend spaces to all lines of append_str. This is
  # necessary for correct markdown generation.
  append_str = "\n".join("    %s" % line for line in append_str.split("\n"))

  # Step 1: Find mention of "Args":
  has_args_ix = [
      ix for ix, line in enumerate(old_str_lines)
      if line.strip().lower() == "args:"]
  if has_args_ix:
    final_args_ix = has_args_ix[-1]
    return ("\n".join(old_str_lines[:final_args_ix])
            + "\n\n" + append_str + "\n\n"
            + "\n".join(old_str_lines[final_args_ix:]))
  else:
    return old_str + "\n\n" + append_str


class _DistributionMeta(abc.ABCMeta):

  def __new__(mcs, classname, baseclasses, attrs):
    """Control the creation of subclasses of the Distribution class.

    The main purpose of this method is to properly propagate docstrings
    from private Distribution methods, like `_log_prob`, into their
    public wrappers as inherited by the Distribution base class
    (e.g. `log_prob`).

    Args:
      classname: The name of the subclass being created.
      baseclasses: A tuple of parent classes.
      attrs: A dict mapping new attributes to their values.

    Returns:
      The class object.

    Raises:
      TypeError: If `Distribution` is not a subclass of `BaseDistribution`, or
        the new class is derived via multiple inheritance and the first
        parent class is not a subclass of `BaseDistribution`.
      AttributeError:  If `Distribution` does not implement e.g. `log_prob`.
      ValueError:  If a `Distribution` public method lacks a docstring.
    """
    if not baseclasses:  # Nothing to be done for Distribution
      raise TypeError("Expected non-empty baseclass.  Does Distribution "
                      "not subclass _BaseDistribution?")
    which_base = [
        base for base in baseclasses
        if base == _BaseDistribution or issubclass(base, Distribution)]
    base = which_base[0]
    if base == _BaseDistribution:  # Nothing to be done for Distribution
      return abc.ABCMeta.__new__(mcs, classname, baseclasses, attrs)
    if not issubclass(base, Distribution):
      raise TypeError("First parent class declared for %s must be "
                      "Distribution, but saw '%s'" % (classname, base.__name__))
    for attr in _DISTRIBUTION_PUBLIC_METHOD_WRAPPERS:
      special_attr = "_%s" % attr
      class_attr_value = attrs.get(attr, None)
      if attr in attrs:
        # The method is being overridden, do not update its docstring
        continue
      base_attr_value = getattr(base, attr, None)
      if not base_attr_value:
        raise AttributeError(
            "Internal error: expected base class '%s' to implement method '%s'"
            % (base.__name__, attr))
      class_special_attr_value = attrs.get(special_attr, None)
      if class_special_attr_value is None:
        # No _special method available, no need to update the docstring.
        continue
      class_special_attr_docstring = inspect.getdoc(class_special_attr_value)
      if not class_special_attr_docstring:
        # No docstring to append.
        continue
      class_attr_value = _copy_fn(base_attr_value)
      class_attr_docstring = inspect.getdoc(base_attr_value)
      if class_attr_docstring is None:
        raise ValueError(
            "Expected base class fn to contain a docstring: %s.%s"
            % (base.__name__, attr))
      class_attr_value.__doc__ = _update_docstring(
          class_attr_value.__doc__,
          ("Additional documentation from `%s`:\n\n%s"
           % (classname, class_special_attr_docstring)))
      attrs[attr] = class_attr_value

    return abc.ABCMeta.__new__(mcs, classname, baseclasses, attrs)


@six.add_metaclass(_DistributionMeta)
class Distribution(_BaseDistribution):
  """A generic probability distribution base class.

  `Distribution` is a base class for constructing and organizing properties
  (e.g., mean, variance) of random variables (e.g, Bernoulli, Gaussian).

  ### Subclassing

  Subclasses are expected to implement a leading-underscore version of the
  same-named function.  The argument signature should be identical except for
  the omission of `name="..."`.  For example, to enable `log_prob(value,
  name="log_prob")` a subclass should implement `_log_prob(value)`.

  Subclasses can append to public-level docstrings by providing
  docstrings for their method specializations. For example:

  ```python
  @distribution_util.AppendDocstring("Some other details.")
  def _log_prob(self, value):
    ...
  ```

  would add the string "Some other details." to the `log_prob` function
  docstring.  This is implemented as a simple decorator to avoid python
  linter complaining about missing Args/Returns/Raises sections in the
  partial docstrings.

  ### Broadcasting, batching, and shapes

  All distributions support batches of independent distributions of that type.
  The batch shape is determined by broadcasting together the parameters.

  The shape of arguments to `__init__`, `cdf`, `log_cdf`, `prob`, and
  `log_prob` reflect this broadcasting, as does the return value of `sample` and
  `sample_n`.

  `sample_n_shape = (n,) + batch_shape + event_shape`, where `sample_n_shape` is
  the shape of the `Tensor` returned from `sample_n`, `n` is the number of
  samples, `batch_shape` defines how many independent distributions there are,
  and `event_shape` defines the shape of samples from each of those independent
  distributions. Samples are independent along the `batch_shape` dimensions, but
  not necessarily so along the `event_shape` dimensions (depending on the
  particulars of the underlying distribution).

  Using the `Uniform` distribution as an example:

  ```python
  minval = 3.0
  maxval = [[4.0, 6.0],
            [10.0, 12.0]]

  # Broadcasting:
  # This instance represents 4 Uniform distributions. Each has a lower bound at
  # 3.0 as the `minval` parameter was broadcasted to match `maxval`'s shape.
  u = Uniform(minval, maxval)

  # `event_shape` is `TensorShape([])`.
  event_shape = u.get_event_shape()
  # `event_shape_t` is a `Tensor` which will evaluate to [].
  event_shape_t = u.event_shape

  # Sampling returns a sample per distribution.  `samples` has shape
  # (5, 2, 2), which is (n,) + batch_shape + event_shape, where n=5,
  # batch_shape=(2, 2), and event_shape=().
  samples = u.sample_n(5)

  # The broadcasting holds across methods. Here we use `cdf` as an example. The
  # same holds for `log_cdf` and the likelihood functions.

  # `cum_prob` has shape (2, 2) as the `value` argument was broadcasted to the
  # shape of the `Uniform` instance.
  cum_prob_broadcast = u.cdf(4.0)

  # `cum_prob`'s shape is (2, 2), one per distribution. No broadcasting
  # occurred.
  cum_prob_per_dist = u.cdf([[4.0, 5.0],
                             [6.0, 7.0]])

  # INVALID as the `value` argument is not broadcastable to the distribution's
  # shape.
  cum_prob_invalid = u.cdf([4.0, 5.0, 6.0])
  ```

  ### Parameter values leading to undefined statistics or distributions.

  Some distributions do not have well-defined statistics for all initialization
  parameter values.  For example, the beta distribution is parameterized by
  positive real numbers `a` and `b`, and does not have well-defined mode if
  `a < 1` or `b < 1`.

  The user is given the option of raising an exception or returning `NaN`.

  ```python
  a = tf.exp(tf.matmul(logits, weights_a))
  b = tf.exp(tf.matmul(logits, weights_b))

  # Will raise exception if ANY batch member has a < 1 or b < 1.
  dist = distributions.beta(a, b, allow_nan_stats=False)
  mode = dist.mode().eval()

  # Will return NaN for batch members with either a < 1 or b < 1.
  dist = distributions.beta(a, b, allow_nan_stats=True)  # Default behavior
  mode = dist.mode().eval()
  ```

  In all cases, an exception is raised if *invalid* parameters are passed, e.g.

  ```python
  # Will raise an exception if any Op is run.
  negative_a = -1.0 * a  # beta distribution by definition has a > 0.
  dist = distributions.beta(negative_a, b, allow_nan_stats=True)
  dist.mean().eval()
  ```

  """

  def __init__(self,
               dtype,
               is_continuous,
               is_reparameterized,
               validate_args,
               allow_nan_stats,
               parameters=None,
               graph_parents=None,
               name=None):
    """Constructs the `Distribution`.

    **This is a private method for subclass use.**

    Args:
      dtype: The type of the event samples. `None` implies no type-enforcement.
      is_continuous: Python boolean. If `True` this
        `Distribution` is continuous over its supported domain.
      is_reparameterized: Python boolean. If `True` this
        `Distribution` can be reparameterized in terms of some standard
        distribution with a function whose Jacobian is constant for the support
        of the standard distribution.
      validate_args: Python boolean.  Whether to validate input with asserts.
        If `validate_args` is `False`, and the inputs are invalid,
        correct behavior is not guaranteed.
      allow_nan_stats: Python boolean.  If `False`, raise an
        exception if a statistic (e.g., mean, mode) is undefined for any batch
        member. If True, batch members with valid parameters leading to
        undefined statistics will return `NaN` for this statistic.
      parameters: Python dictionary of parameters used to instantiate this
        `Distribution`.
      graph_parents: Python list of graph prerequisites of this `Distribution`.
      name: A name for this distribution. Default: subclass name.

    Raises:
      ValueError: if any member of graph_parents is `None` or not a `Tensor`.
    """
    graph_parents = [] if graph_parents is None else graph_parents
    for i, t in enumerate(graph_parents):
      if t is None or not contrib_framework.is_tensor(t):
        raise ValueError("Graph parent item %d is not a Tensor; %s." % (i, t))
    parameters = parameters or {}
    self._dtype = dtype
    self._is_continuous = is_continuous
    self._is_reparameterized = is_reparameterized
    self._allow_nan_stats = allow_nan_stats
    self._validate_args = validate_args
    self._parameters = parameters
    self._graph_parents = graph_parents
    self._name = name or type(self).__name__

  @classmethod
  def param_shapes(cls, sample_shape, name="DistributionParamShapes"):
    """Shapes of parameters given the desired shape of a call to `sample()`.

    Subclasses should override static method `_param_shapes`.

    Args:
      sample_shape: `Tensor` or python list/tuple. Desired shape of a call to
        `sample()`.
      name: name to prepend ops with.

    Returns:
      `dict` of parameter name to `Tensor` shapes.
    """
    with ops.name_scope(name, values=[sample_shape]):
      return cls._param_shapes(sample_shape)

  @classmethod
  def param_static_shapes(cls, sample_shape):
    """param_shapes with static (i.e. TensorShape) shapes.

    Args:
      sample_shape: `TensorShape` or python list/tuple. Desired shape of a call
        to `sample()`.

    Returns:
      `dict` of parameter name to `TensorShape`.

    Raises:
      ValueError: if `sample_shape` is a `TensorShape` and is not fully defined.
    """
    if isinstance(sample_shape, tensor_shape.TensorShape):
      if not sample_shape.is_fully_defined():
        raise ValueError("TensorShape sample_shape must be fully defined")
      sample_shape = sample_shape.as_list()

    params = cls.param_shapes(sample_shape)

    static_params = {}
    for name, shape in params.items():
      static_shape = tensor_util.constant_value(shape)
      if static_shape is None:
        raise ValueError(
            "sample_shape must be a fully-defined TensorShape or list/tuple")
      static_params[name] = tensor_shape.TensorShape(static_shape)

    return static_params

  @staticmethod
  def _param_shapes(sample_shape):
    raise NotImplementedError("_param_shapes not implemented")

  @property
  def name(self):
    """Name prepended to all ops created by this `Distribution`."""
    return self._name

  @property
  def dtype(self):
    """The `DType` of `Tensor`s handled by this `Distribution`."""
    return self._dtype

  @property
  def parameters(self):
    """Dictionary of parameters used to instantiate this `Distribution`."""
    return self._parameters

  @property
  def is_continuous(self):
    return self._is_continuous

  @property
  def is_reparameterized(self):
    return self._is_reparameterized

  @property
  def allow_nan_stats(self):
    """Python boolean describing behavior when a stat is undefined.

    Stats return +/- infinity when it makes sense.  E.g., the variance
    of a Cauchy distribution is infinity.  However, sometimes the
    statistic is undefined, e.g., if a distribution's pdf does not achieve a
    maximum within the support of the distribution, the mode is undefined.
    If the mean is undefined, then by definition the variance is undefined.
    E.g. the mean for Student's T for df = 1 is undefined (no clear way to say
    it is either + or - infinity), so the variance = E[(X - mean)^2] is also
    undefined.

    Returns:
      allow_nan_stats: Python boolean.
    """
    return self._allow_nan_stats

  @property
  def validate_args(self):
    """Python boolean indicated possibly expensive checks are enabled."""
    return self._validate_args

  def copy(self, **override_parameters_kwargs):
    """Creates a deep copy of the distribution.

    Note: the copy distribution may continue to depend on the original
    intialization arguments.

    Args:
      **override_parameters_kwargs: String/value dictionary of initialization
        arguments to override with new values.

    Returns:
      distribution: A new instance of `type(self)` intitialized from the union
        of self.parameters and override_parameters_kwargs, i.e.,
        `dict(self.parameters, **override_parameters_kwargs)`.
    """
    parameters = dict(self.parameters, **override_parameters_kwargs)
    # Python3 leaks "__class__" into `locals()` so we remove if present.
    # TODO(b/32376812): Remove this pop.
    parameters.pop("__class__", None)
    return type(self)(**parameters)

  def _batch_shape(self):
    raise NotImplementedError("batch_shape is not implemented")

  def batch_shape(self, name="batch_shape"):
    """Shape of a single sample from a single event index as a 1-D `Tensor`.

    The product of the dimensions of the `batch_shape` is the number of
    independent distributions of this kind the instance represents.

    Args:
      name: name to give to the op

    Returns:
      batch_shape: `Tensor`.
    """
    with self._name_scope(name):
      if self.get_batch_shape().is_fully_defined():
        return ops.convert_to_tensor(self.get_batch_shape().as_list(),
                                     dtype=dtypes.int32,
                                     name="batch_shape")
      return self._batch_shape()

  def _get_batch_shape(self):
    return tensor_shape.TensorShape(None)

  def get_batch_shape(self):
    """Shape of a single sample from a single event index as a `TensorShape`.

    Same meaning as `batch_shape`. May be only partially defined.

    Returns:
      batch_shape: `TensorShape`, possibly unknown.
    """
    return self._get_batch_shape()

  def _event_shape(self):
    raise NotImplementedError("event_shape is not implemented")

  def event_shape(self, name="event_shape"):
    """Shape of a single sample from a single batch as a 1-D int32 `Tensor`.

    Args:
      name: name to give to the op

    Returns:
      event_shape: `Tensor`.
    """
    with self._name_scope(name):
      if self.get_event_shape().is_fully_defined():
        return ops.convert_to_tensor(self.get_event_shape().as_list(),
                                     dtype=dtypes.int32,
                                     name="event_shape")
      return self._event_shape()

  def _get_event_shape(self):
    return tensor_shape.TensorShape(None)

  def get_event_shape(self):
    """Shape of a single sample from a single batch as a `TensorShape`.

    Same meaning as `event_shape`. May be only partially defined.

    Returns:
      event_shape: `TensorShape`, possibly unknown.
    """
    return self._get_event_shape()

  @property
  def is_scalar_event(self):
    """Indicates that `event_shape==[]`."""
    return ops.convert_to_tensor(
        self._is_scalar_helper(self.get_event_shape, self.event_shape),
        name="is_scalar_event")

  @property
  def is_scalar_batch(self):
    """Indicates that `batch_shape==[]`."""
    return ops.convert_to_tensor(
        self._is_scalar_helper(self.get_batch_shape, self.batch_shape),
        name="is_scalar_batch")

  def _sample_n(self, n, seed=None):
    raise NotImplementedError("sample_n is not implemented")

  def sample(self, sample_shape=(), seed=None, name="sample",
             **condition_kwargs):
    """Generate samples of the specified shape.

    Note that a call to `sample()` without arguments will generate a single
    sample.

    Args:
      sample_shape: 0D or 1D `int32` `Tensor`. Shape of the generated samples.
      seed: Python integer seed for RNG
      name: name to give to the op.
      **condition_kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      samples: a `Tensor` with prepended dimensions `sample_shape`.
    """
    with self._name_scope(name, values=[sample_shape]):
      sample_shape = ops.convert_to_tensor(
          sample_shape, dtype=dtypes.int32, name="sample_shape")
      if sample_shape.get_shape().ndims == 0:
        return self.sample_n(sample_shape, seed, **condition_kwargs)
      sample_shape, total = self._expand_sample_shape(sample_shape)
      samples = self.sample_n(total, seed, **condition_kwargs)
      output_shape = array_ops.concat_v2(
          [sample_shape, array_ops.slice(array_ops.shape(samples), [1], [-1])],
          0)
      output = array_ops.reshape(samples, output_shape)
      output.set_shape(tensor_util.constant_value_as_shape(
          sample_shape).concatenate(samples.get_shape()[1:]))
      return output

  def sample_n(self, n, seed=None, name="sample_n", **condition_kwargs):
    """Generate `n` samples.

    Args:
      n: `Scalar` `Tensor` of type `int32` or `int64`, the number of
        observations to sample.
      seed: Python integer seed for RNG
      name: name to give to the op.
      **condition_kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      samples: a `Tensor` with a prepended dimension (n,).

    Raises:
      TypeError: if `n` is not an integer type.
    """
    warnings.warn("Please use `sample` instead of `sample_n`. `sample_n` "
                  "will be deprecated in December 2016.",
                  PendingDeprecationWarning)
    with self._name_scope(name, values=[n]):
      n = ops.convert_to_tensor(n, name="n")
      if not n.dtype.is_integer:
        raise TypeError("n.dtype=%s is not an integer type" % n.dtype)
      x = self._sample_n(n, seed, **condition_kwargs)

      # Set shape hints.
      sample_shape = tensor_shape.TensorShape(
          tensor_util.constant_value(n))
      batch_ndims = self.get_batch_shape().ndims
      event_ndims = self.get_event_shape().ndims
      if batch_ndims is not None and event_ndims is not None:
        inferred_shape = sample_shape.concatenate(
            self.get_batch_shape().concatenate(
                self.get_event_shape()))
        x.set_shape(inferred_shape)
      elif x.get_shape().ndims is not None and x.get_shape().ndims > 0:
        x.get_shape()[0].merge_with(sample_shape[0])
        if batch_ndims is not None and batch_ndims > 0:
          x.get_shape()[1:1+batch_ndims].merge_with(self.get_batch_shape())
        if event_ndims is not None and event_ndims > 0:
          x.get_shape()[-event_ndims:].merge_with(self.get_event_shape())

      return x

  def _log_prob(self, value):
    raise NotImplementedError("log_prob is not implemented")

  def log_prob(self, value, name="log_prob", **condition_kwargs):
    """Log probability density/mass function (depending on `is_continuous`).

    Args:
      value: `float` or `double` `Tensor`.
      name: The name to give this op.
      **condition_kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      log_prob: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.
    """
    with self._name_scope(name, values=[value]):
      value = ops.convert_to_tensor(value, name="value")
      try:
        return self._log_prob(value, **condition_kwargs)
      except NotImplementedError as original_exception:
        try:
          return math_ops.log(self._prob(value, **condition_kwargs))
        except NotImplementedError:
          raise original_exception

  def prob(self, value, name="prob", **condition_kwargs):
    """Probability density/mass function (depending on `is_continuous`).

    Args:
      value: `float` or `double` `Tensor`.
      name: The name to give this op.
      **condition_kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      prob: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.
    """
    with self._name_scope(name, values=[value]):
      value = ops.convert_to_tensor(value, name="value")
      try:
        return self._prob(value, **condition_kwargs)
      except NotImplementedError as original_exception:
        try:
          return math_ops.exp(self._log_prob(value, **condition_kwargs))
        except NotImplementedError:
          raise original_exception

  def _log_cdf(self, value):
    raise NotImplementedError("log_cdf is not implemented")

  def log_cdf(self, value, name="log_cdf", **condition_kwargs):
    """Log cumulative distribution function.

    Given random variable `X`, the cumulative distribution function `cdf` is:

    ```
    log_cdf(x) := Log[ P[X <= x] ]
    ```

    Often, a numerical approximation can be used for `log_cdf(x)` that yields
    a more accurate answer than simply taking the logarithm of the `cdf` when
    `x << -1`.

    Args:
      value: `float` or `double` `Tensor`.
      name: The name to give this op.
      **condition_kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      logcdf: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.
    """
    with self._name_scope(name, values=[value]):
      value = ops.convert_to_tensor(value, name="value")
      try:
        return self._log_cdf(value, **condition_kwargs)
      except NotImplementedError as original_exception:
        try:
          return math_ops.log(self._cdf(value, **condition_kwargs))
        except NotImplementedError:
          raise original_exception

  def _cdf(self, value):
    raise NotImplementedError("cdf is not implemented")

  def cdf(self, value, name="cdf", **condition_kwargs):
    """Cumulative distribution function.

    Given random variable `X`, the cumulative distribution function `cdf` is:

    ```
    cdf(x) := P[X <= x]
    ```

    Args:
      value: `float` or `double` `Tensor`.
      name: The name to give this op.
      **condition_kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      cdf: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.
    """
    with self._name_scope(name, values=[value]):
      value = ops.convert_to_tensor(value, name="value")
      try:
        return self._cdf(value, **condition_kwargs)
      except NotImplementedError as original_exception:
        try:
          return math_ops.exp(self._log_cdf(value, **condition_kwargs))
        except NotImplementedError:
          raise original_exception

  def _log_survival_function(self, value):
    raise NotImplementedError("log_survival_function is not implemented")

  def log_survival_function(self, value, name="log_survival_function",
                            **condition_kwargs):
    """Log survival function.

    Given random variable `X`, the survival function is defined:

    ```
    log_survival_function(x) = Log[ P[X > x] ]
                             = Log[ 1 - P[X <= x] ]
                             = Log[ 1 - cdf(x) ]
    ```

    Typically, different numerical approximations can be used for the log
    survival function, which are more accurate than `1 - cdf(x)` when `x >> 1`.

    Args:
      value: `float` or `double` `Tensor`.
      name: The name to give this op.
      **condition_kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      `Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type
        `self.dtype`.
    """
    with self._name_scope(name, values=[value]):
      value = ops.convert_to_tensor(value, name="value")
      try:
        return self._log_survival_function(value, **condition_kwargs)
      except NotImplementedError as original_exception:
        try:
          return math_ops.log(1. - self.cdf(value, **condition_kwargs))
        except NotImplementedError:
          raise original_exception

  def _survival_function(self, value):
    raise NotImplementedError("survival_function is not implemented")

  def survival_function(self, value, name="survival_function",
                        **condition_kwargs):
    """Survival function.

    Given random variable `X`, the survival function is defined:

    ```
    survival_function(x) = P[X > x]
                         = 1 - P[X <= x]
                         = 1 - cdf(x).
    ```

    Args:
      value: `float` or `double` `Tensor`.
      name: The name to give this op.
      **condition_kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type
        `self.dtype`.
    """
    with self._name_scope(name, values=[value]):
      value = ops.convert_to_tensor(value, name="value")
      try:
        return self._survival_function(value, **condition_kwargs)
      except NotImplementedError as original_exception:
        try:
          return 1. - self.cdf(value, **condition_kwargs)
        except NotImplementedError:
          raise original_exception

  def _entropy(self):
    raise NotImplementedError("entropy is not implemented")

  def entropy(self, name="entropy"):
    """Shannon entropy in nats."""
    with self._name_scope(name):
      return self._entropy()

  def _mean(self):
    raise NotImplementedError("mean is not implemented")

  def mean(self, name="mean"):
    """Mean."""
    with self._name_scope(name):
      return self._mean()

  def _variance(self):
    raise NotImplementedError("variance is not implemented")

  def variance(self, name="variance"):
    """Variance."""
    with self._name_scope(name):
      return self._variance()

  def _std(self):
    raise NotImplementedError("std is not implemented")

  def std(self, name="std"):
    """Standard deviation."""
    with self._name_scope(name):
      return self._std()

  def _mode(self):
    raise NotImplementedError("mode is not implemented")

  def mode(self, name="mode"):
    """Mode."""
    with self._name_scope(name):
      return self._mode()

  def log_pdf(self, value, name="log_pdf", **condition_kwargs):
    """Log probability density function.

    Args:
      value: `float` or `double` `Tensor`.
      name: The name to give this op.
      **condition_kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      log_prob: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.

    Raises:
      TypeError: if not `is_continuous`.
    """
    warnings.warn("Please use `log_prob` instead of `log_pdf`. `log_pdf` "
                  "will be deprecated in December 2016.",
                  PendingDeprecationWarning)
    if not self.is_continuous:
      raise TypeError("log_pdf is undefined for non-continuous distributions.")
    return self.log_prob(value, name=name, **condition_kwargs)

  def pdf(self, value, name="pdf", **condition_kwargs):
    """Probability density function.

    Args:
      value: `float` or `double` `Tensor`.
      name: The name to give this op.
      **condition_kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      prob: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.

    Raises:
      TypeError: if not `is_continuous`.
    """
    warnings.warn("Please use `prob` instead of `pdf`. `pdf` will be "
                  "deprecated in December 2016.",
                  PendingDeprecationWarning)
    if not self.is_continuous:
      raise TypeError("pdf is undefined for non-continuous distributions.")
    return self.prob(value, name, **condition_kwargs)

  def log_pmf(self, value, name="log_pmf", **condition_kwargs):
    """Log probability mass function.

    Args:
      value: `float` or `double` `Tensor`.
      name: The name to give this op.
      **condition_kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      log_pmf: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.

    Raises:
      TypeError: if `is_continuous`.
    """
    warnings.warn("Please use `log_prob` instead of `log_pmf`. `log_pmf` will "
                  "be deprecated in December 2016.",
                  PendingDeprecationWarning)
    if self.is_continuous:
      raise TypeError("log_pmf is undefined for continuous distributions.")
    return self.log_prob(value, name=name, **condition_kwargs)

  def pmf(self, value, name="pmf", **condition_kwargs):
    """Probability mass function.

    Args:
      value: `float` or `double` `Tensor`.
      name: The name to give this op.
      **condition_kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      pmf: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.

    Raises:
      TypeError: if `is_continuous`.
    """
    warnings.warn("Please use `prob` instead of `pmf`. `pmf` will be "
                  "deprecated in December 2016.",
                  PendingDeprecationWarning)
    if self.is_continuous:
      raise TypeError("pmf is undefined for continuous distributions.")
    return self.prob(value, name=name, **condition_kwargs)

  @contextlib.contextmanager
  def _name_scope(self, name=None, values=None):
    """Helper function to standardize op scope."""
    with ops.name_scope(self.name):
      with ops.name_scope(name, values=(
          (values or []) + self._graph_parents)) as scope:
        yield scope

  def _expand_sample_shape(self, sample_shape):
    """Helper to `sample` which ensures sample_shape is 1D."""
    sample_shape_static_val = tensor_util.constant_value(sample_shape)
    ndims = sample_shape.get_shape().ndims
    if sample_shape_static_val is None:
      if ndims is None or not sample_shape.get_shape().is_fully_defined():
        ndims = array_ops.rank(sample_shape)
      expanded_shape = distribution_util.pick_vector(
          math_ops.equal(ndims, 0),
          np.array((1,), dtype=dtypes.int32.as_numpy_dtype()),
          array_ops.shape(sample_shape))
      sample_shape = array_ops.reshape(sample_shape, expanded_shape)
      total = math_ops.reduce_prod(sample_shape)  # reduce_prod([]) == 1
    else:
      if ndims is None:
        raise ValueError(
            "Shouldn't be here; ndims cannot be none when we have a "
            "tf.constant shape.")
      if ndims == 0:
        sample_shape_static_val = np.reshape(sample_shape_static_val, [1])
        sample_shape = ops.convert_to_tensor(
            sample_shape_static_val,
            dtype=dtypes.int32,
            name="sample_shape")
      total = np.prod(sample_shape_static_val,
                      dtype=dtypes.int32.as_numpy_dtype())
    return sample_shape, total

  def _is_scalar_helper(self, static_shape_fn, dynamic_shape_fn):
    """Implementation for `is_scalar_batch` and `is_scalar_event`."""
    if static_shape_fn().ndims is not None:
      return static_shape_fn().ndims == 0
    shape = dynamic_shape_fn()
    if (shape.get_shape().ndims is not None and
        shape.get_shape()[0].value is not None):
      # If the static_shape_fn is correctly written then we should never execute
      # this branch. We keep it just in case there's some unimagined corner
      # case.
      return shape.get_shape().as_list() == [0]
    return math_ops.equal(array_ops.shape(shape)[0], 0)
