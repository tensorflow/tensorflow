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
import numpy as np
import six

from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


@six.add_metaclass(abc.ABCMeta)
class BaseDistribution(object):
  """Simple abstract base class for probability distributions.

  Implementations of core distributions to be included in the `distributions`
  module should subclass `Distribution`. This base class may be useful to users
  that want to fulfill a simpler distribution contract.
  """

  @abc.abstractmethod
  def sample_n(self, n, seed=None, name="sample"):
    # See `Distribution.sample_n` for docstring.
    pass

  @abc.abstractmethod
  def log_prob(self, value, name="log_prob"):
    # See `Distribution.log_prob` for docstring.
    pass


class Distribution(BaseDistribution):
  """A generic probability distribution base class.

  `Distribution` is a base class for constructing and organizing properties
  (e.g., mean, variance) of random variables (e.g, Bernoulli, Gaussian).

  ### Subclassing

  Subclasess are expected to implement a leading-underscore version of the
  same-named function.  The argument signature should be identical except for
  the omission of `name="..."`.  For example, to enable `log_prob(value,
  name="log_prob")` a subclass should implement `_log_prob(value)`.

  Subclasses can rewrite/append to public-level docstrings. For example,

  ```python
  Subclass.prob.__func__.__doc__ += "Some other details."
  ```

  would add the string "Some other details." to the `prob` function docstring.

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
  not necessarily so along the `event_shape` dimensions (dependending on the
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
               parameters,
               is_continuous,
               is_reparameterized,
               validate_args,
               allow_nan_stats,
               name=None):
    """Constructs the `Distribution`.

    **This is a private method for subclass use.**

    Args:
      dtype: The type of the event samples. `None` implies no type-enforcement.
      parameters: Python dictionary of parameters used by this `Distribution`.
      is_continuous: Python boolean. If `True` this
        `Distribution` is continuous over its supported domain.
      is_reparameterized: Python boolean. If `True` this
        `Distribution` can be reparameterized in terms of some standard
        distribution with a function whose Jacobian is constant for the support
        of the standard distribution.
      validate_args: Python boolean.  Whether to validate input with asserts.
        If `validate_args` is `False`, and the inputs are invalid,
        correct behavior is not guaranteed.
      allow_nan_stats: Pytho nboolean.  If `False`, raise an
        exception if a statistic (e.g., mean, mode) is undefined for any batch
        member. If True, batch members with valid parameters leading to
        undefined statistics will return `NaN` for this statistic.
      name: A name for this distribution (optional).
    """
    self._name = name
    if self._name is None:
      with ops.name_scope(type(self).__name__) as ns:
        self._name = ns
    self._dtype = dtype
    self._parameters = parameters or {}
    self._is_continuous = is_continuous
    self._is_reparameterized = is_reparameterized
    self._allow_nan_stats = allow_nan_stats
    self._validate_args = validate_args

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
    """Dictionary of parameters used by this `Distribution`."""
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

  def batch_shape(self, name="batch_shape"):
    """Shape of a single sample from a single event index as a 1-D `Tensor`.

    The product of the dimensions of the `batch_shape` is the number of
    independent distributions of this kind the instance represents.

    Args:
      name: name to give to the op

    Returns:
      batch_shape: `Tensor`.
    """
    self._check_hasattr(self._batch_shape)
    with self._name_scope(name):
      return self._batch_shape()

  def get_batch_shape(self):
    """Shape of a single sample from a single event index as a `TensorShape`.

    Same meaning as `batch_shape`. May be only partially defined.

    Returns:
      batch_shape: `TensorShape`, possibly unknown.
    """
    self._check_hasattr(self._get_batch_shape)
    return self._get_batch_shape()

  def event_shape(self, name="event_shape"):
    """Shape of a single sample from a single batch as a 1-D int32 `Tensor`.

    Args:
      name: name to give to the op

    Returns:
      event_shape: `Tensor`.
    """
    self._check_hasattr(self._event_shape)
    with self._name_scope(name):
      return self._event_shape()

  def get_event_shape(self):
    """Shape of a single sample from a single batch as a `TensorShape`.

    Same meaning as `event_shape`. May be only partially defined.

    Returns:
      event_shape: `TensorShape`, possibly unknown.
    """
    self._check_hasattr(self._get_event_shape)
    return self._get_event_shape()

  def sample(self, sample_shape=(), seed=None, name="sample"):
    """Generate samples of the specified shape.

    Note that a call to `sample()` without arguments will generate a single
    sample.

    Args:
      sample_shape: 0D or 1D `int32` `Tensor`. Shape of the generated samples.
      seed: Python integer seed for RNG
      name: name to give to the op.

    Returns:
      samples: a `Tensor` with prepended dimensions `sample_shape`.
    """
    with self._name_scope(name, values=[sample_shape]):
      sample_shape = ops.convert_to_tensor(
          sample_shape, dtype=dtypes.int32, name="sample_shape")
      if sample_shape.get_shape().ndims == 0:
        return self.sample_n(sample_shape, seed)
      sample_shape, total = self._expand_sample_shape(sample_shape)
      samples = self.sample_n(total, seed)
      output_shape = array_ops.concat(0, [sample_shape, array_ops.slice(
          array_ops.shape(samples), [1], [-1])])
      output = array_ops.reshape(samples, output_shape)
      output.set_shape(tensor_util.constant_value_as_shape(
          sample_shape).concatenate(samples.get_shape()[1:]))
      return output

  def sample_n(self, n, seed=None, name="sample_n"):
    """Generate `n` samples.

    Args:
      n: `Scalar` `Tensor` of type `int32` or `int64`, the number of
        observations to sample.
      seed: Python integer seed for RNG
      name: name to give to the op.

    Returns:
      samples: a `Tensor` with a prepended dimension (n,).

    Raises:
      TypeError: if `n` is not an integer type.
    """
    self._check_hasattr(self._sample_n)
    with self._name_scope(name, values=[n]):
      n = ops.convert_to_tensor(n, name="n")
      if not n.dtype.is_integer:
        raise TypeError("n.dtype=%s is not an integer type" % n.dtype)
      x = self._sample_n(n, seed)

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
        x.get_shape()[0].merge_with(sample_shape)
        if batch_ndims is not None and batch_ndims > 0:
          x.get_shape()[1:1+batch_ndims].merge_with(self.get_batch_shape())
        if event_ndims is not None and event_ndims > 0:
          x.get_shape()[-event_ndims:].merge_with(self.get_event_shape())

      return x

  def log_prob(self, value, name="log_prob"):
    """Log probability density/mass function (depending on `is_continuous`).

    Args:
      value: `float` or `double` `Tensor`.
      name: The name to give this op.

    Returns:
      log_prob: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.
    """
    self._check_hasattr(self._log_prob)
    with self._name_scope(name, values=[value]):
      value = ops.convert_to_tensor(value, name="value")
      return self._log_prob(value)

  def prob(self, value, name="prob"):
    """Probability density/mass function (depending on `is_continuous`).

    Args:
      value: `float` or `double` `Tensor`.
      name: The name to give this op.

    Returns:
      prob: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.
    """
    self._check_hasattr(self._prob)
    with self._name_scope(name, values=[value]):
      value = ops.convert_to_tensor(value, name="value")
      return self._prob(value)

  def log_cdf(self, value, name="log_cdf"):
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

    Returns:
      logcdf: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.
    """
    self._check_hasattr(self._log_cdf)
    with self._name_scope(name, values=[value]):
      value = ops.convert_to_tensor(value, name="value")
      return self._log_cdf(value)

  def cdf(self, value, name="cdf"):
    """Cumulative distribution function.

    Given random variable `X`, the cumulative distribution function `cdf` is:

    ```
    cdf(x) := P[X <= x]
    ```

    Args:
      value: `float` or `double` `Tensor`.
      name: The name to give this op.

    Returns:
      cdf: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.
    """
    self._check_hasattr(self._cdf)
    with self._name_scope(name, values=[value]):
      value = ops.convert_to_tensor(value, name="value")
      return self._cdf(value)

  def log_survival_function(self, value, name="log_survival_function"):
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

    Returns:
      `Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type
        `self.dtype`.
    """
    self._check_hasattr(self._log_survival_function)
    with self._name_scope(name, values=[value]):
      value = ops.convert_to_tensor(value, name="value")
      return self._log_survival_function(value)

  def survival_function(self, value, name="survival_function"):
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

    Returns:
      Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type
        `self.dtype`.
    """
    self._check_hasattr(self._survival_function)
    with self._name_scope(name, values=[value]):
      value = ops.convert_to_tensor(value, name="value")
      return self._survival_function(value)

  def entropy(self, name="entropy"):
    """Shanon entropy in nats."""
    self._check_hasattr(self._entropy)
    with self._name_scope(name):
      return self._entropy()

  def mean(self, name="mean"):
    """Mean."""
    self._check_hasattr(self._mean)
    with self._name_scope(name):
      return self._mean()

  def variance(self, name="variance"):
    """Variance."""
    self._check_hasattr(self._variance)
    with self._name_scope(name):
      return self._variance()

  def std(self, name="std"):
    """Standard deviation."""
    self._check_hasattr(self._std)
    with self._name_scope(name):
      return self._std()

  def mode(self, name="mode"):
    """Mode."""
    self._check_hasattr(self._mode)
    with self._name_scope(name):
      return self._mode()

  def log_pdf(self, value, name="log_pdf"):
    """Log probability density function.

    Args:
      value: `float` or `double` `Tensor`.
      name: The name to give this op.

    Returns:
      log_prob: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.

    Raises:
      AttributeError: if not `is_continuous`.
    """
    if not self.is_continuous:
      raise AttributeError(
          "log_pdf is undefined for non-continuous distributions.")
    return self.log_prob(value, name=name)

  def pdf(self, value, name="pdf"):
    """Probability density function.

    Args:
      value: `float` or `double` `Tensor`.
      name: The name to give this op.

    Returns:
      prob: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.

    Raises:
      AttributeError: if not `is_continuous`.
    """
    if not self.is_continuous:
      raise AttributeError("pdf is undefined for non-continuous distributions.")
    return self.prob(value, name)

  def log_pmf(self, value, name="log_pmf"):
    """Log probability mass function.

    Args:
      value: `float` or `double` `Tensor`.
      name: The name to give this op.

    Returns:
      log_pmf: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.

    Raises:
      AttributeError: if `is_continuous`.
    """
    if self.is_continuous:
      raise AttributeError("log_pmf is undefined for continuous distributions.")
    return self.log_prob(value, name=name)

  def pmf(self, value, name="pmf"):
    """Probability mass function.

    Args:
      value: `float` or `double` `Tensor`.
      name: The name to give this op.

    Returns:
      pmf: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.

    Raises:
      AttributeError: if `is_continuous`.
    """
    if self.is_continuous:
      raise AttributeError("pmf is undefined for continuous distributions.")
    return self.prob(value, name=name)

  @contextlib.contextmanager
  def _name_scope(self, name=None, values=None):
    """Helper function to standardize op scope."""
    with ops.name_scope(self.name):
      with ops.name_scope(name, values=(
          (values or []) + list(self.parameters.values()))) as scope:
        yield scope

  def _check_hasattr(self, func):
    if hasattr(self, func.__func__.__name__) and callable(func): return
    raise NotImplementedError(
        "Subclass %s does not implement %s" %
        (type(self).__name__, func.__func__.__name__))

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


distribution_util.append_class_fun_doc(BaseDistribution.sample_n,
                                       doc_str=Distribution.sample_n.__doc__)
distribution_util.append_class_fun_doc(BaseDistribution.log_prob,
                                       doc_str=Distribution.log_prob.__doc__)
