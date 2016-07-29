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
"""A helper class for inferring Distribution shape."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops


class _ShapeUtil(object):
  """Class which helps infer/identify subsets of tensor dimensions.

  Terminology:
    Recall that a `Tensor` has:
      shape: sizes of tensor dimensions,
      ndims: size of shape; number of tensor dimensions,
       dims: indexes into shape; useful for transpose, reduce.

    Tensors sampled from a `Distribution` can be partitioned by:
      sample dims: indexes independent, identically distributed (iid) draws,
      batch dims:  indexes non-identical draws,
      event dims:  indexes coordinates of a single draw.

    The sample, batch, and event dimensions constitute the entirety of a
    `Tensor` shape.  The dimensions are always in sample, batch, event order.

  Assumptions:
    We assume that batch_ndims and event_ndims are statically known for both
    creating this object and for inputs to its functions.
    TODO(jvdillon): Relax this assumption and support fully unknown shape.

    We also assume that the `Tensor` rank is static, i.e., `x.get_shape().ndims
    is not None`.

  Possible use-cases:
    ~ Sample dimensions:
      Computing summary statistics, i.e., the average is a reduction over sample
      dimensions.

    ~ Batch dimensions:
      Log-likelihood under model predicted location:
      ```python
      mu = ... # vector of predictions, one for each covariate.
      neg_log_likelihood = -tf.reduce_mean(
        Normal(loc=mu, scale=1).log_pdf(x),
        reduce_dims=[0])
      ```

      Monte Carlo estimation of a marginal probability:
      Average over batch dimensions where batch dimensions are associated with
      random draws of a prior.
      E.g., suppose we want to find the Monte Carlo estimate of the marginal
      distribution of a Normal with a random Laplace location:
      ```
        P(X=x) = integral P(X=x|y) P(Y=y) dy
              ~= 1/n sum_{i=1}^n P(X=x|y_i),   y_i ~iid Laplace(0,1)
               = tf.reduce_mean(Normal(loc=Laplace(0, 1).sample_n(n=1000),
                                       scale=tf.ones([1000, 1])).pdf(x),
                                reduce_dims=[0])
      ```

      The `Laplace` distribution generates a tensor of shape [1000, 1]. When fed
      to a `Normal`, this is interpreted as 1000 different locations, i.e.,
      1000 non-identical Normals.  Therefore a single call to pdf(x) yields 1000
      probabilities, one for every location.  The average over this batch yields
      the marginal.

    ~ Event dimensions:
      Computing the determinant of the Jacobian of a function of a random
      variable involves a reduction over event dimensions.

  Examples:
    Write S, B, E for sample shape, batch shape, and event shape (resp.).

    ```python
    x.get_shape() == S + B + E  # For statically known x shape.

    # 100 iid samples from one multivariate Normal with two
    # degrees of freedom (DF).
    mu = [0., 0]
    sigma = [[1., 0],
             [0,  1]]
    X = MultivariateNormal(loc=mu, scale=sigma).sample_n(n=100)
    # S = [100]
    # B = []
    # E = [2]

    # 100 iid samples from one Wishart with 2x2 DF.
    sigma = [[1., 0],
             [0,  1]]
    X = Wishart(scale=sigma).sample_n(n=100)
    # S = [100]
    # B = []
    # E = [2, 2]

    # 100 iid samples (with shape [2, 50]) from two, non-identical bivariate
    # Normal distributions.
    mu    = ... # shape(2, 2)
    sigma = ... # shape(2, 2, 2)
    X = MultivariateNormal(loc=mu, scale=sigma).sample(shape=[2, 50])
    # S = [2, 50]
    # B = [2]
    # E = [2]
    ```

  """

  def __init__(self, batch_ndims=None, event_ndims=None, name='ShapeUtil'):
    """Construct ShapeUtil with known sample, batch, and/or event ndims.

    Typically, batch_ndims and event_ndims are fixed throughout the lifetime of
    a Distribution.

    Args:
      batch_ndims: number of dims (rank) of the batch portion of indexes of a
        `Tensor`.  A "batch" is a non-identical distribution, i.e, Normal with
        different parameters.
      event_ndims: number of dims (rank) of the event portion of indexes of a
        `Tensor`. An "event" is what is sampled from a distribution, i.e., a
        trivariate Normal has an event shape of [3] and a 4 dimensional Wishart
        has an event shape of [4, 4].
      name: `String`. The name to give Ops created by this class.

    Raises:
      ValueError: if batch_ndims or event_ndims are invalid.
    """
    if batch_ndims < 0:
      raise ValueError('must specify non-negative batch_ndims(%d)', batch_ndims)
    if batch_ndims > 0 and event_ndims < 1:
      raise ValueError('must specify positive event_ndims(%d) when '
                       'batch_ndims(%d) is positive', event_ndims, batch_ndims)
    # TODO(jvdillon): Support batches of scalars.
    self._name = name
    self._batch_ndims = batch_ndims
    self._event_ndims = event_ndims

  @property
  def name(self):
    """Name given to ops created by this class."""
    return self._name

  @property
  def batch_ndims(self):
    """Returns number of dimensions corresponding to non-identical draws."""
    return self._batch_ndims

  @property
  def event_ndims(self):
    """Returns number of dimensions needed to index a sample's coordinates."""
    return self._event_ndims

  def get_ndims(self, x, name='get_ndims'):
    """Get tensor ndims (rank).

    Args:
      x: `Tensor`.
      name: `String`. The name to give this op.

    Raises:
      ValueError: if ndims is not statically known.

    Returns:
      `Scalar` number of dimensions associated with a `Tensor`.
    """
    if x is None:
      raise ValueError('Input was None which does not have known ndims.')
    with ops.name_scope(self.name):
      with ops.op_scope([x], name):
        ndims = ops.convert_to_tensor(x).get_shape().ndims
        if ndims is None:
          raise ValueError('ShapeUtil assumes static number of '
                           'dimensions(%d)', ndims)
        return ndims

  def get_sample_ndims(self, x):
    """Returns number of dimensions corresponding to iid draws.

    Args:
      x: `Tensor`.

    Raises:
      ValueError: if batch_ndims or event_ndims are not statically known.
      ValueError: if static sample_ndims does not match inferred

    Returns:
      Scalar number of dimensions associated with a sample.
    """
    ndims = self.get_ndims(x)
    sample_ndims = ndims - self.batch_ndims - self.event_ndims
    if sample_ndims < 0:
      raise ValueError('expected batch_ndims(%d) + event_ndims(%d) < ndims(%d)',
                       self.batch_ndims, self.event_ndims, ndims)
    return sample_ndims

  def get_dims(self, x, sample=True, batch=True, event=True):
    """Returns subset of tensor's dimension indexes (indexes into shape).

    Args:
      x: `Tensor`.
      sample: `Boolean`. Include sample dimensions or not.
      batch: `Boolean`. Include batch dimensions or not.
      event: `Boolean`. Include event dimensions or not.

    Raises:
      ValueError: if `x.get_shape().ndims` is `None`

    Returns:
      List enumerating requested dimensions.
    """
    ndims = self.get_ndims(x)

    if sample and batch and event:
      return list(range(ndims))

    sample_start = 0
    batch_start = self.get_sample_ndims(x)
    event_start = batch_start + self.batch_ndims

    sample_shape = list(range(sample_start, batch_start)) if sample else []
    batch_shape = list(range(batch_start, event_start)) if batch else []
    event_shape = list(range(event_start, ndims)) if event else []

    return sample_shape + batch_shape + event_shape

  def get_shape(self, x, sample=True, batch=True, event=True, name='get_shape'):
    """Returns subset of tensor's shape (size of dimensions).

    Args:
      x: `Tensor`.
      sample: `Boolean`. Include sample shape or not.
      batch: `Boolean`. Include batch shape or not.
      event: `Boolean`. Include event shape or not.
      name: `String`. The name to give this op.

    Raises:
      ValueError: if `x.get_shape().ndims` is `None`

    Returns:
      List describing event shape if known statically, `Tensor` otherwise.
    """
    if not sample and not batch and not event:
      return []
    with ops.name_scope(self._name):
      with ops.op_scope([x], name):
        x = ops.convert_to_tensor(x)
        shape = (x.get_shape().as_list()
                 if x.get_shape().is_fully_defined()
                 else array_ops.shape(x))

        if sample and batch and event:
          return shape

        sample_start = 0
        batch_start = self.get_sample_ndims(x)
        event_start = batch_start + self.batch_ndims

        sample_shape = shape[sample_start:batch_start] if sample else []
        batch_shape = shape[batch_start:event_start] if batch else []
        event_shape = shape[event_start:] if event else []

        if not batch and not event:
          return sample_shape
        if not sample and not event:
          return batch_shape
        if not sample and not batch:
          return event_shape

        if x.get_shape().is_fully_defined():
          return sample_shape + batch_shape + event_shape
        else:
          return array_ops.concat(0, [sample_shape, batch_shape, event_shape])

  def get_sample_dims(self, x):
    """Returns dimension indexes corresponding to sample.

    Convenience function; identical to:

    ```python
    get_dims(x, sample=True, batch=False, event=False)
    ```

    Args:
      x: `Tensor`.

    Raises:
      ValueError: if `x.get_shape().ndims` is `None`

    Returns:
      List enumerating sample dimensions.
    """
    return self.get_dims(x, sample=True, batch=False, event=False)

  def get_batch_dims(self, x):
    """Returns dimension indexes corresponding to batch.

    Convenience function; identical to:

    ```python
    get_dims(x, sample=False, batch=True, event=False)
    ```

    Args:
      x: `Tensor`.

    Raises:
      ValueError: if `x.get_shape().ndims` is `None`

    Returns:
      List enumerating batch dimensions.
    """
    return self.get_dims(x, sample=False, batch=True, event=False)

  def get_event_dims(self, x):
    """Returns dimension indexes corresponding to event.

    Convenience function; identical to:

    ```python
    get_dims(x, sample=False, batch=False, event=True)
    ```

    Args:
      x: `Tensor`.

    Raises:
      ValueError: if `x.get_shape().ndims` is `None`

    Returns:
      List enumerating event dimensions.
    """
    return self.get_dims(x, sample=False, batch=False, event=True)

  def get_sample_shape(self, x):
    """Returns shape corresponding to sample.

    Convenience function; identical to:

    ```python
    get_shape(x, sample=True, batch=False, event=False)
    ```

    Args:
      x: `Tensor`.

    Returns:
      List describing sample shape if known statically, `Tensor` otherwise.
    """
    return self.get_shape(x, sample=True, batch=False, event=False)

  def get_batch_shape(self, x):
    """Returns shape corresponding to batch.

    Convenience function; identical to:

    ```python
    get_shape(x, sample=False, batch=True, event=False)
    ```

    Args:
      x: `Tensor`.

    Returns:
      List describing batch shape if known statically, `Tensor` otherwise.
    """
    return self.get_shape(x, sample=False, batch=True, event=False)

  def get_event_shape(self, x):
    """Returns shape corresponding to event.

    Convenience function; identical to:

    ```python
    get_shape(x, sample=False, batch=False, event=True)
    ```

    Args:
      x: `Tensor`.

    Returns:
      List describing event shape if known statically, `Tensor` otherwise.
    """
    return self.get_shape(x, sample=False, batch=False, event=True)
