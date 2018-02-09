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
"""Quasi Monte Carlo support: Halton sequence.

@@sample
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


__all__ = [
    'sample',
]


# The maximum dimension we support. This is limited by the number of primes
# in the _PRIMES array.
_MAX_DIMENSION = 1000


def sample(dim, num_samples=None, sample_indices=None, dtype=None, name=None):
  r"""Returns a sample from the `m` dimensional Halton sequence.

  Warning: The sequence elements take values only between 0 and 1. Care must be
  taken to appropriately transform the domain of a function if it differs from
  the unit cube before evaluating integrals using Halton samples. It is also
  important to remember that quasi-random numbers are not a replacement for
  pseudo-random numbers in every context. Quasi random numbers are completely
  deterministic and typically have significant negative autocorrelation (unless
  randomized).

  Computes the members of the low discrepancy Halton sequence in dimension
  `dim`. The d-dimensional sequence takes values in the unit hypercube in d
  dimensions. Currently, only dimensions up to 1000 are supported. The prime
  base for the `k`-th axes is the k-th prime starting from 2. For example,
  if dim = 3, then the bases will be [2, 3, 5] respectively and the first
  element of the sequence will be: [0.5, 0.333, 0.2]. For a more complete
  description of the Halton sequences see:
  https://en.wikipedia.org/wiki/Halton_sequence. For low discrepancy sequences
  and their applications see:
  https://en.wikipedia.org/wiki/Low-discrepancy_sequence.

  The user must supply either `num_samples` or `sample_indices` but not both.
  The former is the number of samples to produce starting from the first
  element. If `sample_indices` is given instead, the specified elements of
  the sequence are generated. For example, sample_indices=tf.range(10) is
  equivalent to specifying n=10.

  Example Use:

  ```python
  bf = tf.contrib.bayesflow

  # Produce the first 1000 members of the Halton sequence in 3 dimensions.
  num_samples = 1000
  dim = 3
  sample = bf.halton_sequence.sample(dim, num_samples=num_samples)

  # Evaluate the integral of x_1 * x_2^2 * x_3^3  over the three dimensional
  # hypercube.
  powers = tf.range(1.0, limit=dim + 1)
  integral = tf.reduce_mean(tf.reduce_prod(sample ** powers, axis=-1))
  true_value = 1.0 / tf.reduce_prod(powers + 1.0)
  with tf.Session() as session:
    values = session.run((integral, true_value))

  # Produces a relative absolute error of 1.7%.
  print ("Estimated: %f, True Value: %f" % values)

  # Now skip the first 1000 samples and recompute the integral with the next
  # thousand samples. The sample_indices argument can be used to do this.


  sample_indices = tf.range(start=1000, limit=1000 + num_samples,
                            dtype=tf.int32)
  sample_leaped = halton.sample(dim, sample_indices=sample_indices)

  integral_leaped = tf.reduce_mean(tf.reduce_prod(sample_leaped ** powers,
                                                  axis=-1))
  with tf.Session() as session:
    values = session.run((integral_leaped, true_value))
  # Now produces a relative absolute error of 0.05%.
  print ("Leaped Estimated: %f, True Value: %f" % values)
  ```

  Args:
    dim: Positive Python `int` representing each sample's `event_size.` Must
      not be greater than 1000.
    num_samples: (Optional) positive Python `int`. The number of samples to
      generate. Either this parameter or sample_indices must be specified but
      not both. If this parameter is None, then the behaviour is determined by
      the `sample_indices`.
    sample_indices: (Optional) `Tensor` of dtype int32 and rank 1. The elements
      of the sequence to compute specified by their position in the sequence.
      The entries index into the Halton sequence starting with 0 and hence,
      must be whole numbers. For example, sample_indices=[0, 5, 6] will produce
      the first, sixth and seventh elements of the sequence. If this parameter
      is None, then the `num_samples` parameter must be specified which gives
      the number of desired samples starting from the first sample.
    dtype: (Optional) The dtype of the sample. One of `float32` or `float64`.
      Default is `float32`.
    name:  (Optional) Python `str` describing ops managed by this function. If
    not supplied the name of this function is used.

  Returns:
    halton_elements: Elements of the Halton sequence. `Tensor` of supplied dtype
    and `shape` `[num_samples, dim]` if `num_samples` was specified or shape
    `[s, dim]` where s is the size of `sample_indices` if `sample_indices`
    were specified.

  Raises:
    ValueError: if both `sample_indices` and `num_samples` were specified or
    if dimension `dim` is less than 1 or greater than 1000.
  """
  if dim < 1 or dim > _MAX_DIMENSION:
    raise ValueError(
        'Dimension must be between 1 and {}. Supplied {}'.format(_MAX_DIMENSION,
                                                                 dim))
  if (num_samples is None) == (sample_indices is None):
    raise ValueError('Either `num_samples` or `sample_indices` must be'
                     ' specified but not both.')

  dtype = dtype or dtypes.float32
  if not dtype.is_floating:
    raise ValueError('dtype must be of `float`-type')

  with ops.name_scope(name, 'sample', values=[sample_indices]):
    # Here and in the following, the shape layout is as follows:
    # [sample dimension, event dimension, coefficient dimension].
    # The coefficient dimension is an intermediate axes which will hold the
    # weights of the starting integer when expressed in the (prime) base for
    # an event dimension.
    indices = _get_indices(num_samples, sample_indices, dtype)
    radixes = array_ops.constant(_PRIMES[0:dim], dtype=dtype, shape=[dim, 1])

    max_sizes_by_axes = _base_expansion_size(math_ops.reduce_max(indices),
                                             radixes)

    max_size = math_ops.reduce_max(max_sizes_by_axes)

    # The powers of the radixes that we will need. Note that there is a bit
    # of an excess here. Suppose we need the place value coefficients of 7
    # in base 2 and 3. For 2, we will have 3 digits but we only need 2 digits
    # for base 3. However, we can only create rectangular tensors so we
    # store both expansions in a [2, 3] tensor. This leads to the problem that
    # we might end up attempting to raise large numbers to large powers. For
    # example, base 2 expansion of 1024 has 10 digits. If we were in 10
    # dimensions, then the 10th prime (29) we will end up computing 29^10 even
    # though we don't need it. We avoid this by setting the exponents for each
    # axes to 0 beyond the maximum value needed for that dimension.
    exponents_by_axes = array_ops.tile([math_ops.range(max_size)], [dim, 1])
    weight_mask = exponents_by_axes > max_sizes_by_axes
    capped_exponents = array_ops.where(
        weight_mask, array_ops.zeros_like(exponents_by_axes), exponents_by_axes)
    weights = radixes ** capped_exponents
    coeffs = math_ops.floor_div(indices, weights)
    coeffs *= 1 - math_ops.cast(weight_mask, dtype)
    coeffs = (coeffs % radixes) / radixes
    return math_ops.reduce_sum(coeffs / weights, axis=-1)


def _get_indices(n, sample_indices, dtype, name=None):
  """Generates starting points for the Halton sequence procedure.

  The k'th element of the sequence is generated starting from a positive integer
  which must be distinct for each `k`. It is conventional to choose the starting
  point as `k` itself (or `k+1` if k is zero based). This function generates
  the starting integers for the required elements and reshapes the result for
  later use.

  Args:
    n: Positive `int`. The number of samples to generate. If this
      parameter is supplied, then `sample_indices` should be None.
    sample_indices: `Tensor` of dtype int32 and rank 1. The entries
      index into the Halton sequence starting with 0 and hence, must be whole
      numbers. For example, sample_indices=[0, 5, 6] will produce the first,
      sixth and seventh elements of the sequence. If this parameter is not None
      then `n` must be None.
    dtype: The dtype of the sample. One of `float32` or `float64`.
      Default is `float32`.
    name: Python `str` name which describes ops created by this function.

  Returns:
    indices: `Tensor` of dtype `dtype` and shape = `[n, 1, 1]`.
  """
  with ops.name_scope(name, 'get_indices', [n, sample_indices]):
    if sample_indices is None:
      sample_indices = math_ops.range(n, dtype=dtype)
    else:
      sample_indices = math_ops.cast(sample_indices, dtype)

    # Shift the indices so they are 1 based.
    indices = sample_indices + 1

    # Reshape to make space for the event dimension and the place value
    # coefficients.
    return array_ops.reshape(indices, [-1, 1, 1])


def _base_expansion_size(num, bases):
  """Computes the number of terms in the place value expansion.

  Let num = a0 + a1 b + a2 b^2 + ... ak b^k be the place value expansion of
  `num` in base b (ak <> 0). This function computes and returns `k` for each
  base `b` specified in `bases`.

  This can be inferred from the base `b` logarithm of `num` as follows:
    $$k = Floor(log_b (num)) + 1  = Floor( log(num) / log(b)) + 1$$

  Args:
    num: Scalar `Tensor` of dtype either `float32` or `float64`. The number to
      compute the base expansion size of.
    bases: `Tensor` of the same dtype as num. The bases to compute the size
      against.

  Returns:
    Tensor of same dtype and shape as `bases` containing the size of num when
    written in that base.
  """
  return math_ops.floor(math_ops.log(num) / math_ops.log(bases)) + 1


def _primes_less_than(n):
  # Based on
  # https://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python/3035188#3035188
  """Returns sorted array of primes such that `2 <= prime < n`."""
  small_primes = np.array((2, 3, 5))
  if n <= 6:
    return small_primes[small_primes < n]
  sieve = np.ones(n // 3 + (n % 6 == 2), dtype=np.bool)
  sieve[0] = False
  m = int(n ** 0.5) // 3 + 1
  for i in range(m):
    if not sieve[i]:
      continue
    k = 3 * i + 1 | 1
    sieve[k ** 2 // 3::2 * k] = False
    sieve[(k ** 2 + 4 * k - 2 * k * (i & 1)) // 3::2 * k] = False
  return np.r_[2, 3, 3 * np.nonzero(sieve)[0] + 1 | 1]

_PRIMES = _primes_less_than(7919+1)

assert len(_PRIMES) == _MAX_DIMENSION
