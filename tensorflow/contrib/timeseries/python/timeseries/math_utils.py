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
"""Miscellaneous utilities used by time series models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.contrib import lookup
from tensorflow.contrib.layers.python.layers import layers

from tensorflow.contrib.timeseries.python.timeseries.feature_keys import TrainEvalFeatures

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest


def clip_covariance(
    covariance_matrix, maximum_variance_ratio, minimum_variance):
  """Enforce constraints on a covariance matrix to improve numerical stability.

  Args:
    covariance_matrix: A [..., N, N] batch of covariance matrices.
    maximum_variance_ratio: The maximum allowed ratio of two diagonal
      entries. Any entries lower than the maximum entry divided by this ratio
      will be set to that value.
    minimum_variance: A floor for diagonal entries in the returned matrix.
  Returns:
    A new covariance matrix with the requested constraints enforced. If the
    input was positive definite, the output will be too.
  """
  # TODO(allenl): Smarter scaling here so that correlations are preserved when
  # fiddling with diagonal elements.
  diagonal = array_ops.matrix_diag_part(covariance_matrix)
  maximum = math_ops.reduce_max(diagonal, axis=-1, keep_dims=True)
  new_diagonal = gen_math_ops.maximum(
      diagonal, maximum / maximum_variance_ratio)
  return array_ops.matrix_set_diag(
      covariance_matrix, math_ops.maximum(new_diagonal, minimum_variance))


def block_diagonal(matrices, dtype=dtypes.float32, name="block_diagonal"):
  r"""Constructs block-diagonal matrices from a list of batched 2D tensors.

  Args:
    matrices: A list of Tensors with shape [..., N_i, M_i] (i.e. a list of
      matrices with the same batch dimension).
    dtype: Data type to use. The Tensors in `matrices` must match this dtype.
    name: A name for the returned op.
  Returns:
    A matrix with the input matrices stacked along its main diagonal, having
    shape [..., \sum_i N_i, \sum_i M_i].
  """
  matrices = [ops.convert_to_tensor(matrix, dtype=dtype) for matrix in matrices]
  blocked_rows = tensor_shape.Dimension(0)
  blocked_cols = tensor_shape.Dimension(0)
  batch_shape = tensor_shape.TensorShape(None)
  for matrix in matrices:
    full_matrix_shape = matrix.get_shape().with_rank_at_least(2)
    batch_shape = batch_shape.merge_with(full_matrix_shape[:-2])
    blocked_rows += full_matrix_shape[-2]
    blocked_cols += full_matrix_shape[-1]
  ret_columns_list = []
  for matrix in matrices:
    matrix_shape = array_ops.shape(matrix)
    ret_columns_list.append(matrix_shape[-1])
  ret_columns = math_ops.add_n(ret_columns_list)
  row_blocks = []
  current_column = 0
  for matrix in matrices:
    matrix_shape = array_ops.shape(matrix)
    row_before_length = current_column
    current_column += matrix_shape[-1]
    row_after_length = ret_columns - current_column
    row_blocks.append(
        array_ops.pad(
            tensor=matrix,
            paddings=array_ops.concat(
                [
                    array_ops.zeros(
                        [array_ops.rank(matrix) - 1, 2], dtype=dtypes.int32), [(
                            row_before_length, row_after_length)]
                ],
                axis=0)))
  blocked = array_ops.concat(row_blocks, -2, name=name)
  blocked.set_shape(batch_shape.concatenate((blocked_rows, blocked_cols)))
  return blocked


def power_sums_tensor(array_size, power_matrix, multiplier):
  r"""Computes \sum_{i=0}^{N-1} A^i B (A^i)^T for N=0..(array_size + 1).

  Args:
    array_size: The number of non-trivial sums to pre-compute.
    power_matrix: The "A" matrix above.
    multiplier: The "B" matrix above
  Returns:
    A Tensor with S[N] = \sum_{i=0}^{N-1} A^i B (A^i)^T
      S[0] is the zero matrix
      S[1] is B
      S[2] is A B A^T + B
      ...and so on
  """
  array_size = math_ops.cast(array_size, dtypes.int32)
  power_matrix = ops.convert_to_tensor(power_matrix)
  identity_like_power_matrix = linalg_ops.eye(
      array_ops.shape(power_matrix)[0], dtype=power_matrix.dtype)
  identity_like_power_matrix.set_shape(
      ops.convert_to_tensor(power_matrix).get_shape())
  transition_powers = functional_ops.scan(
      lambda previous_power, _: math_ops.matmul(previous_power, power_matrix),
      math_ops.range(array_size - 1),
      initializer=identity_like_power_matrix)
  summed = math_ops.cumsum(
      array_ops.concat([
          array_ops.expand_dims(multiplier, 0), math_ops.matmul(
              batch_times_matrix(transition_powers, multiplier),
              transition_powers,
              adjoint_b=True)
      ], 0))
  return array_ops.concat(
      [array_ops.expand_dims(array_ops.zeros_like(multiplier), 0), summed], 0)


def matrix_to_powers(matrix, powers):
  """Raise a single matrix to multiple powers."""
  matrix_tiled = array_ops.tile(
      array_ops.expand_dims(matrix, 0), [array_ops.size(powers), 1, 1])
  return batch_matrix_pow(matrix_tiled, powers)


def batch_matrix_pow(matrices, powers):
  """Compute powers of matrices, e.g. A^3 = matmul(matmul(A, A), A).

  Uses exponentiation by squaring, with O(log(p)) matrix multiplications to
  compute A^p.

  Args:
    matrices: [batch size x N x N]
    powers: Which integer power to raise each matrix to [batch size]
  Returns:
    The matrices raised to their respective powers, same dimensions as the
    "matrices" argument.
  """

  def terminate_when_all_zero(current_argument, residual_powers, accumulator):
    del current_argument, accumulator  # not used for condition
    do_exit = math_ops.reduce_any(
        math_ops.greater(residual_powers, array_ops.ones_like(residual_powers)))
    return do_exit

  def do_iteration(current_argument, residual_powers, accumulator):
    """Compute one step of iterative exponentiation by squaring.

    The recursive form is:
      power(A, p) = { power(matmul(A, A), p / 2) for even p
                    { matmul(A, power(matmul(A, A), (p - 1) / 2)) for odd p
      power(A, 0) = I

    The power(A, 0) = I case is handeled by starting with accumulator set to the
    identity matrix; matrices with zero residual powers are passed through
    unchanged.

    Args:
      current_argument: On this step, what is the first argument (A^2..^2) to
          the (unrolled) recursive function? [batch size x N x N]
      residual_powers: On this step, what is the second argument (residual p)?
          [batch_size]
      accumulator: Accumulates the exterior multiplications from the odd
          powers (initially the identity matrix). [batch_size x N x N]
    Returns:
      Updated versions of each argument for one step of the unrolled
      computation. Does not change parts of the batch which have a residual
      power of zero.
    """
    is_even = math_ops.equal(residual_powers % 2,
                             array_ops.zeros(
                                 array_ops.shape(residual_powers),
                                 dtype=dtypes.int32))
    new_accumulator = array_ops.where(is_even, accumulator,
                                      math_ops.matmul(accumulator,
                                                      current_argument))
    new_argument = math_ops.matmul(current_argument, current_argument)
    do_update = math_ops.greater(residual_powers, 1)
    new_residual_powers = residual_powers - residual_powers % 2
    new_residual_powers //= 2
    # Stop updating if we've reached our base case; some batch elements may
    # finish sooner than others
    accumulator = array_ops.where(do_update, new_accumulator, accumulator)
    current_argument = array_ops.where(do_update, new_argument,
                                       current_argument)
    residual_powers = array_ops.where(do_update, new_residual_powers,
                                      residual_powers)
    return (current_argument, residual_powers, accumulator)

  matrices = ops.convert_to_tensor(matrices)
  powers = math_ops.cast(powers, dtype=dtypes.int32)
  ident = array_ops.expand_dims(
      array_ops.diag(
          array_ops.ones([array_ops.shape(matrices)[1]], dtype=matrices.dtype)),
      0)
  ident_tiled = array_ops.tile(ident, [array_ops.shape(matrices)[0], 1, 1])
  (final_argument,
   final_residual_power, final_accumulator) = control_flow_ops.while_loop(
       terminate_when_all_zero, do_iteration, [matrices, powers, ident_tiled])
  return array_ops.where(
      math_ops.equal(final_residual_power,
                     array_ops.zeros_like(
                         final_residual_power, dtype=dtypes.int32)),
      ident_tiled, math_ops.matmul(final_argument, final_accumulator))


# TODO(allenl): would be useful if this was built into batch_matmul
def batch_times_matrix(batch, matrix, adj_x=False, adj_y=False):
  """Multiply a batch of matrices by a single matrix.

  Functionally equivalent to:
  tf.matmul(batch, array_ops.tile(gen_math_ops.expand_dims(matrix, 0),
                                 [array_ops.shape(batch)[0], 1, 1]),
                  adjoint_a=adj_x, adjoint_b=adj_y)

  Args:
    batch: [batch_size x N x M] after optional transpose
    matrix: [M x P] after optional transpose
    adj_x: If true, transpose the second two dimensions of "batch" before
        multiplying.
    adj_y: If true, transpose "matrix" before multiplying.
  Returns:
    [batch_size x N x P]
  """
  batch = ops.convert_to_tensor(batch)
  matrix = ops.convert_to_tensor(matrix)
  assert batch.get_shape().ndims == 3
  assert matrix.get_shape().ndims == 2
  if adj_x:
    batch = array_ops.transpose(batch, [0, 2, 1])
  batch_dimension = batch.get_shape()[0].value
  first_dimension = batch.get_shape()[1].value
  tensor_batch_shape = array_ops.shape(batch)
  if batch_dimension is None:
    batch_dimension = tensor_batch_shape[0]
  if first_dimension is None:
    first_dimension = tensor_batch_shape[1]
  matrix_first_dimension, matrix_second_dimension = matrix.get_shape().as_list()
  batch_reshaped = array_ops.reshape(batch, [-1, tensor_batch_shape[2]])
  if adj_y:
    if matrix_first_dimension is None:
      matrix_first_dimension = array_ops.shape(matrix)[0]
    result_shape = [batch_dimension, first_dimension, matrix_first_dimension]
  else:
    if matrix_second_dimension is None:
      matrix_second_dimension = array_ops.shape(matrix)[1]
    result_shape = [batch_dimension, first_dimension, matrix_second_dimension]
  return array_ops.reshape(
      math_ops.matmul(batch_reshaped, matrix, adjoint_b=adj_y), result_shape)


def matrix_times_batch(matrix, batch, adj_x=False, adj_y=False):
  """Like batch_times_matrix, but with the multiplication order swapped."""
  return array_ops.transpose(
      batch_times_matrix(
          batch=batch, matrix=matrix, adj_x=not adj_y, adj_y=not adj_x),
      [0, 2, 1])


def make_toeplitz_matrix(inputs, name=None):
  """Make a symmetric Toeplitz matrix from input array of values.

  Args:
    inputs: a 3-D tensor of shape [num_blocks, block_size, block_size].
    name: the name of the operation.

  Returns:
    a symmetric Toeplitz matrix of shape
      [num_blocks*block_size, num_blocks*block_size].
  """
  num_blocks = array_ops.shape(inputs)[0]
  block_size = array_ops.shape(inputs)[1]
  output_size = block_size * num_blocks
  lags = array_ops.reshape(math_ops.range(num_blocks), shape=[1, -1])
  indices = math_ops.abs(lags - array_ops.transpose(lags))
  output = array_ops.gather(inputs, indices)
  output = array_ops.reshape(
      array_ops.transpose(output, [0, 2, 1, 3]), [output_size, output_size])
  return array_ops.identity(output, name=name)


# TODO(allenl): Investigate alternative parameterizations.
def sign_magnitude_positive_definite(
    raw, off_diagonal_scale=0., overall_scale=0.):
  """Constructs a positive definite matrix from an unconstrained input matrix.

  We want to keep the whole matrix on a log scale, but also allow off-diagonal
  elements to be negative, so the sign of off-diagonal elements is modeled
  separately from their magnitude (using the lower and upper triangles
  respectively). Specifically:

  for i < j, we have:
    output_cholesky[i, j] = raw[j, i] / (abs(raw[j, i]) + 1) *
        exp((off_diagonal_scale + overall_scale + raw[i, j]) / 2)

  output_cholesky[i, i] = exp((raw[i, i] + overall_scale) / 2)

  output = output_cholesky^T * output_cholesky

  where raw, off_diagonal_scale, and overall_scale are
  un-constrained real-valued variables. The resulting values are stable
  around zero due to the exponential (and the softsign keeps the function
  smooth).

  Args:
    raw: A [..., M, M] Tensor.
    off_diagonal_scale: A scalar or [...] shaped Tensor controlling the relative
        scale of off-diagonal values in the output matrix.
    overall_scale: A scalar or [...] shaped Tensor controlling the overall scale
        of the output matrix.
  Returns:
    The `output` matrix described above, a [..., M, M] positive definite matrix.

  """
  raw = ops.convert_to_tensor(raw)
  diagonal = array_ops.matrix_diag_part(raw)
  def _right_pad_with_ones(tensor, target_rank):
    # Allow broadcasting even if overall_scale and off_diagonal_scale have batch
    # dimensions
    tensor = ops.convert_to_tensor(tensor, dtype=raw.dtype.base_dtype)
    return array_ops.reshape(tensor,
                             array_ops.concat(
                                 [
                                     array_ops.shape(tensor), array_ops.ones(
                                         [target_rank - array_ops.rank(tensor)],
                                         dtype=target_rank.dtype)
                                 ],
                                 axis=0))
  # We divide the log values by 2 to compensate for the squaring that happens
  # when transforming Cholesky factors into positive definite matrices.
  sign_magnitude = (gen_math_ops.exp(
      (raw + _right_pad_with_ones(off_diagonal_scale, array_ops.rank(raw)) +
       _right_pad_with_ones(overall_scale, array_ops.rank(raw))) / 2.) *
                    nn.softsign(array_ops.matrix_transpose(raw)))
  sign_magnitude.set_shape(raw.get_shape())
  cholesky_factor = array_ops.matrix_set_diag(
      input=array_ops.matrix_band_part(sign_magnitude, 0, -1),
      diagonal=gen_math_ops.exp((diagonal + _right_pad_with_ones(
          overall_scale, array_ops.rank(diagonal))) / 2.))
  return math_ops.matmul(cholesky_factor, cholesky_factor, transpose_a=True)


def transform_to_covariance_matrices(input_vectors, matrix_size):
  """Construct covariance matrices via transformations from input_vectors.

  Args:
    input_vectors: A [batch size x input size] batch of vectors to transform.
    matrix_size: An integer indicating one dimension of the (square) output
        matrix.
  Returns:
    A [batch size x matrix_size x matrix_size] batch of covariance matrices.
  """
  combined_values = layers.fully_connected(
      input_vectors, matrix_size**2 + 2, activation_fn=None)
  return sign_magnitude_positive_definite(
      raw=array_ops.reshape(combined_values[..., :-2],
                            array_ops.concat([
                                array_ops.shape(combined_values)[:-1],
                                [matrix_size, matrix_size]
                            ], 0)),
      off_diagonal_scale=combined_values[..., -2],
      overall_scale=combined_values[..., -1])


def variable_covariance_matrix(
    size, name, dtype, initial_diagonal_values=None,
    initial_overall_scale_log=0.):
  """Construct a Variable-parameterized positive definite matrix.

  Useful for parameterizing covariance matrices.

  Args:
    size: The size of the main diagonal, the returned matrix having shape [size
        x size].
    name: The name to use when defining variables and ops.
    dtype: The floating point data type to use.
    initial_diagonal_values: A Tensor with shape [size] with initial values for
        the diagonal values of the returned matrix. Must be positive.
    initial_overall_scale_log: Initial value of the bias term for every element
        of the matrix in log space.
  Returns:
    A Variable-parameterized covariance matrix with shape [size x size].
  """
  raw_values = variable_scope.get_variable(
      name + "_pre_transform",
      dtype=dtype,
      shape=[size, size],
      initializer=init_ops.zeros_initializer())
  if initial_diagonal_values is not None:
    raw_values += array_ops.matrix_diag(math_ops.log(initial_diagonal_values))
  return array_ops.identity(
      sign_magnitude_positive_definite(
          raw=raw_values,
          off_diagonal_scale=variable_scope.get_variable(
              name + "_off_diagonal_scale",
              dtype=dtype,
              initializer=constant_op.constant(-5., dtype=dtype)),
          overall_scale=ops.convert_to_tensor(
              initial_overall_scale_log, dtype=dtype) +
          variable_scope.get_variable(
              name + "_overall_scale",
              dtype=dtype,
              shape=[],
              initializer=init_ops.zeros_initializer())),
      name=name)


def batch_start_time(times):
  return times[:, 0]


def batch_end_time(times):
  return times[:, -1]


def log_noninformative_covariance_prior(covariance):
  """Compute a relatively uninformative prior for noise parameters.

  Helpful for avoiding noise over-estimation, where noise otherwise decreases
  very slowly during optimization.

  See:
    Villegas, C. On the A Priori Distribution of the Covariance Matrix.
    Ann. Math. Statist. 40 (1969), no. 3, 1098--1099.

  Args:
    covariance: A covariance matrix.
  Returns:
    For a [p x p] matrix:
      log(det(covariance)^(-(p + 1) / 2))
  """
  # Avoid zero/negative determinants due to numerical errors
  covariance += array_ops.diag(1e-8 * array_ops.ones(
      shape=[array_ops.shape(covariance)[0]], dtype=covariance.dtype))
  power = -(math_ops.cast(array_ops.shape(covariance)[0] + 1,
                          covariance.dtype) / 2.)
  return power * math_ops.log(linalg_ops.matrix_determinant(covariance))


def entropy_matched_cauchy_scale(covariance):
  """Approximates a similar Cauchy distribution given a covariance matrix.

  Since Cauchy distributions do not have moments, entropy matching provides one
  way to set a Cauchy's scale parameter in a way that provides a similar
  distribution. The effect is dividing the standard deviation of an independent
  Gaussian by a constant very near 3.

  To set the scale of the Cauchy distribution, we first select the diagonals of
  `covariance`. Since this ignores cross terms, it overestimates the entropy of
  the Gaussian. For each of these variances, we solve for the Cauchy scale
  parameter which gives the same entropy as the Gaussian with that
  variance. This means setting the (univariate) Gaussian entropy
      0.5 * ln(2 * variance * pi * e)
  equal to the Cauchy entropy
      ln(4 * pi * scale)
  Solving, we get scale = sqrt(variance * (e / (8 pi))).

  Args:
    covariance: A [batch size x N x N] batch of covariance matrices to produce
        Cauchy scales for.
  Returns:
    A [batch size x N] set of Cauchy scale parameters for each part of the batch
    and each dimension of the input Gaussians.
  """
  return math_ops.sqrt(math.e / (8. * math.pi) *
                       array_ops.matrix_diag_part(covariance))


class TensorValuedMutableDenseHashTable(lookup.MutableDenseHashTable):
  """A version of MutableDenseHashTable which stores arbitrary Tensor shapes.

  Since MutableDenseHashTable only allows vectors right now, simply adds reshape
  ops on both ends.
  """

  def __init__(self, key_dtype, value_dtype, default_value, *args, **kwargs):
    self._non_vector_value_shape = array_ops.shape(default_value)
    super(TensorValuedMutableDenseHashTable, self).__init__(
        key_dtype=key_dtype,
        value_dtype=value_dtype,
        default_value=array_ops.reshape(default_value, [-1]),
        *args,
        **kwargs)

  def insert(self, keys, values, name=None):
    keys = ops.convert_to_tensor(keys, dtype=self._key_dtype)
    keys_flat = array_ops.reshape(keys, [-1])
    return super(TensorValuedMutableDenseHashTable, self).insert(
        keys=keys_flat,
        # Each key has one corresponding value, so the shape of the tensor of
        # values for every key is key_shape + value_shape
        values=array_ops.reshape(values, [array_ops.shape(keys_flat)[0], -1]),
        name=name)

  def lookup(self, keys, name=None):
    keys_flat = array_ops.reshape(
        ops.convert_to_tensor(keys, dtype=self._key_dtype), [-1])
    return array_ops.reshape(
        super(TensorValuedMutableDenseHashTable, self).lookup(
            keys=keys_flat, name=name),
        array_ops.concat([array_ops.shape(keys), self._non_vector_value_shape],
                         0))


class TupleOfTensorsLookup(lookup.LookupInterface):
  """A LookupInterface with nested tuples of Tensors as values.

  Creates one MutableDenseHashTable per value Tensor, which has some unnecessary
  overhead.
  """

  def __init__(
      self, key_dtype, default_values, empty_key, name, checkpoint=True):
    default_values_flat = nest.flatten(default_values)
    self._hash_tables = nest.pack_sequence_as(
        default_values,
        [TensorValuedMutableDenseHashTable(
            key_dtype=key_dtype,
            value_dtype=default_value.dtype.base_dtype,
            default_value=default_value,
            empty_key=empty_key,
            name=name + "_{}".format(table_number),
            checkpoint=checkpoint)
         for table_number, default_value
         in enumerate(default_values_flat)])
    self._name = name

  def lookup(self, keys):
    return nest.pack_sequence_as(
        self._hash_tables,
        [hash_table.lookup(keys)
         for hash_table in nest.flatten(self._hash_tables)])

  def insert(self, keys, values):
    nest.assert_same_structure(self._hash_tables, values)
    # Avoid race conditions by requiring that all inputs are computed before any
    # inserts happen (an issue if one key's update relies on another's value).
    values_flat = [array_ops.identity(value) for value in nest.flatten(values)]
    with ops.control_dependencies(values_flat):
      insert_ops = [hash_table.insert(keys, value)
                    for hash_table, value
                    in zip(nest.flatten(self._hash_tables),
                           values_flat)]
    return control_flow_ops.group(*insert_ops)

  def check_table_dtypes(self, key_dtype, value_dtype):
    # dtype checking is done in the objects in self._hash_tables
    pass


def replicate_state(start_state, batch_size):
  """Create batch versions of state.

  Takes a list of Tensors, adds a batch dimension, and replicates
  batch_size times across that batch dimension. Used to replicate the
  non-batch state returned by get_start_state in define_loss.

  Args:
    start_state: Model-defined state to replicate.
    batch_size: Batch dimension for data.
  Returns:
    Replicated versions of the state.
  """
  flattened_state = nest.flatten(start_state)
  replicated_state = [
      array_ops.tile(
          array_ops.expand_dims(state_nonbatch, 0),
          array_ops.concat([[batch_size], array_ops.ones(
              [array_ops.rank(state_nonbatch)], dtype=dtypes.int32)], 0))
      for state_nonbatch in flattened_state
  ]
  return nest.pack_sequence_as(start_state, replicated_state)


Moments = collections.namedtuple("Moments", ["mean", "variance"])


# Currently all of these statistics are computed incrementally (i.e. are updated
# every time a new mini-batch of training data is presented) when this object is
# created in InputStatisticsFromMiniBatch.
InputStatistics = collections.namedtuple(
    "InputStatistics",
    ["series_start_moments",  # The mean and variance of each feature in a chunk
                              # (with a size configured in the statistics
                              # object) at the start of the series. A tuple of
                              # (mean, variance), each with shape [number of
                              # features], floating point. One use is in state
                              # space models, to keep priors calibrated even as
                              # earlier parts of the series are presented. If
                              # this object was created by
                              # InputStatisticsFromMiniBatch, these moments are
                              # computed based on the earliest chunk of data
                              # presented so far. However, there is a race
                              # condition in the update, so these may reflect
                              # statistics later in the series, but should
                              # eventually reflect statistics in a chunk at the
                              # series start.
     "overall_feature_moments",  # The mean and variance of each feature over
                                 # the entire series. A tuple of (mean,
                                 # variance), each with shape [number of
                                 # features]. If this object was created by
                                 # InputStatisticsFromMiniBatch, these moments
                                 # are estimates based on the data seen so far.
     "start_time",  # The first (lowest) time in the series, a scalar
                    # integer. If this object was created by
                    # InputStatisticsFromMiniBatch, this is the lowest time seen
                    # so far rather than the lowest time that will ever be seen
                    # (guaranteed to be at least as low as the lowest time
                    # presented in the current minibatch).
     "total_observation_count",  # Count of data points, a scalar integer. If
                                 # this object was created by
                                 # InputStatisticsFromMiniBatch, this is an
                                 # estimate of the total number of observations
                                 # in the whole dataset computed based on the
                                 # density of the series and the minimum and
                                 # maximum times seen.
    ])


# TODO(allenl): It would be nice to do something with full series statistics
# when the user provides that.
class InputStatisticsFromMiniBatch(object):
  """Generate statistics from mini-batch input."""

  def __init__(self, num_features, dtype, starting_variance_window_size=16):
    """Configure the input statistics object.

    Args:
      num_features: Number of features for the time series
      dtype: The floating point data type to use.
      starting_variance_window_size: The number of datapoints to use when
          computing the mean and variance at the start of the series.
    """
    self._starting_variance_window_size = starting_variance_window_size
    self._num_features = num_features
    self._dtype = dtype

  def initialize_graph(self, features, update_statistics=True):
    """Create any ops needed to provide input statistics.

    Should be called before statistics are requested.

    Args:
      features: A dictionary, the output of a `TimeSeriesInputFn` (with keys
          TrainEvalFeatures.TIMES and TrainEvalFeatures.VALUES).
      update_statistics: Whether `features` should be used to update adaptive
          statistics. Typically True for training and false for evaluation.
    Returns:
      An InputStatistics object composed of Variables, which will be updated
      based on mini-batches of data if requested.
    """
    if (TrainEvalFeatures.TIMES in features
        and TrainEvalFeatures.VALUES in features):
      times = features[TrainEvalFeatures.TIMES]
      values = features[TrainEvalFeatures.VALUES]
    else:
      # times and values may not be available, for example during prediction. We
      # still need to retrieve our variables so that they can be read from, even
      # if we're not going to update them.
      times = None
      values = None
    # Create/retrieve variables representing input statistics, initialized
    # without data to avoid deadlocking if variables are initialized before
    # queue runners are started.
    with variable_scope.variable_scope("input_statistics", use_resource=True):
      statistics = self._create_variable_statistics_object()
    with variable_scope.variable_scope(
        "input_statistics_auxiliary", use_resource=True):
      # Secondary statistics, necessary for the incremental computation of the
      # primary statistics (e.g. counts and sums for computing a mean
      # incrementally).
      auxiliary_variables = self._AdaptiveInputAuxiliaryStatistics(
          num_features=self._num_features, dtype=self._dtype)
    if update_statistics and times is not None and values is not None:
      # If we have times and values from mini-batch input, create update ops to
      # take the new data into account.
      assign_op = self._update_statistics_from_mini_batch(
          statistics, auxiliary_variables, times, values)
      with ops.control_dependencies([assign_op]):
        stat_variables = nest.pack_sequence_as(statistics, [
            array_ops.identity(tensor) for tensor in nest.flatten(statistics)
        ])
        # Since start time updates have a race condition, ensure that the
        # reported start time is at least as low as the lowest time in this
        # mini-batch. The start time should converge on the correct value
        # eventually even with the race condition, but for example state space
        # models have an assertion which could fail without this
        # post-processing.
        return stat_variables._replace(start_time=gen_math_ops.minimum(
            stat_variables.start_time, math_ops.reduce_min(times)))
    else:
      return statistics

  class _AdaptiveInputAuxiliaryStatistics(collections.namedtuple(
      "_AdaptiveInputAuxiliaryStatistics",
      ["max_time_seen",  # The maximum time seen (best effort if updated from
                         # multiple workers; see notes about race condition
                         # below).
       "chunk_count",  # The number of chunks seen.
       "inter_observation_duration_sum",  # The sum across chunks of their "time
                                          # density" (number of times per
                                          # example).
       "example_count",  # The number of examples seen (each example has a
                         # single time associated with it and one or more
                         # real-valued features).
       "overall_feature_sum",  # The sum of values for each feature. Shape
                               # [number of features].
       "overall_feature_sum_of_squares",  # The sum of squared values for each
                                          # feature. Shape [number of features]
      ])):
    """Extra statistics used to incrementally update InputStatistics."""

    def __new__(cls, num_features, dtype):
      return super(
          InputStatisticsFromMiniBatch  # pylint: disable=protected-access
          ._AdaptiveInputAuxiliaryStatistics,
          cls).__new__(
              cls,
              max_time_seen=variable_scope.get_variable(
                  name="max_time_seen",
                  initializer=dtypes.int64.min,
                  dtype=dtypes.int64,
                  trainable=False),
              chunk_count=variable_scope.get_variable(
                  name="chunk_count",
                  initializer=init_ops.zeros_initializer(),
                  shape=[],
                  dtype=dtypes.int64,
                  trainable=False),
              inter_observation_duration_sum=variable_scope.get_variable(
                  name="inter_observation_duration_sum",
                  initializer=init_ops.zeros_initializer(),
                  shape=[],
                  dtype=dtype,
                  trainable=False),
              example_count=variable_scope.get_variable(
                  name="example_count",
                  shape=[],
                  dtype=dtypes.int64,
                  trainable=False),
              overall_feature_sum=variable_scope.get_variable(
                  name="overall_feature_sum",
                  shape=[num_features],
                  dtype=dtype,
                  initializer=init_ops.zeros_initializer(),
                  trainable=False),
              overall_feature_sum_of_squares=variable_scope.get_variable(
                  name="overall_feature_sum_of_squares",
                  shape=[num_features],
                  dtype=dtype,
                  initializer=init_ops.zeros_initializer(),
                  trainable=False))

  def _update_statistics_from_mini_batch(
      self, statistics, auxiliary_variables, times, values):
    """Given mini-batch input, update `statistics` and `auxiliary_variables`."""
    values = math_ops.cast(values, self._dtype)
    # The density (measured in times per observation) that we see in each part
    # of the mini-batch.
    batch_inter_observation_duration = (math_ops.cast(
        math_ops.reduce_max(times, axis=1) - math_ops.reduce_min(times, axis=1),
        self._dtype) / math_ops.cast(
            array_ops.shape(times)[1] - 1, self._dtype))
    # Co-locate updates with their variables to minimize race conditions when
    # updating statistics.
    with ops.colocate_with(auxiliary_variables.max_time_seen):
      # There is a race condition if this value is being updated from multiple
      # workers. However, it should eventually reach the correct value if the
      # last chunk is presented enough times.
      max_time_seen_assign = state_ops.assign(
          auxiliary_variables.max_time_seen,
          gen_math_ops.maximum(auxiliary_variables.max_time_seen,
                               math_ops.reduce_max(times)))
    with ops.colocate_with(auxiliary_variables.chunk_count):
      chunk_count_assign = state_ops.assign_add(auxiliary_variables.chunk_count,
                                                array_ops.shape(
                                                    times,
                                                    out_type=dtypes.int64)[0])
    with ops.colocate_with(auxiliary_variables.inter_observation_duration_sum):
      inter_observation_duration_assign = state_ops.assign_add(
          auxiliary_variables.inter_observation_duration_sum,
          math_ops.reduce_sum(batch_inter_observation_duration))
    with ops.colocate_with(auxiliary_variables.example_count):
      example_count_assign = state_ops.assign_add(
          auxiliary_variables.example_count,
          array_ops.size(times, out_type=dtypes.int64))
    # Note: These mean/variance updates assume that all points are equally
    # likely, which is not true if _chunks_ are sampled uniformly from the space
    # of all possible contiguous chunks, since points at the start and end of
    # the series are then members of fewer chunks. For series which are much
    # longer than the chunk size (the usual/expected case), this effect becomes
    # irrelevant.
    with ops.colocate_with(auxiliary_variables.overall_feature_sum):
      overall_feature_sum_assign = state_ops.assign_add(
          auxiliary_variables.overall_feature_sum,
          math_ops.reduce_sum(values, axis=[0, 1]))
    with ops.colocate_with(auxiliary_variables.overall_feature_sum_of_squares):
      overall_feature_sum_of_squares_assign = state_ops.assign_add(
          auxiliary_variables.overall_feature_sum_of_squares,
          math_ops.reduce_sum(values**2, axis=[0, 1]))
    per_chunk_aux_updates = control_flow_ops.group(
        max_time_seen_assign, chunk_count_assign,
        inter_observation_duration_assign, example_count_assign,
        overall_feature_sum_assign, overall_feature_sum_of_squares_assign)
    with ops.control_dependencies([per_chunk_aux_updates]):
      example_count_float = math_ops.cast(auxiliary_variables.example_count,
                                          self._dtype)
      new_feature_mean = (auxiliary_variables.overall_feature_sum /
                          example_count_float)
      overall_feature_mean_update = state_ops.assign(
          statistics.overall_feature_moments.mean, new_feature_mean)
      overall_feature_var_update = state_ops.assign(
          statistics.overall_feature_moments.variance,
          # De-biased n / (n - 1) variance correction
          example_count_float / (example_count_float - 1.) *
          (auxiliary_variables.overall_feature_sum_of_squares /
           example_count_float - new_feature_mean**2))
      # TODO(b/35675805): Remove this cast
      min_time_batch = math_ops.cast(math_ops.argmin(times[:, 0]), dtypes.int32)
      def series_start_updates():
        # If this is the lowest-time chunk that we have seen so far, update
        # series start moments to reflect that. Note that these statistics are
        # "best effort", as there are race conditions in the update (however,
        # they should eventually converge if the start of the series is
        # presented enough times).
        mean, variance = nn.moments(
            values[min_time_batch, :self._starting_variance_window_size],
            axes=[0])
        return control_flow_ops.group(
            state_ops.assign(statistics.series_start_moments.mean, mean),
            state_ops.assign(statistics.series_start_moments.variance,
                             variance))
      with ops.colocate_with(statistics.start_time):
        series_start_update = control_flow_ops.cond(
            # Update moments whenever we even match the lowest time seen so far,
            # to ensure that series start statistics are eventually updated to
            # their correct values, despite race conditions (i.e. eventually
            # statistics.start_time will reflect the global lowest time, and
            # given that we will eventually update the series start moments to
            # their correct values).
            math_ops.less_equal(times[min_time_batch, 0],
                                statistics.start_time),
            series_start_updates,
            control_flow_ops.no_op)
        with ops.control_dependencies([series_start_update]):
          # There is a race condition if this update is performed in parallel on
          # multiple workers. Since models may be sensitive to being presented
          # with times before the putative start time, the value of this
          # variable is post-processed above to guarantee that each worker is
          # presented with a start time which is at least as low as the lowest
          # time in its current mini-batch.
          start_time_update = state_ops.assign(statistics.start_time,
                                               gen_math_ops.minimum(
                                                   statistics.start_time,
                                                   math_ops.reduce_min(times)))
      inter_observation_duration_estimate = (
          auxiliary_variables.inter_observation_duration_sum / math_ops.cast(
              auxiliary_variables.chunk_count, self._dtype))
      # Estimate the total number of observations as:
      #   (end time - start time + 1) * average intra-chunk time density
      total_observation_count_update = state_ops.assign(
          statistics.total_observation_count,
          math_ops.cast(
              gen_math_ops.round(
                  math_ops.cast(auxiliary_variables.max_time_seen -
                                statistics.start_time + 1, self._dtype) /
                  inter_observation_duration_estimate), dtypes.int64))
      per_chunk_stat_updates = control_flow_ops.group(
          overall_feature_mean_update, overall_feature_var_update,
          series_start_update, start_time_update,
          total_observation_count_update)
    return per_chunk_stat_updates

  def _create_variable_statistics_object(self):
    """Creates non-trainable variables representing input statistics."""
    series_start_moments = Moments(
        mean=variable_scope.get_variable(
            name="series_start_mean",
            shape=[self._num_features],
            dtype=self._dtype,
            initializer=init_ops.zeros_initializer(),
            trainable=False),
        variance=variable_scope.get_variable(
            name="series_start_variance",
            shape=[self._num_features],
            dtype=self._dtype,
            initializer=init_ops.ones_initializer(),
            trainable=False))
    overall_feature_moments = Moments(
        mean=variable_scope.get_variable(
            name="overall_feature_mean",
            shape=[self._num_features],
            dtype=self._dtype,
            initializer=init_ops.zeros_initializer(),
            trainable=False),
        variance=variable_scope.get_variable(
            name="overall_feature_var",
            shape=[self._num_features],
            dtype=self._dtype,
            initializer=init_ops.ones_initializer(),
            trainable=False))
    start_time = variable_scope.get_variable(
        name="start_time",
        dtype=dtypes.int64,
        initializer=init_ops.zeros_initializer(),
        shape=[],
        trainable=False)
    total_observation_count = variable_scope.get_variable(
        name="total_observation_count",
        shape=[],
        dtype=dtypes.int64,
        initializer=init_ops.ones_initializer(),
        trainable=False)
    return InputStatistics(
        series_start_moments=series_start_moments,
        overall_feature_moments=overall_feature_moments,
        start_time=start_time,
        total_observation_count=total_observation_count)
