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
"""Utilities for probability distributions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg
from tensorflow.python.ops.distributions import distribution as distribution_lib

# The following two lines are redundant, in a sense. The first enables
# good coding practice  *within* this file (`util.prefer_static_value`
# rather than  `prefer_static_value`). The  second ensures  that users
# also get the core utils when they import this file.
from tensorflow.python.ops.distributions import util
from tensorflow.python.ops.distributions.util import *  # pylint: disable=wildcard-import


def _convert_to_tensor(x, name):
  return None if x is None else ops.convert_to_tensor(x, name=name)


def mixture_stddev(mixture_weight_vector, mean_vector, stddev_vector):
  """Computes the standard deviation of a mixture distribution.

  This function works regardless of the component distribution, so long as
  each component's mean and standard deviation can be provided.

  Args:
    mixture_weight_vector: A 2D tensor with shape [batch_size, num_components]
    mean_vector: A 2D tensor of mixture component means. Has shape
      `[batch_size, num_components]`.
    stddev_vector: A 2D tensor of mixture component standard deviations. Has
      shape `[batch_size, num_components]`.
  Returns:
    A 1D tensor of shape `[batch_size]` representing the standard deviation of
    the mixture distribution with given weights and component means and standard
    deviations.
  Raises:
    ValueError: If the shapes of the input tensors are not as expected.
  """
  mixture_weight_vector.shape.assert_has_rank(2)
  if not mean_vector.shape.is_compatible_with(mixture_weight_vector.shape):
    raise ValueError("Expecting means to have same shape as mixture weights.")
  if not stddev_vector.shape.is_compatible_with(mixture_weight_vector.shape):
    raise ValueError("Expecting stddevs to have same shape as mixture weights.")

  # Reshape the distribution parameters for batched vectorized dot products.
  pi_for_dot_prod = array_ops.expand_dims(mixture_weight_vector, axis=1)
  mu_for_dot_prod = array_ops.expand_dims(mean_vector, axis=2)
  sigma_for_dot_prod = array_ops.expand_dims(stddev_vector, axis=2)

  # weighted average of component means under mixture distribution.
  mean_wa = math_ops.matmul(pi_for_dot_prod, mu_for_dot_prod)
  mean_wa = array_ops.reshape(mean_wa, (-1,))
  # weighted average of component variances under mixture distribution.
  var_wa = math_ops.matmul(pi_for_dot_prod,
                           math_ops.square(sigma_for_dot_prod))
  var_wa = array_ops.reshape(var_wa, (-1,))
  # weighted average of component squared means under mixture distribution.
  sq_mean_wa = math_ops.matmul(pi_for_dot_prod,
                               math_ops.square(mu_for_dot_prod))
  sq_mean_wa = array_ops.reshape(sq_mean_wa, (-1,))
  mixture_variance = var_wa + sq_mean_wa - math_ops.square(mean_wa)
  return math_ops.sqrt(mixture_variance)


def make_tril_scale(
    loc=None,
    scale_tril=None,
    scale_diag=None,
    scale_identity_multiplier=None,
    shape_hint=None,
    validate_args=False,
    assert_positive=False,
    name=None):
  """Creates a LinOp representing a lower triangular matrix.

  Args:
    loc: Floating-point `Tensor`. This is used for inferring shape in the case
      where only `scale_identity_multiplier` is set.
    scale_tril: Floating-point `Tensor` representing the diagonal matrix.
      `scale_diag` has shape [N1, N2, ...  k, k], which represents a k x k
      lower triangular matrix.
      When `None` no `scale_tril` term is added to the LinOp.
      The upper triangular elements above the diagonal are ignored.
    scale_diag: Floating-point `Tensor` representing the diagonal matrix.
      `scale_diag` has shape [N1, N2, ...  k], which represents a k x k
      diagonal matrix.
      When `None` no diagonal term is added to the LinOp.
    scale_identity_multiplier: floating point rank 0 `Tensor` representing a
      scaling done to the identity matrix.
      When `scale_identity_multiplier = scale_diag = scale_tril = None` then
      `scale += IdentityMatrix`. Otherwise no scaled-identity-matrix is added
      to `scale`.
    shape_hint: scalar integer `Tensor` representing a hint at the dimension of
      the identity matrix when only `scale_identity_multiplier` is set.
    validate_args: Python `bool` indicating whether arguments should be
      checked for correctness.
    assert_positive: Python `bool` indicating whether LinOp should be checked
      for being positive definite.
    name: Python `str` name given to ops managed by this object.

  Returns:
    `LinearOperator` representing a lower triangular matrix.

  Raises:
    ValueError:  If only `scale_identity_multiplier` is set and `loc` and
      `shape_hint` are both None.
  """

  def _maybe_attach_assertion(x):
    if not validate_args:
      return x
    if assert_positive:
      return control_flow_ops.with_dependencies([
          check_ops.assert_positive(
              array_ops.matrix_diag_part(x),
              message="diagonal part must be positive"),
      ], x)
    return control_flow_ops.with_dependencies([
        check_ops.assert_none_equal(
            array_ops.matrix_diag_part(x),
            array_ops.zeros([], x.dtype),
            message="diagonal part must be non-zero"),
    ], x)

  with ops.name_scope(name, "make_tril_scale",
                      values=[loc, scale_diag, scale_identity_multiplier]):

    loc = _convert_to_tensor(loc, name="loc")
    scale_tril = _convert_to_tensor(scale_tril, name="scale_tril")
    scale_diag = _convert_to_tensor(scale_diag, name="scale_diag")
    scale_identity_multiplier = _convert_to_tensor(
        scale_identity_multiplier,
        name="scale_identity_multiplier")

  if scale_tril is not None:
    scale_tril = array_ops.matrix_band_part(scale_tril, -1, 0)  # Zero out TriU.
    tril_diag = array_ops.matrix_diag_part(scale_tril)
    if scale_diag is not None:
      tril_diag += scale_diag
    if scale_identity_multiplier is not None:
      tril_diag += scale_identity_multiplier[..., array_ops.newaxis]

    scale_tril = array_ops.matrix_set_diag(scale_tril, tril_diag)

    return linalg.LinearOperatorLowerTriangular(
        tril=_maybe_attach_assertion(scale_tril),
        is_non_singular=True,
        is_self_adjoint=False,
        is_positive_definite=assert_positive)

  return make_diag_scale(
      loc=loc,
      scale_diag=scale_diag,
      scale_identity_multiplier=scale_identity_multiplier,
      shape_hint=shape_hint,
      validate_args=validate_args,
      assert_positive=assert_positive,
      name=name)


def make_diag_scale(
    loc=None,
    scale_diag=None,
    scale_identity_multiplier=None,
    shape_hint=None,
    validate_args=False,
    assert_positive=False,
    name=None):
  """Creates a LinOp representing a diagonal matrix.

  Args:
    loc: Floating-point `Tensor`. This is used for inferring shape in the case
      where only `scale_identity_multiplier` is set.
    scale_diag: Floating-point `Tensor` representing the diagonal matrix.
      `scale_diag` has shape [N1, N2, ...  k], which represents a k x k
      diagonal matrix.
      When `None` no diagonal term is added to the LinOp.
    scale_identity_multiplier: floating point rank 0 `Tensor` representing a
      scaling done to the identity matrix.
      When `scale_identity_multiplier = scale_diag = scale_tril = None` then
      `scale += IdentityMatrix`. Otherwise no scaled-identity-matrix is added
      to `scale`.
    shape_hint: scalar integer `Tensor` representing a hint at the dimension of
      the identity matrix when only `scale_identity_multiplier` is set.
    validate_args: Python `bool` indicating whether arguments should be
      checked for correctness.
    assert_positive: Python `bool` indicating whether LinOp should be checked
      for being positive definite.
    name: Python `str` name given to ops managed by this object.

  Returns:
    `LinearOperator` representing a lower triangular matrix.

  Raises:
    ValueError:  If only `scale_identity_multiplier` is set and `loc` and
      `shape_hint` are both None.
  """

  def _maybe_attach_assertion(x):
    if not validate_args:
      return x
    if assert_positive:
      return control_flow_ops.with_dependencies([
          check_ops.assert_positive(
              x, message="diagonal part must be positive"),
      ], x)
    return control_flow_ops.with_dependencies([
        check_ops.assert_none_equal(
            x,
            array_ops.zeros([], x.dtype),
            message="diagonal part must be non-zero")], x)

  with ops.name_scope(name, "make_diag_scale",
                      values=[loc, scale_diag, scale_identity_multiplier]):
    loc = _convert_to_tensor(loc, name="loc")
    scale_diag = _convert_to_tensor(scale_diag, name="scale_diag")
    scale_identity_multiplier = _convert_to_tensor(
        scale_identity_multiplier,
        name="scale_identity_multiplier")

    if scale_diag is not None:
      if scale_identity_multiplier is not None:
        scale_diag += scale_identity_multiplier[..., array_ops.newaxis]
      return linalg.LinearOperatorDiag(
          diag=_maybe_attach_assertion(scale_diag),
          is_non_singular=True,
          is_self_adjoint=True,
          is_positive_definite=assert_positive)

    if loc is None and shape_hint is None:
      raise ValueError(
          "Cannot infer `event_shape` unless `loc` or "
          "`shape_hint` is specified.")

    if shape_hint is None:
      shape_hint = loc.shape[-1]

    if scale_identity_multiplier is None:
      return linalg.LinearOperatorIdentity(
          num_rows=shape_hint,
          dtype=loc.dtype.base_dtype,
          is_self_adjoint=True,
          is_positive_definite=True,
          assert_proper_shapes=validate_args)

    return linalg.LinearOperatorScaledIdentity(
        num_rows=shape_hint,
        multiplier=_maybe_attach_assertion(scale_identity_multiplier),
        is_non_singular=True,
        is_self_adjoint=True,
        is_positive_definite=assert_positive,
        assert_proper_shapes=validate_args)


def shapes_from_loc_and_scale(loc, scale, name="shapes_from_loc_and_scale"):
  """Infer distribution batch and event shapes from a location and scale.

  Location and scale family distributions determine their batch/event shape by
  broadcasting the `loc` and `scale` args.  This helper does that broadcast,
  statically if possible.

  Batch shape broadcasts as per the normal rules.
  We allow the `loc` event shape to broadcast up to that of `scale`.  We do not
  allow `scale`'s event shape to change.  Therefore, the last dimension of `loc`
  must either be size `1`, or the same as `scale.range_dimension`.

  See `MultivariateNormalLinearOperator` for a usage example.

  Args:
    loc:  `N-D` `Tensor` with `N >= 1` (already converted to tensor) or `None`.
      If `None`, both batch and event shape are determined by `scale`.
    scale:  A `LinearOperator` instance.
    name:  A string name to prepend to created ops.

  Returns:
    batch_shape:  `TensorShape` (if broadcast is done statically), or `Tensor`.
    event_shape:  `TensorShape` (if broadcast is done statically), or `Tensor`.

  Raises:
    ValueError:  If the last dimension of `loc` is determined statically to be
      different than the range of `scale`.
  """
  with ops.name_scope(name, values=[loc] + scale.graph_parents):
    # Get event shape.
    event_size = scale.range_dimension_tensor()
    event_size_const = tensor_util.constant_value(event_size)
    if event_size_const is not None:
      event_shape = event_size_const.reshape([1])
    else:
      event_shape = event_size[array_ops.newaxis]

    # Static check that event shapes match.
    if loc is not None:
      loc_event_size = loc.get_shape()[-1].value
      if loc_event_size is not None and event_size_const is not None:
        if loc_event_size != 1 and loc_event_size != event_size_const:
          raise ValueError(
              "Event size of 'scale' (%d) could not be broadcast up to that of "
              "'loc' (%d)." % (loc_event_size, event_size_const))

    # Get batch shape.
    batch_shape = scale.batch_shape_tensor()
    if loc is None:
      batch_shape_const = tensor_util.constant_value(batch_shape)
      batch_shape = (
          batch_shape_const if batch_shape_const is not None else batch_shape)
    else:
      loc_batch_shape = loc.get_shape().with_rank_at_least(1)[:-1]
      if (loc.get_shape().ndims is None or
          not loc_batch_shape.is_fully_defined()):
        loc_batch_shape = array_ops.shape(loc)[:-1]
      else:
        loc_batch_shape = ops.convert_to_tensor(loc_batch_shape,
                                                name="loc_batch_shape")
      # This is defined in the core util module.
      # pylint: disable=undefined-variable
      batch_shape = prefer_static_broadcast_shape(batch_shape, loc_batch_shape)
      # pylint: enable=undefined-variable

  return batch_shape, event_shape


def get_broadcast_shape(*tensors):
  """Get broadcast shape as a Python list of integers (preferred) or `Tensor`.

  Args:
    *tensors:  One or more `Tensor` objects (already converted!).

  Returns:
    broadcast shape:  Python list (if shapes determined statically), otherwise
      an `int32` `Tensor`.
  """
  # Try static.
  s_shape = tensors[0].shape
  for t in tensors[1:]:
    s_shape = array_ops.broadcast_static_shape(s_shape, t.shape)
  if s_shape.is_fully_defined():
    return s_shape.as_list()

  # Fallback on dynamic.
  d_shape = array_ops.shape(tensors[0])
  for t in tensors[1:]:
    d_shape = array_ops.broadcast_dynamic_shape(d_shape, array_ops.shape(t))
  return d_shape


def is_diagonal_scale(scale):
  """Returns `True` if `scale` is a `LinearOperator` that is known to be diag.

  Args:
    scale:  `LinearOperator` instance.

  Returns:
    Python `bool`.

  Raises:
    TypeError:  If `scale` is not a `LinearOperator`.
  """
  if not isinstance(scale, linalg.LinearOperator):
    raise TypeError("Expected argument 'scale' to be instance of LinearOperator"
                    ". Found: %s" % scale)
  return (isinstance(scale, linalg.LinearOperatorIdentity) or
          isinstance(scale, linalg.LinearOperatorScaledIdentity) or
          isinstance(scale, linalg.LinearOperatorDiag))


def maybe_check_scalar_distribution(
    distribution, expected_base_dtype, validate_args):
  """Helper which checks validity of a scalar `distribution` init arg.

  Valid here means:

  * `distribution` has scalar batch and event shapes.
  * `distribution` is `FULLY_REPARAMETERIZED`
  * `distribution` has expected dtype.

  Args:
    distribution:  `Distribution`-like object.
    expected_base_dtype:  `TensorFlow` `dtype`.
    validate_args:  Python `bool`.  Whether to do additional checks:
      (i)  check that reparameterization_type is `FULLY_REPARAMETERIZED`.
      (ii) add `tf.Assert` ops to the graph to enforce that distribution
           is scalar in the event that this cannot be determined statically.

  Returns:
    List of `tf.Assert` ops to run to enforce validity checks that could not
      be statically determined.  Empty if `not validate_args`.

  Raises:
    ValueError:  If validate_args and distribution is not FULLY_REPARAMETERIZED
    ValueError:  If distribution is statically determined to not have both
      scalar batch and scalar event shapes.
  """
  if distribution.dtype != expected_base_dtype:
    raise TypeError("dtype mismatch; "
                    "distribution.dtype=\"{}\" is not \"{}\"".format(
                        distribution.dtype.name, expected_base_dtype.name))

  # Although `reparameterization_type` is a static property, we guard it by
  # `validate_args`. This allows users to use a `distribution` which is not
  # reparameterized itself. However, we tacitly assume that although the
  # distribution is not reparameterized, it only depends on non-trainable
  # variables.
  if validate_args and (distribution.reparameterization_type
                        != distribution_lib.FULLY_REPARAMETERIZED):
    raise ValueError("Base distribution should be reparameterized or be "
                     "a function of non-trainable variables; "
                     "distribution.reparameterization_type = \"{}\" "
                     "!= \"FULLY_REPARAMETERIZED\".".format(
                         distribution.reparameterization_type))
  with ops.name_scope(name="check_distribution"):
    assertions = []
    def check_is_scalar(is_scalar, name):
      is_scalar_ = static_value(is_scalar)
      if is_scalar_ is not None:
        if not is_scalar_:
          raise ValueError("distribution must be scalar; "
                           "distribution.{}=False is not True".format(name))
      elif validate_args:
        assertions.append(check_ops.assert_equal(
            is_scalar, True,
            message=("distribution must be scalar; "
                     "distribution.{}=False is not True".format(name))))
    check_is_scalar(distribution.is_scalar_event(), "is_scalar_event")
    check_is_scalar(distribution.is_scalar_batch(), "is_scalar_batch")
    return assertions


def pad_mixture_dimensions(x, mixture_distribution, categorical_distribution,
                           event_ndims):
  """Pad dimensions of event tensors for mixture distributions.

  See `Mixture._sample_n` and `MixtureSameFamily._sample_n` for usage examples.

  Args:
    x: event tensor to pad.
    mixture_distribution: Base distribution of the mixture.
    categorical_distribution: `Categorical` distribution that mixes the base
      distribution.
    event_ndims: Integer specifying the number of event dimensions in the event
      tensor.

  Returns:
    A padded version of `x` that can broadcast with `categorical_distribution`.
  """
  with ops.name_scope("pad_mix_dims", values=[x]):
    def _get_ndims(d):
      if d.batch_shape.ndims is not None:
        return d.batch_shape.ndims
      return array_ops.shape(d.batch_shape_tensor())[0]
    dist_batch_ndims = _get_ndims(mixture_distribution)
    cat_batch_ndims = _get_ndims(categorical_distribution)
    pad_ndims = array_ops.where(
        categorical_distribution.is_scalar_batch(),
        dist_batch_ndims,
        dist_batch_ndims - cat_batch_ndims)
    s = array_ops.shape(x)
    x = array_ops.reshape(x, shape=array_ops.concat([
        s[:-1],
        array_ops.ones([pad_ndims], dtype=dtypes.int32),
        s[-1:],
        array_ops.ones([event_ndims], dtype=dtypes.int32),
    ], axis=0))
    return x


def static_value(x):
  """Returns the static value of a `Tensor` or `None`."""
  return tensor_util.constant_value(ops.convert_to_tensor(x))


def move_dimension(x, source_idx, dest_idx):
  """Move a single tensor dimension within its shape.

  This is a special case of `tf.transpose()`, which applies
  arbitrary permutations to tensor dimensions.

  Args:
    x: Tensor of rank `ndims`.
    source_idx: Integer index into `x.shape` (negative indexing is
      supported).
    dest_idx: Integer index into `x.shape` (negative indexing is
      supported).

  Returns:
    x_perm: Tensor of rank `ndims`, in which the dimension at original
     index `source_idx` has been moved to new index `dest_idx`, with
     all other dimensions retained in their original order.

  Example:

  ```python
  x = tf.placeholder(shape=[200, 30, 4, 1, 6])
  x_perm = _move_dimension(x, 1, 1) # no-op
  x_perm = _move_dimension(x, 0, 3) # result shape [30, 4, 1, 200, 6]
  x_perm = _move_dimension(x, 0, -2) # equivalent to previous
  x_perm = _move_dimension(x, 4, 2) # result shape [200, 30, 6, 4, 1]
  ```
  """
  ndims = util.prefer_static_rank(x)
  if isinstance(source_idx, int):
    dtype = dtypes.int32
  else:
    dtype = dtypes.as_dtype(source_idx.dtype)

  # Handle negative indexing. Since ndims might be dynamic, this makes
  # source_idx and dest_idx also possibly dynamic.
  if source_idx < 0:
    source_idx = ndims + source_idx
  if dest_idx < 0:
    dest_idx = ndims + dest_idx

  # Construct the appropriate permutation of dimensions, depending
  # whether the source is before or after the destination.
  def move_left_permutation():
    return util.prefer_static_value(
        array_ops.concat([
            math_ops.range(0, dest_idx, dtype=dtype),
            [source_idx],
            math_ops.range(dest_idx, source_idx, dtype=dtype),
            math_ops.range(source_idx+1, ndims, dtype=dtype)], axis=0))

  def move_right_permutation():
    return util.prefer_static_value(
        array_ops.concat([
            math_ops.range(0, source_idx, dtype=dtype),
            math_ops.range(source_idx+1, dest_idx+1, dtype=dtype),
            [source_idx],
            math_ops.range(dest_idx+1, ndims, dtype=dtype)], axis=0))

  def x_permuted():
    return array_ops.transpose(
        x, perm=smart_cond.smart_cond(source_idx < dest_idx,
                                      move_right_permutation,
                                      move_left_permutation))

  # One final conditional to handle the special case where source
  # and destination indices are equal.
  return smart_cond.smart_cond(math_ops.equal(source_idx, dest_idx),
                               lambda: x,
                               x_permuted)
