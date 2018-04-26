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
"""FisherFactor definitions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import contextlib

import numpy as np
import six

from tensorflow.contrib.kfac.python.ops import utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import moving_averages
from tensorflow.python.util import nest


# Whether to initialize covariance estimators at a zero matrix (or the identity
# matrix).
INIT_COVARIANCES_AT_ZERO = False

# Whether to zero-debias the moving averages.
ZERO_DEBIAS = False

# When the number of inverses requested from a FisherFactor exceeds this value,
# the inverses are computed using an eigenvalue decomposition.
EIGENVALUE_DECOMPOSITION_THRESHOLD = 2

# Numerical eigenvalues computed from covariance matrix estimates are clipped to
# be at least as large as this value before they are used to compute inverses or
# matrix powers. Must be nonnegative.
EIGENVALUE_CLIPPING_THRESHOLD = 0.0

# Used to subsample the flattened extracted image patches. The number of
# outer products per row of the covariance matrix should not exceed this
# value. This parameter is used only if `_SUB_SAMPLE_OUTER_PRODUCTS` is True.
_MAX_NUM_OUTER_PRODUCTS_PER_COV_ROW = 1

# Used to subsample the inputs passed to the extract image patches. The batch
# size of number of inputs to extract image patches is multiplied by this
# factor. This parameter is used only if `_SUB_SAMPLE_INPUTS` is True.
_INPUTS_TO_EXTRACT_PATCHES_FACTOR = 0.5

# If True, then subsamples the tensor passed to compute the covaraince matrix.
_SUB_SAMPLE_OUTER_PRODUCTS = False

# If True, then subsamples the tensor passed to compute the covaraince matrix.
_SUB_SAMPLE_INPUTS = False

# TOWER_STRATEGY can be one of "concat" or "separate".  If "concat", the data
# passed to the factors from the blocks will be concatenated across towers
# (lazilly via PartitionedTensor objects).  Otherwise a tuple of tensors over
# towers will be passed in, and the factors will iterate over this and do the
# cov computations separately for each one, averaging the results together.
TOWER_STRATEGY = "concat"


def set_global_constants(init_covariances_at_zero=None,
                         zero_debias=None,
                         eigenvalue_decomposition_threshold=None,
                         eigenvalue_clipping_threshold=None,
                         max_num_outer_products_per_cov_row=None,
                         sub_sample_outer_products=None,
                         inputs_to_extract_patches_factor=None,
                         sub_sample_inputs=None,
                         tower_strategy=None):
  """Sets various global constants used by the classes in this module."""
  global INIT_COVARIANCES_AT_ZERO
  global ZERO_DEBIAS
  global EIGENVALUE_DECOMPOSITION_THRESHOLD
  global EIGENVALUE_CLIPPING_THRESHOLD
  global _MAX_NUM_OUTER_PRODUCTS_PER_COV_ROW
  global _SUB_SAMPLE_OUTER_PRODUCTS
  global _INPUTS_TO_EXTRACT_PATCHES_FACTOR
  global _SUB_SAMPLE_INPUTS
  global TOWER_STRATEGY

  if init_covariances_at_zero is not None:
    INIT_COVARIANCES_AT_ZERO = init_covariances_at_zero
  if zero_debias is not None:
    ZERO_DEBIAS = zero_debias
  if eigenvalue_decomposition_threshold is not None:
    EIGENVALUE_DECOMPOSITION_THRESHOLD = eigenvalue_decomposition_threshold
  if eigenvalue_clipping_threshold is not None:
    EIGENVALUE_CLIPPING_THRESHOLD = eigenvalue_clipping_threshold
  if max_num_outer_products_per_cov_row is not None:
    _MAX_NUM_OUTER_PRODUCTS_PER_COV_ROW = max_num_outer_products_per_cov_row
  if sub_sample_outer_products is not None:
    _SUB_SAMPLE_OUTER_PRODUCTS = sub_sample_outer_products
  if inputs_to_extract_patches_factor is not None:
    _INPUTS_TO_EXTRACT_PATCHES_FACTOR = inputs_to_extract_patches_factor
  if sub_sample_inputs is not None:
    _SUB_SAMPLE_INPUTS = sub_sample_inputs
  if tower_strategy is not None:
    TOWER_STRATEGY = tower_strategy


def inverse_initializer(shape, dtype, partition_info=None):  # pylint: disable=unused-argument
  return array_ops.diag(array_ops.ones(shape[0], dtype))


def covariance_initializer(shape, dtype, partition_info=None):  # pylint: disable=unused-argument
  if INIT_COVARIANCES_AT_ZERO:
    return array_ops.diag(array_ops.zeros(shape[0], dtype))
  return array_ops.diag(array_ops.ones(shape[0], dtype))


def diagonal_covariance_initializer(shape, dtype, partition_info):  # pylint: disable=unused-argument
  if INIT_COVARIANCES_AT_ZERO:
    return array_ops.zeros(shape, dtype)
  return array_ops.ones(shape, dtype)


@contextlib.contextmanager
def place_on_device(device):
  if device is not None and len(device):
    with tf_ops.device(device):
      yield
  else:
    yield


def compute_cov(tensor, tensor_right=None, normalizer=None):
  """Compute the empirical second moment of the rows of a 2D Tensor.

  This function is meant to be applied to random matrices for which the true row
  mean is zero, so that the true second moment equals the true covariance.

  Args:
    tensor: A 2D Tensor.
    tensor_right: An optional 2D Tensor. If provided, this function computes
      the matrix product tensor^T * tensor_right instead of tensor^T * tensor.
    normalizer: optional scalar for the estimator (by default, the normalizer is
        the number of rows of tensor).

  Returns:
    A square 2D Tensor with as many rows/cols as the number of input columns.
  """
  if normalizer is None:
    normalizer = array_ops.shape(tensor)[0]
  if tensor_right is None:
    cov = (
        math_ops.matmul(tensor, tensor, transpose_a=True) / math_ops.cast(
            normalizer, tensor.dtype))
    return (cov + array_ops.transpose(cov)) / math_ops.cast(2.0, cov.dtype)
  else:
    return (math_ops.matmul(tensor, tensor_right, transpose_a=True) /
            math_ops.cast(normalizer, tensor.dtype))


def append_homog(tensor):
  """Appends a homogeneous coordinate to the last dimension of a Tensor.

  Args:
    tensor: A Tensor.

  Returns:
    A Tensor identical to the input but one larger in the last dimension.  The
    new entries are filled with ones.
  """
  rank = len(tensor.shape.as_list())
  shape = array_ops.concat([array_ops.shape(tensor)[:-1], [1]], axis=0)
  ones = array_ops.ones(shape, dtype=tensor.dtype)
  return array_ops.concat([tensor, ones], axis=rank - 1)


def scope_string_from_params(params):
  """Builds a variable scope string name from the given parameters.

  Supported parameters are:
    * tensors
    * booleans
    * ints
    * strings
    * depth-1 tuples/lists of ints
    * any depth tuples/lists of tensors
  Other parameter types will throw an error.

  Args:
    params: A parameter or list of parameters.

  Returns:
    A string to use for the variable scope.

  Raises:
    ValueError: if params includes an unsupported type.
  """
  params = params if isinstance(params, (tuple, list)) else (params,)

  name_parts = []
  for param in params:
    if param is None:
      name_parts.append("None")
    elif isinstance(param, (tuple, list)):
      if all([isinstance(p, int) for p in param]):
        name_parts.append("-".join([str(p) for p in param]))
      else:
        name_parts.append(scope_string_from_name(param))
    elif isinstance(param, (str, int, bool)):
      name_parts.append(str(param))
    elif isinstance(param, (tf_ops.Tensor, variables.Variable)):
      name_parts.append(scope_string_from_name(param))
    elif isinstance(param, utils.PartitionedTensor):
      name_parts.append(scope_string_from_name(param.tensors))
    else:
      raise ValueError("Encountered an unsupported param type {}".format(
          type(param)))
  return "_".join(name_parts)


def scope_string_from_name(tensor):
  if isinstance(tensor, (tuple, list)):
    return "__".join([scope_string_from_name(t) for t in tensor])
  # "gradients/add_4_grad/Reshape:0" -> "gradients_add_4_grad_Reshape"
  return tensor.name.split(":")[0].replace("/", "_")


def scalar_or_tensor_to_string(val):
  return repr(val) if np.isscalar(val) else scope_string_from_name(val)


def list_to_string(lst):
  return "_".join(val if isinstance(val, six.string_types)
                  else scalar_or_tensor_to_string(val) for val in lst)


def graph_func_to_id(func):
  """Returns a hashable object that represents func's computation."""
  # TODO(b/74201126): replace with Topohash of func's output
  return func.func_id


def graph_func_to_string(func):
  # TODO(b/74201126): replace with Topohash of func's output
  return list_to_string(func.func_id)


def _subsample_for_cov_computation(array, name=None):
  """Subsamples the first dimension of the array.

  `array`(A) is a tensor of shape `[batch_size, dim_2]`. Then the covariance
  matrix(A^TA) is of shape `dim_2 ** 2`. Subsample only if the number of outer
  products per row of the covariance matrix is greater than
  `_MAX_NUM_OUTER_PRODUCTS_PER_COV_ROW`.

  Args:
    array: Tensor, of shape `[batch_size, dim_2]`.
    name: `string`, Default(None)

  Returns:
    A tensor of shape `[max_samples, dim_2]`.

  Raises:
    ValueError: If array's is not matrix-shaped.
    ValueError: If array's batch_size cannot be inferred.

  """
  with tf_ops.name_scope(name, "subsample", [array]):
    array = tf_ops.convert_to_tensor(array)
    if len(array.shape) != 2:
      raise ValueError("Input param array must be a matrix.")

    batch_size = array.shape.as_list()[0]
    if batch_size is None:
      raise ValueError("Unable to get batch_size from input param array.")

    num_cov_rows = array.shape.as_list()[-1]
    max_batch_size = int(_MAX_NUM_OUTER_PRODUCTS_PER_COV_ROW * num_cov_rows)
    if batch_size <= max_batch_size:
      return array

    return _random_tensor_gather(array, max_batch_size)


def _random_tensor_gather(array, max_size):
  """Generates a random set of indices and gathers the value at the indcices.

  Args:
    array: Tensor, of shape `[batch_size, dim_2]`.
    max_size: int, Number of indices to sample.

  Returns:
    A tensor of shape `[max_size, ...]`.
  """
  batch_size = array.shape.as_list()[0]
  indices = random_ops.random_shuffle(math_ops.range(0, batch_size))[:max_size]
  return array_ops.gather(array, indices)


@six.add_metaclass(abc.ABCMeta)
class FisherFactor(object):
  """Base class for objects modeling factors of approximate Fisher blocks.

  A FisherFactor represents part of an approximate Fisher Information matrix.
  For example, one approximation to the Fisher uses the Kronecker product of two
  FisherFactors A and B, F = kron(A, B). FisherFactors are composed with
  FisherBlocks to construct a block-diagonal approximation to the full Fisher.

  FisherFactors are backed by a single, non-trainable variable that is updated
  by running FisherFactor.make_covariance_update_op(). The shape and type of
  this variable is implementation specific.

  Note that for blocks that aren't based on approximations, a 'factor' can
  be the entire block itself, as is the case for the diagonal and full
  representations.
  """

  def __init__(self):
    self._cov = None

  @abc.abstractproperty
  def _var_scope(self):
    """Variable scope for this FisherFactor instance.

    Returns:
      string that unique identifies this FisherFactor instance.
    """
    pass

  @property
  def name(self):
    return self._var_scope

  @abc.abstractproperty
  def _cov_shape(self):
    """The shape of the variable backing this FisherFactor."""
    pass

  @abc.abstractproperty
  def _num_sources(self):
    """The number of things to sum over when updating covariance variable.

    The default make_covariance_update_op function will call _compute_new_cov
    with indices ranging from 0 to _num_sources-1. The typical situation is
    where the factor wants to sum the statistics it computes over multiple
    backpropped "gradients" (typically passed in via "tensors" or
    "outputs_grads" arguments).
    """
    pass

  @abc.abstractproperty
  def _num_towers(self):
    pass

  @abc.abstractproperty
  def _dtype(self):
    """dtype for variable backing this factor."""
    pass

  @property
  def _cov_initializer(self):
    """Function for initializing covariance variable."""
    return covariance_initializer

  def instantiate_cov_variables(self):
    """Makes the internal cov variable(s)."""
    assert self._cov is None
    with variable_scope.variable_scope(self._var_scope):
      self._cov = variable_scope.get_variable(
          "cov",
          initializer=self._cov_initializer,
          shape=self._cov_shape,
          trainable=False,
          dtype=self._dtype)

  @abc.abstractmethod
  def _compute_new_cov(self, source, tower):
    """Computes minibatch-estimated covariance for a single source.

    Args:
      source: int in [0, self._num_sources). Which source to use when computing
        the cov update.
      tower: int in [0, self._num_towers). Which tower to use when computing
        the cov update.

    Returns:
      Tensor of same shape as self.get_cov_var().
    """
    pass

  def make_covariance_update_op(self, ema_decay):
    """Constructs and returns the covariance update Op.

    Args:
      ema_decay: The exponential moving average decay (float or Tensor).
    Returns:
      An Op for updating the covariance Variable referenced by _cov.
    """
    new_cov_contribs = []
    for source in range(self._num_sources):
      for tower in range(self._num_towers):
        device = (self._get_data_device(tower)
                  if TOWER_STRATEGY == "separate" else None)
        with place_on_device(device):
          new_cov_contribs.append(self._compute_new_cov(source, tower))

    new_cov = math_ops.add_n(new_cov_contribs) / float(self._num_towers)

    # Compute average of 'new_cov' across all TPU cores. On a TPU, each
    # instance of 'new_cov' will be based on a different minibatch. This ensures
    # that by the end of assign_moving_average(), all TPU cores see the same
    # value for self._cov.
    #
    # Other implementations of make_covariance_update_op() that accumulate
    # statistics in other variables should mimic this behavior.
    if utils.on_tpu():
      new_cov = utils.cross_replica_mean(new_cov)

    return moving_averages.assign_moving_average(
        self._cov, new_cov, ema_decay, zero_debias=ZERO_DEBIAS)

  @abc.abstractmethod
  def _get_data_device(self, tower):
    pass

  @abc.abstractmethod
  def instantiate_inv_variables(self):
    """Makes the internal "inverse" variable(s)."""
    pass

  @abc.abstractmethod
  def make_inverse_update_ops(self):
    """Create and return update ops corresponding to registered computations."""
    pass

  @abc.abstractmethod
  def get_cov(self):
    """Get full covariance matrix.

    Returns:
      Tensor of shape [n, n]. Represents all parameter-parameter correlations
      captured by this FisherFactor.
    """
    pass

  def get_cov_var(self):
    """Get variable backing this FisherFactor.

    May or may not be the same as self.get_cov()

    Returns:
      Variable of shape self._cov_shape.
    """
    return self._cov

  @abc.abstractmethod
  def left_multiply_matpower(self, x, exp, damping_func):
    """Left multiplies 'x' by matrix power of this factor (w/ damping applied).

    This calculation is essentially:
      (C + damping * I)**exp * x
    where * is matrix-multiplication, ** is matrix power, I is the identity
    matrix, and C is the matrix represented by this factor.

    x can represent either a matrix or a vector.  For some factors, 'x' might
    represent a vector but actually be stored as a 2D matrix for convenience.

    Args:
      x: Tensor. Represents a single vector. Shape depends on implementation.
      exp: float.  The matrix exponent to use.
      damping_func: A function that computes a 0-D Tensor or a float which will
        be the damping value used.  i.e. damping = damping_func().

    Returns:
      Tensor of same shape as 'x' representing the result of the multiplication.
    """
    pass

  @abc.abstractmethod
  def right_multiply_matpower(self, x, exp, damping_func):
    """Right multiplies 'x' by matrix power of this factor (w/ damping applied).

    This calculation is essentially:
      x * (C + damping * I)**exp
    where * is matrix-multiplication, ** is matrix power, I is the identity
    matrix, and C is the matrix represented by this factor.

    Unlike left_multiply_matpower, x will always be a matrix.

    Args:
      x: Tensor. Represents a single vector. Shape depends on implementation.
      exp: float.  The matrix exponent to use.
      damping_func: A function that computes a 0-D Tensor or a float which will
        be the damping value used.  i.e. damping = damping_func().

    Returns:
      Tensor of same shape as 'x' representing the result of the multiplication.
    """
    pass


class InverseProvidingFactor(FisherFactor):
  """Base class for FisherFactors that maintain inverses explicitly.

  This class explicitly calculates and stores inverses of covariance matrices
  provided by the underlying FisherFactor implementation. It is assumed that
  vectors can be represented as 2-D matrices.

  Subclasses must implement the _compute_new_cov method, and the _var_scope and
  _cov_shape properties.
  """

  # TODO(b/69108481): This class (and its subclasses) should be refactored to
  # serve the matrix quantities it computes as both (potentially stale)
  # variables, updated by the inverse update ops, and fresh values stored in
  # tensors that recomputed once every session.run() call.  Currently matpower
  # and damp_inverse have the former behavior, while eigendecomposition has
  # the latter.

  def __init__(self):
    self._matpower_by_exp_and_damping = {}  # { (float, hashable): variable }
    self._matpower_registrations = set()  # { (float, hashable) }
    self._eigendecomp = None
    self._damping_funcs_by_id = {}  # {hashable: lambda}

    super(InverseProvidingFactor, self).__init__()

  def _register_damping(self, damping_func):
    damping_id = graph_func_to_id(damping_func)
    if damping_id not in self._damping_funcs_by_id:
      self._damping_funcs_by_id[damping_id] = damping_func
    return damping_id

  def register_inverse(self, damping_func):
    # Just for backwards compatibility of some old code and tests
    self.register_matpower(-1, damping_func)

  def register_matpower(self, exp, damping_func):
    """Registers a matrix power to be maintained and served on demand.

    This creates a variable and signals make_inverse_update_ops to make the
    corresponding update op.  The variable can be read via the method
    get_matpower.

    Args:
      exp: float.  The exponent to use in the matrix power.
      damping_func: A function that computes a 0-D Tensor or a float which will
        be the damping value used.  i.e. damping = damping_func().
    """
    if exp == 1.0:
      # We don't register these.  The user shouldn't even be calling this
      # function with exp = 1.0.
      return

    damping_id = self._register_damping(damping_func)

    if (exp, damping_id) not in self._matpower_registrations:
      self._matpower_registrations.add((exp, damping_id))

  def instantiate_inv_variables(self):
    """Makes the internal "inverse" variable(s)."""

    for (exp, damping_id) in self._matpower_registrations:
      exp_string = scalar_or_tensor_to_string(exp)
      damping_func = self._damping_funcs_by_id[damping_id]
      damping_string = graph_func_to_string(damping_func)
      with variable_scope.variable_scope(self._var_scope):
        matpower = variable_scope.get_variable(
            "matpower_exp{}_damp{}".format(exp_string, damping_string),
            initializer=inverse_initializer,
            shape=self._cov_shape,
            trainable=False,
            dtype=self._dtype)
      assert (exp, damping_id) not in self._matpower_by_exp_and_damping
      self._matpower_by_exp_and_damping[(exp, damping_id)] = matpower

  def make_inverse_update_ops(self):
    """Create and return update ops corresponding to registered computations."""
    ops = []

    num_inverses = sum(1 for (exp, _) in self._matpower_by_exp_and_damping
                       if exp == -1)

    num_other_matpower = len(self._matpower_by_exp_and_damping) - num_inverses

    other_matrix_power_registered = num_other_matpower >= 1

    use_eig = (
        self._eigendecomp or other_matrix_power_registered or
        num_inverses >= EIGENVALUE_DECOMPOSITION_THRESHOLD)

    # We precompute these so we don't need to evaluate them multiple times (for
    # each matrix power that uses them)
    damping_value_by_id = {damping_id: self._damping_funcs_by_id[damping_id]()
                           for damping_id in self._damping_funcs_by_id}

    if use_eig:
      eigenvalues, eigenvectors = self.get_eigendecomp()  # pylint: disable=unpacking-non-sequence

      for (exp, damping_id), matpower in (
          self._matpower_by_exp_and_damping.items()):
        damping = damping_value_by_id[damping_id]
        ops.append(
            matpower.assign(
                math_ops.matmul(eigenvectors *
                                (eigenvalues + damping)**exp,
                                array_ops.transpose(eigenvectors))))
      # These ops share computation and should be run on a single device.
      ops = [control_flow_ops.group(*ops)]
    else:
      for (exp, damping_id), matpower in (
          self._matpower_by_exp_and_damping.items()):
        assert exp == -1
        damping = damping_value_by_id[damping_id]
        ops.append(matpower.assign(utils.posdef_inv(self._cov, damping)))

    self._eigendecomp = False
    return ops

  def get_inverse(self, damping_func):
    # Just for backwards compatibility of some old code and tests
    damping_id = graph_func_to_id(damping_func)
    return self._matpower_by_exp_and_damping[(-1, damping_id)]

  def get_matpower(self, exp, damping_func):
    # Note that this function returns a variable which gets updated by the
    # inverse ops.  It may be stale / inconsistent with the latest value of
    # get_cov().
    damping_id = graph_func_to_id(damping_func)
    return self._matpower_by_exp_and_damping[(exp, damping_id)]

  def get_eigendecomp(self):
    """Creates or retrieves eigendecomposition of self._cov."""
    # Unlike get_matpower this doesn't retrieve a stored variable, but instead
    # always computes a fresh version from the current value of get_cov().
    if not self._eigendecomp:
      eigenvalues, eigenvectors = linalg_ops.self_adjoint_eig(self._cov)

      # The matrix self._cov is positive semidefinite by construction, but the
      # numerical eigenvalues could be negative due to numerical errors, so here
      # we clip them to be at least FLAGS.eigenvalue_clipping_threshold
      clipped_eigenvalues = math_ops.maximum(eigenvalues,
                                             EIGENVALUE_CLIPPING_THRESHOLD)
      self._eigendecomp = (clipped_eigenvalues, eigenvectors)

    return self._eigendecomp

  def get_cov(self):
    # Variable contains full covariance matrix.
    return self.get_cov_var()

  def left_multiply_matpower(self, x, exp, damping_func):
    if isinstance(x, tf_ops.IndexedSlices):
      raise ValueError("Left-multiply not yet supported for IndexedSlices.")

    if x.shape.ndims != 2:
      raise ValueError(
          "InverseProvidingFactors apply to matrix-shaped vectors. Found: %s."
          % (x,))

    if exp == 1:
      return math_ops.matmul(self.get_cov(), x) + damping_func() * x

    return math_ops.matmul(self.get_matpower(exp, damping_func), x)

  def right_multiply_matpower(self, x, exp, damping_func):
    if isinstance(x, tf_ops.IndexedSlices):
      if exp == 1:
        n = self.get_cov().shape[0]
        damped_cov = self.get_cov() + damping_func() * array_ops.eye(n)
        return utils.matmul_sparse_dense(x, damped_cov)

      return utils.matmul_sparse_dense(x, self.get_matpower(exp, damping_func))

    if x.shape.ndims != 2:
      raise ValueError(
          "InverseProvidingFactors apply to matrix-shaped vectors. Found: %s."
          % (x,))

    if exp == 1:
      return math_ops.matmul(x, self.get_cov()) + damping_func() * x

    return math_ops.matmul(x, self.get_matpower(exp, damping_func))


class FullFactor(InverseProvidingFactor):
  """FisherFactor for a full matrix representation of the Fisher of a parameter.

  Note that this uses the naive "square the sum estimator", and so is applicable
  to any type of parameter in principle, but has very high variance.
  """

  def __init__(self,
               params_grads,
               batch_size):
    self._batch_size = batch_size
    self._params_grads = tuple(utils.ensure_sequence(params_grad)
                               for params_grad in params_grads)
    super(FullFactor, self).__init__()

  @property
  def _var_scope(self):
    return "ff_full_" + scope_string_from_params(
        [self._params_grads, self._batch_size])

  @property
  def _cov_shape(self):
    size = sum(param_grad.shape.num_elements()
               for param_grad in self._params_grads[0])
    return (size, size)

  @property
  def _num_sources(self):
    return len(self._params_grads)

  @property
  def _num_towers(self):
    return 1

  @property
  def _dtype(self):
    return self._params_grads[0][0].dtype

  def _compute_new_cov(self, source, tower):
    assert tower == 0

    # This will be a very basic rank 1 estimate
    params_grads_flat = utils.tensors_to_column(self._params_grads[source])
    return ((params_grads_flat * array_ops.transpose(
        params_grads_flat)) / math_ops.cast(self._batch_size,
                                            params_grads_flat.dtype))

  def _get_data_device(self, tower):
    return None


class DiagonalFactor(FisherFactor):
  """A base class for FisherFactors that use diagonal approximations.

  A DiagonalFactor's covariance variable can be of any shape, but must contain
  exactly one entry per parameter.
  """

  def __init__(self):
    self._damping_funcs_by_id = {}  # { hashable: lambda }
    super(DiagonalFactor, self).__init__()

  @property
  def _cov_initializer(self):
    return diagonal_covariance_initializer

  def make_inverse_update_ops(self):
    return []

  def instantiate_inv_variables(self):
    pass

  def get_cov(self):
    # self.get_cov() could be any shape, but it must have one entry per
    # parameter. Flatten it into a vector.
    cov_diag_vec = array_ops.reshape(self.get_cov_var(), [-1])
    return array_ops.diag(cov_diag_vec)

  def left_multiply_matpower(self, x, exp, damping_func):
    matpower = (self.get_cov_var() + damping_func())**exp

    if isinstance(x, tf_ops.IndexedSlices):
      return utils.matmul_diag_sparse(array_ops.reshape(matpower, [-1]), x)

    if x.shape != matpower.shape:
      raise ValueError("x (%s) and cov (%s) must have same shape." %
                       (x, matpower))
    return matpower * x

  def right_multiply_matpower(self, x, exp, damping_func):
    raise NotImplementedError("Only left-multiply is currently supported.")

  def register_matpower(self, exp, damping_func):
    pass


class NaiveDiagonalFactor(DiagonalFactor):
  """FisherFactor for a diagonal approximation of any type of param's Fisher.

  Note that this uses the naive "square the sum estimator", and so is applicable
  to any type of parameter in principle, but has very high variance.
  """

  def __init__(self,
               params_grads,
               batch_size):
    """Initializes NaiveDiagonalFactor instance.

    Args:
      params_grads: Sequence of Tensors, each with same shape as parameters this
        FisherFactor corresponds to. For example, the gradient of the loss with
        respect to parameters.
      batch_size: int or 0-D Tensor. Size
    """
    self._params_grads = tuple(utils.ensure_sequence(params_grad)
                               for params_grad in params_grads)
    self._batch_size = batch_size
    super(NaiveDiagonalFactor, self).__init__()

  @property
  def _var_scope(self):
    return "ff_naivediag_" + scope_string_from_params(
        [self._params_grads, self._batch_size])

  @property
  def _cov_shape(self):
    size = sum(param_grad.shape.num_elements()
               for param_grad in self._params_grads[0])
    return [size, 1]

  @property
  def _num_sources(self):
    return len(self._params_grads)

  @property
  def _num_towers(self):
    return 1

  @property
  def _dtype(self):
    return self._params_grads[0][0].dtype

  def _compute_new_cov(self, source, tower):
    assert tower == 0

    params_grads_flat = utils.tensors_to_column(self._params_grads[source])
    return (math_ops.square(params_grads_flat) / math_ops.cast(
        self._batch_size, params_grads_flat.dtype))

  def _get_data_device(self, tower):
    return None


class EmbeddingInputKroneckerFactor(DiagonalFactor):
  r"""FisherFactor for input to an embedding layer.

  Given input_ids = [batch_size, input_size] representing indices into an
  [vocab_size, embedding_size] embedding matrix, approximate input covariance by
  a diagonal matrix,

    Cov(input_ids, input_ids) =
        (1/batch_size) sum_{i} diag(n_hot(input[i]) ** 2).

  where n_hot() constructs an n-hot binary vector and diag() constructs a
  diagonal matrix of size [vocab_size, vocab_size].
  """

  def __init__(self, input_ids, vocab_size, dtype=None):
    """Instantiate EmbeddingInputKroneckerFactor.

    Args:
      input_ids: List of Tensors of shape [batch_size, input_size] and dtype
        int32. Indices into embedding matrix. List index is tower.
      vocab_size: int or 0-D Tensor. Maximum value for entries in 'input_ids'.
      dtype: dtype for covariance statistics. Must be a floating point type.
        Defaults to float32.
    """
    self._input_ids = input_ids
    self._vocab_size = vocab_size
    self._cov_dtype = dtype or dtypes.float32

    super(EmbeddingInputKroneckerFactor, self).__init__()

  @property
  def _var_scope(self):
    return "ff_diag_embedding_" + scope_string_from_params(self._input_ids)

  @property
  def _cov_shape(self):
    return [self._vocab_size]

  @property
  def _num_sources(self):
    return 1

  @property
  def _num_towers(self):
    return len(self._input_ids)

  @property
  def _dtype(self):
    return self._cov_dtype

  def _compute_new_cov(self, source, tower):
    assert source == 0

    input_ids = self._input_ids[tower]

    if len(input_ids.shape) > 2:
      raise ValueError(
          "Input to embeddings must have rank <= 2. Found rank %d." % len(
              input_ids.shape))

    batch_size = array_ops.shape(input_ids)[0]

    # Transform indices into one-hot vectors.
    #
    # TODO(b/72714822): There must be a faster way to construct the diagonal
    # covariance matrix! This operation is O(batch_size * vocab_size), where
    # it should be O(batch_size * input_size).
    flat_input_ids = array_ops.reshape(input_ids, [-1])
    one_hots = array_ops.one_hot(flat_input_ids,
                                 self._vocab_size)  # [?, vocab_size]

    # Take average across examples. Note that, because all entries have
    # magnitude zero or one, there's no need to square the entries.
    #
    # TODO(b/72714822): Support for SparseTensor, other kinds of aggregation
    # within an example such as average.
    #
    # TODO(b/72714822): Support for partitioned embeddings.
    new_cov = math_ops.reduce_sum(one_hots, axis=0)  # [vocab_size]
    new_cov /= math_ops.cast(batch_size, new_cov.dtype)

    return new_cov

  def _get_data_device(self, tower):
    return self._input_ids[tower].device


class FullyConnectedDiagonalFactor(DiagonalFactor):
  r"""FisherFactor for a diagonal approx of a fully-connected layer's Fisher.

  Given in = [batch_size, input_size] and out_grad = [batch_size, output_size],
  approximates the covariance as,

    Cov(in, out) = (1/batch_size) sum_{i} outer(in[i], out_grad[i]) ** 2.0

  where the square is taken element-wise.
  """

  def __init__(self,
               inputs,
               outputs_grads,
               has_bias=False):
    """Instantiate FullyConnectedDiagonalFactor.

    Args:
      inputs: List of Tensors of shape [batch_size, input_size]. Inputs to this
        layer.  List index is towers.
      outputs_grads: List of Tensors, each of shape [batch_size, output_size],
        which are the gradients of the loss with respect to the layer's
        outputs. First index is source, second is tower.

      has_bias: bool. If True, append '1' to each input.
    """
    self._inputs = inputs
    self._has_bias = has_bias
    self._outputs_grads = outputs_grads
    self._squared_inputs = None

    super(FullyConnectedDiagonalFactor, self).__init__()

  @property
  def _var_scope(self):
    return "ff_diagfc_" + scope_string_from_params(
        tuple(self._inputs) + tuple(nest.flatten(self._outputs_grads)))

  @property
  def _cov_shape(self):
    input_size = self._inputs[0].shape[1] + self._has_bias
    output_size = self._outputs_grads[0][0].shape[1]
    return [input_size, output_size]

  @property
  def _num_sources(self):
    return len(self._outputs_grads)

  @property
  def _num_towers(self):
    return len(self._inputs)

  @property
  def _dtype(self):
    return self._outputs_grads[0][0].dtype

  def make_covariance_update_op(self, ema_decay):

    self._squared_inputs = []
    for tower in range(self._num_towers):
      inputs = self._inputs[tower]

      with place_on_device(self._get_data_device(tower)):
        if self._has_bias:
          inputs = append_homog(inputs)
        self._squared_inputs.append(math_ops.square(inputs))

    return super(FullyConnectedDiagonalFactor, self).make_covariance_update_op(
        ema_decay)

  def _compute_new_cov(self, source, tower):
    batch_size = array_ops.shape(self._squared_inputs[tower])[0]
    outputs_grad = self._outputs_grads[source][tower]

    # The well-known special formula that uses the fact that the entry-wise
    # square of an outer product is the outer-product of the entry-wise squares.
    # The gradient is the outer product of the input and the output gradients,
    # so we just square both and then take their outer-product.
    new_cov = math_ops.matmul(
        self._squared_inputs[tower],
        math_ops.square(outputs_grad),
        transpose_a=True)
    new_cov /= math_ops.cast(batch_size, new_cov.dtype)
    return new_cov

  def _get_data_device(self, tower):
    return self._inputs[tower].device


class ConvDiagonalFactor(DiagonalFactor):
  """FisherFactor for a diagonal approx of a convolutional layer's Fisher."""

  def __init__(self,
               inputs,
               outputs_grads,
               filter_shape,
               strides,
               padding,
               data_format=None,
               dilations=None,
               has_bias=False):
    """Creates a ConvDiagonalFactor object.

    Args:
      inputs: List of Tensors of shape [batch_size, height, width, in_channels].
        Input activations to this layer.  List index is towers.
      outputs_grads: List of Tensors, each of shape [batch_size,
        height, width, out_channels], which are the gradients of the loss
        with respect to the layer's outputs.  First index is source, second
        index is tower.
      filter_shape: Tuple of 4 ints: (kernel_height, kernel_width, in_channels,
        out_channels). Represents shape of kernel used in this layer.
      strides: The stride size in this layer (1-D Tensor of length 4).
      padding: The padding in this layer (1-D of Tensor length 4).
      data_format: None or str. Format of conv2d inputs.
      dilations: None or tuple of 4 ints.
      has_bias: Python bool. If True, the layer is assumed to have a bias
        parameter in addition to its filter parameter.

    Raises:
      ValueError: If inputs, output_grads, and filter_shape do not agree on
        in_channels or out_channels.
      ValueError: If strides, dilations are not length-4 lists of ints.
      ValueError: If data_format does not put channel last.
    """
    if not utils.is_data_format_channel_last(data_format):
      raise ValueError("Channel must be last.")
    if any(input_.shape.ndims != 4 for input_ in inputs):
      raise ValueError("inputs must be a list of 4-D Tensors.")
    if any(input_.shape.as_list()[-1] != filter_shape[-2] for input_ in inputs):
      raise ValueError("inputs and filter_shape must agree on in_channels.")
    for i, outputs_grad in enumerate(outputs_grads):
      if any(output_grad.shape.ndims != 4 for output_grad in outputs_grad):
        raise ValueError("outputs[%d] must be 4-D Tensor." % i)
      if any(output_grad.shape.as_list()[-1] != filter_shape[-1]
             for output_grad in outputs_grad):
        raise ValueError(
            "outputs[%d] and filter_shape must agree on out_channels." % i)
    if len(strides) != 4:
      raise ValueError("strides must be length-4 list of ints.")
    if dilations is not None and len(dilations) != 4:
      raise ValueError("dilations must be length-4 list of ints.")

    self._inputs = inputs
    self._outputs_grads = outputs_grads
    self._filter_shape = filter_shape
    self._strides = strides
    self._padding = padding
    self._data_format = data_format
    self._dilations = dilations
    self._has_bias = has_bias
    self._patches = None

    super(ConvDiagonalFactor, self).__init__()

  @property
  def _var_scope(self):
    return "ff_convdiag_" + scope_string_from_params(
        tuple(self._inputs) + tuple(nest.flatten(self._outputs_grads)))

  @property
  def _cov_shape(self):
    filter_height, filter_width, in_channels, out_channels = self._filter_shape
    return [
        filter_height * filter_width * in_channels + self._has_bias,
        out_channels
    ]

  @property
  def _num_sources(self):
    return len(self._outputs_grads)

  @property
  def _num_towers(self):
    return len(self._inputs)

  @property
  def _dtype(self):
    return self._inputs[0].dtype

  def make_covariance_update_op(self, ema_decay):
    filter_height, filter_width, _, _ = self._filter_shape

    # TODO(b/64144716): there is potential here for a big savings in terms
    # of memory use.
    if self._dilations is None:
      rates = (1, 1, 1, 1)
    else:
      rates = tuple(self._dilations)

    self._patches = []
    for tower in range(self._num_towers):
      with place_on_device(self._get_data_device(tower)):
        patches = array_ops.extract_image_patches(
            self._inputs[tower],
            ksizes=[1, filter_height, filter_width, 1],
            strides=self._strides,
            rates=rates,
            padding=self._padding)

        if self._has_bias:
          patches = append_homog(patches)

        self._patches.append(patches)

    return super(ConvDiagonalFactor, self).make_covariance_update_op(ema_decay)

  def _compute_new_cov(self, source, tower):
    patches = self._patches[tower]
    batch_size = array_ops.shape(patches)[0]
    outputs_grad = self._outputs_grads[source][tower]

    new_cov = self._convdiag_sum_of_squares(patches, outputs_grad)
    new_cov /= math_ops.cast(batch_size, new_cov.dtype)

    return new_cov

  def _convdiag_sum_of_squares(self, patches, outputs_grad):
    # This computes the sum of the squares of the per-training-case "gradients".
    # It does this simply by computing a giant tensor containing all of these,
    # doing an entry-wise square, and them summing along the batch dimension.
    case_wise_gradients = special_math_ops.einsum("bijk,bijl->bkl", patches,
                                                  outputs_grad)
    return math_ops.reduce_sum(math_ops.square(case_wise_gradients), axis=0)

  def _get_data_device(self, tower):
    return self._inputs[tower].device


class FullyConnectedKroneckerFactor(InverseProvidingFactor):
  """Kronecker factor for the input or output side of a fully-connected layer.
  """

  def __init__(self,
               tensors,
               has_bias=False):
    """Instantiate FullyConnectedKroneckerFactor.

    Args:
      tensors: List of list of Tensors, each of shape [batch_size, n]. The
        Tensors are typically either a layer's inputs or its output's gradients.
        The first list index is source, the second is tower.
      has_bias: bool. If True, append '1' to each row.
    """
    # The tensor argument is either a tensor of input activations or a tensor of
    # output pre-activation gradients.
    self._has_bias = has_bias
    self._tensors = tensors
    super(FullyConnectedKroneckerFactor, self).__init__()

  @property
  def _var_scope(self):
    return "ff_fckron_" + scope_string_from_params(
        tuple(nest.flatten(self._tensors)) + (self._has_bias,))

  @property
  def _cov_shape(self):
    size = self._tensors[0][0].shape[1] + self._has_bias
    return [size, size]

  @property
  def _num_sources(self):
    return len(self._tensors)

  @property
  def _num_towers(self):
    return len(self._tensors[0])

  @property
  def _dtype(self):
    return self._tensors[0][0].dtype

  def _compute_new_cov(self, source, tower):
    tensor = self._tensors[source][tower]
    if self._has_bias:
      tensor = append_homog(tensor)
    return compute_cov(tensor)

  def _get_data_device(self, tower):
    return self._tensors[0][tower].device


class ConvInputKroneckerFactor(InverseProvidingFactor):
  r"""Kronecker factor for the input side of a convolutional layer.

  Estimates E[ a a^T ] where a is the inputs to a convolutional layer given
  example x. Expectation is taken over all examples and locations.

  Equivalent to Omega in https://arxiv.org/abs/1602.01407 for details. See
  Section 3.1 Estimating the factors.
  """

  def __init__(self,
               inputs,
               filter_shape,
               padding,
               strides=None,
               dilation_rate=None,
               data_format=None,
               extract_patches_fn=None,
               has_bias=False,
               sub_sample_inputs=None,
               sub_sample_patches=None):
    """Initializes ConvInputKroneckerFactor.

    Args:
      inputs: List of Tensors of shape [batch_size, ..spatial_input_size..,
        in_channels]. Inputs to layer. List index is tower.
      filter_shape: List of ints. Contains [..spatial_filter_size..,
        in_channels, out_channels]. Shape of convolution kernel.
      padding: str. Padding method for layer. "SAME" or "VALID".
      strides: List of ints or None. Contains [..spatial_filter_strides..] if
        'extract_patches_fn' is compatible with tf.nn.convolution(), else
        [1, ..spatial_filter_strides, 1].
      dilation_rate: List of ints or None. Rate for dilation along each spatial
        dimension if 'extract_patches_fn' is compatible with
        tf.nn.convolution(), else [1, ..spatial_dilation_rates.., 1].
      data_format: str or None. Format of input data.
      extract_patches_fn: str or None. Name of function that extracts image
        patches. One of "extract_convolution_patches", "extract_image_patches",
        "extract_pointwise_conv2d_patches".
      has_bias: bool. If True, append 1 to in_channel.
      sub_sample_inputs: `bool`. If True, then subsample the inputs from which
        the image patches are extracted. (Default: None)
      sub_sample_patches: `bool`, If `True` then subsample the extracted
        patches.(Default: None)
    """
    self._inputs = inputs
    self._filter_shape = filter_shape
    self._strides = strides
    self._padding = padding
    self._dilation_rate = dilation_rate
    self._data_format = data_format
    self._extract_patches_fn = extract_patches_fn
    self._has_bias = has_bias
    if sub_sample_inputs is None:
      self._sub_sample_inputs = _SUB_SAMPLE_INPUTS
    else:
      self._sub_sample_inputs = sub_sample_inputs

    if sub_sample_patches is None:
      self._sub_sample_patches = _SUB_SAMPLE_OUTER_PRODUCTS
    else:
      self._sub_sample_patches = sub_sample_patches
    super(ConvInputKroneckerFactor, self).__init__()

  @property
  def _var_scope(self):
    return "ff_convinkron_" + scope_string_from_params(
        tuple(self._inputs) +
        tuple((self._filter_shape, self._strides, self._padding,
               self._dilation_rate, self._data_format, self._has_bias)))

  @property
  def _cov_shape(self):
    spatial_filter_shape = self._filter_shape[0:-2]
    in_channels = self._filter_shape[-2]
    size = np.prod(spatial_filter_shape) * in_channels + self._has_bias
    return [size, size]

  @property
  def _num_sources(self):
    return 1

  @property
  def _num_towers(self):
    return len(self._inputs)

  @property
  def _dtype(self):
    return self._inputs[0].dtype

  def _compute_new_cov(self, source, tower):
    assert source == 0

    inputs = self._inputs[tower]
    if self._sub_sample_inputs:
      batch_size = inputs.shape.as_list()[0]
      max_size = int(batch_size * _INPUTS_TO_EXTRACT_PATCHES_FACTOR)
      inputs = _random_tensor_gather(inputs, max_size)

    # TODO(b/64144716): there is potential here for a big savings in terms of
    # memory use.
    if self._extract_patches_fn in [None, "extract_convolution_patches"]:
      patches = utils.extract_convolution_patches(
          inputs,
          self._filter_shape,
          padding=self._padding,
          strides=self._strides,
          dilation_rate=self._dilation_rate,
          data_format=self._data_format)

    elif self._extract_patches_fn == "extract_image_patches":
      assert inputs.shape.ndims == 4
      assert len(self._filter_shape) == 4
      assert len(self._strides) == 4, self._strides
      if self._dilation_rate is None:
        rates = [1, 1, 1, 1]
      else:
        rates = self._dilation_rate
        assert len(rates) == 4
        assert rates[0] == rates[-1] == 1
      patches = array_ops.extract_image_patches(
          inputs,
          ksizes=[1] + list(self._filter_shape[0:-2]) + [1],
          strides=self._strides,
          rates=rates,
          padding=self._padding)

    elif self._extract_patches_fn == "extract_pointwise_conv2d_patches":
      assert self._strides in [None, [1, 1, 1, 1], (1, 1, 1, 1)]
      assert self._filter_shape[0] == self._filter_shape[1] == 1
      patches = utils.extract_pointwise_conv2d_patches(
          inputs, self._filter_shape, data_format=None)

    else:
      raise NotImplementedError(self._extract_patches_fn)

    flatten_size = np.prod(self._filter_shape[0:-1])
    # patches_flat below is the matrix [[A_l]] from the KFC paper (tilde
    # omitted over A for clarity). It has shape M|T| x J|Delta| (eq. 14),
    # where M = minibatch size, |T| = number of spatial locations,
    # |Delta| = number of spatial offsets, and J = number of input maps
    # for convolutional layer l.
    patches_flat = array_ops.reshape(patches, [-1, flatten_size])

    # We append a homogenous coordinate to patches_flat if the layer has
    # bias parameters. This gives us [[A_l]]_H from the paper.
    if self._sub_sample_patches:
      patches_flat = _subsample_for_cov_computation(patches_flat)

    if self._has_bias:
      patches_flat = append_homog(patches_flat)
    # We call compute_cov without passing in a normalizer. compute_cov uses
    # the first dimension of patches_flat i.e. M|T| as the normalizer by
    # default. Hence we end up computing 1/M|T| * [[A_l]]^T [[A_l]], with
    # shape J|Delta| x J|Delta|. This is related to hat{Omega}_l from
    # the paper but has a different scale here for consistency with
    # ConvOutputKroneckerFactor.
    # (Tilde omitted over A for clarity.)
    return compute_cov(patches_flat)

  def _get_data_device(self, tower):
    return self._inputs[tower].device


class ConvOutputKroneckerFactor(InverseProvidingFactor):
  r"""Kronecker factor for the output side of a convolutional layer.

  Estimates E[ ds ds^T ] where s is the preactivations of a convolutional layer
  given example x and ds = (d / d s) log(p(y|x, w)). Expectation is taken over
  all examples and locations.

  Equivalent to Gamma in https://arxiv.org/abs/1602.01407 for details. See
  Section 3.1 Estimating the factors.
  """

  def __init__(self, outputs_grads, data_format=None):
    """Initializes ConvOutputKroneckerFactor.

    Args:
      outputs_grads: List of list of Tensors. Each Tensor is of shape
          [batch_size, ..spatial_input_size.., out_channels].  First list index
          is source, the second is tower.
      data_format: None or str. Format of outputs_grads.

    Raises:
      ValueError: If channels are not final dimension.
    """
    if not utils.is_data_format_channel_last(data_format):
      raise ValueError("Channel must be last.")
    self._out_channels = outputs_grads[0][0].shape.as_list()[-1]
    self._outputs_grads = outputs_grads
    super(ConvOutputKroneckerFactor, self).__init__()

  @property
  def _var_scope(self):
    return "ff_convoutkron_" + scope_string_from_params(
        nest.flatten(self._outputs_grads))

  @property
  def _cov_shape(self):
    size = self._out_channels
    return [size, size]

  @property
  def _num_sources(self):
    return len(self._outputs_grads)

  @property
  def _num_towers(self):
    return len(self._outputs_grads[0])

  @property
  def _dtype(self):
    return self._outputs_grads[0][0].dtype

  def _compute_new_cov(self, source, tower):
    outputs_grad = self._outputs_grads[source][tower]

    # reshaped_tensor below is the matrix DS_l defined in the KFC paper
    # (tilde omitted over S for clarity). It has shape M|T| x I, where
    # M = minibatch size, |T| = number of spatial locations, and
    # I = number of output maps for convolutional layer l.
    reshaped_tensor = array_ops.reshape(outputs_grad, [-1, self._out_channels])
    # Following the reasoning in ConvInputKroneckerFactor._compute_new_cov,
    # compute_cov here returns 1/M|T| * DS_l^T DS_l = hat{Gamma}_l
    # as defined in the paper, with shape I x I.
    # (Tilde omitted over S for clarity.)
    return compute_cov(reshaped_tensor)

  def _get_data_device(self, tower):
    return self._outputs_grads[0][tower].device


class FullyConnectedMultiKF(FullyConnectedKroneckerFactor):
  """Kronecker factor for a fully connected layer used multiple times."""

  def __init__(self,
               tensors,
               num_uses=None,
               has_bias=False):
    """Constructs a new `FullyConnectedMultiKF`.

    Args:
      tensors: List of list of Tensors of shape, each of shape
        [num_uses * batch_size, n], and is a reshape version of a Tensor of
        shape [num_uses, batch_size, n]. Each of these tensors is usually a
        layer's inputs or its output's gradients. The first list index is
        sources, the second is towers.
      num_uses: int. The number of time-steps / uses.
      has_bias: bool. If True, '1' is appended to each row.
    """

    self._num_uses = num_uses

    self._cov_dt1 = None
    self._make_cov_dt1 = False
    self._option1quants_by_damping = {}
    self._option2quants_by_damping = {}
    self._option1quants_registrations = set()
    self._option2quants_registrations = set()

    super(FullyConnectedMultiKF, self).__init__(tensors=tensors,
                                                has_bias=has_bias)

  @property
  def _num_timesteps(self):
    return self._num_uses

  @property
  def _var_scope(self):
    return "ff_fc_multi_" + scope_string_from_params(
        tuple(nest.flatten(self._tensors))
        + (self._num_timesteps, self._has_bias,))

  def make_covariance_update_op(self, ema_decay):

    op = super(FullyConnectedMultiKF, self).make_covariance_update_op(ema_decay)

    if self._cov_dt1 is not None:
      new_cov_dt1_contribs = []
      for source in range(self._num_sources):
        for tower in range(self._num_towers):
          with place_on_device(self._get_data_device(tower)):
            new_cov_dt1_contribs.append(self._compute_new_cov_dt1(source,
                                                                  tower))

      new_cov_dt1 = (math_ops.add_n(new_cov_dt1_contribs)
                     / float(self._num_towers))

      # See comments in FisherFactor.make_covariance_update_op() for details.
      if utils.on_tpu():
        new_cov_dt1 = utils.cross_replica_mean(new_cov_dt1)

      op2 = moving_averages.assign_moving_average(
          self._cov_dt1, new_cov_dt1, ema_decay, zero_debias=ZERO_DEBIAS)

      # TODO(b/69112164):
      # It's important that _cov and _cov_dt1 remain consistent with each
      # other while the inverse ops are happening. How can we ensure this?
      # We will need to add explicit synchronization for this to
      # work with asynchronous training.
      op = control_flow_ops.group(op, op2)

    return op

  def _compute_new_cov_dt1(self, source, tower):  # pylint: disable=missing-docstring
    tensor = self._tensors[source][tower]
    if self._has_bias:
      # This appending is technically done twice (the other time is for
      # _compute_new_cov())
      tensor = append_homog(tensor)

    total_len = array_ops.shape(tensor)[0]
    batch_size = total_len // self._num_timesteps

    tensor_present = tensor[:-batch_size, :]
    tensor_future = tensor[batch_size:, :]

    # We specify a normalizer for this computation to ensure a PSD Fisher
    # block estimate.  This is equivalent to padding with zeros, as was done
    # in Section B.2 of the appendix.
    return compute_cov(
        tensor_future, tensor_right=tensor_present, normalizer=total_len)

  def _get_data_device(self, tower):
    return self._tensors[0][tower].device

  @property
  def _vec_shape(self):
    size = self._tensors[0][0].shape[1] + self._has_bias
    return [size]

  def get_option1quants(self, damping_func):
    damping_id = graph_func_to_id(damping_func)
    return self._option1quants_by_damping[damping_id]

  def get_option2quants(self, damping_func):
    damping_id = graph_func_to_id(damping_func)
    return self._option2quants_by_damping[damping_id]

  def get_cov_dt1(self):
    assert self._cov_dt1 is not None
    return self._cov_dt1

  def register_cov_dt1(self):
    self._make_cov_dt1 = True

  def instantiate_cov_variables(self):
    super(FullyConnectedMultiKF, self).instantiate_cov_variables()
    assert self._cov_dt1 is None
    if self._make_cov_dt1:
      with variable_scope.variable_scope(self._var_scope):
        self._cov_dt1 = variable_scope.get_variable(
            "cov_dt1",
            initializer=init_ops.zeros_initializer,
            shape=self._cov_shape,
            trainable=False,
            dtype=self._dtype)

  def register_option1quants(self, damping_func):
    damping_id = self._register_damping(damping_func)
    if damping_id not in self._option1quants_registrations:
      self._option1quants_registrations.add(damping_id)

  def register_option2quants(self, damping_func):
    damping_id = self._register_damping(damping_func)
    if damping_id not in self._option2quants_registrations:
      self._option2quants_registrations.add(damping_id)

  def instantiate_inv_variables(self):
    super(FullyConnectedMultiKF, self).instantiate_inv_variables()

    for damping_id in self._option1quants_registrations:
      damping_func = self._damping_funcs_by_id[damping_id]
      damping_string = graph_func_to_string(damping_func)
      # It's questionable as to whether we should initialize with stuff like
      # this at all.  Ideally these values should never be used until they are
      # updated at least once.
      with variable_scope.variable_scope(self._var_scope):
        Lmat = variable_scope.get_variable(  # pylint: disable=invalid-name
            "Lmat_damp{}".format(damping_string),
            initializer=inverse_initializer,
            shape=self._cov_shape,
            trainable=False,
            dtype=self._dtype)
        psi = variable_scope.get_variable(
            "psi_damp{}".format(damping_string),
            initializer=init_ops.ones_initializer,
            shape=self._vec_shape,
            trainable=False,
            dtype=self._dtype)

      assert damping_id not in self._option1quants_by_damping
      self._option1quants_by_damping[damping_id] = (Lmat, psi)

    for damping_id in self._option2quants_registrations:
      damping_func = self._damping_funcs_by_id[damping_id]
      damping_string = graph_func_to_string(damping_func)
      # It's questionable as to whether we should initialize with stuff like
      # this at all.  Ideally these values should never be used until they are
      # updated at least once.
      with variable_scope.variable_scope(self._var_scope):
        Pmat = variable_scope.get_variable(  # pylint: disable=invalid-name
            "Lmat_damp{}".format(damping_string),
            initializer=inverse_initializer,
            shape=self._cov_shape,
            trainable=False,
            dtype=self._dtype)
        Kmat = variable_scope.get_variable(  # pylint: disable=invalid-name
            "Kmat_damp{}".format(damping_string),
            initializer=inverse_initializer,
            shape=self._cov_shape,
            trainable=False,
            dtype=self._dtype)
        mu = variable_scope.get_variable(
            "mu_damp{}".format(damping_string),
            initializer=init_ops.ones_initializer,
            shape=self._vec_shape,
            trainable=False,
            dtype=self._dtype)

      assert damping_id not in self._option2quants_by_damping
      self._option2quants_by_damping[damping_id] = (Pmat, Kmat, mu)

  def make_inverse_update_ops(self):
    """Create and return update ops corresponding to registered computations."""
    # TODO(b/69918258): Add correctness tests for this method.
    # pylint: disable=invalid-name

    ops = []

    if (len(self._option1quants_by_damping) +
        len(self._option2quants_by_damping)):

      # Note that C0 and C1 are stand-ins for A0 and A1, or G0 and G1, from
      # the pseudo-code in the original paper.  Because the computations for
      # the A and G case are essentially the same they can both be performed by
      # the same class (this one).

      C1 = self.get_cov_dt1()

      # Get the eigendecomposition of C0  (= self.get_cov())
      eigen_e, eigen_V = self.get_eigendecomp()

      # TODO(b/69678661): Note, there is an implicit assumption here that C1
      # and C0 (as represented here by its eigen-decomp) are consistent.  This
      # could fail to be the case if self._cov and self._cov_dt1 are not updated
      # consistently, or are somehow read between or during the cov updates.
      # Can this possibly happen?  Is there a way to prevent it?

      for damping_id, (Lmat_var,
                       psi_var) in self._option1quants_by_damping.items():

        damping = self._damping_funcs_by_id[damping_id]()

        invsqrtC0 = math_ops.matmul(
            eigen_V * (eigen_e + damping)**(-0.5), eigen_V, transpose_b=True)

        # Might need to enforce symmetry lost due to numerical issues.
        invsqrtC0 = (invsqrtC0 + array_ops.transpose(invsqrtC0)) / 2.0

        # The following line imposses the symmetry assumed by "Option 1" on C1.
        # Stangely the code can work okay with this line commented out,
        # depending on how psd_eig is defined.  I'm not sure why.
        C1 = (C1 + array_ops.transpose(C1)) / 2.0

        # hPsi = C0^(-1/2) * C1 * C0^(-1/2)  (hPsi means hat{Psi})
        hPsi = math_ops.matmul(math_ops.matmul(invsqrtC0, C1), invsqrtC0)

        # Compute the decomposition U*diag(psi)*U^T = hPsi
        psi, U = utils.posdef_eig(hPsi)

        # L = C0^(-1/2) * U
        Lmat = math_ops.matmul(invsqrtC0, U)

        ops.append(Lmat_var.assign(Lmat))
        ops.append(psi_var.assign(psi))

      for damping_id, (Pmat_var, Kmat_var,
                       mu_var) in self._option2quants_by_damping.items():

        damping = self._damping_funcs_by_id[damping_id]()

        # compute C0^(-1/2)
        invsqrtC0 = math_ops.matmul(
            eigen_V * (eigen_e + damping)**(-0.5), eigen_V, transpose_b=True)

        # Might need to enforce symmetry lost due to numerical issues.
        invsqrtC0 = (invsqrtC0 + array_ops.transpose(invsqrtC0)) / 2.0

        # Compute the product C0^(-1/2) * C1
        invsqrtC0C1 = math_ops.matmul(invsqrtC0, C1)

        # hPsi = C0^(-1/2) * C1 * C0^(-1/2)  (hPsi means hat{Psi})
        hPsi = math_ops.matmul(invsqrtC0C1, invsqrtC0)

        # Compute the decomposition E*diag(mu)*E^T = hPsi^T * hPsi
        # Note that we using the notation mu instead of "m" for the eigenvalues.
        # Instead of computing the product hPsi^T * hPsi and then doing an
        # eigen-decomposition of this we just compute the SVD of hPsi and then
        # square the singular values to get the eigenvalues. For a justification
        # of this approach, see:
        # https://en.wikipedia.org/wiki/Singular-value_decomposition#Relation_to_eigenvalue_decomposition
        sqrtmu, _, E = linalg_ops.svd(hPsi)
        mu = math_ops.square(sqrtmu)

        # Mathematically, the eigenvalues should not should not exceed 1.0, but
        # due to numerical issues, or possible issues with inconsistent
        # values of C1 and (the eigen-decomposition of) C0 they might. So
        # we enforce this condition.
        mu = math_ops.minimum(mu, 1.0)

        # P = (C0^(-1/2) * C1)^T * C0^(-1/2) = C_1^T * C_0^(-1)
        Pmat = math_ops.matmul(invsqrtC0C1, invsqrtC0, transpose_a=True)

        # K = C_0^(-1/2) * E
        Kmat = math_ops.matmul(invsqrtC0, E)

        ops.append(Pmat_var.assign(Pmat))
        ops.append(Kmat_var.assign(Kmat))
        ops.append(mu_var.assign(mu))

    ops += super(FullyConnectedMultiKF, self).make_inverse_update_ops()
    return [control_flow_ops.group(*ops)]

    # pylint: enable=invalid-name
