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

import numpy as np
import six

from tensorflow.contrib.kfac.python.ops import utils
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import moving_averages

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


def set_global_constants(init_covariances_at_zero=None, zero_debias=None,
                         eigenvalue_decomposition_threshold=None,
                         eigenvalue_clipping_threshold=None):
  """Sets various global constants used by the classes in this module."""
  global INIT_COVARIANCES_AT_ZERO
  global ZERO_DEBIAS
  global EIGENVALUE_DECOMPOSITION_THRESHOLD
  global EIGENVALUE_CLIPPING_THRESHOLD

  if init_covariances_at_zero is not None:
    INIT_COVARIANCES_AT_ZERO = init_covariances_at_zero
  if zero_debias is not None:
    ZERO_DEBIAS = zero_debias
  if eigenvalue_decomposition_threshold is not None:
    EIGENVALUE_DECOMPOSITION_THRESHOLD = eigenvalue_decomposition_threshold
  if eigenvalue_clipping_threshold is not None:
    EIGENVALUE_CLIPPING_THRESHOLD = eigenvalue_clipping_threshold


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


def _compute_cov(tensor, normalizer=None):
  """Compute the empirical second moment of the rows of a 2D Tensor.

  This function is meant to be applied to random matrices for which the true row
  mean is zero, so that the true second moment equals the true covariance.

  Args:
    tensor: A 2D Tensor.
    normalizer: optional scalar for the estimator (by default, the normalizer is
        the number of rows of tensor).

  Returns:
    A square 2D Tensor with as many rows/cols as the number of input columns.
  """
  if normalizer is None:
    normalizer = array_ops.shape(tensor)[0]
  cov = (math_ops.matmul(tensor, tensor, transpose_a=True) / math_ops.cast(
      normalizer, tensor.dtype))
  return (cov + array_ops.transpose(cov)) / math_ops.cast(2, cov.dtype)


def _append_homog(tensor):
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
  return array_ops.concat([tensor, ones], axis=rank-1)


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
    if isinstance(param, (tuple, list)):
      if all([isinstance(p, int) for p in param]):
        name_parts.append("-".join([str(p) for p in param]))
      else:
        name_parts.append(scope_string_from_name(param))
    elif isinstance(param, (str, int, bool)):
      name_parts.append(str(param))
    elif isinstance(param, (tf_ops.Tensor, variables.Variable)):
      name_parts.append(scope_string_from_name(param))
    else:
      raise ValueError(
          "Encountered an unsupported param type {}".format(type(param)))
  return "_".join(name_parts)


def scope_string_from_name(tensor):
  if isinstance(tensor, (tuple, list)):
    return "__".join([scope_string_from_name(t) for t in tensor])
  # "gradients/add_4_grad/Reshape:0" -> "gradients_add_4_grad_Reshape"
  return tensor.name.split(":")[0].replace("/", "_")


def scalar_or_tensor_to_string(val):
  return repr(val) if np.isscalar(val) else scope_string_from_name(val)


@six.add_metaclass(abc.ABCMeta)
class FisherFactor(object):
  """Base class for objects modeling factors of approximate Fisher blocks.

     Note that for blocks that aren't based on approximations, a 'factor' can
     be the entire block itself, as is the case for the diagonal and full
     representations.

     Subclasses must implement the _compute_new_cov method, and the _var_scope
     and _cov_shape properties.
  """

  def __init__(self):
    self.instantiate_covariance()

  @abc.abstractproperty
  def _var_scope(self):
    pass

  @abc.abstractproperty
  def _cov_shape(self):
    """The shape of the cov matrix."""
    pass

  @abc.abstractproperty
  def _num_sources(self):
    """The number of things to sum over when computing cov.

    The default make_covariance_update_op function will call _compute_new_cov
    with indices ranging from 0 to _num_sources-1. The typical situation is
    where the factor wants to sum the statistics it computes over multiple
    backpropped "gradients" (typically passed in via "tensors" or
    "outputs_grads" arguments).
    """
    pass

  @property
  def _cov_initializer(self):
    return covariance_initializer

  def instantiate_covariance(self):
    """Instantiates the covariance Variable as the instance member _cov."""
    with variable_scope.variable_scope(self._var_scope):
      self._cov = variable_scope.get_variable(
          "cov",
          initializer=self._cov_initializer,
          shape=self._cov_shape,
          trainable=False)

  @abc.abstractmethod
  def _compute_new_cov(self, idx=0):
    pass

  def make_covariance_update_op(self, ema_decay):
    """Constructs and returns the covariance update Op.

    Args:
      ema_decay: The exponential moving average decay (float or Tensor).
    Returns:
      An Op for updating the covariance Variable referenced by _cov.
    """
    new_cov = math_ops.add_n(
        tuple(self._compute_new_cov(idx) for idx in range(self._num_sources)))

    return moving_averages.assign_moving_average(
        self._cov, new_cov, ema_decay, zero_debias=ZERO_DEBIAS)

  def make_inverse_update_ops(self):
    """Create and return update ops corresponding to registered computations."""
    return []

  def get_cov(self):
    return self._cov


class InverseProvidingFactor(FisherFactor):
  """Base class for FisherFactors that maintain inverses, powers, etc of _cov.

  Assumes that the _cov property is a square PSD matrix.

  Subclasses must implement the _compute_new_cov method, and the _var_scope and
  _cov_shape properties.
  """

  def __init__(self):
    self._inverses_by_damping = {}
    self._matpower_by_exp_and_damping = {}
    self._eigendecomp = None

    super(InverseProvidingFactor, self).__init__()

  def register_damped_inverse(self, damping):
    """Registers a damped inverse needed by a FisherBlock.

    Args:
      damping: The damping value (float or Tensor) for this factor.
    """
    if damping not in self._inverses_by_damping:
      damping_string = scalar_or_tensor_to_string(damping)
      with variable_scope.variable_scope(self._var_scope):
        inv = variable_scope.get_variable(
            "inv_damp{}".format(damping_string),
            initializer=inverse_initializer,
            shape=self._cov_shape,
            trainable=False)
      self._inverses_by_damping[damping] = inv

  def register_matpower(self, exp, damping):
    """Registers a matrix power needed by a FisherBlock.

    Args:
      exp: The exponent (float or Tensor) to raise the matrix to.
      damping: The damping value (float or Tensor).
    """
    if (exp, damping) not in self._matpower_by_exp_and_damping:
      exp_string = scalar_or_tensor_to_string(exp)
      damping_string = scalar_or_tensor_to_string(damping)
      with variable_scope.variable_scope(self._var_scope):
        matpower = variable_scope.get_variable(
            "matpower_exp{}_damp{}".format(exp_string, damping_string),
            initializer=inverse_initializer,
            shape=self._cov_shape,
            trainable=False)
      self._matpower_by_exp_and_damping[(exp, damping)] = matpower

  def register_eigendecomp(self):
    """Registers that an eigendecomposition is needed by a FisherBlock."""
    if not self._eigendecomp:
      self._eigendecomp = linalg_ops.self_adjoint_eig(self._cov)

  def make_inverse_update_ops(self):
    """Create and return update ops corresponding to registered computations."""
    ops = super(InverseProvidingFactor, self).make_inverse_update_ops()

    num_inverses = len(self._inverses_by_damping)
    matrix_power_registered = bool(self._matpower_by_exp_and_damping)
    use_eig = (self._eigendecomp or matrix_power_registered or
               num_inverses >= EIGENVALUE_DECOMPOSITION_THRESHOLD)

    if use_eig:
      self.register_eigendecomp()  # ensures self._eigendecomp is set
      eigenvalues, eigenvectors = self._eigendecomp  # pylint: disable=unpacking-non-sequence

      # The matrix self._cov is positive semidefinite by construction, but the
      # numerical eigenvalues could be negative due to numerical errors, so here
      # we clip them to be at least EIGENVALUE_CLIPPING_THRESHOLD.
      clipped_eigenvalues = math_ops.maximum(eigenvalues,
                                             EIGENVALUE_CLIPPING_THRESHOLD)

      for damping, inv in self._inverses_by_damping.items():
        ops.append(
            inv.assign(
                math_ops.matmul(eigenvectors / (clipped_eigenvalues + damping),
                                array_ops.transpose(eigenvectors))))

      for (exp, damping), matpower in self._matpower_by_exp_and_damping.items():
        ops.append(
            matpower.assign(
                math_ops.matmul(eigenvectors * (clipped_eigenvalues + damping)**
                                exp, array_ops.transpose(eigenvectors))))
    else:
      for damping, inv in self._inverses_by_damping.items():
        ops.append(inv.assign(utils.posdef_inv(self._cov, damping)))

    return ops

  def get_inverse(self, damping):
    return self._inverses_by_damping[damping]

  def get_matpower(self, exp, damping):
    return self._matpower_by_exp_and_damping[(exp, damping)]

  def get_eigendecomp(self):
    return self._eigendecomp


class FullFactor(InverseProvidingFactor):
  """FisherFactor for a full matrix representation of the Fisher of a parameter.

  Note that this uses the naive "square the sum estimator", and so is applicable
  to any type of parameter in principle, but has very high variance.
  """

  def __init__(self, params_grads, batch_size):
    self._batch_size = batch_size
    self._orig_params_grads_name = scope_string_from_params(
        [params_grads, self._batch_size])
    self._params_grads_flat = tuple(
        utils.tensors_to_column(params_grad) for params_grad in params_grads)
    super(FullFactor, self).__init__()

  @property
  def _var_scope(self):
    return "ff_full/" + self._orig_params_grads_name

  @property
  def _cov_shape(self):
    size = self._params_grads_flat[0].shape[0]
    return [size, size]

  @property
  def _num_sources(self):
    return len(self._params_grads_flat)

  def _compute_new_cov(self, idx=0):
    # This will be a very basic rank 1 estimate
    return ((self._params_grads_flat[idx] * array_ops.transpose(
        self._params_grads_flat[idx])) / math_ops.cast(
            self._batch_size, self._params_grads_flat[idx].dtype))


class DiagonalFactor(FisherFactor):
  """A base class for FisherFactors that use diagonal approximations."""

  def __init__(self):
    super(DiagonalFactor, self).__init__()

  @property
  def _cov_initializer(self):
    return diagonal_covariance_initializer


class NaiveDiagonalFactor(DiagonalFactor):
  """FisherFactor for a diagonal approximation of any type of param's Fisher.

  Note that this uses the naive "square the sum estimator", and so is applicable
  to any type of parameter in principle, but has very high variance.
  """

  def __init__(self, params_grads, batch_size):
    self._batch_size = batch_size
    self._params_grads = tuple(
        utils.tensors_to_column(params_grad) for params_grad in params_grads)
    self._orig_params_grads_name = scope_string_from_params(
        [self._params_grads, self._batch_size])
    super(NaiveDiagonalFactor, self).__init__()

  @property
  def _var_scope(self):
    return "ff_naivediag/" + self._orig_params_grads_name

  @property
  def _cov_shape(self):
    return self._params_grads[0].shape

  @property
  def _num_sources(self):
    return len(self._params_grads)

  def _compute_new_cov(self, idx=0):
    return (math_ops.square(self._params_grads[idx]) / math_ops.cast(
        self._batch_size, self._params_grads[idx].dtype))


class FullyConnectedDiagonalFactor(DiagonalFactor):
  r"""FisherFactor for a diagonal approx of a fully-connected layer's Fisher.

  Given in = [batch_size, input_size] and out_grad = [batch_size, output_size],
  approximates the covariance as,

    Cov(in, out) = (1/batch_size) \sum_{i} outer(in[i], out_grad[i]) ** 2.0

  where the square is taken element-wise.
  """

  # TODO(jamesmartens): add units tests for this class

  def __init__(self, inputs, outputs_grads, has_bias=False):
    """Instantiate FullyConnectedDiagonalFactor.

    Args:
      inputs: Tensor of shape [batch_size, input_size]. Inputs to fully
        connected layer.
      outputs_grads: List of Tensors of shape [batch_size, output_size].
        Gradient of loss with respect to layer's preactivations.
      has_bias: bool. If True, append '1' to each input.
    """
    self._outputs_grads = outputs_grads
    self._batch_size = array_ops.shape(inputs)[0]
    self._orig_tensors_name = scope_string_from_params((inputs,) +
                                                       tuple(outputs_grads))

    # Note that we precompute the required operations on the inputs since the
    # inputs don't change with the 'idx' argument to _compute_new_cov.  (Only
    # the target entry of _outputs_grads changes with idx.)
    if has_bias:
      inputs = _append_homog(inputs)
    self._squared_inputs = math_ops.square(inputs)

    super(FullyConnectedDiagonalFactor, self).__init__()

  @property
  def _var_scope(self):
    return "ff_diagfc/" + self._orig_tensors_name

  @property
  def _cov_shape(self):
    return [self._squared_inputs.shape[1], self._outputs_grads[0].shape[1]]

  @property
  def _num_sources(self):
    return len(self._outputs_grads)

  def _compute_new_cov(self, idx=0):
    # The well-known special formula that uses the fact that the entry-wise
    # square of an outer product is the outer-product of the entry-wise squares.
    # The gradient is the outer product of the input and the output gradients,
    # so we just square both and then take their outer-product.
    new_cov = math_ops.matmul(
        self._squared_inputs,
        math_ops.square(self._outputs_grads[idx]),
        transpose_a=True)
    new_cov /= math_ops.cast(self._batch_size, new_cov.dtype)
    return new_cov


class ConvDiagonalFactor(DiagonalFactor):
  """FisherFactor for a diagonal approx of a convolutional layer's Fisher."""

  # TODO(jamesmartens): add units tests for this class

  def __init__(self, inputs, outputs_grads, filter_shape, strides, padding,
               has_bias=False):
    """Creates a ConvDiagonalFactor object.

    Args:
      inputs: Tensor of shape [batch_size, height, width, in_channels].
        Input activations to this layer.
      outputs_grads: Tensor of shape [batch_size, height, width, out_channels].
        Per-example gradients to the loss with respect to the layer's output
        preactivations.
      filter_shape: Tuple of 4 ints: (kernel_height, kernel_width, in_channels,
        out_channels). Represents shape of kernel used in this layer.
      strides: The stride size in this layer (1-D Tensor of length 4).
      padding: The padding in this layer (1-D of Tensor length 4).
      has_bias: Python bool. If True, the layer is assumed to have a bias
        parameter in addition to its filter parameter.
    """
    self._filter_shape = filter_shape
    self._has_bias = has_bias
    self._outputs_grads = outputs_grads

    self._orig_tensors_name = scope_string_from_name((inputs,)
                                                     + tuple(outputs_grads))

    # Note that we precompute the required operations on the inputs since the
    # inputs don't change with the 'idx' argument to _compute_new_cov.  (Only
    # the target entry of _outputs_grads changes with idx.)
    filter_height, filter_width, _, _ = self._filter_shape
    patches = array_ops.extract_image_patches(
        inputs,
        ksizes=[1, filter_height, filter_width, 1],
        strides=strides,
        rates=[1, 1, 1, 1],
        padding=padding)

    if has_bias:
      patches = _append_homog(patches)

    self._patches = patches

    super(ConvDiagonalFactor, self).__init__()

  @property
  def _var_scope(self):
    return "ff_convdiag/" + self._orig_tensors_name

  @property
  def _cov_shape(self):
    filter_height, filter_width, in_channels, out_channels = self._filter_shape
    return [filter_height * filter_width * in_channels + self._has_bias,
            out_channels]

  @property
  def _num_sources(self):
    return len(self._outputs_grads)

  def _compute_new_cov(self, idx=0):
    outputs_grad = self._outputs_grads[idx]
    batch_size = array_ops.shape(self._patches)[0]

    new_cov = self._convdiag_sum_of_squares(self._patches, outputs_grad)
    new_cov /= math_ops.cast(batch_size, new_cov.dtype)

    return new_cov

  def _convdiag_sum_of_squares(self, patches, outputs_grad):
    # This computes the sum of the squares of the per-training-case "gradients".
    # It does this simply by computing a giant tensor containing all of these,
    # doing an entry-wise square, and them summing along the batch dimension.
    case_wise_gradients = special_math_ops.einsum("bijk,bijl->bkl", patches,
                                                  outputs_grad)
    return math_ops.reduce_sum(math_ops.square(case_wise_gradients), axis=0)


class FullyConnectedKroneckerFactor(InverseProvidingFactor):
  """Kronecker factor for the input or output side of a fully-connected layer.
  """

  def __init__(self, tensors, has_bias=False):
    """Instantiate FullyConnectedKroneckerFactor.

    Args:
      tensors: List of Tensors of shape [batch_size, n]. Represents either a
        layer's inputs or its output's gradients.
      has_bias: bool. If True, assume this factor is for the layer's inputs and
        append '1' to each row.
    """
    # The tensor argument is either a tensor of input activations or a tensor of
    # output pre-activation gradients.
    self._has_bias = has_bias
    self._tensors = tensors
    super(FullyConnectedKroneckerFactor, self).__init__()

  @property
  def _var_scope(self):
    return "ff_fckron/" + scope_string_from_params(
        [self._tensors, self._has_bias])

  @property
  def _cov_shape(self):
    size = self._tensors[0].shape[1] + self._has_bias
    return [size, size]

  @property
  def _num_sources(self):
    return len(self._tensors)

  def _compute_new_cov(self, idx=0):
    tensor = self._tensors[idx]
    if self._has_bias:
      tensor = _append_homog(tensor)
    return _compute_cov(tensor)


class ConvInputKroneckerFactor(InverseProvidingFactor):
  """Kronecker factor for the input side of a convolutional layer."""

  def __init__(self, inputs, filter_shape, strides, padding, has_bias=False):
    self._filter_shape = filter_shape
    self._strides = strides
    self._padding = padding
    self._has_bias = has_bias
    self._inputs = inputs
    super(ConvInputKroneckerFactor, self).__init__()

  @property
  def _var_scope(self):
    return "ff_convinkron/" + scope_string_from_params([
        self._inputs, self._filter_shape, self._strides, self._padding,
        self._has_bias
    ])

  @property
  def _cov_shape(self):
    filter_height, filter_width, in_channels, _ = self._filter_shape
    size = filter_height * filter_width * in_channels + self._has_bias
    return [size, size]

  @property
  def _num_sources(self):
    return 1

  def _compute_new_cov(self, idx=0):
    if idx != 0:
      raise ValueError("ConvInputKroneckerFactor only supports idx = 0")

    # TODO(jamesmartens): factor this patches stuff out into a utility function
    filter_height, filter_width, in_channels, _ = self._filter_shape
    patches = array_ops.extract_image_patches(
        self._inputs,
        ksizes=[1, filter_height, filter_width, 1],
        strides=self._strides,
        rates=[1, 1, 1, 1],
        padding=self._padding)

    flatten_size = (filter_height * filter_width * in_channels)
    patches_flat = array_ops.reshape(patches, [-1, flatten_size])

    if self._has_bias:
      patches_flat = _append_homog(patches_flat)

    return _compute_cov(patches_flat)


class ConvOutputKroneckerFactor(InverseProvidingFactor):
  """Kronecker factor for the output side of a convolutional layer."""

  def __init__(self, outputs_grads):
    self._out_channels = outputs_grads[0].shape.as_list()[3]
    self._outputs_grads = outputs_grads
    super(ConvOutputKroneckerFactor, self).__init__()

  @property
  def _var_scope(self):
    return "ff_convoutkron/" + scope_string_from_params(self._outputs_grads)

  @property
  def _cov_shape(self):
    size = self._out_channels
    return [size, size]

  @property
  def _num_sources(self):
    return len(self._outputs_grads)

  def _compute_new_cov(self, idx=0):
    reshaped_tensor = array_ops.reshape(self._outputs_grads[idx],
                                        [-1, self._out_channels])
    return _compute_cov(reshaped_tensor)
