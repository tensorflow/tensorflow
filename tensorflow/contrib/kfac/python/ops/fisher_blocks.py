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
"""FisherBlock definitions.

This library contains classes for estimating blocks in a model's Fisher
Information matrix. Suppose one has a model that parameterizes a posterior
distribution over 'y' given 'x' with parameters 'params', p(y | x, params). Its
Fisher Information matrix is given by,

  F(params) = E[ v(x, y, params) v(x, y, params)^T ]

where,

  v(x, y, params) = (d / d params) log p(y | x, params)

and the expectation is taken with respect to the data's distribution for 'x' and
the model's posterior distribution for 'y',

  x ~ p(x)
  y ~ p(y | x, params)

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import enum  # pylint: disable=g-bad-import-order

import six

from tensorflow.contrib.kfac.python.ops import fisher_factors
from tensorflow.contrib.kfac.python.ops import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

# For blocks corresponding to convolutional layers, or any type of block where
# the parameters can be thought of as being replicated in time or space,
# we want to adjust the scale of the damping by
#   damping /= num_replications ** NORMALIZE_DAMPING_POWER
NORMALIZE_DAMPING_POWER = 1.0

# Methods for adjusting damping for FisherBlocks. See
# _compute_pi_adjusted_damping() for details.
PI_OFF_NAME = "off"
PI_TRACENORM_NAME = "tracenorm"
PI_TYPE = PI_TRACENORM_NAME


def set_global_constants(normalize_damping_power=None, pi_type=None):
  """Sets various global constants used by the classes in this module."""
  global NORMALIZE_DAMPING_POWER
  global PI_TYPE

  if normalize_damping_power is not None:
    NORMALIZE_DAMPING_POWER = normalize_damping_power

  if pi_type is not None:
    PI_TYPE = pi_type


def _compute_pi_tracenorm(left_cov, right_cov):
  """Computes the scalar constant pi for Tikhonov regularization/damping.

  pi = sqrt( (trace(A) / dim(A)) / (trace(B) / dim(B)) )
  See section 6.3 of https://arxiv.org/pdf/1503.05671.pdf for details.

  Args:
    left_cov: The left Kronecker factor "covariance".
    right_cov: The right Kronecker factor "covariance".

  Returns:
    The computed scalar constant pi for these Kronecker Factors (as a Tensor).
  """
  # Instead of dividing by the dim of the norm, we multiply by the dim of the
  # other norm. This works out the same in the ratio.
  left_norm = math_ops.trace(left_cov) * right_cov.shape.as_list()[0]
  right_norm = math_ops.trace(right_cov) * left_cov.shape.as_list()[0]
  return math_ops.sqrt(left_norm / right_norm)


def _compute_pi_adjusted_damping(left_cov, right_cov, damping):

  if PI_TYPE == PI_TRACENORM_NAME:
    pi = _compute_pi_tracenorm(left_cov, right_cov)
    return (damping * pi, damping / pi)

  elif PI_TYPE == PI_OFF_NAME:
    return (damping, damping)


@six.add_metaclass(abc.ABCMeta)
class FisherBlock(object):
  """Abstract base class for objects modeling approximate Fisher matrix blocks.

  Subclasses must implement multiply_inverse(), instantiate_factors(), and
  tensors_to_compute_grads() methods.
  """

  def __init__(self, layer_collection):
    self._layer_collection = layer_collection

  @abc.abstractmethod
  def instantiate_factors(self, grads_list, damping):
    """Creates and registers the component factors of this Fisher block.

    Args:
      grads_list: A list gradients (each a Tensor or tuple of Tensors) with
          respect to the tensors returned by tensors_to_compute_grads() that
          are to be used to estimate the block.
      damping: The damping factor (float or Tensor).
    """
    pass

  @abc.abstractmethod
  def multiply_inverse(self, vector):
    """Multiplies the vector by the (damped) inverse of the block.

    Args:
      vector: The vector (a Tensor or tuple of Tensors) to be multiplied.

    Returns:
      The vector left-multiplied by the (damped) inverse of the block.
    """
    pass

  @abc.abstractmethod
  def multiply(self, vector):
    """Multiplies the vector by the (damped) block.

    Args:
      vector: The vector (a Tensor or tuple of Tensors) to be multiplied.

    Returns:
      The vector left-multiplied by the (damped) block.
    """
    pass

  @abc.abstractmethod
  def tensors_to_compute_grads(self):
    """Returns the Tensor(s) with respect to which this FisherBlock needs grads.
    """
    pass

  @abc.abstractproperty
  def num_registered_minibatches(self):
    """Number of minibatches registered for this FisherBlock.

    Typically equal to the number of towers in a multi-tower setup.
    """
    pass


class FullFB(FisherBlock):
  """FisherBlock using a full matrix estimate (no approximations).

  FullFB uses a full matrix estimate (no approximations), and should only ever
  be used for very low dimensional parameters.

  Note that this uses the naive "square the sum estimator", and so is applicable
  to any type of parameter in principle, but has very high variance.
  """

  def __init__(self, layer_collection, params):
    """Creates a FullFB block.

    Args:
      layer_collection: The collection of all layers in the K-FAC approximate
          Fisher information matrix to which this FisherBlock belongs.
      params: The parameters of this layer (Tensor or tuple of Tensors).
    """
    self._batch_sizes = []
    self._params = params

    super(FullFB, self).__init__(layer_collection)

  def instantiate_factors(self, grads_list, damping):
    self._damping = damping
    self._factor = self._layer_collection.make_or_get_factor(
        fisher_factors.FullFactor, (grads_list, self._batch_size))
    self._factor.register_damped_inverse(damping)

  def multiply_inverse(self, vector):
    inverse = self._factor.get_damped_inverse(self._damping)
    out_flat = math_ops.matmul(inverse, utils.tensors_to_column(vector))
    return utils.column_to_tensors(vector, out_flat)

  def multiply(self, vector):
    vector_flat = utils.tensors_to_column(vector)
    out_flat = (
        math_ops.matmul(self._factor.get_cov(), vector_flat) +
        self._damping * vector_flat)
    return utils.column_to_tensors(vector, out_flat)

  def full_fisher_block(self):
    """Explicitly constructs the full Fisher block."""
    return self._factor.get_cov()

  def tensors_to_compute_grads(self):
    return self._params

  def register_additional_minibatch(self, batch_size):
    """Register an additional minibatch.

    Args:
      batch_size: The batch size, used in the covariance estimator.
    """
    self._batch_sizes.append(batch_size)

  @property
  def num_registered_minibatches(self):
    return len(self._batch_sizes)

  @property
  def _batch_size(self):
    return math_ops.reduce_sum(self._batch_sizes)


class NaiveDiagonalFB(FisherBlock):
  """FisherBlock using a diagonal matrix approximation.

  This type of approximation is generically applicable but quite primitive.

  Note that this uses the naive "square the sum estimator", and so is applicable
  to any type of parameter in principle, but has very high variance.
  """

  def __init__(self, layer_collection, params):
    """Creates a NaiveDiagonalFB block.

    Args:
      layer_collection: The collection of all layers in the K-FAC approximate
          Fisher information matrix to which this FisherBlock belongs.
      params: The parameters of this layer (Tensor or tuple of Tensors).
    """
    self._params = params
    self._batch_sizes = []

    super(NaiveDiagonalFB, self).__init__(layer_collection)

  def instantiate_factors(self, grads_list, damping):
    self._damping = damping
    self._factor = self._layer_collection.make_or_get_factor(
        fisher_factors.NaiveDiagonalFactor, (grads_list, self._batch_size))

  def multiply_inverse(self, vector):
    vector_flat = utils.tensors_to_column(vector)
    out_flat = vector_flat / (self._factor.get_cov() + self._damping)
    return utils.column_to_tensors(vector, out_flat)

  def multiply(self, vector):
    vector_flat = utils.tensors_to_column(vector)
    out_flat = vector_flat * (self._factor.get_cov() + self._damping)
    return utils.column_to_tensors(vector, out_flat)

  def full_fisher_block(self):
    return array_ops.diag(array_ops.reshape(self._factor.get_cov(), (-1,)))

  def tensors_to_compute_grads(self):
    return self._params

  def register_additional_minibatch(self, batch_size):
    """Register an additional minibatch.

    Args:
      batch_size: The batch size, used in the covariance estimator.
    """
    self._batch_sizes.append(batch_size)

  @property
  def num_registered_minibatches(self):
    return len(self._batch_sizes)

  @property
  def _batch_size(self):
    return math_ops.reduce_sum(self._batch_sizes)


class FullyConnectedDiagonalFB(FisherBlock):
  """FisherBlock for fully-connected (dense) layers using a diagonal approx.

  Estimates the Fisher Information matrix's diagonal entries for a fully
  connected layer. Unlike NaiveDiagonalFB this uses the low-variance "sum of
  squares" estimator.

  Let 'params' be a vector parameterizing a model and 'i' an arbitrary index
  into it. We are interested in Fisher(params)[i, i]. This is,

    Fisher(params)[i, i] = E[ v(x, y, params) v(x, y, params)^T ][i, i]
                         = E[ v(x, y, params)[i] ^ 2 ]

  Consider fully connected layer in this model with (unshared) weight matrix
  'w'. For an example 'x' that produces layer inputs 'a' and output
  preactivations 's',

    v(x, y, w) = vec( a (d loss / d s)^T )

  This FisherBlock tracks Fisher(params)[i, i] for all indices 'i' corresponding
  to the layer's parameters 'w'.
  """

  def __init__(self, layer_collection, has_bias=False):
    """Creates a FullyConnectedDiagonalFB block.

    Args:
      layer_collection: The collection of all layers in the K-FAC approximate
          Fisher information matrix to which this FisherBlock belongs.
      has_bias: Whether the component Kronecker factors have an additive bias.
          (Default: False)
    """
    self._inputs = []
    self._outputs = []
    self._has_bias = has_bias

    super(FullyConnectedDiagonalFB, self).__init__(layer_collection)

  def instantiate_factors(self, grads_list, damping):
    inputs = _concat_along_batch_dim(self._inputs)
    grads_list = tuple(_concat_along_batch_dim(grads) for grads in grads_list)

    self._damping = damping
    self._factor = self._layer_collection.make_or_get_factor(
        fisher_factors.FullyConnectedDiagonalFactor,
        (inputs, grads_list, self._has_bias))

  def multiply_inverse(self, vector):
    """Approximate damped inverse Fisher-vector product.

    Args:
      vector: Tensor or 2-tuple of Tensors. if self._has_bias, Tensor of shape
        [input_size, output_size] corresponding to layer's weights. If not, a
        2-tuple of the former and a Tensor of shape [output_size] corresponding
        to the layer's bias.

    Returns:
      Tensor of the same shape, corresponding to the inverse Fisher-vector
      product.
    """
    reshaped_vect = utils.layer_params_to_mat2d(vector)
    reshaped_out = reshaped_vect / (self._factor.get_cov() + self._damping)
    return utils.mat2d_to_layer_params(vector, reshaped_out)

  def multiply(self, vector):
    """Approximate damped Fisher-vector product.

    Args:
      vector: Tensor or 2-tuple of Tensors. if self._has_bias, Tensor of shape
        [input_size, output_size] corresponding to layer's weights. If not, a
        2-tuple of the former and a Tensor of shape [output_size] corresponding
        to the layer's bias.

    Returns:
      Tensor of the same shape, corresponding to the Fisher-vector product.
    """
    reshaped_vect = utils.layer_params_to_mat2d(vector)
    reshaped_out = reshaped_vect * (self._factor.get_cov() + self._damping)
    return utils.mat2d_to_layer_params(vector, reshaped_out)

  def tensors_to_compute_grads(self):
    """Tensors to compute derivative of loss with respect to."""
    return self._outputs

  def register_additional_minibatch(self, inputs, outputs):
    """Registers an additional minibatch to the FisherBlock.

    Args:
      inputs: Tensor of shape [batch_size, input_size]. Inputs to the
        matrix-multiply.
      outputs: Tensor of shape [batch_size, output_size]. Layer preactivations.
    """
    self._inputs.append(inputs)
    self._outputs.append(outputs)

  @property
  def num_registered_minibatches(self):
    result = len(self._inputs)
    assert result == len(self._outputs)
    return result


class ConvDiagonalFB(FisherBlock):
  """FisherBlock for convolutional layers using a diagonal approx.

  Estimates the Fisher Information matrix's diagonal entries for a convolutional
  layer. Unlike NaiveDiagonalFB this uses the low-variance "sum of squares"
  estimator.

  Let 'params' be a vector parameterizing a model and 'i' an arbitrary index
  into it. We are interested in Fisher(params)[i, i]. This is,

    Fisher(params)[i, i] = E[ v(x, y, params) v(x, y, params)^T ][i, i]
                         = E[ v(x, y, params)[i] ^ 2 ]

  Consider a convoluational layer in this model with (unshared) filter matrix
  'w'. For an example image 'x' that produces layer inputs 'a' and output
  preactivations 's',

    v(x, y, w) = vec( sum_{loc} a_{loc} (d loss / d s_{loc})^T )

  where 'loc' is a single (x, y) location in an image.

  This FisherBlock tracks Fisher(params)[i, i] for all indices 'i' corresponding
  to the layer's parameters 'w'.
  """

  def __init__(self, layer_collection, params, strides, padding):
    """Creates a ConvDiagonalFB block.

    Args:
      layer_collection: The collection of all layers in the K-FAC approximate
          Fisher information matrix to which this FisherBlock belongs.
      params: The parameters (Tensor or tuple of Tensors) of this layer. If
        kernel alone, a Tensor of shape [kernel_height, kernel_width,
        in_channels, out_channels]. If kernel and bias, a tuple of 2 elements
        containing the previous and a Tensor of shape [out_channels].
      strides: The stride size in this layer (1-D Tensor of length 4).
      padding: The padding in this layer (e.g. "SAME").
    """
    self._inputs = []
    self._outputs = []
    self._strides = tuple(strides) if isinstance(strides, list) else strides
    self._padding = padding
    self._has_bias = isinstance(params, (tuple, list))

    fltr = params[0] if self._has_bias else params
    self._filter_shape = tuple(fltr.shape.as_list())

    super(ConvDiagonalFB, self).__init__(layer_collection)

  def instantiate_factors(self, grads_list, damping):
    # Concatenate inputs, grads_list into single Tensors.
    inputs = _concat_along_batch_dim(self._inputs)
    grads_list = tuple(_concat_along_batch_dim(grads) for grads in grads_list)

    # Infer number of locations upon which convolution is applied.
    inputs_shape = tuple(inputs.shape.as_list())
    self._num_locations = (
        inputs_shape[1] * inputs_shape[2] //
        (self._strides[1] * self._strides[2]))

    if NORMALIZE_DAMPING_POWER:
      damping /= self._num_locations**NORMALIZE_DAMPING_POWER
    self._damping = damping

    self._factor = self._layer_collection.make_or_get_factor(
        fisher_factors.ConvDiagonalFactor,
        (inputs, grads_list, self._filter_shape, self._strides, self._padding,
         self._has_bias))

  def multiply_inverse(self, vector):
    reshaped_vect = utils.layer_params_to_mat2d(vector)
    reshaped_out = reshaped_vect / (self._factor.get_cov() + self._damping)
    return utils.mat2d_to_layer_params(vector, reshaped_out)

  def multiply(self, vector):
    reshaped_vect = utils.layer_params_to_mat2d(vector)
    reshaped_out = reshaped_vect * (self._factor.get_cov() + self._damping)
    return utils.mat2d_to_layer_params(vector, reshaped_out)

  def tensors_to_compute_grads(self):
    return self._outputs

  def register_additional_minibatch(self, inputs, outputs):
    """Registers an additional minibatch to the FisherBlock.

    Args:
      inputs: Tensor of shape [batch_size, height, width, input_size]. Inputs to
        the convolution.
      outputs: Tensor of shape [batch_size, height, width, output_size]. Layer
        preactivations.
    """
    self._inputs.append(inputs)
    self._outputs.append(outputs)

  @property
  def num_registered_minibatches(self):
    return len(self._inputs)


class KroneckerProductFB(FisherBlock):
  """A base class for FisherBlocks with separate input and output factors.

  The Fisher block is approximated as a Kronecker product of the input and
  output factors.
  """

  def _register_damped_input_and_output_inverses(self, damping):
    """Registers damped inverses for both the input and output factors.

    Sets the instance members _input_damping and _output_damping. Requires the
    instance members _input_factor and _output_factor.

    Args:
      damping: The base damping factor (float or Tensor) for the damped inverse.
    """
    self._input_damping, self._output_damping = _compute_pi_adjusted_damping(
        self._input_factor.get_cov(),
        self._output_factor.get_cov(),
        damping**0.5)

    self._input_factor.register_damped_inverse(self._input_damping)
    self._output_factor.register_damped_inverse(self._output_damping)

  @property
  def _renorm_coeff(self):
    """Kronecker factor multiplier coefficient.

    If this FisherBlock is represented as 'FB = c * kron(left, right)', then
    this is 'c'.

    Returns:
      0-D Tensor.
    """
    return 1.0

  def multiply_inverse(self, vector):
    left_factor_inv = self._input_factor.get_damped_inverse(self._input_damping)
    right_factor_inv = self._output_factor.get_damped_inverse(
        self._output_damping)
    reshaped_vector = utils.layer_params_to_mat2d(vector)
    reshaped_out = math_ops.matmul(left_factor_inv,
                                   math_ops.matmul(reshaped_vector,
                                                   right_factor_inv))
    if self._renorm_coeff != 1.0:
      reshaped_out /= math_ops.cast(
          self._renorm_coeff, dtype=reshaped_out.dtype)
    return utils.mat2d_to_layer_params(vector, reshaped_out)

  def multiply(self, vector):
    left_factor = self._input_factor.get_cov()
    right_factor = self._output_factor.get_cov()
    reshaped_vector = utils.layer_params_to_mat2d(vector)
    reshaped_out = (
        math_ops.matmul(reshaped_vector, right_factor) +
        self._output_damping * reshaped_vector)
    reshaped_out = (
        math_ops.matmul(left_factor, reshaped_out) +
        self._input_damping * reshaped_out)
    if self._renorm_coeff != 1.0:
      reshaped_out *= math_ops.cast(
          self._renorm_coeff, dtype=reshaped_out.dtype)
    return utils.mat2d_to_layer_params(vector, reshaped_out)

  def full_fisher_block(self):
    """Explicitly constructs the full Fisher block.

    Used for testing purposes. (In general, the result may be very large.)

    Returns:
      The full Fisher block.
    """
    left_factor = self._input_factor.get_cov()
    right_factor = self._output_factor.get_cov()
    return self._renorm_coeff * utils.kronecker_product(left_factor,
                                                        right_factor)


class FullyConnectedKFACBasicFB(KroneckerProductFB):
  """K-FAC FisherBlock for fully-connected (dense) layers.

  This uses the Kronecker-factorized approximation from the original
  K-FAC paper (https://arxiv.org/abs/1503.05671)
  """

  def __init__(self, layer_collection, has_bias=False):
    """Creates a FullyConnectedKFACBasicFB block.

    Args:
      layer_collection: The collection of all layers in the K-FAC approximate
          Fisher information matrix to which this FisherBlock belongs.
      has_bias: Whether the component Kronecker factors have an additive bias.
          (Default: False)
    """
    self._inputs = []
    self._outputs = []
    self._has_bias = has_bias

    super(FullyConnectedKFACBasicFB, self).__init__(layer_collection)

  def instantiate_factors(self, grads_list, damping):
    """Instantiate Kronecker Factors for this FisherBlock.

    Args:
      grads_list: List of list of Tensors. grads_list[i][j] is the
        gradient of the loss with respect to 'outputs' from source 'i' and
        tower 'j'. Each Tensor has shape [tower_minibatch_size, output_size].
      damping: 0-D Tensor or float. 'damping' * identity is approximately added
        to this FisherBlock's Fisher approximation.
    """
    # TODO(b/68033310): Validate which of,
    #   (1) summing on a single device (as below), or
    #   (2) on each device in isolation and aggregating
    # is faster.
    inputs = _concat_along_batch_dim(self._inputs)
    grads_list = tuple(_concat_along_batch_dim(grads) for grads in grads_list)

    self._input_factor = self._layer_collection.make_or_get_factor(  #
        fisher_factors.FullyConnectedKroneckerFactor,  #
        ((inputs,), self._has_bias))
    self._output_factor = self._layer_collection.make_or_get_factor(  #
        fisher_factors.FullyConnectedKroneckerFactor,  #
        (grads_list,))
    self._register_damped_input_and_output_inverses(damping)

  def tensors_to_compute_grads(self):
    return self._outputs

  def register_additional_minibatch(self, inputs, outputs):
    """Registers an additional minibatch to the FisherBlock.

    Args:
      inputs: Tensor of shape [batch_size, input_size]. Inputs to the
        matrix-multiply.
      outputs: Tensor of shape [batch_size, output_size]. Layer preactivations.
    """
    self._inputs.append(inputs)
    self._outputs.append(outputs)

  @property
  def num_registered_minibatches(self):
    return len(self._inputs)


class ConvKFCBasicFB(KroneckerProductFB):
  """FisherBlock for 2D convolutional layers using the basic KFC approx.

  Estimates the Fisher Information matrix's blog for a convolutional
  layer.

  Consider a convoluational layer in this model with (unshared) filter matrix
  'w'. For a minibatch that produces inputs 'a' and output preactivations 's',
  this FisherBlock estimates,

    F(w) = #locations * kronecker(E[flat(a) flat(a)^T],
                                  E[flat(ds) flat(ds)^T])

  where

    ds = (d / ds) log p(y | x, w)
    #locations = number of (x, y) locations where 'w' is applied.

  where the expectation is taken over all examples and locations and flat()
  concatenates an array's leading dimensions.

  See equation 23 in https://arxiv.org/abs/1602.01407 for details.
  """

  def __init__(self, layer_collection, params, strides, padding):
    """Creates a ConvKFCBasicFB block.

    Args:
      layer_collection: The collection of all layers in the K-FAC approximate
          Fisher information matrix to which this FisherBlock belongs.
      params: The parameters (Tensor or tuple of Tensors) of this layer. If
        kernel alone, a Tensor of shape [kernel_height, kernel_width,
        in_channels, out_channels]. If kernel and bias, a tuple of 2 elements
        containing the previous and a Tensor of shape [out_channels].
      strides: The stride size in this layer (1-D Tensor of length 4).
      padding: The padding in this layer (1-D of Tensor length 4).
    """
    self._inputs = []
    self._outputs = []
    self._strides = tuple(strides) if isinstance(strides, list) else strides
    self._padding = padding
    self._has_bias = isinstance(params, (tuple, list))

    fltr = params[0] if self._has_bias else params
    self._filter_shape = tuple(fltr.shape.as_list())

    super(ConvKFCBasicFB, self).__init__(layer_collection)

  def instantiate_factors(self, grads_list, damping):
    # TODO(b/68033310): Validate which of,
    #   (1) summing on a single device (as below), or
    #   (2) on each device in isolation and aggregating
    # is faster.
    inputs = _concat_along_batch_dim(self._inputs)
    grads_list = tuple(_concat_along_batch_dim(grads) for grads in grads_list)

    # Infer number of locations upon which convolution is applied.
    self._num_locations = _num_conv_locations(inputs.shape.as_list(),
                                              self._strides)

    self._input_factor = self._layer_collection.make_or_get_factor(
        fisher_factors.ConvInputKroneckerFactor,
        (inputs, self._filter_shape, self._strides, self._padding,
         self._has_bias))
    self._output_factor = self._layer_collection.make_or_get_factor(
        fisher_factors.ConvOutputKroneckerFactor, (grads_list,))

    if NORMALIZE_DAMPING_POWER:
      damping /= self._num_locations**NORMALIZE_DAMPING_POWER
    self._damping = damping

    self._register_damped_input_and_output_inverses(damping)

  @property
  def _renorm_coeff(self):
    return self._num_locations

  def tensors_to_compute_grads(self):
    return self._outputs

  def register_additional_minibatch(self, inputs, outputs):
    """Registers an additional minibatch to the FisherBlock.

    Args:
      inputs: Tensor of shape [batch_size, height, width, input_size]. Inputs to
        the convolution.
      outputs: Tensor of shape [batch_size, height, width, output_size]. Layer
        preactivations.
    """
    self._inputs.append(inputs)
    self._outputs.append(outputs)

  @property
  def num_registered_minibatches(self):
    return len(self._inputs)


def _concat_along_batch_dim(tensor_list):
  """Concatenate tensors along batch (first) dimension.

  Args:
    tensor_list: list of Tensors or list of tuples of Tensors.

  Returns:
    Tensor or tuple of Tensors.

  Raises:
    ValueError: If 'tensor_list' is empty.

  """
  if not tensor_list:
    raise ValueError(
        "Cannot concatenate Tensors if there are no Tensors to concatenate.")

  if isinstance(tensor_list[0], (tuple, list)):
    # [(tensor1a, tensor1b),
    #  (tensor2a, tensor2b), ...] --> (tensor_a, tensor_b)
    return tuple(
        array_ops.concat(tensors, axis=0) for tensors in zip(*tensor_list))
  else:
    # [tensor1, tensor2] --> tensor
    return array_ops.concat(tensor_list, axis=0)


def _num_conv_locations(input_shape, strides):
  """Returns the number of locations a Conv kernel is applied to."""
  return input_shape[1] * input_shape[2] // (strides[1] * strides[2])


class FullyConnectedMultiIndepFB(KroneckerProductFB):
  """FisherBlock for fully-connected layers that share parameters.
  """

  def __init__(self, layer_collection, inputs, outputs, has_bias=False):
    """Creates a FullyConnectedMultiIndepFB block.

    Args:
      layer_collection: LayerCollection instance.
      inputs: list or tuple of Tensors. Each Tensor has shape [batch_size,
        inputs_size].
      outputs: list or tuple of Tensors. Each Tensor has shape [batch_size,
        outputs_size].
      has_bias: bool. If True, estimates Fisher with respect to a bias
        parameter as well as the layer's parameters.
    """

    assert len(inputs) == len(outputs)
    # We need to make sure inputs and outputs are tuples and not lists so that
    # they get hashed by layer_collection.make_or_get_factor properly.
    self._inputs = tuple(inputs)
    self._outputs = tuple(outputs)
    self._has_bias = has_bias
    self._num_uses = len(inputs)

    super(FullyConnectedMultiIndepFB, self).__init__(layer_collection)

  @property
  def num_registered_minibatches(self):
    # TODO(b/69411207): Add support for registering additional minibatches.
    return 1

  def instantiate_factors(self, grads_list, damping):

    self._input_factor = self._layer_collection.make_or_get_factor(
        fisher_factors.FullyConnectedMultiKF,
        ((self._inputs,), self._has_bias))

    self._output_factor = self._layer_collection.make_or_get_factor(
        fisher_factors.FullyConnectedMultiKF, (grads_list,))

    if NORMALIZE_DAMPING_POWER:
      damping /= self._num_uses**NORMALIZE_DAMPING_POWER

    self._register_damped_input_and_output_inverses(damping)

  @property
  def _renorm_coeff(self):
    return self._num_uses

  def tensors_to_compute_grads(self):
    return self._outputs

  def num_inputs(self):
    return len(self._inputs)


class SeriesFBApproximation(enum.IntEnum):
  """See FullyConnectedSeriesFB.__init__ for description and usage."""
  option1 = 1
  option2 = 2


class FullyConnectedSeriesFB(FisherBlock):
  """FisherBlock for fully-connected layers that share parameters across time.

  See the following preprint for details:
    https://openreview.net/pdf?id=HyMTkQZAb

  See the end of the appendix of the paper for a pseudo-code of the
  algorithm being implemented by multiply_inverse here.  Note that we are
  using pre-computed versions of certain matrix-matrix products to speed
  things up.  This is explicitly explained wherever it is done.
  """

  def __init__(self,
               layer_collection,
               inputs,
               outputs,
               has_bias=False,
               option=SeriesFBApproximation.option2):
    """Constructs a new `FullyConnectedSeriesFB`.

    Args:
      layer_collection: The collection of all layers in the K-FAC approximate
        Fisher information matrix to which this FisherBlock belongs.
      inputs: List of tensors of shape [batch_size, input_size].
        Inputs to the layer.
      outputs: List of tensors of shape [batch_size, input_size].
        Outputs of the layer (before activations).
      has_bias: Whether the layer includes a bias parameter.
      option: A `SeriesFBApproximation` specifying the simplifying assumption
        to be used in this block. `option1` approximates the cross-covariance
        over time as a symmetric matrix, while `option2` makes
        the assumption that training sequences are infinitely long. See section
        3.5 of the paper for more details.
    """

    assert len(inputs) == len(outputs)
    # We need to make sure inputs and outputs are tuples and not lists so that
    # they get hashed by layer_collection.make_or_get_factor properly.
    self._inputs = tuple(inputs)
    self._outputs = tuple(outputs)
    self._has_bias = has_bias
    self._num_timesteps = len(inputs)
    self._option = option

    super(FullyConnectedSeriesFB, self).__init__(layer_collection)

  @property
  def num_registered_minibatches(self):
    # TODO(b/69411207): Add support for registering additional minibatches.
    return 1

  def instantiate_factors(self, grads_list, damping):

    self._input_factor = self._layer_collection.make_or_get_factor(
        fisher_factors.FullyConnectedMultiKF, ((self._inputs,), self._has_bias))

    self._output_factor = self._layer_collection.make_or_get_factor(
        fisher_factors.FullyConnectedMultiKF, (grads_list,))

    if NORMALIZE_DAMPING_POWER:
      damping /= self._num_timesteps**NORMALIZE_DAMPING_POWER

    self._damping_input, self._damping_output = _compute_pi_adjusted_damping(
        self._input_factor.get_cov(),
        self._output_factor.get_cov(),
        damping**0.5)

    if self._option == SeriesFBApproximation.option1:
      self._input_factor.register_option1quants(self._damping_input)
      self._output_factor.register_option1quants(self._damping_output)
    elif self._option == SeriesFBApproximation.option2:
      self._input_factor.register_option2quants(self._damping_input)
      self._output_factor.register_option2quants(self._damping_output)
    else:
      raise ValueError(
          "Unrecognized FullyConnectedSeriesFB approximation: {}".format(
              self._option))

  def multiply_inverse(self, vector):
    # pylint: disable=invalid-name

    Z = utils.layer_params_to_mat2d(vector)

    # Derivations were done for "batch_dim==1" case so we need to convert to
    # that orientation:
    Z = array_ops.transpose(Z)

    if self._option == SeriesFBApproximation.option1:

      # Note that L_A = A0^(-1/2) * U_A and L_G = G0^(-1/2) * U_G.
      L_A, psi_A = self._input_factor.get_option1quants(self._damping_input)
      L_G, psi_G = self._output_factor.get_option1quants(self._damping_output)

      def gamma(x):
        # We are assuming that each case has the same number of time-steps.
        # If this stops being the case one shouldn't simply replace this T
        # with its average value.  Instead, one needs to go back to the
        # definition of the gamma function from the paper.
        T = self._num_timesteps
        return (1 - x)**2 / (T * (1 - x**2) - 2 * x * (1 - x**T))

      # Y = gamma( psi_G*psi_A^T ) (computed element-wise)
      # Even though Y is Z-independent we are recomputing it from the psi's
      # each since Y depends on both A and G quantities, and it is relatively
      # cheap to compute.
      Y = gamma(array_ops.reshape(psi_G, [int(psi_G.shape[0]), -1]) * psi_A)

      # Z = L_G^T * Z * L_A
      # This is equivalent to the following computation from the original
      # pseudo-code:
      # Z = G0^(-1/2) * Z * A0^(-1/2)
      # Z = U_G^T * Z * U_A
      Z = math_ops.matmul(L_G, math_ops.matmul(Z, L_A), transpose_a=True)

      # Z = Z .* Y
      Z *= Y

      # Z = L_G * Z * L_A^T
      # This is equivalent to the following computation from the original
      # pseudo-code:
      # Z = U_G * Z * U_A^T
      # Z = G0^(-1/2) * Z * A0^(-1/2)
      Z = math_ops.matmul(L_G, math_ops.matmul(Z, L_A, transpose_b=True))

    elif self._option == SeriesFBApproximation.option2:

      # Note that P_A = A_1^T * A_0^(-1) and P_G = G_1^T * G_0^(-1),
      # and K_A = A_0^(-1/2) * E_A and K_G = G_0^(-1/2) * E_G.
      P_A, K_A, mu_A = self._input_factor.get_option2quants(self._damping_input)
      P_G, K_G, mu_G = self._output_factor.get_option2quants(
          self._damping_output)

      # Our approach differs superficially from the pseudo-code in the paper
      # in order to reduce the total number of matrix-matrix multiplies.
      # In particular, the first three computations in the pseudo code are
      # Z = G0^(-1/2) * Z * A0^(-1/2)
      # Z = Z - hPsi_G^T * Z * hPsi_A
      # Z = E_G^T * Z * E_A
      # Noting that hPsi = C0^(-1/2) * C1 * C0^(-1/2), so that
      # C0^(-1/2) * hPsi = C0^(-1) * C1 * C0^(-1/2) = P^T * C0^(-1/2)
      # the entire computation can be written as
      # Z = E_G^T * (G0^(-1/2) * Z * A0^(-1/2)
      #     - hPsi_G^T * G0^(-1/2) * Z * A0^(-1/2) * hPsi_A) * E_A
      #   = E_G^T * (G0^(-1/2) * Z * A0^(-1/2)
      #     - G0^(-1/2) * P_G * Z * P_A^T * A0^(-1/2)) * E_A
      #   = E_G^T * G0^(-1/2) * Z * A0^(-1/2) * E_A
      #     -  E_G^T* G0^(-1/2) * P_G * Z * P_A^T * A0^(-1/2) * E_A
      #   = K_G^T * Z * K_A  -  K_G^T * P_G * Z * P_A^T * K_A
      # This final expression is computed by the following two lines:
      # Z = Z - P_G * Z * P_A^T
      Z -= math_ops.matmul(P_G, math_ops.matmul(Z, P_A, transpose_b=True))
      # Z = K_G^T * Z * K_A
      Z = math_ops.matmul(K_G, math_ops.matmul(Z, K_A), transpose_a=True)

      # Z = Z ./ (1*1^T - mu_G*mu_A^T)
      # Be careful with the outer product.  We don't want to accidentally
      # make it an inner-product instead.
      tmp = 1.0 - array_ops.reshape(mu_G, [int(mu_G.shape[0]), -1]) * mu_A
      # Prevent some numerical issues by setting any 0.0 eigs to 1.0
      tmp += 1.0 * math_ops.cast(math_ops.equal(tmp, 0.0), dtype=tmp.dtype)
      Z /= tmp

      # We now perform the transpose/reverse version of the operations
      # derived above, whose derivation from the original pseudo-code is
      # analgous.
      # Z = K_G * Z * K_A^T
      Z = math_ops.matmul(K_G, math_ops.matmul(Z, K_A, transpose_b=True))

      # Z = Z - P_G^T * Z * P_A
      Z -= math_ops.matmul(P_G, math_ops.matmul(Z, P_A), transpose_a=True)

      # Z = normalize (1/E[T]) * Z
      # Note that this normalization is done because we compute the statistics
      # by averaging, not summing, over time. (And the gradient is presumably
      # summed over time, not averaged, and thus their scales are different.)
      Z /= math_ops.cast(self._num_timesteps, Z.dtype)

    # Convert back to the "batch_dim==0" orientation.
    Z = array_ops.transpose(Z)

    return utils.mat2d_to_layer_params(vector, Z)

    # pylint: enable=invalid-name

  def multiply(self, vector):
    raise NotImplementedError

  def tensors_to_compute_grads(self):
    return self._outputs

  def num_inputs(self):
    return len(self._inputs)
