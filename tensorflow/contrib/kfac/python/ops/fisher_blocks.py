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
# compute_pi_adjusted_damping() for details.
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


def normalize_damping(damping, num_replications):
  """Normalize damping after adjusting scale by NORMALIZE_DAMPING_POWER."""
  if NORMALIZE_DAMPING_POWER:
    return damping / (num_replications ** NORMALIZE_DAMPING_POWER)
  return damping


def compute_pi_tracenorm(left_cov, right_cov):
  """Computes the scalar constant pi for Tikhonov regularization/damping.

  pi = sqrt( (trace(A) / dim(A)) / (trace(B) / dim(B)) )
  See section 6.3 of https://arxiv.org/pdf/1503.05671.pdf for details.

  Args:
    left_cov: The left Kronecker factor "covariance".
    right_cov: The right Kronecker factor "covariance".

  Returns:
    The computed scalar constant pi for these Kronecker Factors (as a Tensor).
  """

  def _trace(cov):
    if len(cov.shape) == 1:
      # Diagonal matrix.
      return math_ops.reduce_sum(cov)
    elif len(cov.shape) == 2:
      # Full matrix.
      return math_ops.trace(cov)
    else:
      raise ValueError(
          "What's the trace of a Tensor of rank %d?" % len(cov.shape))

  # Instead of dividing by the dim of the norm, we multiply by the dim of the
  # other norm. This works out the same in the ratio.
  left_norm = _trace(left_cov) * right_cov.shape.as_list()[0]
  right_norm = _trace(right_cov) * left_cov.shape.as_list()[0]
  return math_ops.sqrt(left_norm / right_norm)


def compute_pi_adjusted_damping(left_cov, right_cov, damping):

  if PI_TYPE == PI_TRACENORM_NAME:
    pi = compute_pi_tracenorm(left_cov, right_cov)
    return (damping * pi, damping / pi)

  elif PI_TYPE == PI_OFF_NAME:
    return (damping, damping)


class PackagedFunc(object):
  """A Python thunk with a stable ID.

  Enables stable names for lambdas.
  """

  def __init__(self, func, func_id):
    """Initializes PackagedFunc.

    Args:
      func: a zero-arg Python function.
      func_id: a hashable, function that produces a hashable, or a list/tuple
        thereof.
    """
    self._func = func
    func_id = func_id if isinstance(func_id, (tuple, list)) else (func_id,)
    self._func_id = func_id

  def __call__(self):
    return self._func()

  @property
  def func_id(self):
    """A hashable identifier for this function."""
    return tuple(elt() if callable(elt) else elt for elt in self._func_id)


def _package_func(func, func_id):
  return PackagedFunc(func, func_id)


@six.add_metaclass(abc.ABCMeta)
class FisherBlock(object):
  """Abstract base class for objects modeling approximate Fisher matrix blocks.

  Subclasses must implement register_matpower, multiply_matpower,
  instantiate_factors, tensors_to_compute_grads, and num_registered_minibatches
  methods.
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
  def register_matpower(self, exp):
    """Registers a matrix power to be computed by the block.

    Args:
      exp: A float representing the power to raise the block by.
    """
    pass

  def register_inverse(self):
    """Registers a matrix inverse to be computed by the block."""
    self.register_matpower(-1)

  @abc.abstractmethod
  def multiply_matpower(self, vector, exp):
    """Multiplies the vector by the (damped) matrix-power of the block.

    Args:
      vector: The vector (a Tensor or tuple of Tensors) to be multiplied.
      exp: A float representing the power to raise the block by before
        multiplying it by the vector.

    Returns:
      The vector left-multiplied by the (damped) matrix-power of the block.
    """
    pass

  def multiply_inverse(self, vector):
    """Multiplies the vector by the (damped) inverse of the block.

    Args:
      vector: The vector (a Tensor or tuple of Tensors) to be multiplied.

    Returns:
      The vector left-multiplied by the (damped) inverse of the block.
    """
    return self.multiply_matpower(vector, -1)

  def multiply(self, vector):
    """Multiplies the vector by the (damped) block.

    Args:
      vector: The vector (a Tensor or tuple of Tensors) to be multiplied.

    Returns:
      The vector left-multiplied by the (damped) block.
    """
    return self.multiply_matpower(vector, 1)

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
    self._damping_func = _package_func(lambda: damping, (damping,))

    self._factor = self._layer_collection.make_or_get_factor(
        fisher_factors.FullFactor, (grads_list, self._batch_size))

  def register_matpower(self, exp):
    self._factor.register_matpower(exp, self._damping_func)

  def multiply_matpower(self, vector, exp):
    vector_flat = utils.tensors_to_column(vector)
    out_flat = self._factor.left_multiply_matpower(
        vector_flat, exp, self._damping_func)
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
    self._damping_func = _package_func(lambda: damping, (damping,))

    self._factor = self._layer_collection.make_or_get_factor(
        fisher_factors.NaiveDiagonalFactor, (grads_list, self._batch_size))

  def register_matpower(self, exp):
    # Not needed for this.  Matrix powers are computed on demand in the
    # diagonal case
    pass

  def multiply_matpower(self, vector, exp):
    vector_flat = utils.tensors_to_column(vector)
    out_flat = self._factor.left_multiply_matpower(
        vector_flat, exp, self._damping_func)
    return utils.column_to_tensors(vector, out_flat)

  def full_fisher_block(self):
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


class InputOutputMultiMinibatch(object):
  """Mix-in class for blocks with inputs & outputs and multiple mini-batches."""

  def __init__(self, *args, **kwargs):
    self.__inputs = []
    self.__outputs = []
    super(InputOutputMultiMinibatch, self).__init__(*args, **kwargs)

  def tensors_to_compute_grads(self):
    """Tensors to compute derivative of loss with respect to."""
    return self._outputs

  def register_additional_minibatch(self, inputs, outputs):
    self._inputs.append(inputs)
    self._outputs.append(outputs)

  @property
  def num_registered_minibatches(self):
    result = len(self._inputs)
    assert result == len(self._outputs)
    return result

  @property
  def _inputs(self):
    return self.__inputs

  @property
  def _outputs(self):
    return self.__outputs

  def _package_minibatches(self, grads_list):
    """Constructs PartitionedTensor for inputs, grads_list.

    The purpose of this method is to package up the towers/minibatch dimension
    of these arrays into PartitionedTensor objects.

    Args:
      grads_list: 2-D list of Tensors. First index is for source, second
        index for tower.

    Returns:
      inputs: PartitionedTensor.
      grads_list: Tuple of PartitionedTensors, one per source.
    """
    inputs = utils.PartitionedTensor(self._inputs)
    grads_list = tuple(utils.PartitionedTensor(grads) for grads in grads_list)

    return inputs, grads_list

  def _package_minibatches_multi(self, grads_list):
    """Constructs PartitionedTensors for inputs, grads_list.

    The purpose of this method is to package up the towers/minibatch dimension
    of these arrays into PartitionedTensor objects.

    This version of this function is for use with FisherBlocks that deal with
    multiple uses or time-steps. One PartitionedTensor is created for each
    use/time-step.

    Args:
      grads_list: 3-D tuple of Tensors. First index is for source, second
        index is for tower, third is for use/time-step.

    Returns:
      inputs: A tuple of PartitionedTensor's, one per use/time-step.
      grads_list: 2-D tuple of PartitionedTensors. First index is for source,
        second is for use/time-step.
    """
    # self._inputs is a 2-D tuple.  First index is tower/mini-batch, second is
    # use/time-step.
    inputs = self._inputs
    num_uses = len(inputs[0])
    assert all(len(input_) == num_uses for input_ in inputs)
    assert all(len(grad) == num_uses for grads in grads_list for grad in grads)

    inputs = tuple(utils.PartitionedTensor(input_) for input_ in zip(*inputs))
    grads_list = tuple(tuple(utils.PartitionedTensor(grad)
                             for grad in zip(*grads)) for grads in grads_list)

    return inputs, grads_list


class FullyConnectedDiagonalFB(InputOutputMultiMinibatch, FisherBlock):
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
    self._has_bias = has_bias

    super(FullyConnectedDiagonalFB, self).__init__(layer_collection)

  def instantiate_factors(self, grads_list, damping):
    inputs, grads_list = self._package_minibatches(grads_list)

    self._factor = self._layer_collection.make_or_get_factor(
        fisher_factors.FullyConnectedDiagonalFactor,
        (inputs, grads_list, self._has_bias))

    self._damping_func = _package_func(lambda: damping, (damping,))

  def register_matpower(self, exp):
    # Not needed for this.  Matrix powers are computed on demand in the
    # diagonal case
    pass

  def multiply_matpower(self, vector, exp):
    """Multiplies the vector by the (damped) matrix-power of the block.

    Args:
      vector: Tensor or 2-tuple of Tensors. if self._has_bias, Tensor of shape
        [input_size, output_size] corresponding to layer's weights. If not, a
        2-tuple of the former and a Tensor of shape [output_size] corresponding
        to the layer's bias.
      exp: A scalar representing the power to raise the block before multiplying
           it by the vector.

    Returns:
      The vector left-multiplied by the (damped) matrix-power of the block.
    """
    reshaped_vec = utils.layer_params_to_mat2d(vector)
    reshaped_out = self._factor.left_multiply_matpower(
        reshaped_vec, exp, self._damping_func)
    return utils.mat2d_to_layer_params(vector, reshaped_out)


class ConvDiagonalFB(InputOutputMultiMinibatch, FisherBlock):
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
    self._strides = tuple(strides) if isinstance(strides, list) else strides
    self._padding = padding
    self._has_bias = isinstance(params, (tuple, list))

    fltr = params[0] if self._has_bias else params
    self._filter_shape = tuple(fltr.shape.as_list())

    super(ConvDiagonalFB, self).__init__(layer_collection)

  def instantiate_factors(self, grads_list, damping):
    # Infer number of locations upon which convolution is applied.
    inputs_shape = tuple(self._inputs[0].shape.as_list())
    self._num_locations = (
        inputs_shape[1] * inputs_shape[2] //
        (self._strides[1] * self._strides[2]))

    inputs, grads_list = self._package_minibatches(grads_list)

    self._factor = self._layer_collection.make_or_get_factor(
        fisher_factors.ConvDiagonalFactor,
        (inputs, grads_list, self._filter_shape, self._strides,
         self._padding, self._has_bias))

    def damping_func():
      return self._num_locations * normalize_damping(damping,
                                                     self._num_locations)

    damping_id = (self._num_locations, "mult", "normalize_damping", damping,
                  self._num_locations)
    self._damping_func = _package_func(damping_func, damping_id)

  def register_matpower(self, exp):
    # Not needed for this.  Matrix powers are computed on demand in the
    # diagonal case
    pass

  def multiply_matpower(self, vector, exp):
    reshaped_vect = utils.layer_params_to_mat2d(vector)
    reshaped_out = self._factor.left_multiply_matpower(
        reshaped_vect, exp, self._damping_func)
    return utils.mat2d_to_layer_params(vector, reshaped_out)


class KroneckerProductFB(FisherBlock):
  """A base class for FisherBlocks with separate input and output factors.

  The Fisher block is approximated as a Kronecker product of the input and
  output factors.
  """

  def __init__(self, layer_collection):
    super(KroneckerProductFB, self).__init__(layer_collection)

  def _setup_damping(self, damping, normalization=None):
    """Makes functions that compute the damping values for both factors."""
    def compute_damping():
      if normalization is not None:
        maybe_normalized_damping = normalize_damping(damping, normalization)
      else:
        maybe_normalized_damping = damping

      return compute_pi_adjusted_damping(self._input_factor.get_cov(),
                                         self._output_factor.get_cov(),
                                         maybe_normalized_damping**0.5)

    if normalization is not None:
      damping_id = ("compute_pi_adjusted_damping",
                    "cov", self._input_factor.name,
                    "cov", self._output_factor.name,
                    "normalize_damping", damping, normalization, "power", 0.5)
    else:
      damping_id = ("compute_pi_adjusted_damping",
                    "cov", self._input_factor.name,
                    "cov", self._output_factor.name,
                    damping, "power", 0.5)

    self._input_damping_func = _package_func(lambda: compute_damping()[0],
                                             damping_id + ("ref", 0))
    self._output_damping_func = _package_func(lambda: compute_damping()[1],
                                              damping_id + ("ref", 1))

  def register_matpower(self, exp):
    self._input_factor.register_matpower(exp, self._input_damping_func)
    self._output_factor.register_matpower(exp, self._output_damping_func)

  @property
  def _renorm_coeff(self):
    """Kronecker factor multiplier coefficient.

    If this FisherBlock is represented as 'FB = c * kron(left, right)', then
    this is 'c'.

    Returns:
      0-D Tensor.
    """
    return 1.0

  def multiply_matpower(self, vector, exp):
    reshaped_vector = utils.layer_params_to_mat2d(vector)
    reshaped_out = self._output_factor.right_multiply_matpower(
        reshaped_vector, exp, self._output_damping_func)
    reshaped_out = self._input_factor.left_multiply_matpower(
        reshaped_out, exp, self._input_damping_func)
    if self._renorm_coeff != 1.0:
      reshaped_out *= math_ops.cast(
          self._renorm_coeff**exp, dtype=reshaped_out.dtype)
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


class EmbeddingKFACFB(InputOutputMultiMinibatch, KroneckerProductFB):
  """K-FAC FisherBlock for embedding layers.

  This FisherBlock is similar to EmbeddingKFACFB, except that its
  input factor is approximated by a diagonal matrix. In the case that each
  example references exactly one embedding, this approximation is exact.

  Does not support bias parameters.
  """

  def __init__(self, layer_collection, vocab_size):
    """Creates a EmbeddingKFACFB block.

    Args:
      layer_collection: The collection of all layers in the K-FAC approximate
          Fisher information matrix to which this FisherBlock belongs.
      vocab_size: int. Size of vocabulary for this embedding layer.
    """
    self._vocab_size = vocab_size

    super(EmbeddingKFACFB, self).__init__(layer_collection)

  def instantiate_factors(self, grads_list, damping):
    """Instantiate Kronecker Factors for this FisherBlock.

    Args:
      grads_list: List of list of Tensors. grads_list[i][j] is the
        gradient of the loss with respect to 'outputs' from source 'i' and
        tower 'j'. Each Tensor has shape [tower_minibatch_size, output_size].
      damping: 0-D Tensor or float. 'damping' * identity is approximately added
        to this FisherBlock's Fisher approximation.
    """
    inputs, grads_list = self._package_minibatches(grads_list)

    self._input_factor = self._layer_collection.make_or_get_factor(  #
        fisher_factors.EmbeddingInputKroneckerFactor,  #
        (inputs, self._vocab_size))
    self._output_factor = self._layer_collection.make_or_get_factor(  #
        fisher_factors.FullyConnectedKroneckerFactor,  #
        (grads_list,))
    self._setup_damping(damping)


class FullyConnectedKFACBasicFB(InputOutputMultiMinibatch, KroneckerProductFB):
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
    inputs, grads_list = self._package_minibatches(grads_list)

    self._input_factor = self._layer_collection.make_or_get_factor(  #
        fisher_factors.FullyConnectedKroneckerFactor,  #
        ((inputs,), self._has_bias))
    self._output_factor = self._layer_collection.make_or_get_factor(  #
        fisher_factors.FullyConnectedKroneckerFactor,  #
        (grads_list,))
    self._setup_damping(damping)


class ConvKFCBasicFB(InputOutputMultiMinibatch, KroneckerProductFB):
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
    self._strides = tuple(strides) if isinstance(strides, list) else strides
    self._padding = padding
    self._has_bias = isinstance(params, (tuple, list))

    fltr = params[0] if self._has_bias else params
    self._filter_shape = tuple(fltr.shape.as_list())

    super(ConvKFCBasicFB, self).__init__(layer_collection)

  def instantiate_factors(self, grads_list, damping):
    # Infer number of locations upon which convolution is applied.
    self._num_locations = num_conv_locations(self._inputs[0].shape.as_list(),
                                             self._strides)

    inputs, grads_list = self._package_minibatches(grads_list)

    self._input_factor = self._layer_collection.make_or_get_factor(
        fisher_factors.ConvInputKroneckerFactor,
        (inputs, self._filter_shape, self._strides, self._padding,
         self._has_bias))
    self._output_factor = self._layer_collection.make_or_get_factor(
        fisher_factors.ConvOutputKroneckerFactor, (grads_list,))

    self._setup_damping(damping, normalization=self._num_locations)

  @property
  def _renorm_coeff(self):
    return self._num_locations


def num_conv_locations(input_shape, strides):
  """Returns the number of spatial locations a 2D Conv kernel is applied to.

  Args:
    input_shape: list representing shape of inputs to the Conv layer.
    strides: list representing strides for the Conv kernel.

  Returns:
    A scalar |T| denoting the number of spatial locations for the Conv layer.
  """
  return input_shape[1] * input_shape[2] // (strides[1] * strides[2])


class FullyConnectedMultiIndepFB(InputOutputMultiMinibatch, KroneckerProductFB):
  """FisherBlock for fully-connected layers that share parameters.
  """

  def __init__(self, layer_collection, has_bias=False):
    """Creates a FullyConnectedMultiIndepFB block.

    Args:
      layer_collection: LayerCollection instance.
      has_bias: bool. If True, estimates Fisher with respect to a bias
        parameter as well as the layer's parameters.
    """
    self._has_bias = has_bias

    super(FullyConnectedMultiIndepFB, self).__init__(layer_collection)

  def instantiate_factors(self, grads_list, damping):

    self._num_uses = len(self._inputs[0])
    inputs, grads_list = self._package_minibatches_multi(grads_list)

    self._input_factor = self._layer_collection.make_or_get_factor(
        fisher_factors.FullyConnectedMultiKF,
        ((inputs,), self._has_bias))

    self._output_factor = self._layer_collection.make_or_get_factor(
        fisher_factors.FullyConnectedMultiKF, (grads_list,))

    self._setup_damping(damping, normalization=self._num_uses)

  @property
  def _renorm_coeff(self):
    return self._num_uses

  def tensors_to_compute_grads(self):
    return self._outputs


class SeriesFBApproximation(enum.IntEnum):
  """See FullyConnectedSeriesFB.__init__ for description and usage."""
  option1 = 1
  option2 = 2


class FullyConnectedSeriesFB(InputOutputMultiMinibatch, FisherBlock):
  """FisherBlock for fully-connected layers that share parameters across time.

  See the following preprint for details:
    https://openreview.net/pdf?id=HyMTkQZAb

  See the end of the appendix of the paper for a pseudo-code of the
  algorithm being implemented by multiply_matpower here.  Note that we are
  using pre-computed versions of certain matrix-matrix products to speed
  things up.  This is explicitly explained wherever it is done.
  """

  def __init__(self,
               layer_collection,
               has_bias=False,
               option=SeriesFBApproximation.option2):
    """Constructs a new `FullyConnectedSeriesFB`.

    Args:
      layer_collection: The collection of all layers in the K-FAC approximate
        Fisher information matrix to which this FisherBlock belongs.
      has_bias: Whether the layer includes a bias parameter.
      option: A `SeriesFBApproximation` specifying the simplifying assumption
        to be used in this block. `option1` approximates the cross-covariance
        over time as a symmetric matrix, while `option2` makes
        the assumption that training sequences are infinitely long. See section
        3.5 of the paper for more details.
    """

    self._has_bias = has_bias
    self._option = option

    super(FullyConnectedSeriesFB, self).__init__(layer_collection)

  def instantiate_factors(self, grads_list, damping):

    self._num_timesteps = len(self._inputs[0])
    inputs, grads_list = self._package_minibatches_multi(grads_list)

    self._input_factor = self._layer_collection.make_or_get_factor(
        fisher_factors.FullyConnectedMultiKF, ((inputs,), self._has_bias))
    self._input_factor.register_cov_dt1()

    self._output_factor = self._layer_collection.make_or_get_factor(
        fisher_factors.FullyConnectedMultiKF, (grads_list,))
    self._output_factor.register_cov_dt1()

    def compute_damping():
      normalized_damping = normalize_damping(damping, self._num_timesteps)
      return compute_pi_adjusted_damping(self._input_factor.get_cov(),
                                         self._output_factor.get_cov(),
                                         normalized_damping**0.5)

    damping_id = ("compute_pi_adjusted_damping",
                  "cov", self._input_factor.name,
                  "cov", self._output_factor.name,
                  "normalize_damping",
                  damping, self._num_timesteps, "power", 0.5)
    self._input_damping_func = _package_func(lambda: compute_damping()[0],
                                             damping_id + ("ref", 0))
    self._output_damping_func = _package_func(lambda: compute_damping()[1],
                                              damping_id + ("ref", 1))

  def register_matpower(self, exp):
    if exp != -1:
      raise NotImplementedError("FullyConnectedSeriesFB only supports inverse"
                                "multiplications.")

    if self._option == SeriesFBApproximation.option1:
      self._input_factor.register_option1quants(self._input_damping_func)
      self._output_factor.register_option1quants(self._output_damping_func)
    elif self._option == SeriesFBApproximation.option2:
      self._input_factor.register_option2quants(self._input_damping_func)
      self._output_factor.register_option2quants(self._output_damping_func)
    else:
      raise ValueError(
          "Unrecognized FullyConnectedSeriesFB approximation: {}".format(
              self._option))

  def multiply_matpower(self, vector, exp):
    if exp != -1:
      raise NotImplementedError("FullyConnectedSeriesFB only supports inverse"
                                "multiplications.")

    # pylint: disable=invalid-name

    Z = utils.layer_params_to_mat2d(vector)

    # Derivations were done for "batch_dim==1" case so we need to convert to
    # that orientation:
    Z = array_ops.transpose(Z)

    if self._option == SeriesFBApproximation.option1:

      # Note that L_A = A0^(-1/2) * U_A and L_G = G0^(-1/2) * U_G.
      L_A, psi_A = self._input_factor.get_option1quants(
          self._input_damping_func)
      L_G, psi_G = self._output_factor.get_option1quants(
          self._output_damping_func)

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
      P_A, K_A, mu_A = self._input_factor.get_option2quants(
          self._input_damping_func)
      P_G, K_G, mu_G = self._output_factor.get_option2quants(
          self._output_damping_func)

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

  def tensors_to_compute_grads(self):
    return self._outputs
