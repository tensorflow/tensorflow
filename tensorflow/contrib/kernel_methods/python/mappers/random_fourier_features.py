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
"""Approximate kernel mapper for RBF kernel based on Random Fourier Features."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from tensorflow.contrib.kernel_methods.python.mappers import dense_kernel_mapper as dkm
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops


# TODO(sibyl-vie3Poto,felixyu): add an option to control whether the parameters in the
# kernel map are trainable.
class RandomFourierFeatureMapper(dkm.DenseKernelMapper):
  r"""Class that implements Random Fourier Feature Mapping (RFFM) in TensorFlow.

  The RFFM mapping is used to approximate the Gaussian (RBF) kernel:
  ```
  $$(exp(-||x-y||_2^2 / (2 * \sigma^2))$$
  ```

  The implementation of RFFM is based on the following paper:
  "Random Features for Large-Scale Kernel Machines" by Ali Rahimi and Ben Recht.
  (link: https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf)

  The mapping uses a matrix `\\(Omega \in R^{d x D}\\)` and a bias vector
  `\\(b \in R^D\\)` where `d` is the input dimension (number of dense input
  features) and `D` is the output dimension (i.e., dimension of the feature
  space the input is mapped to). Each entry of `Omega` is sampled i.i.d. from a
  (scaled) Gaussian distribution and each entry of `b` is sampled independently
  and uniformly from [0, \\(2 * pi\\)].

  For a single input feature vector x in R^d, its RFFM is defined as:
  ```
      $$sqrt(2/D) * cos(x * Omega + b)$$
  ```
  where `cos` is the element-wise cosine function and `x, b` are represented as
  row vectors. The aforementioned paper shows that the linear kernel of
  RFFM-mapped vectors approximates the Gaussian kernel of the initial vectors.

  """

  def __init__(self, input_dim, output_dim, stddev=1.0, seed=1, name=None):
    """Constructs a RandomFourierFeatureMapper instance.

    Args:
      input_dim: The dimension (number of features) of the tensors to be mapped.
      output_dim: The output dimension of the mapping.
      stddev: The standard deviation of the Gaussian kernel to be approximated.
        The error of the classifier trained using this approximation is very
        sensitive to this parameter.
      seed: An integer used to initialize the parameters (`Omega` and `b`) of
        the mapper. For repeatable sequences across different invocations of the
        mapper object (for instance, to ensure consistent mapping both at
        training and eval/inference if these happen in different invocations),
        set this to the same integer.
      name: name for the mapper object.
    """
    # TODO(sibyl-vie3Poto): Maybe infer input_dim and/or output_dim (if not explicitly
    # provided). input_dim can be inferred lazily, the first time map is called.
    # output_dim can be inferred from input_dim using heuristics on the error of
    # the approximation (and, by extension, the error of the classification
    # based on the approximation).
    self._input_dim = input_dim
    self._output_dim = output_dim
    self._stddev = stddev
    self._seed = seed
    self._name = name

  @property
  def name(self):
    """Returns a name for the `RandomFourierFeatureMapper` instance.

    If the name provided in the constructor is `None`, then the object's unique
    id is returned.

    Returns:
      A name for the `RandomFourierFeatureMapper` instance.
    """
    return self._name or str(id(self))

  @property
  def input_dim(self):
    return self._input_dim

  @property
  def output_dim(self):
    return self._output_dim

  def map(self, input_tensor):
    """Maps each row of input_tensor using random Fourier features.

    Args:
      input_tensor: a `Tensor` containing input features. It's shape is
      [batch_size, self._input_dim].

    Returns:
      A `Tensor` of shape [batch_size, self._output_dim] containing RFFM-mapped
      features.

    Raises:
      InvalidShapeError: if the shape of the `input_tensor` is inconsistent with
        expected input dimension.
    """
    input_tensor_shape = input_tensor.get_shape()
    if len(input_tensor_shape) != 2:
      raise dkm.InvalidShapeError(
          'The shape of the tensor should be 2. Got %d instead.' %
          len(input_tensor_shape))

    features_dim = input_tensor_shape[1]
    if features_dim != self._input_dim:
      raise dkm.InvalidShapeError(
          'Invalid dimension: expected %d input features, got %d instead.' %
          (self._input_dim, features_dim))

    # Add ops that compute (deterministically) omega_matrix and bias based on
    # the provided seed.
    # TODO(sibyl-vie3Poto): Storing the mapper's parameters (omega_matrix and bias) as
    # constants incurs no RPC calls to the parameter server during distributed
    # training. However, if the parameters grow too large (for instance if they
    # don't fit into memory or if they blow up the size of the GraphDef proto),
    # stroring them as constants is no longer an option. In this case, we should
    # have a heuristic to choose out of one of the following alternatives:
    # a) store them as variables (in the parameter server)
    # b) store them as worker local variables
    # c) generating on the fly the omega matrix at each step
    np.random.seed(self._seed)
    omega_matrix_shape = [self._input_dim, self._output_dim]
    bias_shape = [self._output_dim]

    omega_matrix = constant_op.constant(
        np.random.normal(
            scale=1.0 / self._stddev, size=omega_matrix_shape),
        dtype=dtypes.float32)
    bias = constant_op.constant(
        np.random.uniform(
            low=0.0, high=2 * np.pi, size=bias_shape),
        dtype=dtypes.float32)

    x_omega_plus_bias = math_ops.add(
        math_ops.matmul(input_tensor, omega_matrix), bias)
    return math.sqrt(2.0 / self._output_dim) * math_ops.cos(x_omega_plus_bias)
