# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Keras layers that implement explicit (approximate) kernel feature maps."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn

_SUPPORTED_RBF_KERNEL_TYPES = ['gaussian', 'laplacian']


class RandomFourierFeatures(base_layer.Layer):
  r"""Layer that maps its inputs using random Fourier features.

  This layer implements a feature map \\(\phi: \mathbb{R}^d \rightarrow
  \mathbb{R}^D\\) which approximates shift-invariant kernels. A kernel function
  K(x, y) defined over \\(\mathbb{R}^d x \mathbb{R}^d\\) is shift-invariant if
  K(x, y) = k(x-y) for some function defined over \\(\mathbb{R}^d\\). Many
  popular Radial Basis Functions (in short RBF), including gaussian and
  laplacian kernels are shift-invariant.

  The layer approximates a (shift invariant) kernel K in the following sense:
    up to a scaling factor, for all inputs \\(x, y \in \mathbb{R}^d\\)
        \\(\phi(x)^T \cdot \phi(y) \approx K(x, y)\\)

  The implementation of this layer is based on the following paper:
  "Random Features for Large-Scale Kernel Machines" by Ali Rahimi and Ben Recht.
  (link: https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf)

  The distribution from which the parameters of the random features map (layer)
  are sampled, determines which shift-invariant kernel the layer approximates
  (see paper for more details). The users can use the distribution of their
  choice. Due to their popularity, the layer supports the out-of-the-box
  approximation of the following RBF kernels:
  - Gaussian: \\(K(x, y) = e^{-\frac{\|x-y\|_2^2}{2 \cdot scale^2}}\\)
  - Laplacian: \\(K(x, y) = e^{-\frac{\|x-y\|_1}{scale}}\\)

  NOTE: Unlike the map described in the paper and the scikit-learn
  implementation, the output of this layer does not apply the sqrt(2/D)
  normalization factor.

  Usage for ML: Typically, this layer is used to "kernelize" linear models by
  applying a non-linear transformation (this layer) to the input features and
  then training a linear model on top of the transformed features. Depending on
  the loss function of the linear model, the composition of this layer and the
  linear model results to models that are equivalent (up to approximation) to
  kernel SVMs (for hinge loss), kernel logistic regression (for logistic loss),
  kernel linear regression (for squared loss) etc.

  Example of building a kernel multinomial logistic regression model with
  Gaussian kernel in keras:
  ```python
  random_features_layer = RandomFourierFeatures(
      output_dim=500,
      kernel_initializer='gaussian',
      scale=5.0,
      ...)

  model = tf.keras.models.Sequential()
  model.add(random_features_layer)
  model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax')

  model.compile(
    loss=tf.keras.losses.categorical_crossentropy, optimizer=..., metrics=...)
  ```

  To use another kernel, replace the layer creation command with:
  ```python
  random_features_layer = RandomFourierFeatures(
      output_dim=500,
      kernel_initializer=<my_initializer>,
      scale=...,
      ...)
  ```

  Arguments:
    output_dim: Positive integer, the dimension of the layer's output, i.e., the
      number of random features used to approximate the kernel.
    kernel_initializer: Determines the distribution of the parameters of the
      random features map (and therefore the kernel approximated by the layer).
      It can be either a string or an instance of TensorFlow's Initializer
      class. Currently only 'gaussian' and 'laplacian' are supported as string
      initializers (case insensitive). Note that these parameters are not
      trainable.
    scale: For gaussian and laplacian kernels, this corresponds to a scaling
      factor of the corresponding kernel approximated by the layer (see concrete
      definitions above). When provided, it should be a positive float. If None,
      the implementation chooses a default value (1.0 typically). Both the
      approximation error of the kernel and the classification quality are
      sensitive to this parameter. If trainable is set to True, this parameter
      is learned end-to-end during training and the provided value serves as an
      initialization value.
      NOTE: When this layer is used to map the initial features and then the
        transformed features are fed to a linear model, by making `scale`
        trainable, the resulting optimization problem is no longer convex (even
        if the loss function used by the linear model is convex).
    trainable: Whether the scaling parameter of th layer is trainable. Defaults
      to False.
    name: name for the RandomFourierFeatures layer.

  Raises:
    ValueError: if output_dim or stddev are not positive or if the provided
      kernel_initializer is not supported.
  """

  def __init__(self,
               output_dim,
               kernel_initializer='gaussian',
               scale=None,
               trainable=False,
               name=None,
               **kwargs):
    if output_dim <= 0:
      raise ValueError(
          '`output_dim` should be a positive integer. Given: {}.'.format(
              output_dim))
    if isinstance(kernel_initializer, six.string_types):
      if kernel_initializer.lower() not in _SUPPORTED_RBF_KERNEL_TYPES:
        raise ValueError(
            'Unsupported kernel type: \'{}\'. Supported kernel types: {}.'
            .format(kernel_initializer, _SUPPORTED_RBF_KERNEL_TYPES))
    if scale is not None and scale <= 0.0:
      raise ValueError('When provided, `scale` should be a positive float. '
                       'Given: {}.'.format(scale))
    super(RandomFourierFeatures, self).__init__(
        trainable=trainable, name=name, **kwargs)
    self.output_dim = output_dim
    self.kernel_initializer = kernel_initializer
    self.scale = scale

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    # TODO(sibyl-vie3Poto): Allow higher dimension inputs. Currently the input is expected
    # to have shape [batch_size, dimension].
    if input_shape.rank != 2:
      raise ValueError(
          'The rank of the input tensor should be 2. Got {} instead.'.format(
              input_shape.ndims))
    if input_shape.dims[1].value is None:
      raise ValueError(
          'The last dimension of the inputs to `RandomFourierFeatures` '
          'should be defined. Found `None`.')
    self.input_spec = input_spec.InputSpec(
        ndim=2, axes={1: input_shape.dims[1].value})
    input_dim = input_shape.dims[1].value

    kernel_initializer = _get_random_features_initializer(
        self.kernel_initializer, shape=(input_dim, self.output_dim))

    unscaled_kernel = self.add_weight(
        name='unscaled_random_features',
        shape=(input_dim, self.output_dim),
        dtype=dtypes.float32,
        initializer=kernel_initializer,
        trainable=False)

    self.bias = self.add_weight(
        name='random_features_bias',
        shape=(self.output_dim,),
        dtype=dtypes.float32,
        initializer=init_ops.random_uniform_initializer(
            minval=0.0, maxval=2 * np.pi, dtype=dtypes.float32),
        trainable=False)

    if self.scale is None:
      self.scale = _get_default_scale(self.kernel_initializer, input_dim)
    scale = self.add_weight(
        name='random_features_scale',
        shape=(1,),
        dtype=dtypes.float32,
        initializer=init_ops.constant_initializer(self.scale),
        trainable=True,
        constraint='NonNeg')
    self.kernel = (1.0 / scale) * unscaled_kernel
    super(RandomFourierFeatures, self).build(input_shape)

  def call(self, inputs):
    inputs = ops.convert_to_tensor_v2(inputs, dtype=self.dtype)
    inputs = gen_math_ops.cast(inputs, dtypes.float32)
    outputs = gen_math_ops.mat_mul(inputs, self.kernel)
    outputs = nn.bias_add(outputs, self.bias)
    return gen_math_ops.cos(outputs)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank(2)
    if input_shape.dims[-1].value is None:
      raise ValueError(
          'The innermost dimension of input shape must be defined. Given: %s' %
          input_shape)
    return input_shape[:-1].concatenate(self.output_dim)

  def get_config(self):
    kernel_initializer = self.kernel_initializer
    if isinstance(self.kernel_initializer, init_ops.Initializer):
      kernel_initializer = initializers.serialize(self.kernel_initializer)
    config = {
        'output_dim': self.output_dim,
        'kernel_initializer': kernel_initializer,
        'scale': self.scale,
    }
    base_config = super(RandomFourierFeatures, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def _get_random_features_initializer(initializer, shape):
  """Returns Initializer object for random features."""

  def _get_cauchy_samples(loc, scale, shape):
    probs = np.random.uniform(low=0., high=1., size=shape)
    return loc + scale * np.tan(np.pi * (probs - 0.5))

  random_features_initializer = initializer
  if isinstance(initializer, six.string_types):
    if initializer.lower() == 'gaussian':
      random_features_initializer = init_ops.random_normal_initializer(
          stddev=1.0)
    elif initializer.lower() == 'laplacian':
      random_features_initializer = init_ops.constant_initializer(
          _get_cauchy_samples(loc=0.0, scale=1.0, shape=shape))

    else:
      raise ValueError(
          'Unsupported kernel type: \'{}\'. Supported kernel types: {}.'.format(
              random_features_initializer, _SUPPORTED_RBF_KERNEL_TYPES))
  return random_features_initializer


def _get_default_scale(initializer, input_dim):
  if (isinstance(initializer, six.string_types) and
      initializer.lower() == 'gaussian'):
    return np.sqrt(input_dim / 2.0)
  return 1.0
