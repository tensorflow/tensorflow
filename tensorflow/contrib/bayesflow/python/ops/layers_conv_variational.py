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
"""Convolutional variational layer classes and their functional aliases.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.bayesflow.python.ops import layers_util
from tensorflow.contrib.distributions.python.ops import independent as independent_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_lib
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops.distributions import kullback_leibler as kl_lib
from tensorflow.python.ops.distributions import normal as normal_lib
from tensorflow.python.ops.distributions import util as distribution_util


class _ConvVariational(layers_lib.Layer):
  """Abstract nD convolution layer (private, used as implementation base).

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. It may also include a bias addition and activation function
  on the outputs. It assumes the `kernel` and/or `bias` are drawn from
  distributions.

  By default, the layer implements a stochastic forward pass via
  sampling from the kernel and bias posteriors,
  ```none
  outputs = f(inputs; kernel, bias), kernel, bias ~ posterior
  ```
  where f denotes the layer's calculation.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.

  Arguments:
    rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: An integer or tuple/list of n integers, specifying the
      length of the convolution window.
    strides: An integer or tuple/list of n integers,
      specifying the stride length of the convolution.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, ...)`.
    dilation_rate: An integer or tuple/list of n integers, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any `strides` value != 1.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    activity_regularizer: Optional regularizer function for the output.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    kernel_posterior_fn: Python `callable` which creates
      `tf.distributions.Distribution` instance representing the surrogate
      posterior of the `kernel` parameter. Default value:
      `default_mean_field_normal_fn()`.
    kernel_posterior_tensor_fn: Python `callable` which takes a
      `tf.distributions.Distribution` instance and returns a representative
      value. Default value: `lambda d: d.sample()`.
    kernel_prior_fn: Python `callable` which creates `tf.distributions`
      instance. See `default_mean_field_normal_fn` docstring for required
      parameter signature.
      Default value: `tf.distributions.Normal(loc=0., scale=1.)`.
    kernel_divergence_fn: Python `callable` which takes the surrogate posterior
      distribution, prior distribution and random variate sample(s) from the
      surrogate posterior and computes or approximates the KL divergence. The
      distributions are `tf.distributions.Distribution`-like instances and the
      sample is a `Tensor`.
    bias_posterior_fn: Python `callable` which creates
      `tf.distributions.Distribution` instance representing the surrogate
      posterior of the `bias` parameter. Default value:
      `default_mean_field_normal_fn(is_singular=True)` (which creates an
      instance of `tf.distributions.Deterministic`).
    bias_posterior_tensor_fn: Python `callable` which takes a
      `tf.distributions.Distribution` instance and returns a representative
      value. Default value: `lambda d: d.sample()`.
    bias_prior_fn: Python `callable` which creates `tf.distributions` instance.
      See `default_mean_field_normal_fn` docstring for required parameter
      signature. Default value: `None` (no prior, no variational inference)
    bias_divergence_fn: Python `callable` which takes the surrogate posterior
      distribution, prior distribution and random variate sample(s) from the
      surrogate posterior and computes or approximates the KL divergence. The
      distributions are `tf.distributions.Distribution`-like instances and the
      sample is a `Tensor`.
    name: A string, the name of the layer.

  Properties:
    rank: Python integer, dimensionality of convolution.
    filters: Python integer, dimensionality of the output space.
    kernel_size: Size of the convolution window.
    strides: Stride length of convolution.
    padding: Python string describing padding approach.
    data_format: Python string describing input data's dimensions.
    dilation_rate: Dilation rate for an atrous convolution.
    activation: Activation function (`callable`).
    activity_regularizer: Regularizer function for the output.
    kernel_posterior_fn: `callable` returning posterior.
    kernel_posterior_tensor_fn: `callable` operating on posterior.
    kernel_prior_fn: `callable` returning prior.
    kernel_divergence_fn: `callable` returning divergence.
    bias_posterior_fn: `callable` returning posterior.
    bias_posterior_tensor_fn: `callable` operating on posterior.
    bias_prior_fn: `callable` returning prior.
    bias_divergence_fn: `callable` returning divergence.
  """

  def __init__(
      self,
      rank,
      filters,
      kernel_size,
      strides=1,
      padding="valid",
      data_format="channels_last",
      dilation_rate=1,
      activation=None,
      activity_regularizer=None,
      trainable=True,
      kernel_posterior_fn=layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=lambda dtype, *args: normal_lib.Normal(  # pylint: disable=g-long-lambda
          loc=dtype.as_numpy_dtype(0.), scale=dtype.as_numpy_dtype(1.)),
      kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      bias_posterior_fn=layers_util.default_mean_field_normal_fn(is_singular=True),  # pylint: disable=line-too-long
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      name=None,
      **kwargs):
    super(_ConvVariational, self).__init__(
        trainable=trainable,
        name=name,
        activity_regularizer=activity_regularizer,
        **kwargs)
    self.rank = rank
    self.filters = filters
    self.kernel_size = utils.normalize_tuple(kernel_size, rank, "kernel_size")
    self.strides = utils.normalize_tuple(strides, rank, "strides")
    self.padding = utils.normalize_padding(padding)
    self.data_format = utils.normalize_data_format(data_format)
    self.dilation_rate = utils.normalize_tuple(
        dilation_rate, rank, "dilation_rate")
    self.activation = activation
    self.input_spec = layers_lib.InputSpec(ndim=self.rank + 2)
    self.kernel_posterior_fn = kernel_posterior_fn
    self.kernel_posterior_tensor_fn = kernel_posterior_tensor_fn
    self.kernel_prior_fn = kernel_prior_fn
    self.kernel_divergence_fn = kernel_divergence_fn
    self.bias_posterior_fn = bias_posterior_fn
    self.bias_posterior_tensor_fn = bias_posterior_tensor_fn
    self.bias_prior_fn = bias_prior_fn
    self.bias_divergence_fn = bias_divergence_fn

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if self.data_format == "channels_first":
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape[channel_axis].value is None:
      raise ValueError("The channel dimension of the inputs "
                       "should be defined. Found `None`.")
    input_dim = input_shape[channel_axis].value
    kernel_shape = self.kernel_size + (input_dim, self.filters)
    dtype = dtypes.as_dtype(self.dtype)

    # Must have a posterior kernel.
    self.kernel_posterior = self.kernel_posterior_fn(
        dtype, kernel_shape, "kernel_posterior",
        self.trainable, self.add_variable)

    if self.kernel_prior_fn is None:
      self.kernel_prior = None
    else:
      self.kernel_prior = self.kernel_prior_fn(
          dtype, kernel_shape, "kernel_prior",
          self.trainable, self.add_variable)
    self._built_kernel_divergence = False

    if self.bias_posterior_fn is None:
      self.bias_posterior = None
    else:
      self.bias_posterior = self.bias_posterior_fn(
          dtype, (self.filters,), "bias_posterior",
          self.trainable, self.add_variable)

    if self.bias_prior_fn is None:
      self.bias_prior = None
    else:
      self.bias_prior = self.bias_prior_fn(
          dtype, (self.filters,), "bias_prior",
          self.trainable, self.add_variable)
    self._built_bias_divergence = False

    self.input_spec = layers_lib.InputSpec(ndim=self.rank + 2,
                                           axes={channel_axis: input_dim})
    self._convolution_op = nn_ops.Convolution(
        input_shape,
        filter_shape=tensor_shape.TensorShape(kernel_shape),
        dilation_rate=self.dilation_rate,
        strides=self.strides,
        padding=self.padding.upper(),
        data_format=utils.convert_data_format(self.data_format,
                                              self.rank + 2))

    self.built = True

  def call(self, inputs):
    inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)

    outputs = self._apply_variational_kernel(inputs)
    outputs = self._apply_variational_bias(outputs)
    if self.activation is not None:
      outputs = self.activation(outputs)
    if not self._built_kernel_divergence:
      kernel_posterior = self.kernel_posterior
      kernel_prior = self.kernel_prior
      if isinstance(self.kernel_posterior, independent_lib.Independent):
        kernel_posterior = kernel_posterior.distribution
      if isinstance(self.kernel_prior, independent_lib.Independent):
        kernel_prior = kernel_prior.distribution
      self._apply_divergence(self.kernel_divergence_fn,
                             kernel_posterior,
                             kernel_prior,
                             self.kernel_posterior_tensor,
                             name="divergence_kernel")
      self._built_kernel_divergence = True
    if not self._built_bias_divergence:
      bias_posterior = self.bias_posterior
      bias_prior = self.bias_prior
      if isinstance(self.bias_posterior, independent_lib.Independent):
        bias_posterior = bias_posterior.distribution
      if isinstance(self.bias_prior, independent_lib.Independent):
        bias_prior = bias_prior.distribution
      self._apply_divergence(self.bias_divergence_fn,
                             bias_posterior,
                             bias_prior,
                             self.bias_posterior_tensor,
                             name="divergence_bias")
      self._built_bias_divergence = True
    return outputs

  def _apply_variational_bias(self, inputs):
    if self.bias_posterior is None:
      self.bias_posterior_tensor = None
      return inputs
    self.bias_posterior_tensor = self.bias_posterior_tensor_fn(
        self.bias_posterior)
    outputs = inputs
    if self.data_format == "channels_first":
      if self.rank == 1:
        # nn.bias_add does not accept a 1D input tensor.
        bias = array_ops.reshape(self.bias_posterior_tensor,
                                 (1, self.filters, 1))
        outputs += bias
      if self.rank == 2:
        outputs = nn.bias_add(outputs,
                              self.bias_posterior_tensor,
                              data_format="NCHW")
      if self.rank == 3:
        # As of Mar 2017, direct addition is significantly slower than
        # bias_add when computing gradients. To use bias_add, we collapse Z
        # and Y into a single dimension to obtain a 4D input tensor.
        outputs_shape = outputs.shape.as_list()
        outputs_4d = array_ops.reshape(outputs,
                                       [outputs_shape[0], outputs_shape[1],
                                        outputs_shape[2] * outputs_shape[3],
                                        outputs_shape[4]])
        outputs_4d = nn.bias_add(outputs_4d,
                                 self.bias_posterior_tensor,
                                 data_format="NCHW")
        outputs = array_ops.reshape(outputs_4d, outputs_shape)
    else:
      outputs = nn.bias_add(outputs,
                            self.bias_posterior_tensor,
                            data_format="NHWC")
    return outputs

  def _apply_divergence(self, divergence_fn, posterior, prior,
                        posterior_tensor, name):
    if (divergence_fn is None or
        posterior is None or
        prior is None):
      divergence = None
      return
    divergence = standard_ops.identity(
        divergence_fn(
            posterior, prior, posterior_tensor),
        name=name)
    self.add_loss(divergence)

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == "channels_last":
      space = input_shape[1:-1]
      new_space = []
      for i in range(len(space)):
        new_dim = utils.conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      return tensor_shape.TensorShape([input_shape[0]] + new_space +
                                      [self.filters])
    else:
      space = input_shape[2:]
      new_space = []
      for i in range(len(space)):
        new_dim = utils.conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      return tensor_shape.TensorShape([input_shape[0], self.filters] +
                                      new_space)


class _ConvReparameterization(_ConvVariational):
  """Abstract nD convolution layer (private, used as implementation base).

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. It may also include a bias addition and activation function
  on the outputs. It assumes the `kernel` and/or `bias` are drawn from
  distributions.

  By default, the layer implements a stochastic forward pass via
  sampling from the kernel and bias posteriors,
  ```none
  outputs = f(inputs; kernel, bias), kernel, bias ~ posterior
  ```
  where f denotes the layer's calculation. It uses the reparameterization
  estimator [1], which performs a Monte Carlo approximation of the
  distribution integrating over the `kernel` and `bias`.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.

  Arguments:
    rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: An integer or tuple/list of n integers, specifying the
      length of the convolution window.
    strides: An integer or tuple/list of n integers,
      specifying the stride length of the convolution.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, ...)`.
    dilation_rate: An integer or tuple/list of n integers, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any `strides` value != 1.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    activity_regularizer: Optional regularizer function for the output.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    kernel_posterior_fn: Python `callable` which creates
      `tf.distributions.Distribution` instance representing the surrogate
      posterior of the `kernel` parameter. Default value:
      `default_mean_field_normal_fn()`.
    kernel_posterior_tensor_fn: Python `callable` which takes a
      `tf.distributions.Distribution` instance and returns a representative
      value. Default value: `lambda d: d.sample()`.
    kernel_prior_fn: Python `callable` which creates `tf.distributions`
      instance. See `default_mean_field_normal_fn` docstring for required
      parameter signature.
      Default value: `tf.distributions.Normal(loc=0., scale=1.)`.
    kernel_divergence_fn: Python `callable` which takes the surrogate posterior
      distribution, prior distribution and random variate sample(s) from the
      surrogate posterior and computes or approximates the KL divergence. The
      distributions are `tf.distributions.Distribution`-like instances and the
      sample is a `Tensor`.
    bias_posterior_fn: Python `callable` which creates
      `tf.distributions.Distribution` instance representing the surrogate
      posterior of the `bias` parameter. Default value:
      `default_mean_field_normal_fn(is_singular=True)` (which creates an
      instance of `tf.distributions.Deterministic`).
    bias_posterior_tensor_fn: Python `callable` which takes a
      `tf.distributions.Distribution` instance and returns a representative
      value. Default value: `lambda d: d.sample()`.
    bias_prior_fn: Python `callable` which creates `tf.distributions` instance.
      See `default_mean_field_normal_fn` docstring for required parameter
      signature. Default value: `None` (no prior, no variational inference)
    bias_divergence_fn: Python `callable` which takes the surrogate posterior
      distribution, prior distribution and random variate sample(s) from the
      surrogate posterior and computes or approximates the KL divergence. The
      distributions are `tf.distributions.Distribution`-like instances and the
      sample is a `Tensor`.
    name: A string, the name of the layer.

  Properties:
    rank: Python integer, dimensionality of convolution.
    filters: Python integer, dimensionality of the output space.
    kernel_size: Size of the convolution window.
    strides: Stride length of convolution.
    padding: Python string describing padding approach.
    data_format: Python string describing input data's dimensions.
    dilation_rate: Dilation rate for an atrous convolution.
    activation: Activation function (`callable`).
    activity_regularizer: Regularizer function for the output.
    kernel_posterior_fn: `callable` returning posterior.
    kernel_posterior_tensor_fn: `callable` operating on posterior.
    kernel_prior_fn: `callable` returning prior.
    kernel_divergence_fn: `callable` returning divergence.
    bias_posterior_fn: `callable` returning posterior.
    bias_posterior_tensor_fn: `callable` operating on posterior.
    bias_prior_fn: `callable` returning prior.
    bias_divergence_fn: `callable` returning divergence.

  [1]: "Auto-Encoding Variational Bayes."
        Diederik P. Kingma, Max Welling.
        International Conference on Learning Representations, 2014.
  """

  def __init__(
      self,
      rank,
      filters,
      kernel_size,
      strides=1,
      padding="valid",
      data_format="channels_last",
      dilation_rate=1,
      activation=None,
      activity_regularizer=None,
      trainable=True,
      kernel_posterior_fn=layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=lambda dtype, *args: normal_lib.Normal(  # pylint: disable=g-long-lambda
          loc=dtype.as_numpy_dtype(0.), scale=dtype.as_numpy_dtype(1.)),
      kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      bias_posterior_fn=layers_util.default_mean_field_normal_fn(is_singular=True),  # pylint: disable=line-too-long
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      name=None,
      **kwargs):
    super(_ConvReparameterization, self).__init__(
        rank=rank,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        kernel_posterior_fn=kernel_posterior_fn,
        kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
        kernel_prior_fn=kernel_prior_fn,
        kernel_divergence_fn=kernel_divergence_fn,
        bias_posterior_fn=bias_posterior_fn,
        bias_posterior_tensor_fn=bias_posterior_tensor_fn,
        bias_prior_fn=bias_prior_fn,
        bias_divergence_fn=bias_divergence_fn,
        name=name, **kwargs)

  def _apply_variational_kernel(self, inputs):
    self.kernel_posterior_tensor = self.kernel_posterior_tensor_fn(
        self.kernel_posterior)
    self.kernel_posterior_affine = None
    self.kernel_posterior_affine_tensor = None
    outputs = self._convolution_op(inputs, self.kernel_posterior_tensor)
    return outputs


class Conv1DReparameterization(_ConvReparameterization):
  """1D convolution layer (e.g. temporal convolution).

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. It may also include a bias addition and activation function
  on the outputs. It assumes the `kernel` and/or `bias` are drawn from
  distributions.

  By default, the layer implements a stochastic forward pass via
  sampling from the kernel and bias posteriors,
  ```none
  outputs = f(inputs; kernel, bias), kernel, bias ~ posterior
  ```
  where f denotes the layer's calculation. It uses the reparameterization
  estimator [1], which performs a Monte Carlo approximation of the
  distribution integrating over the `kernel` and `bias`.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.

  Arguments:
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: An integer or tuple/list of a single integer, specifying the
      length of the 1D convolution window.
    strides: An integer or tuple/list of a single integer,
      specifying the stride length of the convolution.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, length, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, length)`.
    dilation_rate: An integer or tuple/list of a single integer, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any `strides` value != 1.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    activity_regularizer: Optional regularizer function for the output.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    kernel_posterior_fn: Python `callable` which creates
      `tf.distributions.Distribution` instance representing the surrogate
      posterior of the `kernel` parameter. Default value:
      `default_mean_field_normal_fn()`.
    kernel_posterior_tensor_fn: Python `callable` which takes a
      `tf.distributions.Distribution` instance and returns a representative
      value. Default value: `lambda d: d.sample()`.
    kernel_prior_fn: Python `callable` which creates `tf.distributions`
      instance. See `default_mean_field_normal_fn` docstring for required
      parameter signature.
      Default value: `tf.distributions.Normal(loc=0., scale=1.)`.
    kernel_divergence_fn: Python `callable` which takes the surrogate posterior
      distribution, prior distribution and random variate sample(s) from the
      surrogate posterior and computes or approximates the KL divergence. The
      distributions are `tf.distributions.Distribution`-like instances and the
      sample is a `Tensor`.
    bias_posterior_fn: Python `callable` which creates
      `tf.distributions.Distribution` instance representing the surrogate
      posterior of the `bias` parameter. Default value:
      `default_mean_field_normal_fn(is_singular=True)` (which creates an
      instance of `tf.distributions.Deterministic`).
    bias_posterior_tensor_fn: Python `callable` which takes a
      `tf.distributions.Distribution` instance and returns a representative
      value. Default value: `lambda d: d.sample()`.
    bias_prior_fn: Python `callable` which creates `tf.distributions` instance.
      See `default_mean_field_normal_fn` docstring for required parameter
      signature. Default value: `None` (no prior, no variational inference)
    bias_divergence_fn: Python `callable` which takes the surrogate posterior
      distribution, prior distribution and random variate sample(s) from the
      surrogate posterior and computes or approximates the KL divergence. The
      distributions are `tf.distributions.Distribution`-like instances and the
    name: A string, the name of the layer.

  Properties:
    filters: Python integer, dimensionality of the output space.
    kernel_size: Size of the convolution window.
    strides: Stride length of convolution.
    padding: Python string describing padding approach.
    data_format: Python string describing input data's dimensions.
    dilation_rate: Dilation rate for an atrous convolution.
    activation: Activation function (`callable`).
    activity_regularizer: Regularizer function for the output.
    kernel_posterior_fn: `callable` returning posterior.
    kernel_posterior_tensor_fn: `callable` operating on posterior.
    kernel_prior_fn: `callable` returning prior.
    kernel_divergence_fn: `callable` returning divergence.
    bias_posterior_fn: `callable` returning posterior.
    bias_posterior_tensor_fn: `callable` operating on posterior.
    bias_prior_fn: `callable` returning prior.
    bias_divergence_fn: `callable` returning divergence.

  #### Examples

  We illustrate a Bayesian neural network with [variational inference](
  https://en.wikipedia.org/wiki/Variational_Bayesian_methods),
  assuming a dataset of `features` and `labels`.

  ```python
  tfp = tf.contrib.bayesflow

  net = tf.reshape(features, [-1, 128, 1])
  net = tfp.layers.Conv1DReparameterization(64,
                                            kernel_size=5,
                                            padding="SAME",
                                            activation=tf.nn.relu)(net)
  net = tf.reshape(net, [-1, 128 * 64])
  logits = tfp.layers.DenseReparameterization(10)(net)
  neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  kl = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
  loss = neg_log_likelihood + kl
  train_op = tf.train.AdamOptimizer().minimize(loss)
  ```

  It uses reparameterization gradients to minimize the
  Kullback-Leibler divergence up to a constant, also known as the
  negative Evidence Lower Bound. It consists of the sum of two terms:
  the expected negative log-likelihood, which we approximate via
  Monte Carlo; and the KL divergence, which is added via regularizer
  terms which are arguments to the layer.

  [1]: "Auto-Encoding Variational Bayes."
        Diederik P. Kingma, Max Welling.
        International Conference on Learning Representations, 2014.
  """

  def __init__(
      self,
      filters,
      kernel_size,
      strides=1,
      padding="valid",
      data_format="channels_last",
      dilation_rate=1,
      activation=None,
      activity_regularizer=None,
      trainable=True,
      kernel_posterior_fn=layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=lambda dtype, *args: normal_lib.Normal(  # pylint: disable=g-long-lambda
          loc=dtype.as_numpy_dtype(0.), scale=dtype.as_numpy_dtype(1.)),
      kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      bias_posterior_fn=layers_util.default_mean_field_normal_fn(is_singular=True),  # pylint: disable=line-too-long
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      name=None,
      **kwargs):
    super(Conv1DReparameterization, self).__init__(
        rank=1,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        kernel_posterior_fn=kernel_posterior_fn,
        kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
        kernel_prior_fn=kernel_prior_fn,
        kernel_divergence_fn=kernel_divergence_fn,
        bias_posterior_fn=bias_posterior_fn,
        bias_posterior_tensor_fn=bias_posterior_tensor_fn,
        bias_prior_fn=bias_prior_fn,
        bias_divergence_fn=bias_divergence_fn,
        name=name, **kwargs)


def conv1d_reparameterization(
    inputs,
    filters,
    kernel_size,
    strides=1,
    padding="valid",
    data_format="channels_last",
    dilation_rate=1,
    activation=None,
    activity_regularizer=None,
    trainable=True,
    kernel_posterior_fn=layers_util.default_mean_field_normal_fn(),
    kernel_posterior_tensor_fn=lambda d: d.sample(),
    kernel_prior_fn=lambda dtype, *args: normal_lib.Normal(  # pylint: disable=g-long-lambda
        loc=dtype.as_numpy_dtype(0.), scale=dtype.as_numpy_dtype(1.)),
    kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
    bias_posterior_fn=layers_util.default_mean_field_normal_fn(is_singular=True),  # pylint: disable=line-too-long
    bias_posterior_tensor_fn=lambda d: d.sample(),
    bias_prior_fn=None,
    bias_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
    name=None,
    reuse=None):
  """Functional interface for 1D convolution layer (e.g. temporal convolution).

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. It may also include a bias addition and activation function
  on the outputs. It assumes the `kernel` and/or `bias` are drawn from
  distributions.

  By default, the layer implements a stochastic forward pass via
  sampling from the kernel and bias posteriors,
  ```none
  outputs = f(inputs; kernel, bias), kernel, bias ~ posterior
  ```
  where f denotes the layer's calculation. It uses the reparameterization
  estimator [1], which performs a Monte Carlo approximation of the
  distribution integrating over the `kernel` and `bias`.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.

  Arguments:
    inputs: Tensor input.
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: An integer or tuple/list of a single integer, specifying the
      length of the 1D convolution window.
    strides: An integer or tuple/list of a single integer,
      specifying the stride length of the convolution.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, length, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, length)`.
    dilation_rate: An integer or tuple/list of a single integer, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any `strides` value != 1.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    activity_regularizer: Optional regularizer function for the output.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    kernel_posterior_fn: Python `callable` which creates
      `tf.distributions.Distribution` instance representing the surrogate
      posterior of the `kernel` parameter. Default value:
      `default_mean_field_normal_fn()`.
    kernel_posterior_tensor_fn: Python `callable` which takes a
      `tf.distributions.Distribution` instance and returns a representative
      value. Default value: `lambda d: d.sample()`.
    kernel_prior_fn: Python `callable` which creates `tf.distributions`
      instance. See `default_mean_field_normal_fn` docstring for required
      parameter signature.
      Default value: `tf.distributions.Normal(loc=0., scale=1.)`.
    kernel_divergence_fn: Python `callable` which takes the surrogate posterior
      distribution, prior distribution and random variate sample(s) from the
      surrogate posterior and computes or approximates the KL divergence. The
      distributions are `tf.distributions.Distribution`-like instances and the
      sample is a `Tensor`.
    bias_posterior_fn: Python `callable` which creates
      `tf.distributions.Distribution` instance representing the surrogate
      posterior of the `bias` parameter. Default value:
      `default_mean_field_normal_fn(is_singular=True)` (which creates an
      instance of `tf.distributions.Deterministic`).
    bias_posterior_tensor_fn: Python `callable` which takes a
      `tf.distributions.Distribution` instance and returns a representative
      value. Default value: `lambda d: d.sample()`.
    bias_prior_fn: Python `callable` which creates `tf.distributions` instance.
      See `default_mean_field_normal_fn` docstring for required parameter
      signature. Default value: `None` (no prior, no variational inference)
    bias_divergence_fn: Python `callable` which takes the surrogate posterior
      distribution, prior distribution and random variate sample(s) from the
      surrogate posterior and computes or approximates the KL divergence. The
      distributions are `tf.distributions.Distribution`-like instances and the
    name: A string, the name of the layer.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

  Returns:
    Output tensor.

  Raises:
    ValueError: if eager execution is enabled.

  #### Examples

  We illustrate a Bayesian neural network with [variational inference](
  https://en.wikipedia.org/wiki/Variational_Bayesian_methods),
  assuming a dataset of `features` and `labels`.

  ```python
  tfp = tf.contrib.bayesflow

  net = tf.reshape(features, [-1, 128, 1])
  net = tfp.layers.conv1d_reparameterization(net,
                                             filters=64,
                                             kernel_size=5,
                                             padding="SAME",
                                             activation=tf.nn.relu)
  net = tf.reshape(net, [-1, 128 * 64])
  logits = tfp.layers.dense_reparameterization(net, 10)
  neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  kl = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
  loss = neg_log_likelihood + kl
  train_op = tf.train.AdamOptimizer().minimize(loss)
  ```

  It uses reparameterization gradients to minimize the
  Kullback-Leibler divergence up to a constant, also known as the
  negative Evidence Lower Bound. It consists of the sum of two terms:
  the expected negative log-likelihood, which we approximate via
  Monte Carlo; and the KL divergence, which is added via regularizer
  terms which are arguments to the layer.

  [1]: "Auto-Encoding Variational Bayes."
        Diederik P. Kingma, Max Welling.
        International Conference on Learning Representations, 2014.
  """
  layer = Conv1DReparameterization(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilation_rate=dilation_rate,
      activation=activation,
      activity_regularizer=activity_regularizer,
      trainable=trainable,
      kernel_posterior_fn=kernel_posterior_fn,
      kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
      kernel_prior_fn=kernel_prior_fn,
      kernel_divergence_fn=kernel_divergence_fn,
      bias_posterior_fn=bias_posterior_fn,
      bias_posterior_tensor_fn=bias_posterior_tensor_fn,
      bias_prior_fn=bias_prior_fn,
      bias_divergence_fn=bias_divergence_fn,
      name=name,
      dtype=inputs.dtype.base_dtype,
      _scope=name,
      _reuse=reuse)
  return layer.apply(inputs)


class Conv2DReparameterization(_ConvReparameterization):
  """2D convolution layer (e.g. spatial convolution over images).

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. It may also include a bias addition and activation function
  on the outputs. It assumes the `kernel` and/or `bias` are drawn from
  distributions.

  By default, the layer implements a stochastic forward pass via
  sampling from the kernel and bias posteriors,
  ```none
  outputs = f(inputs; kernel, bias), kernel, bias ~ posterior
  ```
  where f denotes the layer's calculation. It uses the reparameterization
  estimator [1], which performs a Monte Carlo approximation of the
  distribution integrating over the `kernel` and `bias`.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.

  Arguments:
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: An integer or tuple/list of 2 integers, specifying the
      height and width of the 2D convolution window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the convolution along the height and width.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, height, width)`.

    dilation_rate: An integer or tuple/list of 2 integers, specifying
      the dilation rate to use for dilated convolution.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any stride value != 1.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    activity_regularizer: Optional regularizer function for the output.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    kernel_posterior_fn: Python `callable` which creates
      `tf.distributions.Distribution` instance representing the surrogate
      posterior of the `kernel` parameter. Default value:
      `default_mean_field_normal_fn()`.
    kernel_posterior_tensor_fn: Python `callable` which takes a
      `tf.distributions.Distribution` instance and returns a representative
      value. Default value: `lambda d: d.sample()`.
    kernel_prior_fn: Python `callable` which creates `tf.distributions`
      instance. See `default_mean_field_normal_fn` docstring for required
      parameter signature.
      Default value: `tf.distributions.Normal(loc=0., scale=1.)`.
    kernel_divergence_fn: Python `callable` which takes the surrogate posterior
      distribution, prior distribution and random variate sample(s) from the
      surrogate posterior and computes or approximates the KL divergence. The
      distributions are `tf.distributions.Distribution`-like instances and the
      sample is a `Tensor`.
    bias_posterior_fn: Python `callable` which creates
      `tf.distributions.Distribution` instance representing the surrogate
      posterior of the `bias` parameter. Default value:
      `default_mean_field_normal_fn(is_singular=True)` (which creates an
      instance of `tf.distributions.Deterministic`).
    bias_posterior_tensor_fn: Python `callable` which takes a
      `tf.distributions.Distribution` instance and returns a representative
      value. Default value: `lambda d: d.sample()`.
    bias_prior_fn: Python `callable` which creates `tf.distributions` instance.
      See `default_mean_field_normal_fn` docstring for required parameter
      signature. Default value: `None` (no prior, no variational inference)
    bias_divergence_fn: Python `callable` which takes the surrogate posterior
      distribution, prior distribution and random variate sample(s) from the
      surrogate posterior and computes or approximates the KL divergence. The
      distributions are `tf.distributions.Distribution`-like instances and the
    name: A string, the name of the layer.

  Properties:
    filters: Python integer, dimensionality of the output space.
    kernel_size: Size of the convolution window.
    strides: Stride length of convolution.
    padding: Python string describing padding approach.
    data_format: Python string describing input data's dimensions.
    dilation_rate: Dilation rate for an atrous convolution.
    activation: Activation function (`callable`).
    activity_regularizer: Regularizer function for the output.
    kernel_posterior_fn: `callable` returning posterior.
    kernel_posterior_tensor_fn: `callable` operating on posterior.
    kernel_prior_fn: `callable` returning prior.
    kernel_divergence_fn: `callable` returning divergence.
    bias_posterior_fn: `callable` returning posterior.
    bias_posterior_tensor_fn: `callable` operating on posterior.
    bias_prior_fn: `callable` returning prior.
    bias_divergence_fn: `callable` returning divergence.

  #### Examples

  We illustrate a Bayesian neural network with [variational inference](
  https://en.wikipedia.org/wiki/Variational_Bayesian_methods),
  assuming a dataset of `features` and `labels`.

  ```python
  tfp = tf.contrib.bayesflow

  net = tf.reshape(features, [-1, 32, 32, 3])
  net = tfp.layers.Conv2DReparameterization(64,
                                            kernel_size=5,
                                            padding="SAME",
                                            activation=tf.nn.relu)(net)
  net = tf.layers.MaxPooling2D(pool_size=2,
                               strides=2,
                               padding="SAME")(net)
  net = tf.reshape(net, [-1, 8 * 8 * 64])
  logits = tfp.layers.DenseReparameterization(10)(net)
  neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  kl = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
  loss = neg_log_likelihood + kl
  train_op = tf.train.AdamOptimizer().minimize(loss)
  ```

  It uses reparameterization gradients to minimize the
  Kullback-Leibler divergence up to a constant, also known as the
  negative Evidence Lower Bound. It consists of the sum of two terms:
  the expected negative log-likelihood, which we approximate via
  Monte Carlo; and the KL divergence, which is added via regularizer
  terms which are arguments to the layer.

  [1]: "Auto-Encoding Variational Bayes."
        Diederik P. Kingma, Max Welling.
        International Conference on Learning Representations, 2014.
  """

  def __init__(
      self,
      filters,
      kernel_size,
      strides=(1, 1),
      padding="valid",
      data_format="channels_last",
      dilation_rate=(1, 1),
      activation=None,
      activity_regularizer=None,
      trainable=True,
      kernel_posterior_fn=layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=lambda dtype, *args: normal_lib.Normal(  # pylint: disable=g-long-lambda
          loc=dtype.as_numpy_dtype(0.), scale=dtype.as_numpy_dtype(1.)),
      kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      bias_posterior_fn=layers_util.default_mean_field_normal_fn(is_singular=True),  # pylint: disable=line-too-long
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      name=None,
      **kwargs):
    super(Conv2DReparameterization, self).__init__(
        rank=2,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        kernel_posterior_fn=kernel_posterior_fn,
        kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
        kernel_prior_fn=kernel_prior_fn,
        kernel_divergence_fn=kernel_divergence_fn,
        bias_posterior_fn=bias_posterior_fn,
        bias_posterior_tensor_fn=bias_posterior_tensor_fn,
        bias_prior_fn=bias_prior_fn,
        bias_divergence_fn=bias_divergence_fn,
        name=name, **kwargs)


def conv2d_reparameterization(
    inputs,
    filters,
    kernel_size,
    strides=(1, 1),
    padding="valid",
    data_format="channels_last",
    dilation_rate=(1, 1),
    activation=None,
    activity_regularizer=None,
    trainable=True,
    kernel_posterior_fn=layers_util.default_mean_field_normal_fn(),
    kernel_posterior_tensor_fn=lambda d: d.sample(),
    kernel_prior_fn=lambda dtype, *args: normal_lib.Normal(  # pylint: disable=g-long-lambda
        loc=dtype.as_numpy_dtype(0.), scale=dtype.as_numpy_dtype(1.)),
    kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
    bias_posterior_fn=layers_util.default_mean_field_normal_fn(is_singular=True),  # pylint: disable=line-too-long
    bias_posterior_tensor_fn=lambda d: d.sample(),
    bias_prior_fn=None,
    bias_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
    name=None,
    reuse=None):
  """Functional interface for the 2D convolution layer.

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. It may also include a bias addition and activation function
  on the outputs. It assumes the `kernel` and/or `bias` are drawn from
  distributions.

  By default, the layer implements a stochastic forward pass via
  sampling from the kernel and bias posteriors,
  ```none
  outputs = f(inputs; kernel, bias), kernel, bias ~ posterior
  ```
  where f denotes the layer's calculation. It uses the reparameterization
  estimator [1], which performs a Monte Carlo approximation of the
  distribution integrating over the `kernel` and `bias`.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.

  Arguments:
    inputs: Tensor input.
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: An integer or tuple/list of 2 integers, specifying the
      height and width of the 2D convolution window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the convolution along the height and width.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, height, width)`.

    dilation_rate: An integer or tuple/list of 2 integers, specifying
      the dilation rate to use for dilated convolution.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any stride value != 1.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    activity_regularizer: Optional regularizer function for the output.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    kernel_posterior_fn: Python `callable` which creates
      `tf.distributions.Distribution` instance representing the surrogate
      posterior of the `kernel` parameter. Default value:
      `default_mean_field_normal_fn()`.
    kernel_posterior_tensor_fn: Python `callable` which takes a
      `tf.distributions.Distribution` instance and returns a representative
      value. Default value: `lambda d: d.sample()`.
    kernel_prior_fn: Python `callable` which creates `tf.distributions`
      instance. See `default_mean_field_normal_fn` docstring for required
      parameter signature.
      Default value: `tf.distributions.Normal(loc=0., scale=1.)`.
    kernel_divergence_fn: Python `callable` which takes the surrogate posterior
      distribution, prior distribution and random variate sample(s) from the
      surrogate posterior and computes or approximates the KL divergence. The
      distributions are `tf.distributions.Distribution`-like instances and the
      sample is a `Tensor`.
    bias_posterior_fn: Python `callable` which creates
      `tf.distributions.Distribution` instance representing the surrogate
      posterior of the `bias` parameter. Default value:
      `default_mean_field_normal_fn(is_singular=True)` (which creates an
      instance of `tf.distributions.Deterministic`).
    bias_posterior_tensor_fn: Python `callable` which takes a
      `tf.distributions.Distribution` instance and returns a representative
      value. Default value: `lambda d: d.sample()`.
    bias_prior_fn: Python `callable` which creates `tf.distributions` instance.
      See `default_mean_field_normal_fn` docstring for required parameter
      signature. Default value: `None` (no prior, no variational inference)
    bias_divergence_fn: Python `callable` which takes the surrogate posterior
      distribution, prior distribution and random variate sample(s) from the
      surrogate posterior and computes or approximates the KL divergence. The
      distributions are `tf.distributions.Distribution`-like instances and the
    name: A string, the name of the layer.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

  Returns:
    Output tensor.

  Raises:
    ValueError: if eager execution is enabled.

  #### Examples

  We illustrate a Bayesian neural network with [variational inference](
  https://en.wikipedia.org/wiki/Variational_Bayesian_methods),
  assuming a dataset of `features` and `labels`.

  ```python
  tfp = tf.contrib.bayesflow

  net = tf.reshape(features, [-1, 32, 32, 3])
  net = tfp.layers.conv2d_reparameterization(net,
                                             filters=64,
                                             kernel_size=5,
                                             padding="SAME",
                                             activation=tf.nn.relu)
  net = tf.layers.max_pooling2d(net,
                                pool_size=2,
                                strides=2,
                                padding="SAME")
  net = tf.reshape(net, [-1, 8 * 8 * 64])
  logits = tfp.layers.dense_reparameterization(net, 10)
  neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  kl = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
  loss = neg_log_likelihood + kl
  train_op = tf.train.AdamOptimizer().minimize(loss)
  ```

  It uses reparameterization gradients to minimize the
  Kullback-Leibler divergence up to a constant, also known as the
  negative Evidence Lower Bound. It consists of the sum of two terms:
  the expected negative log-likelihood, which we approximate via
  Monte Carlo; and the KL divergence, which is added via regularizer
  terms which are arguments to the layer.

  [1]: "Auto-Encoding Variational Bayes."
        Diederik P. Kingma, Max Welling.
        International Conference on Learning Representations, 2014.
  """
  layer = Conv2DReparameterization(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilation_rate=dilation_rate,
      activation=activation,
      activity_regularizer=activity_regularizer,
      trainable=trainable,
      kernel_posterior_fn=kernel_posterior_fn,
      kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
      kernel_prior_fn=kernel_prior_fn,
      kernel_divergence_fn=kernel_divergence_fn,
      bias_posterior_fn=bias_posterior_fn,
      bias_posterior_tensor_fn=bias_posterior_tensor_fn,
      bias_prior_fn=bias_prior_fn,
      bias_divergence_fn=bias_divergence_fn,
      name=name,
      dtype=inputs.dtype.base_dtype,
      _scope=name,
      _reuse=reuse)
  return layer.apply(inputs)


class Conv3DReparameterization(_ConvReparameterization):
  """3D convolution layer (e.g. spatial convolution over volumes).

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. It may also include a bias addition and activation function
  on the outputs. It assumes the `kernel` and/or `bias` are drawn from
  distributions.

  By default, the layer implements a stochastic forward pass via
  sampling from the kernel and bias posteriors,
  ```none
  outputs = f(inputs; kernel, bias), kernel, bias ~ posterior
  ```
  where f denotes the layer's calculation. It uses the reparameterization
  estimator [1], which performs a Monte Carlo approximation of the
  distribution integrating over the `kernel` and `bias`.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.

  Arguments:
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: An integer or tuple/list of 3 integers, specifying the
      depth, height and width of the 3D convolution window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 3 integers,
      specifying the strides of the convolution along the depth,
      height and width.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, depth, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch, channels, depth, height, width)`.
    dilation_rate: An integer or tuple/list of 3 integers, specifying
      the dilation rate to use for dilated convolution.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any stride value != 1.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    activity_regularizer: Optional regularizer function for the output.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    kernel_posterior_fn: Python `callable` which creates
      `tf.distributions.Distribution` instance representing the surrogate
      posterior of the `kernel` parameter. Default value:
      `default_mean_field_normal_fn()`.
    kernel_posterior_tensor_fn: Python `callable` which takes a
      `tf.distributions.Distribution` instance and returns a representative
      value. Default value: `lambda d: d.sample()`.
    kernel_prior_fn: Python `callable` which creates `tf.distributions`
      instance. See `default_mean_field_normal_fn` docstring for required
      parameter signature.
      Default value: `tf.distributions.Normal(loc=0., scale=1.)`.
    kernel_divergence_fn: Python `callable` which takes the surrogate posterior
      distribution, prior distribution and random variate sample(s) from the
      surrogate posterior and computes or approximates the KL divergence. The
      distributions are `tf.distributions.Distribution`-like instances and the
      sample is a `Tensor`.
    bias_posterior_fn: Python `callable` which creates
      `tf.distributions.Distribution` instance representing the surrogate
      posterior of the `bias` parameter. Default value:
      `default_mean_field_normal_fn(is_singular=True)` (which creates an
      instance of `tf.distributions.Deterministic`).
    bias_posterior_tensor_fn: Python `callable` which takes a
      `tf.distributions.Distribution` instance and returns a representative
      value. Default value: `lambda d: d.sample()`.
    bias_prior_fn: Python `callable` which creates `tf.distributions` instance.
      See `default_mean_field_normal_fn` docstring for required parameter
      signature. Default value: `None` (no prior, no variational inference)
    bias_divergence_fn: Python `callable` which takes the surrogate posterior
      distribution, prior distribution and random variate sample(s) from the
      surrogate posterior and computes or approximates the KL divergence. The
      distributions are `tf.distributions.Distribution`-like instances and the
    name: A string, the name of the layer.

  Properties:
    filters: Python integer, dimensionality of the output space.
    kernel_size: Size of the convolution window.
    strides: Stride length of convolution.
    padding: Python string describing padding approach.
    data_format: Python string describing input data's dimensions.
    dilation_rate: Dilation rate for an atrous convolution.
    activation: Activation function (`callable`).
    activity_regularizer: Regularizer function for the output.
    kernel_posterior_fn: `callable` returning posterior.
    kernel_posterior_tensor_fn: `callable` operating on posterior.
    kernel_prior_fn: `callable` returning prior.
    kernel_divergence_fn: `callable` returning divergence.
    bias_posterior_fn: `callable` returning posterior.
    bias_posterior_tensor_fn: `callable` operating on posterior.
    bias_prior_fn: `callable` returning prior.
    bias_divergence_fn: `callable` returning divergence.

  #### Examples

  We illustrate a Bayesian neural network with [variational inference](
  https://en.wikipedia.org/wiki/Variational_Bayesian_methods),
  assuming a dataset of `features` and `labels`.

  ```python
  tfp = tf.contrib.bayesflow

  net = tf.reshape(features, [-1, 256, 32, 32, 3])
  net = tfp.layers.Conv3DReparameterization(64,
                                            kernel_size=5,
                                            padding="SAME",
                                            activation=tf.nn.relu)(net)
  net = tf.layers.MaxPooling2D(pool_size=2,
                               strides=2,
                               padding="SAME")(net)
  net = tf.reshape(net, [-1, 256 * 8 * 8 * 64])
  logits = tfp.layers.DenseReparameterization(10)(net)
  neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  kl = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
  loss = neg_log_likelihood + kl
  train_op = tf.train.AdamOptimizer().minimize(loss)
  ```

  It uses reparameterization gradients to minimize the
  Kullback-Leibler divergence up to a constant, also known as the
  negative Evidence Lower Bound. It consists of the sum of two terms:
  the expected negative log-likelihood, which we approximate via
  Monte Carlo; and the KL divergence, which is added via regularizer
  terms which are arguments to the layer.

  [1]: "Auto-Encoding Variational Bayes."
        Diederik P. Kingma, Max Welling.
        International Conference on Learning Representations, 2014.
  """

  def __init__(
      self,
      filters,
      kernel_size,
      strides=(1, 1, 1),
      padding="valid",
      data_format="channels_last",
      dilation_rate=(1, 1, 1),
      activation=None,
      activity_regularizer=None,
      trainable=True,
      kernel_posterior_fn=layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=lambda dtype, *args: normal_lib.Normal(  # pylint: disable=g-long-lambda
          loc=dtype.as_numpy_dtype(0.), scale=dtype.as_numpy_dtype(1.)),
      kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      bias_posterior_fn=layers_util.default_mean_field_normal_fn(is_singular=True),  # pylint: disable=line-too-long
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      name=None,
      **kwargs):
    super(Conv3DReparameterization, self).__init__(
        rank=3,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        kernel_posterior_fn=kernel_posterior_fn,
        kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
        kernel_prior_fn=kernel_prior_fn,
        kernel_divergence_fn=kernel_divergence_fn,
        bias_posterior_fn=bias_posterior_fn,
        bias_posterior_tensor_fn=bias_posterior_tensor_fn,
        bias_prior_fn=bias_prior_fn,
        bias_divergence_fn=bias_divergence_fn,
        name=name, **kwargs)


def conv3d_reparameterization(
    inputs,
    filters,
    kernel_size,
    strides=(1, 1, 1),
    padding="valid",
    data_format="channels_last",
    dilation_rate=(1, 1, 1),
    activation=None,
    activity_regularizer=None,
    trainable=True,
    kernel_posterior_fn=layers_util.default_mean_field_normal_fn(),
    kernel_posterior_tensor_fn=lambda d: d.sample(),
    kernel_prior_fn=lambda dtype, *args: normal_lib.Normal(  # pylint: disable=g-long-lambda
        loc=dtype.as_numpy_dtype(0.), scale=dtype.as_numpy_dtype(1.)),
    kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
    bias_posterior_fn=layers_util.default_mean_field_normal_fn(is_singular=True),  # pylint: disable=line-too-long
    bias_posterior_tensor_fn=lambda d: d.sample(),
    bias_prior_fn=None,
    bias_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
    name=None,
    reuse=None):
  """Functional interface for the 3D convolution layer.

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. It may also include a bias addition and activation function
  on the outputs. It assumes the `kernel` and/or `bias` are drawn from
  distributions.

  By default, the layer implements a stochastic forward pass via
  sampling from the kernel and bias posteriors,
  ```none
  outputs = f(inputs; kernel, bias), kernel, bias ~ posterior
  ```
  where f denotes the layer's calculation. It uses the reparameterization
  estimator [1], which performs a Monte Carlo approximation of the
  distribution integrating over the `kernel` and `bias`.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.

  Arguments:
    inputs: Tensor input.
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: An integer or tuple/list of 3 integers, specifying the
      depth, height and width of the 3D convolution window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 3 integers,
      specifying the strides of the convolution along the depth,
      height and width.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, depth, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch, channels, depth, height, width)`.
    dilation_rate: An integer or tuple/list of 3 integers, specifying
      the dilation rate to use for dilated convolution.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any stride value != 1.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    activity_regularizer: Optional regularizer function for the output.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    kernel_posterior_fn: Python `callable` which creates
      `tf.distributions.Distribution` instance representing the surrogate
      posterior of the `kernel` parameter. Default value:
      `default_mean_field_normal_fn()`.
    kernel_posterior_tensor_fn: Python `callable` which takes a
      `tf.distributions.Distribution` instance and returns a representative
      value. Default value: `lambda d: d.sample()`.
    kernel_prior_fn: Python `callable` which creates `tf.distributions`
      instance. See `default_mean_field_normal_fn` docstring for required
      parameter signature.
      Default value: `tf.distributions.Normal(loc=0., scale=1.)`.
    kernel_divergence_fn: Python `callable` which takes the surrogate posterior
      distribution, prior distribution and random variate sample(s) from the
      surrogate posterior and computes or approximates the KL divergence. The
      distributions are `tf.distributions.Distribution`-like instances and the
      sample is a `Tensor`.
    bias_posterior_fn: Python `callable` which creates
      `tf.distributions.Distribution` instance representing the surrogate
      posterior of the `bias` parameter. Default value:
      `default_mean_field_normal_fn(is_singular=True)` (which creates an
      instance of `tf.distributions.Deterministic`).
    bias_posterior_tensor_fn: Python `callable` which takes a
      `tf.distributions.Distribution` instance and returns a representative
      value. Default value: `lambda d: d.sample()`.
    bias_prior_fn: Python `callable` which creates `tf.distributions` instance.
      See `default_mean_field_normal_fn` docstring for required parameter
      signature. Default value: `None` (no prior, no variational inference)
    bias_divergence_fn: Python `callable` which takes the surrogate posterior
      distribution, prior distribution and random variate sample(s) from the
      surrogate posterior and computes or approximates the KL divergence. The
      distributions are `tf.distributions.Distribution`-like instances and the
    name: A string, the name of the layer.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

  Returns:
    Output tensor.

  Raises:
    ValueError: if eager execution is enabled.

  #### Examples

  We illustrate a Bayesian neural network with [variational inference](
  https://en.wikipedia.org/wiki/Variational_Bayesian_methods),
  assuming a dataset of `features` and `labels`.

  ```python
  tfp = tf.contrib.bayesflow

  net = tf.reshape(features, [-1, 256, 32, 32, 3])
  net = tfp.layers.conv3d_reparameterization(net,
                                             filters=64,
                                             kernel_size=5,
                                             padding="SAME",
                                             activation=tf.nn.relu)
  net = tf.layers.max_pooling2d(net,
                                pool_size=2,
                                strides=2,
                                padding="SAME")
  net = tf.reshape(net, [-1, 256 * 8 * 8 * 64])
  logits = tfp.layers.dense_reparameterization(net, 10)
  neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  kl = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
  loss = neg_log_likelihood + kl
  train_op = tf.train.AdamOptimizer().minimize(loss)
  ```

  It uses reparameterization gradients to minimize the
  Kullback-Leibler divergence up to a constant, also known as the
  negative Evidence Lower Bound. It consists of the sum of two terms:
  the expected negative log-likelihood, which we approximate via
  Monte Carlo; and the KL divergence, which is added via regularizer
  terms which are arguments to the layer.

  [1]: "Auto-Encoding Variational Bayes."
        Diederik P. Kingma, Max Welling.
        International Conference on Learning Representations, 2014.
  """
  layer = Conv3DReparameterization(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilation_rate=dilation_rate,
      activation=activation,
      activity_regularizer=activity_regularizer,
      trainable=trainable,
      kernel_posterior_fn=kernel_posterior_fn,
      kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
      kernel_prior_fn=kernel_prior_fn,
      kernel_divergence_fn=kernel_divergence_fn,
      bias_posterior_fn=bias_posterior_fn,
      bias_posterior_tensor_fn=bias_posterior_tensor_fn,
      bias_prior_fn=bias_prior_fn,
      bias_divergence_fn=bias_divergence_fn,
      name=name,
      dtype=inputs.dtype.base_dtype,
      _scope=name,
      _reuse=reuse)
  return layer.apply(inputs)


class _ConvFlipout(_ConvVariational):
  """Abstract nD convolution layer (private, used as implementation base).

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. It may also include a bias addition and activation function
  on the outputs. It assumes the `kernel` and/or `bias` are drawn from
  distributions.

  By default, the layer implements a stochastic forward pass via
  sampling from the kernel and bias posteriors,
  ```none
  outputs = f(inputs; kernel, bias), kernel, bias ~ posterior
  ```
  where f denotes the layer's calculation. It uses the Flipout
  estimator [1], which performs a Monte Carlo approximation of the
  distribution integrating over the `kernel` and `bias`. Flipout uses
  roughly twice as many floating point operations as the
  reparameterization estimator but has the advantage of significantly
  lower variance.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.

  Arguments:
    rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: An integer or tuple/list of n integers, specifying the
      length of the convolution window.
    strides: An integer or tuple/list of n integers,
      specifying the stride length of the convolution.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, ...)`.
    dilation_rate: An integer or tuple/list of n integers, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any `strides` value != 1.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    activity_regularizer: Optional regularizer function for the output.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    kernel_posterior_fn: Python `callable` which creates
      `tf.distributions.Distribution` instance representing the surrogate
      posterior of the `kernel` parameter. Default value:
      `default_mean_field_normal_fn()`.
    kernel_posterior_tensor_fn: Python `callable` which takes a
      `tf.distributions.Distribution` instance and returns a representative
      value. Default value: `lambda d: d.sample()`.
    kernel_prior_fn: Python `callable` which creates `tf.distributions`
      instance. See `default_mean_field_normal_fn` docstring for required
      parameter signature.
      Default value: `tf.distributions.Normal(loc=0., scale=1.)`.
    kernel_divergence_fn: Python `callable` which takes the surrogate posterior
      distribution, prior distribution and random variate sample(s) from the
      surrogate posterior and computes or approximates the KL divergence. The
      distributions are `tf.distributions.Distribution`-like instances and the
      sample is a `Tensor`.
    bias_posterior_fn: Python `callable` which creates
      `tf.distributions.Distribution` instance representing the surrogate
      posterior of the `bias` parameter. Default value:
      `default_mean_field_normal_fn(is_singular=True)` (which creates an
      instance of `tf.distributions.Deterministic`).
    bias_posterior_tensor_fn: Python `callable` which takes a
      `tf.distributions.Distribution` instance and returns a representative
      value. Default value: `lambda d: d.sample()`.
    bias_prior_fn: Python `callable` which creates `tf.distributions` instance.
      See `default_mean_field_normal_fn` docstring for required parameter
      signature. Default value: `None` (no prior, no variational inference)
    bias_divergence_fn: Python `callable` which takes the surrogate posterior
      distribution, prior distribution and random variate sample(s) from the
      surrogate posterior and computes or approximates the KL divergence. The
      distributions are `tf.distributions.Distribution`-like instances and the
      sample is a `Tensor`.
    seed: Python scalar `int` which initializes the random number
      generator. Default value: `None` (i.e., use global seed).
    name: A string, the name of the layer.

  Properties:
    rank: Python integer, dimensionality of convolution.
    filters: Python integer, dimensionality of the output space.
    kernel_size: Size of the convolution window.
    strides: Stride length of convolution.
    padding: Python string describing padding approach.
    data_format: Python string describing input data's dimensions.
    dilation_rate: Dilation rate for an atrous convolution.
    activation: Activation function (`callable`).
    activity_regularizer: Regularizer function for the output.
    kernel_posterior_fn: `callable` returning posterior.
    kernel_posterior_tensor_fn: `callable` operating on posterior.
    kernel_prior_fn: `callable` returning prior.
    kernel_divergence_fn: `callable` returning divergence.
    bias_posterior_fn: `callable` returning posterior.
    bias_posterior_tensor_fn: `callable` operating on posterior.
    bias_prior_fn: `callable` returning prior.
    bias_divergence_fn: `callable` returning divergence.
    seed: Python integer, used to create random seeds.

  [1]: "Flipout: Efficient Pseudo-Independent Weight Perturbations on
        Mini-Batches."
        Anonymous. OpenReview, 2017.
        https://openreview.net/forum?id=rJnpifWAb
  """

  def __init__(
      self,
      rank,
      filters,
      kernel_size,
      strides=1,
      padding="valid",
      data_format="channels_last",
      dilation_rate=1,
      activation=None,
      activity_regularizer=None,
      trainable=True,
      kernel_posterior_fn=layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=lambda dtype, *args: normal_lib.Normal(  # pylint: disable=g-long-lambda
          loc=dtype.as_numpy_dtype(0.), scale=dtype.as_numpy_dtype(1.)),
      kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      bias_posterior_fn=layers_util.default_mean_field_normal_fn(is_singular=True),  # pylint: disable=line-too-long
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      seed=None,
      name=None,
      **kwargs):
    super(_ConvFlipout, self).__init__(
        rank=rank,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        kernel_posterior_fn=kernel_posterior_fn,
        kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
        kernel_prior_fn=kernel_prior_fn,
        kernel_divergence_fn=kernel_divergence_fn,
        bias_posterior_fn=bias_posterior_fn,
        bias_posterior_tensor_fn=bias_posterior_tensor_fn,
        bias_prior_fn=bias_prior_fn,
        bias_divergence_fn=bias_divergence_fn,
        name=name, **kwargs)
    self.seed = seed

  def _apply_variational_kernel(self, inputs):
    if (not isinstance(self.kernel_posterior, independent_lib.Independent) or
        not isinstance(self.kernel_posterior.distribution, normal_lib.Normal)):
      raise TypeError(
          "`{}` requires "
          "`kernel_posterior_fn` produce an instance of "
          "`tf.distributions.Independent(tf.distributions.Normal)` "
          "(saw: \"{}\").".format(
              type(self).__name__, self.kernel_posterior.name))
    self.kernel_posterior_affine = normal_lib.Normal(
        loc=array_ops.zeros_like(self.kernel_posterior.distribution.loc),
        scale=self.kernel_posterior.distribution.scale)
    self.kernel_posterior_affine_tensor = (
        self.kernel_posterior_tensor_fn(self.kernel_posterior_affine))
    self.kernel_posterior_tensor = None

    outputs = self._convolution_op(
        inputs, self.kernel_posterior.distribution.loc)

    input_shape = array_ops.shape(inputs)
    output_shape = array_ops.shape(outputs)
    batch_shape = array_ops.expand_dims(input_shape[0], 0)
    channels = input_shape[-1]

    sign_input = layers_util.random_sign(
        array_ops.concat([batch_shape,
                          array_ops.expand_dims(channels, 0)], 0),
        dtype=inputs.dtype,
        seed=self.seed)
    sign_output = layers_util.random_sign(
        array_ops.concat([batch_shape,
                          array_ops.expand_dims(self.filters, 0)], 0),
        dtype=inputs.dtype,
        seed=distribution_util.gen_new_seed(
            self.seed, salt="conv_flipout"))
    for _ in range(self.rank):
      sign_input = array_ops.expand_dims(sign_input, 1)  # 2D ex: (B, 1, 1, C)
      sign_output = array_ops.expand_dims(sign_output, 1)

    sign_input = array_ops.tile(  # tile for element-wise op broadcasting
        sign_input,
        [1] + [input_shape[i + 1] for i in range(self.rank)] + [1])
    sign_output = array_ops.tile(
        sign_output,
        [1] + [output_shape[i + 1] for i in range(self.rank)] + [1])

    perturbed_inputs = self._convolution_op(
        inputs * sign_input, self.kernel_posterior_affine_tensor) * sign_output

    outputs += perturbed_inputs
    return outputs


class Conv1DFlipout(_ConvFlipout):
  """1D convolution layer (e.g. temporal convolution) with Flipout.

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. It may also include a bias addition and activation function
  on the outputs. It assumes the `kernel` and/or `bias` are drawn from
  distributions.

  By default, the layer implements a stochastic forward pass via
  sampling from the kernel and bias posteriors,
  ```none
  outputs = f(inputs; kernel, bias), kernel, bias ~ posterior
  ```
  where f denotes the layer's calculation. It uses the Flipout
  estimator [1], which performs a Monte Carlo approximation of the
  distribution integrating over the `kernel` and `bias`. Flipout uses
  roughly twice as many floating point operations as the
  reparameterization estimator but has the advantage of significantly
  lower variance.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.

  Arguments:
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: An integer or tuple/list of a single integer, specifying the
      length of the 1D convolution window.
    strides: An integer or tuple/list of a single integer,
      specifying the stride length of the convolution.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, length, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, length)`.
    dilation_rate: An integer or tuple/list of a single integer, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any `strides` value != 1.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    activity_regularizer: Optional regularizer function for the output.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    kernel_posterior_fn: Python `callable` which creates
      `tf.distributions.Distribution` instance representing the surrogate
      posterior of the `kernel` parameter. Default value:
      `default_mean_field_normal_fn()`.
    kernel_posterior_tensor_fn: Python `callable` which takes a
      `tf.distributions.Distribution` instance and returns a representative
      value. Default value: `lambda d: d.sample()`.
    kernel_prior_fn: Python `callable` which creates `tf.distributions`
      instance. See `default_mean_field_normal_fn` docstring for required
      parameter signature.
      Default value: `tf.distributions.Normal(loc=0., scale=1.)`.
    kernel_divergence_fn: Python `callable` which takes the surrogate posterior
      distribution, prior distribution and random variate sample(s) from the
      surrogate posterior and computes or approximates the KL divergence. The
      distributions are `tf.distributions.Distribution`-like instances and the
      sample is a `Tensor`.
    bias_posterior_fn: Python `callable` which creates
      `tf.distributions.Distribution` instance representing the surrogate
      posterior of the `bias` parameter. Default value:
      `default_mean_field_normal_fn(is_singular=True)` (which creates an
      instance of `tf.distributions.Deterministic`).
    bias_posterior_tensor_fn: Python `callable` which takes a
      `tf.distributions.Distribution` instance and returns a representative
      value. Default value: `lambda d: d.sample()`.
    bias_prior_fn: Python `callable` which creates `tf.distributions` instance.
      See `default_mean_field_normal_fn` docstring for required parameter
      signature. Default value: `None` (no prior, no variational inference)
    bias_divergence_fn: Python `callable` which takes the surrogate posterior
      distribution, prior distribution and random variate sample(s) from the
      surrogate posterior and computes or approximates the KL divergence. The
      distributions are `tf.distributions.Distribution`-like instances and the
    seed: Python scalar `int` which initializes the random number
      generator. Default value: `None` (i.e., use global seed).
    name: A string, the name of the layer.

  Properties:
    filters: Python integer, dimensionality of the output space.
    kernel_size: Size of the convolution window.
    strides: Stride length of convolution.
    padding: Python string describing padding approach.
    data_format: Python string describing input data's dimensions.
    dilation_rate: Dilation rate for an atrous convolution.
    activation: Activation function (`callable`).
    activity_regularizer: Regularizer function for the output.
    kernel_posterior_fn: `callable` returning posterior.
    kernel_posterior_tensor_fn: `callable` operating on posterior.
    kernel_prior_fn: `callable` returning prior.
    kernel_divergence_fn: `callable` returning divergence.
    bias_posterior_fn: `callable` returning posterior.
    bias_posterior_tensor_fn: `callable` operating on posterior.
    bias_prior_fn: `callable` returning prior.
    bias_divergence_fn: `callable` returning divergence.
    seed: Python integer, used to create random seeds.

  #### Examples

  We illustrate a Bayesian neural network with [variational inference](
  https://en.wikipedia.org/wiki/Variational_Bayesian_methods),
  assuming a dataset of `features` and `labels`.

  ```python
  tfp = tf.contrib.bayesflow

  net = tf.reshape(features, [-1, 128, 1])
  net = tfp.layers.Conv1DFlipout(64,
                                 kernel_size=5,
                                 padding="SAME",
                                 activation=tf.nn.relu)(net)
  net = tf.reshape(net, [-1, 128 * 64])
  logits = tfp.layers.DenseFlipout(10)(net)
  neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  kl = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
  loss = neg_log_likelihood + kl
  train_op = tf.train.AdamOptimizer().minimize(loss)
  ```

  It uses the Flipout gradient estimator to minimize the
  Kullback-Leibler divergence up to a constant, also known as the
  negative Evidence Lower Bound. It consists of the sum of two terms:
  the expected negative log-likelihood, which we approximate via
  Monte Carlo; and the KL divergence, which is added via regularizer
  terms which are arguments to the layer.

  [1]: "Flipout: Efficient Pseudo-Independent Weight Perturbations on
        Mini-Batches."
        Anonymous. OpenReview, 2017.
        https://openreview.net/forum?id=rJnpifWAb
  """

  def __init__(
      self,
      filters,
      kernel_size,
      strides=1,
      padding="valid",
      data_format="channels_last",
      dilation_rate=1,
      activation=None,
      activity_regularizer=None,
      trainable=True,
      kernel_posterior_fn=layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=lambda dtype, *args: normal_lib.Normal(  # pylint: disable=g-long-lambda
          loc=dtype.as_numpy_dtype(0.), scale=dtype.as_numpy_dtype(1.)),
      kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      bias_posterior_fn=layers_util.default_mean_field_normal_fn(is_singular=True),  # pylint: disable=line-too-long
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      seed=None,
      name=None,
      **kwargs):
    super(Conv1DFlipout, self).__init__(
        rank=1,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        kernel_posterior_fn=kernel_posterior_fn,
        kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
        kernel_prior_fn=kernel_prior_fn,
        kernel_divergence_fn=kernel_divergence_fn,
        bias_posterior_fn=bias_posterior_fn,
        bias_posterior_tensor_fn=bias_posterior_tensor_fn,
        bias_prior_fn=bias_prior_fn,
        bias_divergence_fn=bias_divergence_fn,
        seed=seed,
        name=name, **kwargs)


def conv1d_flipout(
    inputs,
    filters,
    kernel_size,
    strides=1,
    padding="valid",
    data_format="channels_last",
    dilation_rate=1,
    activation=None,
    activity_regularizer=None,
    trainable=True,
    kernel_posterior_fn=layers_util.default_mean_field_normal_fn(),
    kernel_posterior_tensor_fn=lambda d: d.sample(),
    kernel_prior_fn=lambda dtype, *args: normal_lib.Normal(  # pylint: disable=g-long-lambda
        loc=dtype.as_numpy_dtype(0.), scale=dtype.as_numpy_dtype(1.)),
    kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
    bias_posterior_fn=layers_util.default_mean_field_normal_fn(is_singular=True),  # pylint: disable=line-too-long
    bias_posterior_tensor_fn=lambda d: d.sample(),
    bias_prior_fn=None,
    bias_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
    seed=None,
    name=None,
    reuse=None):
  """Functional interface for 1D convolution layer (e.g. temporal convolution).

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. It may also include a bias addition and activation function
  on the outputs. It assumes the `kernel` and/or `bias` are drawn from
  distributions.

  By default, the layer implements a stochastic forward pass via
  sampling from the kernel and bias posteriors,
  ```none
  outputs = f(inputs; kernel, bias), kernel, bias ~ posterior
  ```
  where f denotes the layer's calculation. It uses the Flipout
  estimator [1], which performs a Monte Carlo approximation of the
  distribution integrating over the `kernel` and `bias`. Flipout uses
  roughly twice as many floating point operations as the
  reparameterization estimator but has the advantage of significantly
  lower variance.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.

  Arguments:
    inputs: Tensor input.
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: An integer or tuple/list of a single integer, specifying the
      length of the 1D convolution window.
    strides: An integer or tuple/list of a single integer,
      specifying the stride length of the convolution.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, length, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, length)`.
    dilation_rate: An integer or tuple/list of a single integer, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any `strides` value != 1.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    activity_regularizer: Optional regularizer function for the output.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    kernel_posterior_fn: Python `callable` which creates
      `tf.distributions.Distribution` instance representing the surrogate
      posterior of the `kernel` parameter. Default value:
      `default_mean_field_normal_fn()`.
    kernel_posterior_tensor_fn: Python `callable` which takes a
      `tf.distributions.Distribution` instance and returns a representative
      value. Default value: `lambda d: d.sample()`.
    kernel_prior_fn: Python `callable` which creates `tf.distributions`
      instance. See `default_mean_field_normal_fn` docstring for required
      parameter signature.
      Default value: `tf.distributions.Normal(loc=0., scale=1.)`.
    kernel_divergence_fn: Python `callable` which takes the surrogate posterior
      distribution, prior distribution and random variate sample(s) from the
      surrogate posterior and computes or approximates the KL divergence. The
      distributions are `tf.distributions.Distribution`-like instances and the
      sample is a `Tensor`.
    bias_posterior_fn: Python `callable` which creates
      `tf.distributions.Distribution` instance representing the surrogate
      posterior of the `bias` parameter. Default value:
      `default_mean_field_normal_fn(is_singular=True)` (which creates an
      instance of `tf.distributions.Deterministic`).
    bias_posterior_tensor_fn: Python `callable` which takes a
      `tf.distributions.Distribution` instance and returns a representative
      value. Default value: `lambda d: d.sample()`.
    bias_prior_fn: Python `callable` which creates `tf.distributions` instance.
      See `default_mean_field_normal_fn` docstring for required parameter
      signature. Default value: `None` (no prior, no variational inference)
    bias_divergence_fn: Python `callable` which takes the surrogate posterior
      distribution, prior distribution and random variate sample(s) from the
      surrogate posterior and computes or approximates the KL divergence. The
      distributions are `tf.distributions.Distribution`-like instances and the
    seed: Python scalar `int` which initializes the random number
      generator. Default value: `None` (i.e., use global seed).
    name: A string, the name of the layer.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

  Returns:
    Output tensor.

  Raises:
    ValueError: if eager execution is enabled.

  #### Examples

  We illustrate a Bayesian neural network with [variational inference](
  https://en.wikipedia.org/wiki/Variational_Bayesian_methods),
  assuming a dataset of `features` and `labels`.

  ```python
  tfp = tf.contrib.bayesflow

  net = tf.reshape(features, [-1, 128, 1])
  net = tfp.layers.conv1d_flipout(net,
                                  filters=64,
                                  kernel_size=5,
                                  padding="SAME",
                                  activation=tf.nn.relu)
  net = tf.reshape(net, [-1, 128 * 64])
  logits = tfp.layers.dense_flipout(net, 10)
  neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  kl = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
  loss = neg_log_likelihood + kl
  train_op = tf.train.AdamOptimizer().minimize(loss)
  ```

  It uses the Flipout gradient estimator to minimize the
  Kullback-Leibler divergence up to a constant, also known as the
  negative Evidence Lower Bound. It consists of the sum of two terms:
  the expected negative log-likelihood, which we approximate via
  Monte Carlo; and the KL divergence, which is added via regularizer
  terms which are arguments to the layer.

  [1]: "Flipout: Efficient Pseudo-Independent Weight Perturbations on
        Mini-Batches."
        Anonymous. OpenReview, 2017.
        https://openreview.net/forum?id=rJnpifWAb
  """
  layer = Conv1DFlipout(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilation_rate=dilation_rate,
      activation=activation,
      activity_regularizer=activity_regularizer,
      trainable=trainable,
      kernel_posterior_fn=kernel_posterior_fn,
      kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
      kernel_prior_fn=kernel_prior_fn,
      kernel_divergence_fn=kernel_divergence_fn,
      bias_posterior_fn=bias_posterior_fn,
      bias_posterior_tensor_fn=bias_posterior_tensor_fn,
      bias_prior_fn=bias_prior_fn,
      bias_divergence_fn=bias_divergence_fn,
      seed=seed,
      name=name,
      dtype=inputs.dtype.base_dtype,
      _scope=name,
      _reuse=reuse)
  return layer.apply(inputs)


class Conv2DFlipout(_ConvFlipout):
  """2D convolution layer (e.g. spatial convolution over images) with Flipout.

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. It may also include a bias addition and activation function
  on the outputs. It assumes the `kernel` and/or `bias` are drawn from
  distributions.

  By default, the layer implements a stochastic forward pass via
  sampling from the kernel and bias posteriors,
  ```none
  outputs = f(inputs; kernel, bias), kernel, bias ~ posterior
  ```
  where f denotes the layer's calculation. It uses the Flipout
  estimator [1], which performs a Monte Carlo approximation of the
  distribution integrating over the `kernel` and `bias`. Flipout uses
  roughly twice as many floating point operations as the
  reparameterization estimator but has the advantage of significantly
  lower variance.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.

  Arguments:
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: An integer or tuple/list of 2 integers, specifying the
      height and width of the 2D convolution window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the convolution along the height and width.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, height, width)`.

    dilation_rate: An integer or tuple/list of 2 integers, specifying
      the dilation rate to use for dilated convolution.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any stride value != 1.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    activity_regularizer: Optional regularizer function for the output.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    kernel_posterior_fn: Python `callable` which creates
      `tf.distributions.Distribution` instance representing the surrogate
      posterior of the `kernel` parameter. Default value:
      `default_mean_field_normal_fn()`.
    kernel_posterior_tensor_fn: Python `callable` which takes a
      `tf.distributions.Distribution` instance and returns a representative
      value. Default value: `lambda d: d.sample()`.
    kernel_prior_fn: Python `callable` which creates `tf.distributions`
      instance. See `default_mean_field_normal_fn` docstring for required
      parameter signature.
      Default value: `tf.distributions.Normal(loc=0., scale=1.)`.
    kernel_divergence_fn: Python `callable` which takes the surrogate posterior
      distribution, prior distribution and random variate sample(s) from the
      surrogate posterior and computes or approximates the KL divergence. The
      distributions are `tf.distributions.Distribution`-like instances and the
      sample is a `Tensor`.
    bias_posterior_fn: Python `callable` which creates
      `tf.distributions.Distribution` instance representing the surrogate
      posterior of the `bias` parameter. Default value:
      `default_mean_field_normal_fn(is_singular=True)` (which creates an
      instance of `tf.distributions.Deterministic`).
    bias_posterior_tensor_fn: Python `callable` which takes a
      `tf.distributions.Distribution` instance and returns a representative
      value. Default value: `lambda d: d.sample()`.
    bias_prior_fn: Python `callable` which creates `tf.distributions` instance.
      See `default_mean_field_normal_fn` docstring for required parameter
      signature. Default value: `None` (no prior, no variational inference)
    bias_divergence_fn: Python `callable` which takes the surrogate posterior
      distribution, prior distribution and random variate sample(s) from the
      surrogate posterior and computes or approximates the KL divergence. The
      distributions are `tf.distributions.Distribution`-like instances and the
    seed: Python scalar `int` which initializes the random number
      generator. Default value: `None` (i.e., use global seed).
    name: A string, the name of the layer.

  Properties:
    filters: Python integer, dimensionality of the output space.
    kernel_size: Size of the convolution window.
    strides: Stride length of convolution.
    padding: Python string describing padding approach.
    data_format: Python string describing input data's dimensions.
    dilation_rate: Dilation rate for an atrous convolution.
    activation: Activation function (`callable`).
    activity_regularizer: Regularizer function for the output.
    kernel_posterior_fn: `callable` returning posterior.
    kernel_posterior_tensor_fn: `callable` operating on posterior.
    kernel_prior_fn: `callable` returning prior.
    kernel_divergence_fn: `callable` returning divergence.
    bias_posterior_fn: `callable` returning posterior.
    bias_posterior_tensor_fn: `callable` operating on posterior.
    bias_prior_fn: `callable` returning prior.
    bias_divergence_fn: `callable` returning divergence.
    seed: Python integer, used to create random seeds.

  #### Examples

  We illustrate a Bayesian neural network with [variational inference](
  https://en.wikipedia.org/wiki/Variational_Bayesian_methods),
  assuming a dataset of `features` and `labels`.

  ```python
  tfp = tf.contrib.bayesflow

  net = tf.reshape(features, [-1, 32, 32, 3])
  net = tfp.layers.Conv2DFlipout(64,
                                 kernel_size=5,
                                 padding="SAME",
                                 activation=tf.nn.relu)(net)
  net = tf.layers.MaxPooling2D(pool_size=2,
                               strides=2,
                               padding="SAME")(net)
  net = tf.reshape(net, [-1, 8 * 8 * 64])
  logits = tfp.layers.DenseFlipout(10)(net)
  neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  kl = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
  loss = neg_log_likelihood + kl
  train_op = tf.train.AdamOptimizer().minimize(loss)
  ```

  It uses the Flipout gradient estimator to minimize the
  Kullback-Leibler divergence up to a constant, also known as the
  negative Evidence Lower Bound. It consists of the sum of two terms:
  the expected negative log-likelihood, which we approximate via
  Monte Carlo; and the KL divergence, which is added via regularizer
  terms which are arguments to the layer.

  [1]: "Flipout: Efficient Pseudo-Independent Weight Perturbations on
        Mini-Batches."
        Anonymous. OpenReview, 2017.
        https://openreview.net/forum?id=rJnpifWAb
  """

  def __init__(
      self,
      filters,
      kernel_size,
      strides=(1, 1),
      padding="valid",
      data_format="channels_last",
      dilation_rate=(1, 1),
      activation=None,
      activity_regularizer=None,
      trainable=True,
      kernel_posterior_fn=layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=lambda dtype, *args: normal_lib.Normal(  # pylint: disable=g-long-lambda
          loc=dtype.as_numpy_dtype(0.), scale=dtype.as_numpy_dtype(1.)),
      kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      bias_posterior_fn=layers_util.default_mean_field_normal_fn(is_singular=True),  # pylint: disable=line-too-long
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      seed=None,
      name=None,
      **kwargs):
    super(Conv2DFlipout, self).__init__(
        rank=2,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        kernel_posterior_fn=kernel_posterior_fn,
        kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
        kernel_prior_fn=kernel_prior_fn,
        kernel_divergence_fn=kernel_divergence_fn,
        bias_posterior_fn=bias_posterior_fn,
        bias_posterior_tensor_fn=bias_posterior_tensor_fn,
        bias_prior_fn=bias_prior_fn,
        bias_divergence_fn=bias_divergence_fn,
        seed=seed,
        name=name, **kwargs)


def conv2d_flipout(
    inputs,
    filters,
    kernel_size,
    strides=(1, 1),
    padding="valid",
    data_format="channels_last",
    dilation_rate=(1, 1),
    activation=None,
    activity_regularizer=None,
    trainable=True,
    kernel_posterior_fn=layers_util.default_mean_field_normal_fn(),
    kernel_posterior_tensor_fn=lambda d: d.sample(),
    kernel_prior_fn=lambda dtype, *args: normal_lib.Normal(  # pylint: disable=g-long-lambda
        loc=dtype.as_numpy_dtype(0.), scale=dtype.as_numpy_dtype(1.)),
    kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
    bias_posterior_fn=layers_util.default_mean_field_normal_fn(is_singular=True),  # pylint: disable=line-too-long
    bias_posterior_tensor_fn=lambda d: d.sample(),
    bias_prior_fn=None,
    bias_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
    seed=None,
    name=None,
    reuse=None):
  """Functional interface for the 2D convolution layer.

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. It may also include a bias addition and activation function
  on the outputs. It assumes the `kernel` and/or `bias` are drawn from
  distributions.

  By default, the layer implements a stochastic forward pass via
  sampling from the kernel and bias posteriors,
  ```none
  outputs = f(inputs; kernel, bias), kernel, bias ~ posterior
  ```
  where f denotes the layer's calculation. It uses the Flipout
  estimator [1], which performs a Monte Carlo approximation of the
  distribution integrating over the `kernel` and `bias`. Flipout uses
  roughly twice as many floating point operations as the
  reparameterization estimator but has the advantage of significantly
  lower variance.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.

  Arguments:
    inputs: Tensor input.
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: An integer or tuple/list of 2 integers, specifying the
      height and width of the 2D convolution window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the convolution along the height and width.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, height, width)`.

    dilation_rate: An integer or tuple/list of 2 integers, specifying
      the dilation rate to use for dilated convolution.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any stride value != 1.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    activity_regularizer: Optional regularizer function for the output.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    kernel_posterior_fn: Python `callable` which creates
      `tf.distributions.Distribution` instance representing the surrogate
      posterior of the `kernel` parameter. Default value:
      `default_mean_field_normal_fn()`.
    kernel_posterior_tensor_fn: Python `callable` which takes a
      `tf.distributions.Distribution` instance and returns a representative
      value. Default value: `lambda d: d.sample()`.
    kernel_prior_fn: Python `callable` which creates `tf.distributions`
      instance. See `default_mean_field_normal_fn` docstring for required
      parameter signature.
      Default value: `tf.distributions.Normal(loc=0., scale=1.)`.
    kernel_divergence_fn: Python `callable` which takes the surrogate posterior
      distribution, prior distribution and random variate sample(s) from the
      surrogate posterior and computes or approximates the KL divergence. The
      distributions are `tf.distributions.Distribution`-like instances and the
      sample is a `Tensor`.
    bias_posterior_fn: Python `callable` which creates
      `tf.distributions.Distribution` instance representing the surrogate
      posterior of the `bias` parameter. Default value:
      `default_mean_field_normal_fn(is_singular=True)` (which creates an
      instance of `tf.distributions.Deterministic`).
    bias_posterior_tensor_fn: Python `callable` which takes a
      `tf.distributions.Distribution` instance and returns a representative
      value. Default value: `lambda d: d.sample()`.
    bias_prior_fn: Python `callable` which creates `tf.distributions` instance.
      See `default_mean_field_normal_fn` docstring for required parameter
      signature. Default value: `None` (no prior, no variational inference)
    bias_divergence_fn: Python `callable` which takes the surrogate posterior
      distribution, prior distribution and random variate sample(s) from the
      surrogate posterior and computes or approximates the KL divergence. The
      distributions are `tf.distributions.Distribution`-like instances and the
    seed: Python scalar `int` which initializes the random number
      generator. Default value: `None` (i.e., use global seed).
    name: A string, the name of the layer.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

  Returns:
    Output tensor.

  Raises:
    ValueError: if eager execution is enabled.

  #### Examples

  We illustrate a Bayesian neural network with [variational inference](
  https://en.wikipedia.org/wiki/Variational_Bayesian_methods),
  assuming a dataset of `features` and `labels`.

  ```python
  tfp = tf.contrib.bayesflow

  net = tf.reshape(features, [-1, 32, 32, 3])
  net = tfp.layers.conv2d_flipout(net,
                                  filters=64,
                                  kernel_size=5,
                                  padding="SAME",
                                  activation=tf.nn.relu)
  net = tf.layers.max_pooling2d(net,
                                pool_size=2,
                                strides=2,
                                padding="SAME")
  net = tf.reshape(net, [-1, 8 * 8 * 64])
  logits = tfp.layers.dense_flipout(net, 10)
  neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  kl = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
  loss = neg_log_likelihood + kl
  train_op = tf.train.AdamOptimizer().minimize(loss)
  ```

  It uses the Flipout gradient estimator to minimize the
  Kullback-Leibler divergence up to a constant, also known as the
  negative Evidence Lower Bound. It consists of the sum of two terms:
  the expected negative log-likelihood, which we approximate via
  Monte Carlo; and the KL divergence, which is added via regularizer
  terms which are arguments to the layer.

  [1]: "Flipout: Efficient Pseudo-Independent Weight Perturbations on
        Mini-Batches."
        Anonymous. OpenReview, 2017.
        https://openreview.net/forum?id=rJnpifWAb
  """
  layer = Conv2DFlipout(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilation_rate=dilation_rate,
      activation=activation,
      activity_regularizer=activity_regularizer,
      trainable=trainable,
      kernel_posterior_fn=kernel_posterior_fn,
      kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
      kernel_prior_fn=kernel_prior_fn,
      kernel_divergence_fn=kernel_divergence_fn,
      bias_posterior_fn=bias_posterior_fn,
      bias_posterior_tensor_fn=bias_posterior_tensor_fn,
      bias_prior_fn=bias_prior_fn,
      bias_divergence_fn=bias_divergence_fn,
      seed=seed,
      name=name,
      dtype=inputs.dtype.base_dtype,
      _scope=name,
      _reuse=reuse)
  return layer.apply(inputs)


class Conv3DFlipout(_ConvFlipout):
  """3D convolution layer (e.g. spatial convolution over volumes) with Flipout.

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. It may also include a bias addition and activation function
  on the outputs. It assumes the `kernel` and/or `bias` are drawn from
  distributions.

  By default, the layer implements a stochastic forward pass via
  sampling from the kernel and bias posteriors,
  ```none
  outputs = f(inputs; kernel, bias), kernel, bias ~ posterior
  ```
  where f denotes the layer's calculation. It uses the Flipout
  estimator [1], which performs a Monte Carlo approximation of the
  distribution integrating over the `kernel` and `bias`. Flipout uses
  roughly twice as many floating point operations as the
  reparameterization estimator but has the advantage of significantly
  lower variance.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.

  Arguments:
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: An integer or tuple/list of 3 integers, specifying the
      depth, height and width of the 3D convolution window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 3 integers,
      specifying the strides of the convolution along the depth,
      height and width.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, depth, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch, channels, depth, height, width)`.
    dilation_rate: An integer or tuple/list of 3 integers, specifying
      the dilation rate to use for dilated convolution.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any stride value != 1.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    activity_regularizer: Optional regularizer function for the output.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    kernel_posterior_fn: Python `callable` which creates
      `tf.distributions.Distribution` instance representing the surrogate
      posterior of the `kernel` parameter. Default value:
      `default_mean_field_normal_fn()`.
    kernel_posterior_tensor_fn: Python `callable` which takes a
      `tf.distributions.Distribution` instance and returns a representative
      value. Default value: `lambda d: d.sample()`.
    kernel_prior_fn: Python `callable` which creates `tf.distributions`
      instance. See `default_mean_field_normal_fn` docstring for required
      parameter signature.
      Default value: `tf.distributions.Normal(loc=0., scale=1.)`.
    kernel_divergence_fn: Python `callable` which takes the surrogate posterior
      distribution, prior distribution and random variate sample(s) from the
      surrogate posterior and computes or approximates the KL divergence. The
      distributions are `tf.distributions.Distribution`-like instances and the
      sample is a `Tensor`.
    bias_posterior_fn: Python `callable` which creates
      `tf.distributions.Distribution` instance representing the surrogate
      posterior of the `bias` parameter. Default value:
      `default_mean_field_normal_fn(is_singular=True)` (which creates an
      instance of `tf.distributions.Deterministic`).
    bias_posterior_tensor_fn: Python `callable` which takes a
      `tf.distributions.Distribution` instance and returns a representative
      value. Default value: `lambda d: d.sample()`.
    bias_prior_fn: Python `callable` which creates `tf.distributions` instance.
      See `default_mean_field_normal_fn` docstring for required parameter
      signature. Default value: `None` (no prior, no variational inference)
    bias_divergence_fn: Python `callable` which takes the surrogate posterior
      distribution, prior distribution and random variate sample(s) from the
      surrogate posterior and computes or approximates the KL divergence. The
      distributions are `tf.distributions.Distribution`-like instances and the
    seed: Python scalar `int` which initializes the random number
      generator. Default value: `None` (i.e., use global seed).
    name: A string, the name of the layer.

  Properties:
    filters: Python integer, dimensionality of the output space.
    kernel_size: Size of the convolution window.
    strides: Stride length of convolution.
    padding: Python string describing padding approach.
    data_format: Python string describing input data's dimensions.
    dilation_rate: Dilation rate for an atrous convolution.
    activation: Activation function (`callable`).
    activity_regularizer: Regularizer function for the output.
    kernel_posterior_fn: `callable` returning posterior.
    kernel_posterior_tensor_fn: `callable` operating on posterior.
    kernel_prior_fn: `callable` returning prior.
    kernel_divergence_fn: `callable` returning divergence.
    bias_posterior_fn: `callable` returning posterior.
    bias_posterior_tensor_fn: `callable` operating on posterior.
    bias_prior_fn: `callable` returning prior.
    bias_divergence_fn: `callable` returning divergence.
    seed: Python integer, used to create random seeds.

  #### Examples

  We illustrate a Bayesian neural network with [variational inference](
  https://en.wikipedia.org/wiki/Variational_Bayesian_methods),
  assuming a dataset of `features` and `labels`.

  ```python
  tfp = tf.contrib.bayesflow

  net = tf.reshape(features, [-1, 256, 32, 32, 3])
  net = tfp.layers.Conv3DFlipout(64,
                                 kernel_size=5,
                                 padding="SAME",
                                 activation=tf.nn.relu)(net)
  net = tf.layers.MaxPooling2D(pool_size=2,
                               strides=2,
                               padding="SAME")(net)
  net = tf.reshape(net, [-1, 256 * 8 * 8 * 64])
  logits = tfp.layers.DenseFlipout(10)(net)
  neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  kl = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
  loss = neg_log_likelihood + kl
  train_op = tf.train.AdamOptimizer().minimize(loss)
  ```

  It uses the Flipout gradient estimator to minimize the
  Kullback-Leibler divergence up to a constant, also known as the
  negative Evidence Lower Bound. It consists of the sum of two terms:
  the expected negative log-likelihood, which we approximate via
  Monte Carlo; and the KL divergence, which is added via regularizer
  terms which are arguments to the layer.

  [1]: "Flipout: Efficient Pseudo-Independent Weight Perturbations on
        Mini-Batches."
        Anonymous. OpenReview, 2017.
        https://openreview.net/forum?id=rJnpifWAb
  """

  def __init__(
      self,
      filters,
      kernel_size,
      strides=(1, 1, 1),
      padding="valid",
      data_format="channels_last",
      dilation_rate=(1, 1, 1),
      activation=None,
      activity_regularizer=None,
      trainable=True,
      kernel_posterior_fn=layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=lambda dtype, *args: normal_lib.Normal(  # pylint: disable=g-long-lambda
          loc=dtype.as_numpy_dtype(0.), scale=dtype.as_numpy_dtype(1.)),
      kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      bias_posterior_fn=layers_util.default_mean_field_normal_fn(is_singular=True),  # pylint: disable=line-too-long
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      seed=None,
      name=None,
      **kwargs):
    super(Conv3DFlipout, self).__init__(
        rank=3,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        kernel_posterior_fn=kernel_posterior_fn,
        kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
        kernel_prior_fn=kernel_prior_fn,
        kernel_divergence_fn=kernel_divergence_fn,
        bias_posterior_fn=bias_posterior_fn,
        bias_posterior_tensor_fn=bias_posterior_tensor_fn,
        bias_prior_fn=bias_prior_fn,
        bias_divergence_fn=bias_divergence_fn,
        seed=seed,
        name=name, **kwargs)


def conv3d_flipout(
    inputs,
    filters,
    kernel_size,
    strides=(1, 1, 1),
    padding="valid",
    data_format="channels_last",
    dilation_rate=(1, 1, 1),
    activation=None,
    activity_regularizer=None,
    trainable=True,
    kernel_posterior_fn=layers_util.default_mean_field_normal_fn(),
    kernel_posterior_tensor_fn=lambda d: d.sample(),
    kernel_prior_fn=lambda dtype, *args: normal_lib.Normal(  # pylint: disable=g-long-lambda
        loc=dtype.as_numpy_dtype(0.), scale=dtype.as_numpy_dtype(1.)),
    kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
    bias_posterior_fn=layers_util.default_mean_field_normal_fn(is_singular=True),  # pylint: disable=line-too-long
    bias_posterior_tensor_fn=lambda d: d.sample(),
    bias_prior_fn=None,
    bias_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
    seed=None,
    name=None,
    reuse=None):
  """Functional interface for the 3D convolution layer.

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. It may also include a bias addition and activation function
  on the outputs. It assumes the `kernel` and/or `bias` are drawn from
  distributions.

  By default, the layer implements a stochastic forward pass via
  sampling from the kernel and bias posteriors,
  ```none
  outputs = f(inputs; kernel, bias), kernel, bias ~ posterior
  ```
  where f denotes the layer's calculation. It uses the Flipout
  estimator [1], which performs a Monte Carlo approximation of the
  distribution integrating over the `kernel` and `bias`. Flipout uses
  roughly twice as many floating point operations as the
  reparameterization estimator but has the advantage of significantly
  lower variance.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.

  Arguments:
    inputs: Tensor input.
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: An integer or tuple/list of 3 integers, specifying the
      depth, height and width of the 3D convolution window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 3 integers,
      specifying the strides of the convolution along the depth,
      height and width.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, depth, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch, channels, depth, height, width)`.
    dilation_rate: An integer or tuple/list of 3 integers, specifying
      the dilation rate to use for dilated convolution.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any stride value != 1.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    activity_regularizer: Optional regularizer function for the output.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    kernel_posterior_fn: Python `callable` which creates
      `tf.distributions.Distribution` instance representing the surrogate
      posterior of the `kernel` parameter. Default value:
      `default_mean_field_normal_fn()`.
    kernel_posterior_tensor_fn: Python `callable` which takes a
      `tf.distributions.Distribution` instance and returns a representative
      value. Default value: `lambda d: d.sample()`.
    kernel_prior_fn: Python `callable` which creates `tf.distributions`
      instance. See `default_mean_field_normal_fn` docstring for required
      parameter signature.
      Default value: `tf.distributions.Normal(loc=0., scale=1.)`.
    kernel_divergence_fn: Python `callable` which takes the surrogate posterior
      distribution, prior distribution and random variate sample(s) from the
      surrogate posterior and computes or approximates the KL divergence. The
      distributions are `tf.distributions.Distribution`-like instances and the
      sample is a `Tensor`.
    bias_posterior_fn: Python `callable` which creates
      `tf.distributions.Distribution` instance representing the surrogate
      posterior of the `bias` parameter. Default value:
      `default_mean_field_normal_fn(is_singular=True)` (which creates an
      instance of `tf.distributions.Deterministic`).
    bias_posterior_tensor_fn: Python `callable` which takes a
      `tf.distributions.Distribution` instance and returns a representative
      value. Default value: `lambda d: d.sample()`.
    bias_prior_fn: Python `callable` which creates `tf.distributions` instance.
      See `default_mean_field_normal_fn` docstring for required parameter
      signature. Default value: `None` (no prior, no variational inference)
    bias_divergence_fn: Python `callable` which takes the surrogate posterior
      distribution, prior distribution and random variate sample(s) from the
      surrogate posterior and computes or approximates the KL divergence. The
      distributions are `tf.distributions.Distribution`-like instances and the
    seed: Python scalar `int` which initializes the random number
      generator. Default value: `None` (i.e., use global seed).
    name: A string, the name of the layer.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

  Returns:
    Output tensor.

  Raises:
    ValueError: if eager execution is enabled.

  #### Examples

  We illustrate a Bayesian neural network with [variational inference](
  https://en.wikipedia.org/wiki/Variational_Bayesian_methods),
  assuming a dataset of `features` and `labels`.

  ```python
  tfp = tf.contrib.bayesflow

  net = tf.reshape(features, [-1, 256, 32, 32, 3])
  net = tfp.layers.conv3d_flipout(net,
                                  filters=64,
                                  kernel_size=5,
                                  padding="SAME",
                                  activation=tf.nn.relu)
  net = tf.layers.max_pooling2d(net,
                                pool_size=2,
                                strides=2,
                                padding="SAME")
  net = tf.reshape(net, [-1, 256 * 8 * 8 * 64])
  logits = tfp.layers.dense_flipout(net, 10)
  neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  kl = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
  loss = neg_log_likelihood + kl
  train_op = tf.train.AdamOptimizer().minimize(loss)
  ```

  It uses the Flipout gradient estimator to minimize the
  Kullback-Leibler divergence up to a constant, also known as the
  negative Evidence Lower Bound. It consists of the sum of two terms:
  the expected negative log-likelihood, which we approximate via
  Monte Carlo; and the KL divergence, which is added via regularizer
  terms which are arguments to the layer.

  [1]: "Flipout: Efficient Pseudo-Independent Weight Perturbations on
        Mini-Batches."
        Anonymous. OpenReview, 2017.
        https://openreview.net/forum?id=rJnpifWAb
  """
  layer = Conv3DFlipout(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilation_rate=dilation_rate,
      activation=activation,
      activity_regularizer=activity_regularizer,
      trainable=trainable,
      kernel_posterior_fn=kernel_posterior_fn,
      kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
      kernel_prior_fn=kernel_prior_fn,
      kernel_divergence_fn=kernel_divergence_fn,
      bias_posterior_fn=bias_posterior_fn,
      bias_posterior_tensor_fn=bias_posterior_tensor_fn,
      bias_prior_fn=bias_prior_fn,
      bias_divergence_fn=bias_divergence_fn,
      seed=seed,
      name=name,
      dtype=inputs.dtype.base_dtype,
      _scope=name,
      _reuse=reuse)
  return layer.apply(inputs)


# Aliases

Convolution1DReparameterization = Conv1DReparameterization
Convolution2DReparameterization = Conv2DReparameterization
Convolution3DReparameterization = Conv3DReparameterization
convolution1d_reparameterization = conv1d_reparameterization
convolution2d_reparameterization = conv2d_reparameterization
convolution3d_reparameterization = conv3d_reparameterization
Convolution1DFlipout = Conv1DFlipout
Convolution2DFlipout = Conv2DFlipout
Convolution3DFlipout = Conv3DFlipout
convolution1d_flipout = conv1d_flipout
convolution2d_flipout = conv2d_flipout
convolution3d_flipout = conv3d_flipout
