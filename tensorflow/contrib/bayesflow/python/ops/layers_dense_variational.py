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
"""Dense Bayesian layer using KL-divergence based variational inference.
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
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops.distributions import kullback_leibler as kl_lib
from tensorflow.python.ops.distributions import normal as normal_lib
from tensorflow.python.ops.distributions import util as distribution_util


class _DenseVariational(layers_lib.Layer):
  """Abstract densely-connected class (private, used as implementation base).

  This layer implements the Bayesian variational inference analogue to
  a dense layer by assuming the `kernel` and/or the `bias` are drawn
  from distributions. By default, the layer implements a stochastic
  forward pass via sampling from the kernel and bias posteriors,

  ```none
  kernel, bias ~ posterior
  outputs = activation(matmul(inputs, kernel) + bias)
  ```

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.

  Args:
    units: Integer or Long, dimensionality of the output space.
    activation: Activation function (`callable`). Set it to None to maintain a
      linear activation.
    activity_regularizer: Regularizer function for the output.
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
    name: Python `str`, the name of the layer. Layers with the same name will
      share `tf.Variable`s, but to avoid mistakes we require `reuse=True` in
      such cases.
    reuse: Python `bool`, whether to reuse the `tf.Variable`s of a previous
      layer by the same name.

  Properties:
    units: Python integer, dimensionality of the output space.
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
      units,
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
    super(_DenseVariational, self).__init__(
        trainable=trainable,
        name=name,
        activity_regularizer=activity_regularizer,
        **kwargs)
    self.units = units
    self.activation = activation
    self.input_spec = layers_lib.InputSpec(min_ndim=2)
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
    in_size = input_shape.with_rank_at_least(2)[-1].value
    if in_size is None:
      raise ValueError("The last dimension of the inputs to `Dense` "
                       "should be defined. Found `None`.")
    self._input_spec = layers_lib.InputSpec(min_ndim=2, axes={-1: in_size})
    dtype = dtypes.as_dtype(self.dtype)

    # Must have a posterior kernel.
    self.kernel_posterior = self.kernel_posterior_fn(
        dtype, [in_size, self.units], "kernel_posterior",
        self.trainable, self.add_variable)

    if self.kernel_prior_fn is None:
      self.kernel_prior = None
    else:
      self.kernel_prior = self.kernel_prior_fn(
          dtype, [in_size, self.units], "kernel_prior",
          self.trainable, self.add_variable)
    self._built_kernel_divergence = False

    if self.bias_posterior_fn is None:
      self.bias_posterior = None
    else:
      self.bias_posterior = self.bias_posterior_fn(
          dtype, [self.units], "bias_posterior",
          self.trainable, self.add_variable)

    if self.bias_prior_fn is None:
      self.bias_prior = None
    else:
      self.bias_prior = self.bias_prior_fn(
          dtype, [self.units], "bias_prior",
          self.trainable, self.add_variable)
    self._built_bias_divergence = False

    self.built = True

  def call(self, inputs):
    inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)

    outputs = self._apply_variational_kernel(inputs)
    outputs = self._apply_variational_bias(outputs)
    if self.activation is not None:
      outputs = self.activation(outputs)  # pylint: disable=not-callable
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
    return nn.bias_add(inputs, self.bias_posterior_tensor)

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

  def _matmul(self, inputs, kernel):
    if inputs.shape.ndims <= 2:
      return standard_ops.matmul(inputs, kernel)
    # To handle broadcasting, we must use `tensordot`.
    return standard_ops.tensordot(inputs, kernel, axes=[[-1], [0]])

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).with_rank_at_least(2)
    if input_shape[-1].value is None:
      raise ValueError(
          "The innermost dimension of input_shape must be defined, "
          "but saw: {}".format(input_shape))
    return input_shape[:-1].concatenate(self.units)


class DenseReparameterization(_DenseVariational):
  """Densely-connected layer class with reparameterization estimator.

  This layer implements the Bayesian variational inference analogue to
  a dense layer by assuming the `kernel` and/or the `bias` are drawn
  from distributions. By default, the layer implements a stochastic
  forward pass via sampling from the kernel and bias posteriors,

  ```none
  kernel, bias ~ posterior
  outputs = activation(matmul(inputs, kernel) + bias)
  ```

  It uses the reparameterization estimator [1], which performs a Monte Carlo
  approximation of the distribution integrating over the `kernel` and
  `bias`.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.

  Args:
    units: Integer or Long, dimensionality of the output space.
    activation: Activation function (`callable`). Set it to None to maintain a
      linear activation.
    activity_regularizer: Regularizer function for the output.
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
    name: Python `str`, the name of the layer. Layers with the same name will
      share `tf.Variable`s, but to avoid mistakes we require `reuse=True` in
      such cases.
    reuse: Python `bool`, whether to reuse the `tf.Variable`s of a previous
      layer by the same name.

  Properties:
    units: Python integer, dimensionality of the output space.
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

  net = tfp.layers.DenseReparameterization(
      512, activation=tf.nn.relu)(features)
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
      units,
      activation=None,
      activity_regularizer=None,
      trainable=True,
      kernel_posterior_fn=layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=lambda dtype, *args: normal_lib.Normal(  # pylint: disable=g-long-lambda
          loc=dtype.as_numpy_dtype(0.), scale=dtype.as_numpy_dtype(1.)),
      kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      bias_posterior_fn=layers_util.default_mean_field_normal_fn(
          is_singular=True),
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      name=None,
      **kwargs):
    super(DenseReparameterization, self).__init__(
        units=units,
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
        **kwargs)

  def _apply_variational_kernel(self, inputs):
    self.kernel_posterior_tensor = self.kernel_posterior_tensor_fn(
        self.kernel_posterior)
    self.kernel_posterior_affine = None
    self.kernel_posterior_affine_tensor = None
    return self._matmul(inputs, self.kernel_posterior_tensor)


def dense_reparameterization(
    inputs,
    units,
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
  """Densely-connected layer with reparameterization estimator.

  This layer implements the Bayesian variational inference analogue to
  a dense layer by assuming the `kernel` and/or the `bias` are drawn
  from distributions. By default, the layer implements a stochastic
  forward pass via sampling from the kernel and bias posteriors,

  ```none
  kernel, bias ~ posterior
  outputs = activation(matmul(inputs, kernel) + bias)
  ```

  It uses the reparameterization estimator [1], which performs a Monte Carlo
  approximation of the distribution integrating over the `kernel` and
  `bias`.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.

  Args:
    inputs: Tensor input.
    units: Integer or Long, dimensionality of the output space.
    activation: Activation function (`callable`). Set it to None to maintain a
      linear activation.
    activity_regularizer: Regularizer function for the output.
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
    name: Python `str`, the name of the layer. Layers with the same name will
      share `tf.Variable`s, but to avoid mistakes we require `reuse=True` in
      such cases.
    reuse: Python `bool`, whether to reuse the `tf.Variable`s of a previous
      layer by the same name.

  Returns:
    output: `Tensor` representing a the affine transformed input under a random
      draw from the surrogate posterior distribution.

  #### Examples

  We illustrate a Bayesian neural network with [variational inference](
  https://en.wikipedia.org/wiki/Variational_Bayesian_methods),
  assuming a dataset of `features` and `labels`.

  ```python
  tfp = tf.contrib.bayesflow

  net = tfp.layers.dense_reparameterization(
      features, 512, activation=tf.nn.relu)
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
  layer = DenseReparameterization(
      units,
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


class DenseLocalReparameterization(_DenseVariational):
  """Densely-connected layer class with local reparameterization estimator.

  This layer implements the Bayesian variational inference analogue to
  a dense layer by assuming the `kernel` and/or the `bias` are drawn
  from distributions. By default, the layer implements a stochastic
  forward pass via sampling from the kernel and bias posteriors,

  ```none
  kernel, bias ~ posterior
  outputs = activation(matmul(inputs, kernel) + bias)
  ```

  It uses the local reparameterization estimator [1], which performs a
  Monte Carlo approximation of the distribution on the hidden units
  induced by the `kernel` and `bias`.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.

  Args:
    units: Integer or Long, dimensionality of the output space.
    activation: Activation function (`callable`). Set it to None to maintain a
      linear activation.
    activity_regularizer: Regularizer function for the output.
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
    name: Python `str`, the name of the layer. Layers with the same name will
      share `tf.Variable`s, but to avoid mistakes we require `reuse=True` in
      such cases.
    reuse: Python `bool`, whether to reuse the `tf.Variable`s of a previous
      layer by the same name.

  Properties:
    units: Python integer, dimensionality of the output space.
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

  net = tfp.layers.DenseLocalReparameterization(
      512, activation=tf.nn.relu)(features)
  logits = tfp.layers.DenseLocalReparameterization(10)(net)
  neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  kl = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
  loss = neg_log_likelihood + kl
  train_op = tf.train.AdamOptimizer().minimize(loss)
  ```

  It uses local reparameterization gradients to minimize the
  Kullback-Leibler divergence up to a constant, also known as the
  negative Evidence Lower Bound. It consists of the sum of two terms:
  the expected negative log-likelihood, which we approximate via
  Monte Carlo; and the KL divergence, which is added via regularizer
  terms which are arguments to the layer.

  [1]: "Variational Dropout and the Local Reparameterization Trick."
        Diederik P. Kingma, Tim Salimans, Max Welling.
        Neural Information Processing Systems, 2015.
  """

  def __init__(
      self,
      units,
      activation=None,
      activity_regularizer=None,
      trainable=True,
      kernel_posterior_fn=layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=lambda dtype, *args: normal_lib.Normal(  # pylint: disable=g-long-lambda
          loc=dtype.as_numpy_dtype(0.), scale=dtype.as_numpy_dtype(1.)),
      kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      bias_posterior_fn=layers_util.default_mean_field_normal_fn(
          is_singular=True),
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      name=None,
      **kwargs):
    super(DenseLocalReparameterization, self).__init__(
        units=units,
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
        **kwargs)

  def _apply_variational_kernel(self, inputs):
    if (not isinstance(self.kernel_posterior, independent_lib.Independent) or
        not isinstance(self.kernel_posterior.distribution, normal_lib.Normal)):
      raise TypeError(
          "`DenseLocalReparameterization` requires "
          "`kernel_posterior_fn` produce an instance of "
          "`tf.distributions.Independent(tf.distributions.Normal)` "
          "(saw: \"{}\").".format(self.kernel_posterior.name))
    self.kernel_posterior_affine = normal_lib.Normal(
        loc=self._matmul(inputs, self.kernel_posterior.distribution.loc),
        scale=standard_ops.sqrt(self._matmul(
            standard_ops.square(inputs),
            standard_ops.square(self.kernel_posterior.distribution.scale))))
    self.kernel_posterior_affine_tensor = (
        self.kernel_posterior_tensor_fn(self.kernel_posterior_affine))
    self.kernel_posterior_tensor = None
    return self.kernel_posterior_affine_tensor


def dense_local_reparameterization(
    inputs,
    units,
    activation=None,
    activity_regularizer=None,
    trainable=True,
    kernel_posterior_fn=layers_util.default_mean_field_normal_fn(),
    kernel_posterior_tensor_fn=lambda d: d.sample(),
    kernel_prior_fn=lambda dtype, *args: normal_lib.Normal(  # pylint: disable=g-long-lambda
        loc=dtype.as_numpy_dtype(0.), scale=dtype.as_numpy_dtype(1.)),
    kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
    bias_posterior_fn=layers_util.default_mean_field_normal_fn(
        is_singular=True),
    bias_posterior_tensor_fn=lambda d: d.sample(),
    bias_prior_fn=None,
    bias_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
    name=None,
    reuse=None):
  """Densely-connected layer with local reparameterization estimator.

  This layer implements the Bayesian variational inference analogue to
  a dense layer by assuming the `kernel` and/or the `bias` are drawn
  from distributions. By default, the layer implements a stochastic
  forward pass via sampling from the kernel and bias posteriors,

  ```none
  kernel, bias ~ posterior
  outputs = activation(matmul(inputs, kernel) + bias)
  ```

  It uses the local reparameterization estimator [1], which performs a
  Monte Carlo approximation of the distribution on the hidden units
  induced by the `kernel` and `bias`.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.

  Args:
    inputs: Tensor input.
    units: Integer or Long, dimensionality of the output space.
    activation: Activation function (`callable`). Set it to None to maintain a
      linear activation.
    activity_regularizer: Regularizer function for the output.
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
    name: Python `str`, the name of the layer. Layers with the same name will
      share `tf.Variable`s, but to avoid mistakes we require `reuse=True` in
      such cases.
    reuse: Python `bool`, whether to reuse the `tf.Variable`s of a previous
      layer by the same name.

  Returns:
    output: `Tensor` representing a the affine transformed input under a random
      draw from the surrogate posterior distribution.

  #### Examples

  We illustrate a Bayesian neural network with [variational inference](
  https://en.wikipedia.org/wiki/Variational_Bayesian_methods),
  assuming a dataset of `features` and `labels`.

  ```python
  tfp = tf.contrib.bayesflow

  net = tfp.layers.dense_local_reparameterization(
      features, 512, activation=tf.nn.relu)
  logits = tfp.layers.dense_local_reparameterization(net, 10)
  neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  kl = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
  loss = neg_log_likelihood + kl
  train_op = tf.train.AdamOptimizer().minimize(loss)
  ```

  It uses local reparameterization gradients to minimize the
  Kullback-Leibler divergence up to a constant, also known as the
  negative Evidence Lower Bound. It consists of the sum of two terms:
  the expected negative log-likelihood, which we approximate via
  Monte Carlo; and the KL divergence, which is added via regularizer
  terms which are arguments to the layer.

  [1]: "Variational Dropout and the Local Reparameterization Trick."
        Diederik P. Kingma, Tim Salimans, Max Welling.
        Neural Information Processing Systems, 2015.
  """
  layer = DenseLocalReparameterization(
      units,
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


class DenseFlipout(_DenseVariational):
  """Densely-connected layer class with Flipout estimator.

  This layer implements the Bayesian variational inference analogue to
  a dense layer by assuming the `kernel` and/or the `bias` are drawn
  from distributions. By default, the layer implements a stochastic
  forward pass via sampling from the kernel and bias posteriors,

  ```none
  kernel, bias ~ posterior
  outputs = activation(matmul(inputs, kernel) + bias)
  ```

  It uses the Flipout estimator [1], which performs a Monte Carlo
  approximation of the distribution integrating over the `kernel` and
  `bias`. Flipout uses roughly twice as many floating point operations
  as the reparameterization estimator but has the advantage of
  significantly lower variance.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.

  Args:
    units: Integer or Long, dimensionality of the output space.
    activation: Activation function (`callable`). Set it to None to maintain a
      linear activation.
    activity_regularizer: Regularizer function for the output.
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
    name: Python `str`, the name of the layer. Layers with the same name will
      share `tf.Variable`s, but to avoid mistakes we require `reuse=True` in
      such cases.
    reuse: Python `bool`, whether to reuse the `tf.Variable`s of a previous
      layer by the same name.

  Properties:
    units: Python integer, dimensionality of the output space.
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

  net = tfp.layers.DenseFlipout(
      512, activation=tf.nn.relu)(features)
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
      units,
      activation=None,
      activity_regularizer=None,
      trainable=True,
      kernel_posterior_fn=layers_util.default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=lambda dtype, *args: normal_lib.Normal(  # pylint: disable=g-long-lambda
          loc=dtype.as_numpy_dtype(0.), scale=dtype.as_numpy_dtype(1.)),
      kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      bias_posterior_fn=layers_util.default_mean_field_normal_fn(
          is_singular=True),
      bias_posterior_tensor_fn=lambda d: d.sample(),
      bias_prior_fn=None,
      bias_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      seed=None,
      name=None,
      **kwargs):
    super(DenseFlipout, self).__init__(
        units=units,
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
        **kwargs)
    self.seed = seed

  def _apply_variational_kernel(self, inputs):
    if (not isinstance(self.kernel_posterior, independent_lib.Independent) or
        not isinstance(self.kernel_posterior.distribution, normal_lib.Normal)):
      raise TypeError(
          "`DenseFlipout` requires "
          "`kernel_posterior_fn` produce an instance of "
          "`tf.distributions.Independent(tf.distributions.Normal)` "
          "(saw: \"{}\").".format(self.kernel_posterior.name))
    self.kernel_posterior_affine = normal_lib.Normal(
        loc=array_ops.zeros_like(self.kernel_posterior.distribution.loc),
        scale=self.kernel_posterior.distribution.scale)
    self.kernel_posterior_affine_tensor = (
        self.kernel_posterior_tensor_fn(self.kernel_posterior_affine))
    self.kernel_posterior_tensor = None

    input_shape = array_ops.shape(inputs)
    batch_shape = input_shape[:-1]

    sign_input = layers_util.random_sign(
        input_shape,
        dtype=inputs.dtype,
        seed=self.seed)
    sign_output = layers_util.random_sign(
        array_ops.concat([batch_shape,
                          array_ops.expand_dims(self.units, 0)], 0),
        dtype=inputs.dtype,
        seed=distribution_util.gen_new_seed(
            self.seed, salt="dense_flipout"))
    perturbed_inputs = self._matmul(
        inputs * sign_input, self.kernel_posterior_affine_tensor) * sign_output

    outputs = self._matmul(inputs, self.kernel_posterior.distribution.loc)
    outputs += perturbed_inputs
    return outputs


def dense_flipout(
    inputs,
    units,
    activation=None,
    activity_regularizer=None,
    trainable=True,
    kernel_posterior_fn=layers_util.default_mean_field_normal_fn(),
    kernel_posterior_tensor_fn=lambda d: d.sample(),
    kernel_prior_fn=lambda dtype, *args: normal_lib.Normal(  # pylint: disable=g-long-lambda
        loc=dtype.as_numpy_dtype(0.), scale=dtype.as_numpy_dtype(1.)),
    kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
    bias_posterior_fn=layers_util.default_mean_field_normal_fn(
        is_singular=True),
    bias_posterior_tensor_fn=lambda d: d.sample(),
    bias_prior_fn=None,
    bias_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
    seed=None,
    name=None,
    reuse=None):
  """Densely-connected layer with Flipout estimator.

  This layer implements the Bayesian variational inference analogue to
  a dense layer by assuming the `kernel` and/or the `bias` are drawn
  from distributions. By default, the layer implements a stochastic
  forward pass via sampling from the kernel and bias posteriors,

  ```none
  kernel, bias ~ posterior
  outputs = activation(matmul(inputs, kernel) + bias)
  ```

  It uses the Flipout estimator [1], which performs a Monte Carlo
  approximation of the distribution integrating over the `kernel` and
  `bias`. Flipout uses roughly twice as many floating point operations
  as the reparameterization estimator but has the advantage of
  significantly lower variance.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.

  Args:
    inputs: Tensor input.
    units: Integer or Long, dimensionality of the output space.
    activation: Activation function (`callable`). Set it to None to maintain a
      linear activation.
    activity_regularizer: Regularizer function for the output.
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
    name: Python `str`, the name of the layer. Layers with the same name will
      share `tf.Variable`s, but to avoid mistakes we require `reuse=True` in
      such cases.
    reuse: Python `bool`, whether to reuse the `tf.Variable`s of a previous
      layer by the same name.

  Returns:
    output: `Tensor` representing a the affine transformed input under a random
      draw from the surrogate posterior distribution.

  #### Examples

  We illustrate a Bayesian neural network with [variational inference](
  https://en.wikipedia.org/wiki/Variational_Bayesian_methods),
  assuming a dataset of `features` and `labels`.

  ```python
  tfp = tf.contrib.bayesflow

  net = tfp.layers.dense_flipout(
      features, 512, activation=tf.nn.relu)
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
  layer = DenseFlipout(
      units,
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
