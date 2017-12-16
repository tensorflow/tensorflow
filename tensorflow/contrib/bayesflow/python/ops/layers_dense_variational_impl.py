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

@@DenseReparameterization
@@DenseLocalReparameterization
@@DenseFlipout
@@dense_reparameterization
@@dense_local_reparameterization
@@dense_flipout

@@default_loc_scale_fn
@@default_mean_field_normal_fn
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.ops import deterministic as deterministic_lib
from tensorflow.contrib.distributions.python.ops import independent as independent_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops.distributions import kullback_leibler as kl_lib
from tensorflow.python.ops.distributions import normal as normal_lib
from tensorflow.python.ops.distributions import util as distribution_util


__all__ = [
    "DenseReparameterization",
    "DenseLocalReparameterization",
    "DenseFlipout",
    "dense_reparameterization",
    "dense_local_reparameterization",
    "dense_flipout",
    "default_loc_scale_fn",
    "default_mean_field_normal_fn",
]


def default_loc_scale_fn(
    is_singular=False,
    loc_initializer=init_ops.random_normal_initializer(stddev=0.1),
    untransformed_scale_initializer=init_ops.random_normal_initializer(
        mean=-3., stddev=0.1),
    loc_regularizer=None,
    untransformed_scale_regularizer=None,
    loc_constraint=None,
    untransformed_scale_constraint=None):
  """Makes closure which creates `loc`, `scale` params from `tf.get_variable`.

  This function produces a closure which produces `loc`, `scale` using
  `tf.get_variable`. The closure accepts the following arguments:

    dtype: Type of parameter's event.
    shape: Python `list`-like representing the parameter's event shape.
    name: Python `str` name prepended to any created (or existing)
      `tf.Variable`s.
    trainable: Python `bool` indicating all created `tf.Variable`s should be
      added to the graph collection `GraphKeys.TRAINABLE_VARIABLES`.
    add_variable_fn: `tf.get_variable`-like `callable` used to create (or
      access existing) `tf.Variable`s.

  Args:
    is_singular: Python `bool` indicating if `scale is None`. Default: `False`.
    loc_initializer: Initializer function for the `loc` parameters.
      The default is `tf.random_normal_initializer(mean=0., stddev=0.1)`.
    untransformed_scale_initializer: Initializer function for the `scale`
      parameters. Default value: `tf.random_normal_initializer(mean=-3.,
      stddev=0.1)`. This implies the softplus transformed result has mean
      approximately `0.05` and std. deviation approximately `0.005`.
    loc_regularizer: Regularizer function for the `loc` parameters.
      The default (`None`) is to use the `tf.get_variable` default.
    untransformed_scale_regularizer: Regularizer function for the `scale`
      parameters. The default (`None`) is to use the `tf.get_variable` default.
    loc_constraint: An optional projection function to be applied to the
      loc after being updated by an `Optimizer`. The function must take as input
      the unprojected variable and must return the projected variable (which
      must have the same shape). Constraints are not safe to use when doing
      asynchronous distributed training.
      The default (`None`) is to use the `tf.get_variable` default.
    untransformed_scale_constraint: An optional projection function to be
      applied to the `scale` parameters after being updated by an `Optimizer`
      (e.g. used to implement norm constraints or value constraints). The
      function must take as input the unprojected variable and must return the
      projected variable (which must have the same shape). Constraints are not
      safe to use when doing asynchronous distributed training. The default
      (`None`) is to use the `tf.get_variable` default.

  Returns:
    default_loc_scale_fn: Python `callable` which instantiates `loc`, `scale`
    parameters from args: `dtype, shape, name, trainable, add_variable_fn`.
  """
  def _fn(dtype, shape, name, trainable, add_variable_fn):
    """Creates `loc`, `scale` parameters."""
    loc = add_variable_fn(
        name=name + "_loc",
        shape=shape,
        initializer=loc_initializer,
        regularizer=loc_regularizer,
        constraint=loc_constraint,
        dtype=dtype,
        trainable=trainable)
    if is_singular:
      return loc, None
    untransformed_scale = add_variable_fn(
        name=name + "_untransformed_scale",
        shape=shape,
        initializer=untransformed_scale_initializer,
        regularizer=untransformed_scale_regularizer,
        constraint=untransformed_scale_constraint,
        dtype=dtype,
        trainable=trainable)
    scale = (np.finfo(dtype.as_numpy_dtype).eps +
             nn_ops.softplus(untransformed_scale))
    return loc, scale
  return _fn


def default_mean_field_normal_fn(
    is_singular=False,
    loc_initializer=None,
    untransformed_scale_initializer=None,
    loc_regularizer=None,
    untransformed_scale_regularizer=None,
    loc_constraint=None,
    untransformed_scale_constraint=None):
  """Creates a function to build Normal distributions with trainable params.

  This function produces a closure which produces `tf.distributions.Normal`
  parameterized by a loc` and `scale` each created using `tf.get_variable`. The
  produced closure accepts the following arguments:

    name: Python `str` name prepended to any created (or existing)
      `tf.Variable`s.
    shape: Python `list`-like representing the parameter's event shape.
    dtype: Type of parameter's event.
    trainable: Python `bool` indicating all created `tf.Variable`s should be
      added to the graph collection `GraphKeys.TRAINABLE_VARIABLES`.
    add_variable_fn: `tf.get_variable`-like `callable` used to create (or
      access existing) `tf.Variable`s.

  Args:
    is_singular: Python `bool` if `True`, forces the special case limit of
      `scale->0`, i.e., a `Deterministic` distribution.
    loc_initializer: Initializer function for the `loc` parameters.
      If `None` (default), values are initialized using the default
      initializer used by `tf.get_variable`.
    untransformed_scale_initializer: Initializer function for the `scale`
      parameters. If `None` (default), values are initialized using the default
      initializer used by `tf.get_variable`.
    loc_regularizer: Regularizer function for the `loc` parameters.
    untransformed_scale_regularizer: Regularizer function for the `scale`
      parameters.
    loc_constraint: An optional projection function to be applied to the
      loc after being updated by an `Optimizer`. The function must take as input
      the unprojected variable and must return the projected variable (which
      must have the same shape). Constraints are not safe to use when doing
      asynchronous distributed training.
    untransformed_scale_constraint: An optional projection function to be
      applied to the `scale` parameters after being updated by an `Optimizer`
      (e.g. used to implement norm constraints or value constraints). The
      function must take as input the unprojected variable and must return the
      projected variable (which must have the same shape). Constraints are not
      safe to use when doing asynchronous distributed training.

  Returns:
    make_normal_fn: Python `callable` which creates a `tf.distributions.Normal`
      using from args: `dtype, shape, name, trainable, add_variable_fn`.
  """
  loc_scale_fn_ = default_loc_scale_fn(
      is_singular,
      loc_initializer,
      untransformed_scale_initializer,
      loc_regularizer,
      untransformed_scale_regularizer,
      loc_constraint,
      untransformed_scale_constraint)
  def _fn(dtype, shape, name, trainable, add_variable_fn):
    """Creates multivariate `Deterministic` or `Normal` distribution."""
    loc, scale = loc_scale_fn_(dtype, shape, name, trainable, add_variable_fn)
    if scale is None:
      dist = deterministic_lib.Deterministic(loc=loc)
    else:
      dist = normal_lib.Normal(loc=loc, scale=scale)
    reinterpreted_batch_ndims = array_ops.shape(dist.batch_shape_tensor())[0]
    return independent_lib.Independent(
        dist, reinterpreted_batch_ndims=reinterpreted_batch_ndims)
  return _fn


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
      kernel_posterior_fn=default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=lambda dtype, *args: normal_lib.Normal(  # pylint: disable=g-long-lambda
          loc=dtype.as_numpy_dtype(0.), scale=dtype.as_numpy_dtype(1.)),
      kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      bias_posterior_fn=default_mean_field_normal_fn(is_singular=True),
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
  """

  def __init__(
      self,
      units,
      activation=None,
      activity_regularizer=None,
      trainable=True,
      kernel_posterior_fn=default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=lambda dtype, *args: normal_lib.Normal(  # pylint: disable=g-long-lambda
          loc=dtype.as_numpy_dtype(0.), scale=dtype.as_numpy_dtype(1.)),
      kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      bias_posterior_fn=default_mean_field_normal_fn(is_singular=True),
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
    kernel_posterior_fn=default_mean_field_normal_fn(),
    kernel_posterior_tensor_fn=lambda d: d.sample(),
    kernel_prior_fn=lambda dtype, *args: normal_lib.Normal(  # pylint: disable=g-long-lambda
        loc=dtype.as_numpy_dtype(0.), scale=dtype.as_numpy_dtype(1.)),
    kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
    bias_posterior_fn=default_mean_field_normal_fn(is_singular=True),
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
  """

  def __init__(
      self,
      units,
      activation=None,
      activity_regularizer=None,
      trainable=True,
      kernel_posterior_fn=default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=lambda dtype, *args: normal_lib.Normal(  # pylint: disable=g-long-lambda
          loc=dtype.as_numpy_dtype(0.), scale=dtype.as_numpy_dtype(1.)),
      kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      bias_posterior_fn=default_mean_field_normal_fn(is_singular=True),
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
          "(saw: \"{}\").".format(type(self.kernel_posterior).__name__))
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
    kernel_posterior_fn=default_mean_field_normal_fn(),
    kernel_posterior_tensor_fn=lambda d: d.sample(),
    kernel_prior_fn=lambda dtype, *args: normal_lib.Normal(  # pylint: disable=g-long-lambda
        loc=dtype.as_numpy_dtype(0.), scale=dtype.as_numpy_dtype(1.)),
    kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
    bias_posterior_fn=default_mean_field_normal_fn(is_singular=True),
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
  """

  def __init__(
      self,
      units,
      activation=None,
      activity_regularizer=None,
      trainable=True,
      kernel_posterior_fn=default_mean_field_normal_fn(),
      kernel_posterior_tensor_fn=lambda d: d.sample(),
      kernel_prior_fn=lambda dtype, *args: normal_lib.Normal(  # pylint: disable=g-long-lambda
          loc=dtype.as_numpy_dtype(0.), scale=dtype.as_numpy_dtype(1.)),
      kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
      bias_posterior_fn=default_mean_field_normal_fn(is_singular=True),
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
          "(saw: \"{}\").".format(type(self.kernel_posterior).__name__))
    self.kernel_posterior_affine = normal_lib.Normal(
        loc=array_ops.zeros_like(self.kernel_posterior.distribution.loc),
        scale=self.kernel_posterior.distribution.scale)
    self.kernel_posterior_affine_tensor = (
        self.kernel_posterior_tensor_fn(self.kernel_posterior_affine))
    self.kernel_posterior_tensor = None

    input_shape = array_ops.shape(inputs)
    batch_shape = input_shape[:-1]

    sign_input = random_sign(input_shape, dtype=inputs.dtype, seed=self.seed)
    sign_output = random_sign(
        array_ops.concat([batch_shape,
                          array_ops.expand_dims(self.units, 0)], 0),
        dtype=inputs.dtype,
        seed=distribution_util.gen_new_seed(
            self.seed, salt="conv_variational"))
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
    kernel_posterior_fn=default_mean_field_normal_fn(),
    kernel_posterior_tensor_fn=lambda d: d.sample(),
    kernel_prior_fn=lambda dtype, *args: normal_lib.Normal(  # pylint: disable=g-long-lambda
        loc=dtype.as_numpy_dtype(0.), scale=dtype.as_numpy_dtype(1.)),
    kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
    bias_posterior_fn=default_mean_field_normal_fn(is_singular=True),
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


def random_sign(shape, dtype=dtypes.float32, seed=None):
  """Draw values from {-1, 1} uniformly, i.e., Rademacher distribution."""
  random_bernoulli = random_ops.random_uniform(shape, minval=0, maxval=2,
                                               dtype=dtypes.int32,
                                               seed=seed)
  return math_ops.cast(2 * random_bernoulli - 1, dtype)
