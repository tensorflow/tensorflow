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

@@DenseVariational
@@dense_variational

@@default_loc_scale_fn
@@default_mean_field_normal_fn
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.ops import deterministic as deterministic_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_lib
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops.distributions import kullback_leibler as kl_lib
from tensorflow.python.ops.distributions import normal as normal_lib


__all__ = [
    "DenseVariational",
    "dense_variational",
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
    """Creates a batch of `Deterministic` or `Normal` distributions."""
    loc, scale = loc_scale_fn_(dtype, shape, name, trainable, add_variable_fn)
    if scale is None:
      return deterministic_lib.Deterministic(loc=loc)
    return normal_lib.Normal(loc=loc, scale=scale)
  return _fn


class DenseVariational(layers_lib.Layer):
  """Densely-connected variational class.

  This layer implements the Bayesian variational inference analogue to:
  `outputs = activation(matmul(inputs, kernel) + bias)`
  by assuming the `kernel` and/or the `bias` are random variables.

  The layer implements a stochastic dense calculation by making a Monte Carlo
  approximation of a [variational Bayesian method based on KL divergence](
  https://en.wikipedia.org/wiki/Variational_Bayesian_methods), i.e.,

  ```none
  -log p(y|x) = -log int_{R**d} p(y|x,w) p(w) dw
              = -log int_{R**d} p(y,w|x) q(w|x) / q(w|x) dw
             <= E_q(W|x)[-log p(y,W|x) + log q(W|x)]       # Jensen's
              = E_q(W|x)[-log p(y|x,W)] + KL[q(W|x), p(W)]
             ~= m**-1 sum{ -log(y|x,w[j]) : w[j] ~ q(W|x), j=1..m }
                 + KL[q(W|x), p(W)]
  ```

  where `W` denotes the (independent) `kernel` and `bias` random variables, `w`
  is a random variate or outcome of `W`, `y` is the label, `x` is the evidence`,
  and `~=` denotes an approximation which becomes exact as `m->inf`. The above
  bound is sometimes referred to as the negative Evidence Lower BOund or
  negative [ELBO](https://arxiv.org/abs/1601.00670). In context of a DNN, this
  layer is appropriate to use when the final loss is a negative log-likelihood.

  The Monte-Carlo sum portion is used for the feed-forward calculation of the
  DNN. The KL divergence portion can be added to the final loss via:
  `loss += sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))`.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  random variables (which together comprise `W`).

  Args:
    units: Integer or Long, dimensionality of the output space.
    activation: Activation function (`callable`). Set it to None to maintain a
      linear activation.
    activity_regularizer: Regularizer function for the output.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    kernel_use_local_reparameterization: Python `bool` indicating whether
      `kernel` calculation should employ the Local Reparameterization Trick.
      When `True`, `kernel_posterior_fn` must create an instance of
      `tf.distributions.Normal`.
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
    kernel_use_local_reparameterization: Python `bool` indicating whether
      `kernel` calculation should employ the Local Reparameterization Trick.
    kernel: `VariationalKernelParamater` instance containing all `kernel`
      related properties and `callable`s.
    bias: `VariationalParameter` instance containing all `kernel`
      related properties and `callable`s.
  """

  def __init__(
      self,
      units,
      activation=None,
      activity_regularizer=None,
      trainable=True,
      kernel_use_local_reparameterization=True,
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
    super(DenseVariational, self).__init__(
        trainable=trainable,
        name=name,
        activity_regularizer=activity_regularizer,
        **kwargs)
    self._units = units
    self._activation = activation
    self._input_spec = layers_lib.InputSpec(min_ndim=2)
    self._kernel_use_local_reparameterization = (
        kernel_use_local_reparameterization)
    self._kernel = VariationalKernelParameter(
        kernel_posterior_fn,
        kernel_posterior_tensor_fn,
        kernel_prior_fn,
        kernel_divergence_fn)
    self._bias = VariationalParameter(
        bias_posterior_fn,
        bias_posterior_tensor_fn,
        bias_prior_fn,
        bias_divergence_fn)

  @property
  def units(self):
    return self._units

  @property
  def activation(self):
    return self._activation

  @property
  def input_spec(self):
    return self._input_spec

  @input_spec.setter
  def input_spec(self, value):
    self._input_spec = value

  @property
  def kernel_use_local_reparameterization(self):
    return self._kernel_use_local_reparameterization

  @property
  def kernel(self):
    return self._kernel

  @property
  def bias(self):
    return self._bias

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    in_size = input_shape.with_rank_at_least(2)[-1].value
    if in_size is None:
      raise ValueError("The last dimension of the inputs to `Dense` "
                       "should be defined. Found `None`.")
    self._input_spec = layers_lib.InputSpec(min_ndim=2, axes={-1: in_size})
    dtype = dtypes.as_dtype(self.dtype)

    # Must have a posterior kernel.
    self.kernel.posterior = self.kernel.posterior_fn(
        dtype, [in_size, self.units], "kernel_posterior",
        self.trainable, self.add_variable)

    if self.kernel.prior_fn is None:
      self.kernel_prior = None
    else:
      self.kernel.prior = self.kernel.prior_fn(
          dtype, [in_size, self.units], "kernel_prior",
          self.trainable, self.add_variable)
    self._built_kernel_divergence = False

    if self.bias.posterior_fn is None:
      self.bias.posterior = None
    else:
      self.bias.posterior = self.bias.posterior_fn(
          dtype, [self.units], "bias_posterior",
          self.trainable, self.add_variable)

    if self.bias.prior_fn is None:
      self.bias.prior = None
    else:
      self.bias.prior = self.bias.prior_fn(
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
      self._apply_divergence(self.kernel, name="divergence_kernel")
      self._built_kernel_divergence = True
    if not self._built_bias_divergence:
      self._apply_divergence(self.bias, name="divergence_bias")
      self._built_bias_divergence = True
    return outputs

  def _apply_variational_kernel(self, inputs):
    if not self.kernel_use_local_reparameterization:
      self.kernel.posterior_tensor = self.kernel.posterior_tensor_fn(
          self.kernel.posterior)
      self.kernel.posterior_affine = None
      self.kernel.posterior_affine_tensor = None
      return self._matmul(inputs, self.kernel.posterior_tensor)
    if not isinstance(self.kernel.posterior, normal_lib.Normal):
      raise TypeError("`kernel_use_local_reparameterization=True` requires "
                      "`kernel_posterior_fn` produce an instance of "
                      "`tf.distributions.Normal` (saw: \"{}\").".format(
                          type(self.kernel.posterior).__name__))
    self.kernel.posterior_affine = normal_lib.Normal(
        loc=self._matmul(inputs, self.kernel.posterior.loc),
        scale=standard_ops.sqrt(self._matmul(
            standard_ops.square(inputs),
            standard_ops.square(self.kernel.posterior.scale))))
    self.kernel.posterior_affine_tensor = (
        self.kernel.posterior_tensor_fn(self.kernel.posterior_affine))
    self.kernel.posterior_tensor = None
    return self.kernel.posterior_affine_tensor

  def _apply_variational_bias(self, inputs):
    if self.bias.posterior is None:
      self.bias.posterior_tensor = None
      return inputs
    self.bias.posterior_tensor = self.bias.posterior_tensor_fn(
        self.bias.posterior)
    return nn.bias_add(inputs, self.bias.posterior_tensor)

  def _apply_divergence(self, param, name):
    if (param.divergence_fn is None or
        param.posterior is None or
        param.prior is None):
      param.divergence = None
      return
    param.divergence = standard_ops.identity(
        param.divergence_fn(
            param.posterior, param.prior, param.posterior_tensor),
        name=name)
    self.add_loss(param.divergence)

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


def dense_variational(
    inputs,
    units,
    activation=None,
    activity_regularizer=None,
    trainable=True,
    kernel_use_local_reparameterization=True,
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
  """Densely-connected variational layer.

  This layer implements the Bayesian variational inference analogue to:
  `outputs = activation(matmul(inputs, kernel) + bias)`
  by assuming the `kernel` and/or the `bias` are random variables.

  The layer implements a stochastic dense calculation by making a Monte Carlo
  approximation of a [variational Bayesian method based on KL divergence](
  https://en.wikipedia.org/wiki/Variational_Bayesian_methods), i.e.,

  ```none
  -log p(y|x) = -log int_{R**d} p(y|x,w) p(w) dw
              = -log int_{R**d} p(y,w|x) q(w|x) / q(w|x) dw
             <= E_q(W|x)[-log p(y,W|x) + log q(W|x)]       # Jensen's
              = E_q(W|x)[-log p(y|x,W)] + KL[q(W|x), p(W)]
             ~= m**-1 sum{ -log(y|x,w[j]) : w[j] ~ q(W|x), j=1..m }
                 + KL[q(W|x), p(W)]
  ```

  where `W` denotes the (independent) `kernel` and `bias` random variables, `w`
  is a random variate or outcome of `W`, `y` is the label, `x` is the evidence`,
  and `~=` denotes an approximation which becomes exact as `m->inf`. The above
  bound is sometimes referred to as the negative Evidence Lower BOund or
  negative [ELBO](https://arxiv.org/abs/1601.00670). In context of a DNN, this
  layer is appropriate to use when the final loss is a negative log-likelihood.

  The Monte-Carlo sum portion is used for the feed-forward calculation of the
  DNN. The KL divergence portion can be added to the final loss via:
  `loss += sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))`.

  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  random variables (which together comprise `W`).

  Args:
    inputs: Tensor input.
    units: Integer or Long, dimensionality of the output space.
    activation: Activation function (`callable`). Set it to None to maintain a
      linear activation.
    activity_regularizer: Regularizer function for the output.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    kernel_use_local_reparameterization: Python `bool` indicating whether
      `kernel` calculation should employ the Local Reparameterization Trick.
      When `True`, `kernel_posterior_fn` must create an instance of
      `tf.distributions.Normal`.
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
  """
  layer = DenseVariational(
      units,
      activation=activation,
      activity_regularizer=activity_regularizer,
      trainable=trainable,
      kernel_use_local_reparameterization=(
          kernel_use_local_reparameterization),
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


class NotSet(object):
  """Helper to track whether a `VariationalParameter` value has been set."""
  pass


class VariationalParameter(object):
  """Struct-like container of variational parameter properties.

  A `VariationalParameter` is intitialized with Python `callable`s which set the
  value of correspondingly named members. Corresponding values have "set once"
  semantics, i.e., once set to any value they are immutable.
  """

  def __init__(
      self,
      posterior_fn,
      posterior_tensor_fn,
      prior_fn,
      divergence_fn):
    """Creates the `VariationalParameter` struct-like object.

    Args:
      posterior_fn: Python `callable` which creates a
        `tf.distribution.Distribution` like object representing the posterior
        distribution. See `VariationalParameter.posterior_fn` for `callable`'s
        required parameters.
      posterior_tensor_fn: Python `callable` which computes a `Tensor`
        which represents the `posterior`.
      prior_fn: Python `callable` which creates a
        `tf.distribution.Distribution` like object representing the prior
        distribution. See `VariationalParameter.prior_fn` for `callable`'s
        required parameters.
      divergence_fn: Python `callable` which computes the KL divergence from
        `posterior` to `prior`. See `VariationalParameter.divergence_fn` for
        required `callable`'s parameters.
    """
    self._posterior_fn = posterior_fn
    self._posterior = NotSet()
    self._posterior_tensor_fn = posterior_tensor_fn
    self._posterior_tensor = NotSet()
    self._prior_fn = prior_fn
    self._prior = NotSet()
    self._divergence_fn = divergence_fn
    self._divergence = NotSet()
    self._init_helper()

  @property
  def posterior_fn(self):
    """`callable` which creates `tf.distributions.Distribution`-like posterior.

    The `callable` must accept the following parameters:
      name: Python `str` name prepended to any created (or existing)
        `tf.Variable`s.
      shape: Python `list`-like representing the parameter's event shape.
      dtype: Type of parameter's event.
      trainable: Python `bool` indicating all created `tf.Variable`s should be
        added to the graph collection `GraphKeys.TRAINABLE_VARIABLES`.
      add_variable_fn: `tf.get_variable`-like `callable` used to create (or
        access existing) `tf.Variable`s.

    Returns:
      posterior_fn: The Python `callable` specified in `__init__`.
    """
    return self._posterior_fn

  @property
  def posterior(self):
    """`tf.distributions.Distribution`-like instance representing posterior."""
    return self._posterior

  @posterior.setter
  def posterior(self, value):
    """One-time setter of the `posterior` distribution."""
    if not isinstance(self._posterior, NotSet):
      raise ValueError("Cannot override already set attribute.")
    self._posterior = value

  @property
  def posterior_tensor_fn(self):
    """Creates `Tensor` representing the `posterior` distribution.

    The `callable` must accept the following parameters:
      posterior: `tf.distributions.Distribution`-like instance.

    Returns:
      posterior_tensor_fn: The Python `callable` specified in
        `__init__`.
    """
    return self._posterior_tensor_fn

  @property
  def posterior_tensor(self):
    """`Tensor` representing the `posterior` distribution."""
    return self._posterior_tensor

  @posterior_tensor.setter
  def posterior_tensor(self, value):
    """One-time setter of the `posterior_tensor`."""
    if not isinstance(self._posterior_tensor, NotSet):
      raise ValueError("Cannot override already set attribute.")
    self._posterior_tensor = value

  @property
  def prior_fn(self):
    """`callable` which creates `tf.distributions.Distribution`-like prior.

    The `callable` must accept the following parameters:
      name: Python `str` name prepended to any created (or existing)
        `tf.Variable`s.
      shape: Python `list`-like representing the parameter's event shape.
      dtype: Type of parameter's event.
      trainable: Python `bool` indicating all created `tf.Variable`s should be
        added to the graph collection `GraphKeys.TRAINABLE_VARIABLES`.
      add_variable_fn: `tf.get_variable`-like `callable` used to create (or
        access existing) `tf.Variable`s.

    Returns:
      prior_fn: The Python `callable` specified in `__init__`.
    """
    return self._prior_fn

  @property
  def prior(self):
    """`tf.distributions.Distribution`-like instance representing posterior."""
    return self._prior

  @prior.setter
  def prior(self, value):
    """One-time setter of the `prior` distribution."""
    if not isinstance(self._prior, NotSet):
      raise ValueError("Cannot override already set attribute.")
    self._prior = value

  @property
  def divergence_fn(self):
    """`callable` which computes KL-divergence `Tensor` from posterior to prior.

    The `callable` must accept the following parameters:
      posterior: `tf.distributions.Distribution`-like instance.
      prior: `tf.distributions.Distribution`-like instance.
      posterior_tensor: `Tensor` representing value of posterior.

    Returns:
      divergence_fn: The Python `callable` specified in `__init__`.
    """
    return self._divergence_fn

  @property
  def divergence(self):
    """`Tensor` representing KL-divergence from posterior to prior."""
    return self._divergence

  @divergence.setter
  def divergence(self, value):
    """One-time setter of the `divergence`."""
    if not isinstance(self._divergence, NotSet):
      raise ValueError("Cannot override already set attribute.")
    self._divergence = value

  def _init_helper(self):
    pass


class VariationalKernelParameter(VariationalParameter):
  """Struct-like container of variational kernel properties.

  A `VariationalKernelParameter` is intitialized with Python `callable`s which
  set the value of correspondingly named members. Corresponding values have "set
  once" semantics, i.e., once set to any value they are immutable.
  """

  @property
  def posterior_affine(self):
    """`tf.distributions.Distribution` affine transformed posterior."""
    return self._posterior_affine

  @posterior_affine.setter
  def posterior_affine(self, value):
    """One-time setter of `posterior_affine`."""
    if not isinstance(self._posterior_affine, NotSet):
      raise ValueError("Cannot override already set attribute.")
    self._posterior_affine = value

  @property
  def posterior_affine_tensor(self):
    """`Tensor` representing the `posterior_affine` distribution."""
    return self._posterior_affine_tensor

  @posterior_affine_tensor.setter
  def posterior_affine_tensor(self, value):
    """One-time setter of the `posterior_affine_tensor`."""
    if not isinstance(self._posterior_affine_tensor, NotSet):
      raise ValueError("Cannot override already set attribute.")
    self._posterior_affine_tensor = value

  def _init_helper(self):
    self._posterior_affine = NotSet()
    self._posterior_affine_tensor = NotSet()
