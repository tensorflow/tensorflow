# -*- coding: utf-8 -*-
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Entropy bottleneck layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.coder.python.ops import coder_ops

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import engine
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.summary import summary


class EntropyBottleneck(engine.Layer):
  """Entropy bottleneck layer.

  This layer can be used to model the entropy (the amount of information
  conveyed) of the tensor passing through it. During training, this can be used
  to impose a (soft) entropy constraint on its activations, limiting the amount
  of information flowing through the layer. Note that this is distinct from
  other types of bottlenecks, which reduce the dimensionality of the space, for
  example. Dimensionality reduction does not limit the amount of information,
  and does not enable efficient data compression per se.

  After training, this layer can be used to compress any input tensor to a
  string, which may be written to a file, and to decompress a file which it
  previously generated back to a reconstructed tensor (possibly on a different
  machine having access to the same model checkpoint). The entropies estimated
  during training or evaluation are approximately equal to the average length of
  the strings in bits.

  The layer implements a flexible probability density model to estimate entropy,
  which is described in the appendix of the paper (please cite the paper if you
  use this code for scientific work):

  "Variational image compression with a scale hyperprior"

  Johannes Ballé, David Minnen, Saurabh Singh, Sung Jin Hwang, Nick Johnston

  https://arxiv.org/abs/1802.01436

  The layer assumes that the input tensor is at least 2D, with a batch dimension
  at the beginning and a channel dimension as specified by `data_format`. The
  layer trains an independent probability density model for each channel, but
  assumes that across all other dimensions, the inputs are i.i.d. (independent
  and identically distributed). Because the entropy (and hence, average
  codelength) is a function of the densities, this assumption may have a direct
  effect on the compression performance.

  Because data compression always involves discretization, the outputs of the
  layer are generally only approximations of its inputs. During training,
  discretization is modeled using additive uniform noise to ensure
  differentiability. The entropies computed during training are differential
  entropies. During evaluation, the data is actually quantized, and the
  entropies are discrete (Shannon entropies). To make sure the approximated
  tensor values are good enough for practical purposes, the training phase must
  be used to balance the quality of the approximation with the entropy, by
  adding an entropy term to the training loss, as in the following example.

  Here, we use the entropy bottleneck to compress the latent representation of
  an autoencoder. The data vectors `x` in this case are 4D tensors in
  `'channels_last'` format (for example, 16x16 pixel grayscale images).

  The layer always produces exactly one auxiliary loss and one update op which
  are only significant for compression and decompression. To use the compression
  feature, the auxiliary loss must be minimized during or after training. After
  that, the update op must be executed at least once. Here, we simply attach
  them to the main training step.

  Training:
  ```
  # Build autoencoder.
  x = tf.placeholder(tf.float32, shape=[None, 16, 16, 1])
  y = forward_transform(x)
  entropy_bottleneck = EntropyBottleneck()
  y_, likelihoods = entropy_bottleneck(y, training=True)
  x_ = backward_transform(y_)

  # Information content (= predicted codelength) in bits of each batch element
  # (note that taking the natural logarithm and dividing by `log(2)` is
  # equivalent to taking base-2 logarithms):
  bits = tf.reduce_sum(tf.log(likelihoods), axis=(1, 2, 3)) / -np.log(2)

  # Squared difference of each batch element:
  squared_error = tf.reduce_sum(tf.squared_difference(x, x_), axis=(1, 2, 3))

  # The loss is a weighted sum of mean squared error and entropy (average
  # information content), where the weight controls the trade-off between
  # approximation error and entropy.
  main_loss = 0.5 * tf.reduce_mean(squared_error) + tf.reduce_mean(bits)

  # Minimize loss and auxiliary loss, and execute update op.
  main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
  main_step = optimizer.minimize(main_loss)
  # 1e-2 is a good starting point for the learning rate of the auxiliary loss,
  # assuming Adam is used.
  aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
  aux_step = optimizer.minimize(entropy_bottleneck.losses[0])
  step = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])
  ```

  Evaluation:
  ```
  # Build autoencoder.
  x = tf.placeholder(tf.float32, shape=[None, 16, 16, 1])
  y = forward_transform(x)
  y_, likelihoods = EntropyBottleneck()(y, training=False)
  x_ = backward_transform(y_)

  # Information content (= predicted codelength) in bits of each batch element:
  bits = tf.reduce_sum(tf.log(likelihoods), axis=(1, 2, 3)) / -np.log(2)

  # Squared difference of each batch element:
  squared_error = tf.reduce_sum(tf.squared_difference(x, x_), axis=(1, 2, 3))

  # The loss is a weighted sum of mean squared error and entropy (average
  # information content), where the weight controls the trade-off between
  # approximation error and entropy.
  loss = 0.5 * tf.reduce_mean(squared_error) + tf.reduce_mean(bits)
  ```

  To be able to compress the bottleneck tensor and decompress it in a different
  session, or on a different machine, you need three items:
  - The compressed representations stored as strings.
  - The shape of the bottleneck for these string representations as a `Tensor`,
    as well as the number of channels of the bottleneck at graph construction
    time.
  - The checkpoint of the trained model that was used for compression. Note:
    It is crucial that the auxiliary loss produced by this layer is minimized
    during or after training, and that the update op is run after training and
    minimization of the auxiliary loss, but *before* the checkpoint is saved.

  Compression:
  ```
  x = tf.placeholder(tf.float32, shape=[None, 16, 16, 1])
  y = forward_transform(x)
  strings = EntropyBottleneck().compress(y)
  shape = tf.shape(y)[1:]
  ```

  Decompression:
  ```
  strings = tf.placeholder(tf.string, shape=[None])
  shape = tf.placeholder(tf.int32, shape=[3])
  entropy_bottleneck = EntropyBottleneck(dtype=tf.float32)
  y_ = entropy_bottleneck.decompress(strings, shape, channels=5)
  x_ = backward_transform(y_)
  ```
  Here, we assumed that the tensor produced by the forward transform has 5
  channels.

  The above four use cases can also be implemented within the same session (i.e.
  on the same `EntropyBottleneck` instance), for testing purposes, etc., by
  calling the object more than once.

  Arguments:
    init_scale: Float. A scaling factor determining the initial width of the
      probability densities. This should be chosen big enough so that the
      range of values of the layer inputs roughly falls within the interval
      [`-init_scale`, `init_scale`] at the beginning of training.
    filters: An iterable of ints, giving the number of filters at each layer of
      the density model. Generally, the more filters and layers, the more
      expressive is the density model in terms of modeling more complicated
      distributions of the layer inputs. For details, refer to the paper
      referenced above. The default is `[3, 3, 3]`, which should be sufficient
      for most practical purposes.
    tail_mass: Float, between 0 and 1. The bottleneck layer automatically
      determines the range of input values that should be represented based on
      their frequency of occurrence. Values occurring in the tails of the
      distributions will be clipped to that range during compression.
      `tail_mass` determines the amount of probability mass in the tails which
      is cut off in the worst case. For example, the default value of `1e-9`
      means that at most 1 in a billion input samples will be clipped to the
      range.
    optimize_integer_offset: Boolean. Typically, the input values of this layer
      are floats, which means that quantization during evaluation can be
      performed with an arbitrary offset. By default, the layer determines that
      offset automatically. In special situations, such as when it is known that
      the layer will receive only full integer values during evaluation, it can
      be desirable to set this argument to `False` instead, in order to always
      quantize to full integer values.
    likelihood_bound: Float. If positive, the returned likelihood values are
      ensured to be greater than or equal to this value. This prevents very
      large gradients with a typical entropy loss (defaults to 1e-9).
    range_coder_precision: Integer, between 1 and 16. The precision of the range
      coder used for compression and decompression. This trades off computation
      speed with compression efficiency, where 16 is the slowest but most
      efficient setting. Choosing lower values may increase the average
      codelength slightly compared to the estimated entropies.
    data_format: Either `'channels_first'` or `'channels_last'` (default).
    trainable: Boolean. Whether the layer should be trained.
    name: String. The name of the layer.
    dtype: Default dtype of the layer's parameters (default of `None` means use
      the type of the first input).

  Read-only properties:
    init_scale: See above.
    filters: See above.
    tail_mass: See above.
    optimize_integer_offset: See above.
    likelihood_bound: See above.
    range_coder_precision: See above.
    data_format: See above.
    name: String. See above.
    dtype: See above.
    trainable_variables: List of trainable variables.
    non_trainable_variables: List of non-trainable variables.
    variables: List of all variables of this layer, trainable and non-trainable.
    updates: List of update ops of this layer. Always contains exactly one
      update op, which must be run once after the last training step, before
      `compress` or `decompress` is used.
    losses: List of losses added by this layer. Always contains exactly one
      auxiliary loss, which must be added to the training loss.

  Mutable properties:
    trainable: Boolean. Whether the layer should be trained.
    input_spec: Optional `InputSpec` object specifying the constraints on inputs
      that can be accepted by the layer.
  """

  def __init__(self, init_scale=10, filters=(3, 3, 3), tail_mass=1e-9,
               optimize_integer_offset=True, likelihood_bound=1e-9,
               range_coder_precision=16, data_format="channels_last", **kwargs):
    super(EntropyBottleneck, self).__init__(**kwargs)
    self._init_scale = float(init_scale)
    self._filters = tuple(int(f) for f in filters)
    self._tail_mass = float(tail_mass)
    if not 0 < self.tail_mass < 1:
      raise ValueError(
          "`tail_mass` must be between 0 and 1, got {}.".format(self.tail_mass))
    self._optimize_integer_offset = bool(optimize_integer_offset)
    self._likelihood_bound = float(likelihood_bound)
    self._range_coder_precision = int(range_coder_precision)
    self._data_format = data_format
    self._channel_axis(2)  # trigger ValueError early
    self.input_spec = engine.InputSpec(min_ndim=2)

  @property
  def init_scale(self):
    return self._init_scale

  @property
  def filters(self):
    return self._filters

  @property
  def tail_mass(self):
    return self._tail_mass

  @property
  def optimize_integer_offset(self):
    return self._optimize_integer_offset

  @property
  def likelihood_bound(self):
    return self._likelihood_bound

  @property
  def range_coder_precision(self):
    return self._range_coder_precision

  @property
  def data_format(self):
    return self._data_format

  def _channel_axis(self, ndim):
    try:
      return {"channels_first": 1, "channels_last": ndim - 1}[self.data_format]
    except KeyError:
      raise ValueError("Unsupported `data_format` for {} layer: {}.".format(
          self.__class__.__name__, self.data_format))

  def _logits_cumulative(self, inputs, stop_gradient):
    """Evaluate logits of the cumulative densities.

    Args:
      inputs: The values at which to evaluate the cumulative densities, expected
        to be a `Tensor` of shape `(channels, 1, batch)`.
      stop_gradient: Boolean. Whether to add `array_ops.stop_gradient` calls so
        that the gradient of the output with respect to the density model
        parameters is disconnected (the gradient with respect to `inputs` is
        left untouched).

    Returns:
      A `Tensor` of the same shape as `inputs`, containing the logits of the
      cumulative densities evaluated at the given inputs.
    """
    logits = inputs

    for i in range(len(self.filters) + 1):
      matrix = self._matrices[i]
      if stop_gradient:
        matrix = array_ops.stop_gradient(matrix)
      logits = math_ops.matmul(matrix, logits)

      bias = self._biases[i]
      if stop_gradient:
        bias = array_ops.stop_gradient(bias)
      logits += bias

      if i < len(self._factors):
        factor = self._factors[i]
        if stop_gradient:
          factor = array_ops.stop_gradient(factor)
        logits += factor * math_ops.tanh(logits)

    return logits

  def build(self, input_shape):
    """Builds the layer.

    Creates the variables for the network modeling the densities, creates the
    auxiliary loss estimating the median and tail quantiles of the densities,
    and then uses that to create the probability mass functions and the update
    op that produces the discrete cumulative density functions used by the range
    coder.

    Args:
      input_shape: Shape of the input tensor, used to get the number of
        channels.

    Raises:
      ValueError: if `input_shape` doesn't specify the length of the channel
        dimension.
    """
    input_shape = tensor_shape.TensorShape(input_shape)
    channel_axis = self._channel_axis(input_shape.ndims)
    channels = input_shape[channel_axis].value
    if channels is None:
      raise ValueError("The channel dimension of the inputs must be defined.")
    self.input_spec = engine.InputSpec(
        ndim=input_shape.ndims, axes={channel_axis: channels})
    filters = (1,) + self.filters + (1,)
    scale = self.init_scale ** (1 / (len(self.filters) + 1))

    # Create variables.
    self._matrices = []
    self._biases = []
    self._factors = []
    for i in range(len(self.filters) + 1):
      init = np.log(np.expm1(1 / scale / filters[i + 1]))
      matrix = self.add_variable(
          "matrix_{}".format(i), dtype=self.dtype,
          shape=(channels, filters[i + 1], filters[i]),
          initializer=init_ops.Constant(init))
      matrix = nn.softplus(matrix)
      self._matrices.append(matrix)

      bias = self.add_variable(
          "bias_{}".format(i), dtype=self.dtype,
          shape=(channels, filters[i + 1], 1),
          initializer=init_ops.RandomUniform(-.5, .5))
      self._biases.append(bias)

      if i < len(self.filters):
        factor = self.add_variable(
            "factor_{}".format(i), dtype=self.dtype,
            shape=(channels, filters[i + 1], 1),
            initializer=init_ops.Zeros())
        factor = math_ops.tanh(factor)
        self._factors.append(factor)

    # To figure out what range of the densities to sample, we need to compute
    # the quantiles given by `tail_mass / 2` and `1 - tail_mass / 2`. Since we
    # can't take inverses of the cumulative directly, we make it an optimization
    # problem:
    # `quantiles = argmin(|logit(cumulative) - target|)`
    # where `target` is `logit(tail_mass / 2)` or `logit(1 - tail_mass / 2)`.
    # Taking the logit (inverse of sigmoid) of the cumulative makes the
    # representation of the right target more numerically stable.

    # Numerically stable way of computing logits of `tail_mass / 2`
    # and `1 - tail_mass / 2`.
    target = np.log(2 / self.tail_mass - 1)
    # Compute lower and upper tail quantile as well as median.
    target = constant_op.constant([-target, 0, target], dtype=self.dtype)

    def quantiles_initializer(shape, dtype=None, partition_info=None):
      del partition_info  # unused
      assert tuple(shape[1:]) == (1, 3)
      init = constant_op.constant(
          [[[-self.init_scale, 0, self.init_scale]]], dtype=dtype)
      return array_ops.tile(init, (shape[0], 1, 1))

    quantiles = self.add_variable(
        "quantiles", shape=(channels, 1, 3), dtype=self.dtype,
        initializer=quantiles_initializer)
    logits = self._logits_cumulative(quantiles, stop_gradient=True)
    loss = math_ops.reduce_sum(abs(logits - target))
    self.add_loss(loss, inputs=None)

    # Save medians for `call`, `compress`, and `decompress`.
    self._medians = quantiles[:, :, 1:2]
    if not self.optimize_integer_offset:
      self._medians = math_ops.round(self._medians)

    # Largest distance observed between lower tail quantile and median,
    # or between median and upper tail quantile.
    minima = math_ops.reduce_max(self._medians - quantiles[:, :, 0:1])
    maxima = math_ops.reduce_max(quantiles[:, :, 2:3] - self._medians)
    minmax = math_ops.maximum(minima, maxima)
    minmax = math_ops.ceil(minmax)
    minmax = math_ops.maximum(minmax, 1)

    # Sample the density up to `minmax` around the median.
    samples = math_ops.range(-minmax, minmax + 1, dtype=self.dtype)
    samples += self._medians

    half = constant_op.constant(.5, dtype=self.dtype)
    # We strip the sigmoid from the end here, so we can use the special rule
    # below to only compute differences in the left tail of the sigmoid.
    # This increases numerical stability (see explanation in `call`).
    lower = self._logits_cumulative(samples - half, stop_gradient=True)
    upper = self._logits_cumulative(samples + half, stop_gradient=True)
    # Flip signs if we can move more towards the left tail of the sigmoid.
    sign = -math_ops.sign(math_ops.add_n([lower, upper]))
    pmf = abs(math_ops.sigmoid(sign * upper) - math_ops.sigmoid(sign * lower))
    # Add tail masses to first and last bin of pmf, as we clip values for
    # compression, meaning that out-of-range values get mapped to these bins.
    pmf = array_ops.concat([
        math_ops.add_n([pmf[:, 0, :1], math_ops.sigmoid(lower[:, 0, :1])]),
        pmf[:, 0, 1:-1],
        math_ops.add_n([pmf[:, 0, -1:], math_ops.sigmoid(-upper[:, 0, -1:])]),
        ], axis=-1)
    self._pmf = pmf

    cdf = coder_ops.pmf_to_quantized_cdf(
        pmf, precision=self.range_coder_precision)
    def cdf_getter(*args, **kwargs):
      del args, kwargs  # ignored
      return variable_scope.get_variable(
          "quantized_cdf", dtype=dtypes.int32, initializer=cdf,
          trainable=False, validate_shape=False, collections=())
    # Need to provide a fake shape here since add_variable insists on it.
    self._quantized_cdf = self.add_variable(
        "quantized_cdf", shape=(channels, 1), dtype=dtypes.int32,
        getter=cdf_getter, trainable=False)

    update_op = state_ops.assign(
        self._quantized_cdf, cdf, validate_shape=False)
    self.add_update(update_op, inputs=None)

    super(EntropyBottleneck, self).build(input_shape)

  def call(self, inputs, training):
    """Pass a tensor through the bottleneck.

    Args:
      inputs: The tensor to be passed through the bottleneck.
      training: Boolean. If `True`, returns a differentiable approximation of
        the inputs, and their likelihoods under the modeled probability
        densities. If `False`, returns the quantized inputs and their
        likelihoods under the corresponding probability mass function. These
        quantities can't be used for training, as they are not differentiable,
        but represent actual compression more closely.

    Returns:
      values: `Tensor` with the same shape as `inputs` containing the perturbed
        or quantized input values.
      likelihood: `Tensor` with the same shape as `inputs` containing the
        likelihood of `values` under the modeled probability distributions.

    Raises:
      ValueError: if `inputs` has different `dtype` or number of channels than
        a previous set of inputs the model was invoked with earlier.
    """
    inputs = ops.convert_to_tensor(inputs)
    ndim = self.input_spec.ndim
    channel_axis = self._channel_axis(ndim)
    half = constant_op.constant(.5, dtype=self.dtype)

    # Convert to (channels, 1, batch) format by commuting channels to front
    # and then collapsing.
    order = list(range(ndim))
    order.pop(channel_axis)
    order.insert(0, channel_axis)
    values = array_ops.transpose(inputs, order)
    shape = array_ops.shape(values)
    values = array_ops.reshape(values, (shape[0], 1, -1))

    # Add noise or quantize.
    if training:
      noise = random_ops.random_uniform(array_ops.shape(values), -half, half)
      values = math_ops.add_n([values, noise])
    elif self.optimize_integer_offset:
      values = math_ops.round(values - self._medians) + self._medians
    else:
      values = math_ops.round(values)

    # Evaluate densities.
    # We can use the special rule below to only compute differences in the left
    # tail of the sigmoid. This increases numerical stability: sigmoid(x) is 1
    # for large x, 0 for small x. Subtracting two numbers close to 0 can be done
    # with much higher precision than subtracting two numbers close to 1.
    lower = self._logits_cumulative(values - half, stop_gradient=False)
    upper = self._logits_cumulative(values + half, stop_gradient=False)
    # Flip signs if we can move more towards the left tail of the sigmoid.
    sign = -math_ops.sign(math_ops.add_n([lower, upper]))
    sign = array_ops.stop_gradient(sign)
    likelihood = abs(
        math_ops.sigmoid(sign * upper) - math_ops.sigmoid(sign * lower))
    if self.likelihood_bound > 0:
      likelihood_bound = constant_op.constant(
          self.likelihood_bound, dtype=self.dtype)
      # TODO(jballe): Override gradients.
      likelihood = math_ops.maximum(likelihood, likelihood_bound)

    # Convert back to input tensor shape.
    order = list(range(1, ndim))
    order.insert(channel_axis, 0)
    values = array_ops.reshape(values, shape)
    values = array_ops.transpose(values, order)
    likelihood = array_ops.reshape(likelihood, shape)
    likelihood = array_ops.transpose(likelihood, order)

    if not context.executing_eagerly():
      values_shape, likelihood_shape = self.compute_output_shape(inputs.shape)
      values.set_shape(values_shape)
      likelihood.set_shape(likelihood_shape)

    return values, likelihood

  def compress(self, inputs):
    """Compress inputs and store their binary representations into strings.

    Args:
      inputs: `Tensor` with values to be compressed.

    Returns:
      String `Tensor` vector containing the compressed representation of each
      batch element of `inputs`.
    """
    with ops.name_scope(self._name_scope()):
      inputs = ops.convert_to_tensor(inputs)
      if not self.built:
        # Check input assumptions set before layer building, e.g. input rank.
        self._assert_input_compatibility(inputs)
        if self.dtype is None:
          self._dtype = inputs.dtype.base_dtype.name
        self.build(inputs.shape)

      # Check input assumptions set after layer building, e.g. input shape.
      if not context.executing_eagerly():
        self._assert_input_compatibility(inputs)

      ndim = self.input_spec.ndim
      channel_axis = self._channel_axis(ndim)
      # Tuple of slices for expanding dimensions of tensors below.
      slices = ndim * [None] + [slice(None)]
      slices[channel_axis] = slice(None)
      slices = tuple(slices)

      # Expand dimensions of CDF to input dimensions, keeping the channels along
      # the right dimension.
      cdf = self._quantized_cdf[slices[1:]]
      num_levels = array_ops.shape(cdf)[-1] - 1

      # Bring inputs to the right range by centering the range on the medians.
      half = constant_op.constant(.5, dtype=self.dtype)
      medians = array_ops.squeeze(self._medians, [1, 2])
      offsets = (math_ops.cast(num_levels // 2, self.dtype) + half) - medians
      # Expand offsets to input dimensions and add to inputs.
      values = inputs + offsets[slices[:-1]]

      # Clip to range and cast to integers. Because we have added .5 above, and
      # all values are positive, the cast effectively implements rounding.
      values = math_ops.maximum(values, half)
      values = math_ops.minimum(
          values, math_ops.cast(num_levels, self.dtype) - half)
      values = math_ops.cast(values, dtypes.int16)

      def loop_body(tensor):
        return coder_ops.range_encode(
            tensor, cdf, precision=self.range_coder_precision)
      strings = functional_ops.map_fn(
          loop_body, values, dtype=dtypes.string, back_prop=False)

      if not context.executing_eagerly():
        strings.set_shape(inputs.shape[:1])

      return strings

  def decompress(self, strings, shape, channels=None):
    """Decompress values from their compressed string representations.

    Args:
      strings: A string `Tensor` vector containing the compressed data.
      shape: A `Tensor` vector of int32 type. Contains the shape of the tensor
        to be decompressed, excluding the batch dimension.
      channels: Integer. Specifies the number of channels statically. Needs only
        be set if the layer hasn't been built yet (i.e., this is the first input
        it receives).

    Returns:
      The decompressed `Tensor`. Its shape will be equal to `shape` prepended
      with the batch dimension from `strings`.

    Raises:
      ValueError: If the length of `shape` isn't available at graph construction
        time.
    """
    with ops.name_scope(self._name_scope()):
      strings = ops.convert_to_tensor(strings)
      shape = ops.convert_to_tensor(shape)
      if self.built:
        ndim = self.input_spec.ndim
        channel_axis = self._channel_axis(ndim)
        if channels is None:
          channels = self.input_spec.axes[channel_axis]
      else:
        if not (shape.shape.is_fully_defined() and shape.shape.ndims == 1):
          raise ValueError("`shape` must be a vector with known length.")
        ndim = shape.shape[0].value + 1
        channel_axis = self._channel_axis(ndim)
        input_shape = ndim * [None]
        input_shape[channel_axis] = channels
        self.build(input_shape)

      # Tuple of slices for expanding dimensions of tensors below.
      slices = ndim * [None] + [slice(None)]
      slices[channel_axis] = slice(None)
      slices = tuple(slices)

      # Expand dimensions of CDF to input dimensions, keeping the channels along
      # the right dimension.
      cdf = self._quantized_cdf[slices[1:]]
      num_levels = array_ops.shape(cdf)[-1] - 1

      def loop_body(string):
        return coder_ops.range_decode(
            string, shape, cdf, precision=self.range_coder_precision)
      outputs = functional_ops.map_fn(
          loop_body, strings, dtype=dtypes.int16, back_prop=False)
      outputs = math_ops.cast(outputs, self.dtype)

      medians = array_ops.squeeze(self._medians, [1, 2])
      offsets = math_ops.cast(num_levels // 2, self.dtype) - medians
      outputs -= offsets[slices[:-1]]

      if not context.executing_eagerly():
        outputs_shape = ndim * [None]
        outputs_shape[0] = strings.shape[0]
        outputs_shape[channel_axis] = channels
        outputs.set_shape(outputs_shape)

      return outputs

  def visualize(self):
    """Multi-channel visualization of densities as images.

    Creates and returns an image summary visualizing the current probabilty
    density estimates. The image contains one row for each channel. Within each
    row, the pixel intensities are proportional to probability values, and each
    row is centered on the median of the corresponding distribution.

    Returns:
      The created image summary.
    """
    with ops.name_scope(self._name_scope()):
      image = self._pmf
      image *= 255 / math_ops.reduce_max(image, axis=1, keepdims=True)
      image = math_ops.cast(image + .5, dtypes.uint8)
      image = image[None, :, :, None]
    return summary.image("pmf", image, max_outputs=1)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    return input_shape, input_shape
