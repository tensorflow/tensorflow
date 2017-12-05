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
"""MaskedAutoregressiveFlow bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.layers import core as layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import template as template_ops
from tensorflow.python.ops import variable_scope as variable_scope_lib
from tensorflow.python.ops.distributions import bijector as bijector_lib


__all__ = [
    "MaskedAutoregressiveFlow",
    "masked_autoregressive_default_template",
    "masked_dense",
]


class MaskedAutoregressiveFlow(bijector_lib.Bijector):
  """Affine MaskedAutoregressiveFlow bijector for vector-valued events.

  The affine autoregressive flow [1] provides a relatively simple framework for
  user-specified (deep) architectures to learn a distribution over vector-valued
  events. Regarding terminology,

    "Autoregressive models decompose the joint density as a product of
    conditionals, and model each conditional in turn. Normalizing flows
    transform a base density (e.g. a standard Gaussian) into the target density
    by an invertible transformation with tractable Jacobian." [1]

  In other words, the "autoregressive property" is equivalent to the
  decomposition, `p(x) = prod{ p(x[i] | x[0:i]) : i=0, ..., d }`. The provided
  `shift_and_log_scale_fn`, `masked_autoregressive_default_template`, achieves
  this property by zeroing out weights in its `masked_dense` layers.

  In the `tf.distributions` framework, a "normalizing flow" is implemented as a
  `tf.distributions.bijectors.Bijector`. The `forward` "autoregression"
  is implemented using a `tf.while_loop` and a deep neural network (DNN) with
  masked weights such that the autoregressive property is automatically met in
  the `inverse`.

  A `TransformedDistribution` using `MaskedAutoregressiveFlow(...)` uses the
  (expensive) forward-mode calculation to draw samples and the (cheap)
  reverse-mode calculation to compute log-probabilities. Conversely, a
  `TransformedDistribution` using `Invert(MaskedAutoregressiveFlow(...))` uses
  the (expensive) forward-mode calculation to compute log-probabilities and the
  (cheap) reverse-mode calculation to compute samples.  See "Example Use"
  [below] for more details.

  Given a `shift_and_log_scale_fn`, the forward and inverse transformations are
  (a sequence of) affine transformations. A "valid" `shift_and_log_scale_fn`
  must compute each `shift` (aka `loc` or "mu" [2]) and `log(scale)` (aka
  "alpha" [2]) such that each are broadcastable with the arguments to `forward`
  and `inverse`, i.e., such that the calculations in `forward`, `inverse`
  [below] are possible.

  For convenience, `masked_autoregressive_default_template` is offered as a
  possible `shift_and_log_scale_fn` function. It implements the MADE
  architecture [2]. MADE is a feed-forward network that computes a `shift` and
  `log(scale)` using `masked_dense` layers in a deep neural network. Weights are
  masked to ensure the autoregressive property. It is possible that this
  architecture is suboptimal for your task. To build alternative networks,
  either change the arguments to `masked_autoregressive_default_template`, use
  the `masked_dense` function to roll-out your own, or use some other
  architecture, e.g., using `tf.layers`.

  Warning: no attempt is made to validate that the `shift_and_log_scale_fn`
  enforces the "autoregressive property".

  Assuming `shift_and_log_scale_fn` has valid shape and autoregressive
  semantics, the forward transformation is,

  ```python
  def forward(x):
    y = zeros_like(x)
    event_size = x.shape[-1]
    for _ in range(event_size):
      shift, log_scale = shift_and_log_scale_fn(y)
      y = x * math_ops.exp(log_scale) + shift
    return y
  ```

  and the inverse transformation is,

  ```python
  def inverse(y):
    shift, log_scale = shift_and_log_scale_fn(y)
    return (y - shift) / math_ops.exp(log_scale)
  ```

  Notice that the `inverse` does not need a for-loop. This is because in the
  forward pass each calculation of `shift` and `log_scale` is based on the `y`
  calculated so far (not `x`). In the `inverse`, the `y` is fully known, thus is
  equivalent to the scaling used in `forward` after `event_size` passes, i.e.,
  the "last" `y` used to compute `shift`, `log_scale`. (Roughly speaking, this
  also proves the transform is bijective.)

  #### Example Use

  ```python
  tfd = tf.contrib.distributions
  tfb = tfd.bijectors

  dims = 5

  # A common choice for a normalizing flow is to use a Gaussian for the base
  # distribution. (However, any continuous distribution would work.) E.g.,
  maf = tfd.TransformedDistribution(
      distribution=tfd.Normal(loc=0., scale=1.),
      bijector=tfb.MaskedAutoregressiveFlow(
          shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
              hidden_layers=[512, 512])),
      event_shape=[dims])

  x = maf.sample()  # Expensive; uses `tf.while_loop`, no Bijector caching.
  maf.log_prob(x)   # Almost free; uses Bijector caching.
  maf.log_prob(0.)  # Cheap; no `tf.while_loop` despite no Bijector caching.

  # [1] also describes an "Inverse Autoregressive Flow", e.g.,
  iaf = tfd.TransformedDistribution(
      distribution=tfd.Normal(loc=0., scale=1.),
      bijector=tfb.Invert(tfb.MaskedAutoregressiveFlow(
          shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
              hidden_layers=[512, 512]))),
      event_shape=[dims])

  x = iaf.sample()  # Cheap; no `tf.while_loop` despite no Bijector caching.
  iaf.log_prob(x)   # Almost free; uses Bijector caching.
  iaf.log_prob(0.)  # Expensive; uses `tf.while_loop`, no Bijector caching.

  # In many (if not most) cases the default `shift_and_log_scale_fn` will be a
  # poor choice. Here's an example of using a "shift only" version and with a
  # different number/depth of hidden layers.
  shift_only = True
  maf_no_scale_hidden2 = tfd.TransformedDistribution(
      distribution=tfd.Normal(loc=0., scale=1.),
      bijector=tfb.MaskedAutoregressiveFlow(
          tfb.masked_autoregressive_default_template(
              hidden_layers=[32],
              shift_only=shift_only),
          is_constant_jacobian=shift_only),
      event_shape=[dims])
  ```

  [1]: "Masked Autoregressive Flow for Density Estimation."
       George Papamakarios, Theo Pavlakou, Iain Murray. Arxiv. 2017.
       https://arxiv.org/abs/1705.07057

  [2]: "MADE: Masked Autoencoder for Distribution Estimation."
       Mathieu Germain, Karol Gregor, Iain Murray, Hugo Larochelle. ICML. 2015.
       https://arxiv.org/abs/1502.03509

  """

  def __init__(self,
               shift_and_log_scale_fn,
               is_constant_jacobian=False,
               validate_args=False,
               name=None):
    """Creates the MaskedAutoregressiveFlow bijector.

    Args:
      shift_and_log_scale_fn: Python `callable` which computes `shift` and
        `log_scale` from both the forward domain (`x`) and the inverse domain
        (`y`). Calculation must respect the "autoregressive property" (see class
        docstring). Suggested default
        `masked_autoregressive_default_template(hidden_layers=...)`.
        Typically the function contains `tf.Variables` and is wrapped using
        `tf.make_template`. Returning `None` for either (both) `shift`,
        `log_scale` is equivalent to (but more efficient than) returning zero.
      is_constant_jacobian: Python `bool`. Default: `False`. When `True` the
        implementation assumes `log_scale` does not depend on the forward domain
        (`x`) or inverse domain (`y`) values. (No validation is made;
        `is_constant_jacobian=False` is always safe but possibly computationally
        inefficient.)
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str`, name given to ops managed by this object.
    """
    name = name or "masked_autoregressive_flow"
    self._shift_and_log_scale_fn = shift_and_log_scale_fn
    super(MaskedAutoregressiveFlow, self).__init__(
        is_constant_jacobian=is_constant_jacobian,
        validate_args=validate_args,
        name=name)

  def _forward(self, x):
    event_size = array_ops.shape(x)[-1]
    y0 = array_ops.zeros_like(x, name="y0")
    # call the template once to ensure creation
    _ = self._shift_and_log_scale_fn(y0)
    def _loop_body(index, y0):
      """While-loop body for autoregression calculation."""
      # Set caching device to avoid re-getting the tf.Variable for every while
      # loop iteration.
      with variable_scope_lib.variable_scope(
          variable_scope_lib.get_variable_scope()) as vs:
        if vs.caching_device is None:
          vs.set_caching_device(lambda op: op.device)
        shift, log_scale = self._shift_and_log_scale_fn(y0)
      y = x
      if log_scale is not None:
        y *= math_ops.exp(log_scale)
      if shift is not None:
        y += shift
      return index + 1, y
    _, y = control_flow_ops.while_loop(
        cond=lambda index, _: index < event_size,
        body=_loop_body,
        loop_vars=[0, y0])
    return y

  def _inverse(self, y):
    shift, log_scale = self._shift_and_log_scale_fn(y)
    x = y
    if shift is not None:
      x -= shift
    if log_scale is not None:
      x *= math_ops.exp(-log_scale)
    return x

  def _inverse_log_det_jacobian(self, y):
    _, log_scale = self._shift_and_log_scale_fn(y)
    if log_scale is None:
      return constant_op.constant(0., dtype=y.dtype, name="ildj")
    return -math_ops.reduce_sum(log_scale, axis=-1)


MASK_INCLUSIVE = "inclusive"
MASK_EXCLUSIVE = "exclusive"


def _gen_slices(num_blocks, n_in, n_out, mask_type=MASK_EXCLUSIVE):
  """Generate the slices for building an autoregressive mask."""
  # TODO(b/67594795): Better support of dynamic shape.
  slices = []
  col = 0
  d_in = n_in // num_blocks
  d_out = n_out // num_blocks
  row = d_out if mask_type == MASK_EXCLUSIVE else 0
  for _ in range(num_blocks):
    row_slice = slice(row, None)
    col_slice = slice(col, col + d_in)
    slices.append([row_slice, col_slice])
    col += d_in
    row += d_out
  return slices


def _gen_mask(num_blocks,
              n_in,
              n_out,
              mask_type=MASK_EXCLUSIVE,
              dtype=dtypes.float32):
  """Generate the mask for building an autoregressive dense layer."""
  # TODO(b/67594795): Better support of dynamic shape.
  mask = np.zeros([n_out, n_in], dtype=dtype.as_numpy_dtype())
  slices = _gen_slices(num_blocks, n_in, n_out, mask_type=mask_type)
  for [row_slice, col_slice] in slices:
    mask[row_slice, col_slice] = 1
  return mask


def masked_dense(inputs,
                 units,
                 num_blocks=None,
                 exclusive=False,
                 kernel_initializer=None,
                 reuse=None,
                 name=None,
                 *args,
                 **kwargs):
  """A autoregressively masked dense layer. Analogous to `tf.layers.dense`.

  See [1] for detailed explanation.

  [1]: "MADE: Masked Autoencoder for Distribution Estimation."
       Mathieu Germain, Karol Gregor, Iain Murray, Hugo Larochelle. ICML. 2015.
       https://arxiv.org/abs/1502.03509

  Arguments:
    inputs: Tensor input.
    units: Python `int` scalar representing the dimensionality of the output
      space.
    num_blocks: Python `int` scalar representing the number of blocks for the
      MADE masks.
    exclusive: Python `bool` scalar representing whether to zero the diagonal of
      the mask, used for the first layer of a MADE.
    kernel_initializer: Initializer function for the weight matrix.
      If `None` (default), weights are initialized using the
      `tf.glorot_random_initializer`.
    reuse: Python `bool` scalar representing whether to reuse the weights of a
      previous layer by the same name.
    name: Python `str` used to describe ops managed by this function.
    *args: `tf.layers.dense` arguments.
    **kwargs: `tf.layers.dense` keyword arguments.

  Returns:
    Output tensor.

  Raises:
    NotImplementedError: if rightmost dimension of `inputs` is unknown prior to
      graph execution.
  """
  # TODO(b/67594795): Better support of dynamic shape.
  input_depth = inputs.shape.with_rank_at_least(1)[-1].value
  if input_depth is None:
    raise NotImplementedError(
        "Rightmost dimension must be known prior to graph execution.")

  mask = _gen_mask(num_blocks, input_depth, units,
                   MASK_EXCLUSIVE if exclusive else MASK_INCLUSIVE).T

  if kernel_initializer is None:
    kernel_initializer = init_ops.glorot_normal_initializer()

  def masked_initializer(shape, dtype=None, partition_info=None):
    return mask * kernel_initializer(shape, dtype, partition_info)

  with ops.name_scope(name, "masked_dense", [inputs, units, num_blocks]):
    layer = layers.Dense(
        units,
        kernel_initializer=masked_initializer,
        kernel_constraint=lambda x: mask * x,
        name=name,
        dtype=inputs.dtype.base_dtype,
        _scope=name,
        _reuse=reuse,
        *args,
        **kwargs)
    return layer.apply(inputs)


def masked_autoregressive_default_template(
    hidden_layers,
    shift_only=False,
    activation=nn_ops.relu,
    log_scale_min_clip=-5.,
    log_scale_max_clip=3.,
    log_scale_clip_gradient=False,
    name=None,
    *args,
    **kwargs):
  """Build the MADE Model [1].

  This will be wrapped in a make_template to ensure the variables are only
  created once. It takes the input and returns the `loc` ("mu" [1]) and
  `log_scale` ("alpha" [1]) from the MADE network.

  Warning: This function uses `masked_dense` to create randomly initialized
  `tf.Variables`. It is presumed that these will be fit, just as you would any
  other neural architecture which uses `tf.layers.dense`.

  #### About Hidden Layers:

  Each element of `hidden_layers` should be greater than the `input_depth`
  (i.e., `input_depth = tf.shape(input)[-1]` where `input` is the input to the
  neural network). This is necessary to ensure the autoregressivity property.

  #### About Clipping:

  This function also optionally clips the `log_scale` (but possibly not its
  gradient). This is useful because if `log_scale` is too small/large it might
  underflow/overflow making it impossible for the `MaskedAutoregressiveFlow`
  bijector to implement a bijection. Additionally, the `log_scale_clip_gradient`
  `bool` indicates whether the gradient should also be clipped. The default does
  not clip the gradient; this is useful because it still provides gradient
  information (for fitting) yet solves the numerical stability problem. I.e.,
  `log_scale_clip_gradient = False` means
  `grad[exp(clip(x))] = grad[x] exp(clip(x))` rather than the usual
  `grad[clip(x)] exp(clip(x))`.

  [1]: "MADE: Masked Autoencoder for Distribution Estimation."
       Mathieu Germain, Karol Gregor, Iain Murray, Hugo Larochelle. ICML. 2015.
       https://arxiv.org/abs/1502.03509

  Arguments:
    hidden_layers: Python `list`-like of non-negative integer, scalars
      indicating the number of units in each hidden layer. Default: `[512, 512].
    shift_only: Python `bool` indicating if only the `shift` term shall be
      computed. Default: `False`.
    activation: Activation function (callable). Explicitly setting to `None`
      implies a linear activation.
    log_scale_min_clip: `float`-like scalar `Tensor`, or a `Tensor` with the
      same shape as `log_scale`. The minimum value to clip by. Default: -5.
    log_scale_max_clip: `float`-like scalar `Tensor`, or a `Tensor` with the
      same shape as `log_scale`. The maximum value to clip by. Default: 3.
    log_scale_clip_gradient: Python `bool` indicating that the gradient of
      `tf.clip_by_value` should be preserved. Default: `False`.
    name: A name for ops managed by this function. Default:
      "masked_autoregressive_default_template".
    *args: `tf.layers.dense` arguments.
    **kwargs: `tf.layers.dense` keyword arguments.

  Returns:
    shift: `Float`-like `Tensor` of shift terms (the "mu" in [2]).
    log_scale: `Float`-like `Tensor` of log(scale) terms (the "alpha" in [2]).

  Raises:
    NotImplementedError: if rightmost dimension of `inputs` is unknown prior to
      graph execution.
  """

  with ops.name_scope(name, "masked_autoregressive_default_template",
                      values=[log_scale_min_clip, log_scale_max_clip]):
    def _fn(x):
      """MADE parameterized via `masked_autoregressive_default_template`."""
      # TODO(b/67594795): Better support of dynamic shape.
      input_depth = x.shape.with_rank_at_least(1)[-1].value
      if input_depth is None:
        raise NotImplementedError(
            "Rightmost dimension must be known prior to graph execution.")
      input_shape = (np.int32(x.shape.as_list()) if x.shape.is_fully_defined()
                     else array_ops.shape(x))
      for i, units in enumerate(hidden_layers):
        x = masked_dense(
            inputs=x,
            units=units,
            num_blocks=input_depth,
            exclusive=True if i == 0 else False,
            activation=activation,
            *args,
            **kwargs)
      x = masked_dense(
          inputs=x,
          units=(1 if shift_only else 2) * input_depth,
          num_blocks=input_depth,
          activation=None,
          *args,
          **kwargs)
      if shift_only:
        x = array_ops.reshape(x, shape=input_shape)
        return x, None
      x = array_ops.reshape(
          x, shape=array_ops.concat([input_shape, [2]], axis=0))
      shift, log_scale = array_ops.unstack(x, num=2, axis=-1)
      which_clip = (math_ops.clip_by_value if log_scale_clip_gradient
                    else _clip_by_value_preserve_grad)
      log_scale = which_clip(log_scale, log_scale_min_clip, log_scale_max_clip)
      return shift, log_scale
    return template_ops.make_template(
        "masked_autoregressive_default_template", _fn)


def _clip_by_value_preserve_grad(x, clip_value_min, clip_value_max, name=None):
  """Clips input while leaving gradient unaltered."""
  with ops.name_scope(name, "clip_by_value_preserve_grad",
                      [x, clip_value_min, clip_value_max]):
    clip_x = clip_ops.clip_by_value(x, clip_value_min, clip_value_max)
    return x + array_ops.stop_gradient(clip_x - x)
