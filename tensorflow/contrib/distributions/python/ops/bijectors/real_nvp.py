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
"""Real NVP bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.layers import core as layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import template as template_ops
from tensorflow.python.ops.distributions import bijector as bijector_lib


__all__ = [
    "RealNVP",
    "real_nvp_default_template"
]


class RealNVP(bijector_lib.Bijector):
  """RealNVP "affine coupling layer" for vector-valued events.

  Real NVP models a normalizing flow on a `D`-dimensional distribution via a
  single `D-d`-dimensional conditional distribution [(Dinh et al., 2017)][1]:

  `y[d:D] = y[d:D] * math_ops.exp(log_scale_fn(y[d:D])) + shift_fn(y[d:D])`
  `y[0:d] = x[0:d]`

  The last `D-d` units are scaled and shifted based on the first `d` units only,
  while the first `d` units are 'masked' and left unchanged. Real NVP's
  `shift_and_log_scale_fn` computes vector-valued quantities. For
  scale-and-shift transforms that do not depend on any masked units, i.e.
  `d=0`, use the `tfb.Affine` bijector with learned parameters instead.

  Masking is currently only supported for base distributions with
  `event_ndims=1`. For more sophisticated masking schemes like checkerboard or
  channel-wise masking [(Papamakarios et al., 2016)[4], use the `tfb.Permute`
  bijector to re-order desired masked units into the first `d` units. For base
  distributions with `event_ndims > 1`, use the `tfb.Reshape` bijector to
  flatten the event shape.

  Recall that the MAF bijector [(Papamakarios et al., 2016)][4] implements a
  normalizing flow via an autoregressive transformation. MAF and IAF have
  opposite computational tradeoffs - MAF can train all units in parallel but
  must sample units sequentially, while IAF must train units sequentially but
  can sample in parallel. In contrast, Real NVP can compute both forward and
  inverse computations in parallel. However, the lack of an autoregressive
  transformations makes it less expressive on a per-bijector basis.

  A "valid" `shift_and_log_scale_fn` must compute each `shift` (aka `loc` or
  "mu" in [Papamakarios et al. (2016)][4]) and `log(scale)` (aka "alpha" in
  [Papamakarios et al. (2016)][4]) such that each are broadcastable with the
  arguments to `forward` and `inverse`, i.e., such that the calculations in
  `forward`, `inverse` [below] are possible. For convenience,
  `real_nvp_default_nvp` is offered as a possible `shift_and_log_scale_fn`
  function.

  NICE [(Dinh et al., 2014)][2] is a special case of the Real NVP bijector
  which discards the scale transformation, resulting in a constant-time
  inverse-log-determinant-Jacobian. To use a NICE bijector instead of Real
  NVP, `shift_and_log_scale_fn` should return `(shift, None)`, and
  `is_constant_jacobian` should be set to `True` in the `RealNVP` constructor.
  Calling `real_nvp_default_template` with `shift_only=True` returns one such
  NICE-compatible `shift_and_log_scale_fn`.

  Caching: the scalar input depth `D` of the base distribution is not known at
  construction time. The first call to any of `forward(x)`, `inverse(x)`,
  `inverse_log_det_jacobian(x)`, or `forward_log_det_jacobian(x)` memoizes
  `D`, which is re-used in subsequent calls. This shape must be known prior to
  graph execution (which is the case if using tf.layers).

  #### Example Use

  ```python
  tfd = tf.contrib.distributions
  tfb = tfd.bijectors

  # A common choice for a normalizing flow is to use a Gaussian for the base
  # distribution. (However, any continuous distribution would work.) E.g.,
  nvp = tfd.TransformedDistribution(
      distribution=tfd.MultivariateNormalDiag(loc=[0., 0., 0.])),
      bijector=tfb.RealNVP(
          num_masked=2,
          shift_and_log_scale_fn=tfb.real_nvp_default_template(
              hidden_layers=[512, 512])))

  x = nvp.sample()
  nvp.log_prob(x)
  nvp.log_prob(0.)
  ```

  For more examples, see [Jang (2018)][3].

  #### References

  [1]: Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density Estimation
       using Real NVP. In _International Conference on Learning
       Representations_, 2017. https://arxiv.org/abs/1605.08803

  [2]: Laurent Dinh, David Krueger, and Yoshua Bengio. NICE: Non-linear
       Independent Components Estimation. _arXiv preprint arXiv:1410.8516_,
       2014. https://arxiv.org/abs/1410.8516

  [3]: Eric Jang. Normalizing Flows Tutorial, Part 2: Modern Normalizing Flows.
       _Technical Report_, 2018. http://blog.evjang.com/2018/01/nf2.html

  [4]: George Papamakarios, Theo Pavlakou, and Iain Murray. Masked
       Autoregressive Flow for Density Estimation. In _Neural Information
       Processing Systems_, 2017. https://arxiv.org/abs/1705.07057
  """

  def __init__(self,
               num_masked,
               shift_and_log_scale_fn,
               is_constant_jacobian=False,
               validate_args=False,
               name=None):
    """Creates the Real NVP or NICE bijector.

    Args:
      num_masked: Python `int` indicating that the first `d` units of the event
        should be masked. Must be in the closed interval `[1, D-1]`, where `D`
        is the event size of the base distribution.
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

    Raises:
      ValueError: If num_masked < 1.
    """
    name = name or "real_nvp"
    if num_masked <= 0:
      raise ValueError("num_masked must be a positive integer.")
    self._num_masked = num_masked
    # At construction time, we don't know input_depth.
    self._input_depth = None
    self._shift_and_log_scale_fn = shift_and_log_scale_fn
    super(RealNVP, self).__init__(
        forward_min_event_ndims=1,
        is_constant_jacobian=is_constant_jacobian,
        validate_args=validate_args,
        name=name)

  def _cache_input_depth(self, x):
    if self._input_depth is None:
      self._input_depth = x.shape.with_rank_at_least(1)[-1].value
      if self._input_depth is None:
        raise NotImplementedError(
            "Rightmost dimension must be known prior to graph execution.")
      if self._num_masked >= self._input_depth:
        raise ValueError(
            "Number of masked units must be smaller than the event size.")

  def _forward(self, x):
    self._cache_input_depth(x)
    # Performs scale and shift.
    x0, x1 = x[:, :self._num_masked], x[:, self._num_masked:]
    shift, log_scale = self._shift_and_log_scale_fn(
        x0, self._input_depth - self._num_masked)
    y1 = x1
    if log_scale is not None:
      y1 *= math_ops.exp(log_scale)
    if shift is not None:
      y1 += shift
    y = array_ops.concat([x0, y1], axis=-1)
    return y

  def _inverse(self, y):
    self._cache_input_depth(y)
    # Performs un-shift and un-scale.
    y0, y1 = y[:, :self._num_masked], y[:, self._num_masked:]
    shift, log_scale = self._shift_and_log_scale_fn(
        y0, self._input_depth - self._num_masked)
    x1 = y1
    if shift is not None:
      x1 -= shift
    if log_scale is not None:
      x1 *= math_ops.exp(-log_scale)
    x = array_ops.concat([y0, x1], axis=-1)
    return x

  def _inverse_log_det_jacobian(self, y):
    self._cache_input_depth(y)
    y0 = y[:, :self._num_masked]
    _, log_scale = self._shift_and_log_scale_fn(
        y0, self._input_depth - self._num_masked)
    if log_scale is None:
      return constant_op.constant(0., dtype=y.dtype, name="ildj")
    return -math_ops.reduce_sum(log_scale, axis=-1)

  def _forward_log_det_jacobian(self, x):
    self._cache_input_depth(x)
    x0 = x[:, :self._num_masked]
    _, log_scale = self._shift_and_log_scale_fn(
        x0, self._input_depth - self._num_masked)
    if log_scale is None:
      return constant_op.constant(0., dtype=x.dtype, name="fldj")
    return math_ops.reduce_sum(log_scale, axis=-1)


def real_nvp_default_template(
    hidden_layers,
    shift_only=False,
    activation=nn_ops.relu,
    name=None,
    *args,
    **kwargs):
  """Build a scale-and-shift function using a multi-layer neural network.

  This will be wrapped in a make_template to ensure the variables are only
  created once. It takes the `d`-dimensional input x[0:d] and returns the `D-d`
  dimensional outputs `loc` ("mu") and `log_scale` ("alpha").

  Arguments:
    hidden_layers: Python `list`-like of non-negative integer, scalars
      indicating the number of units in each hidden layer. Default: `[512, 512].
    shift_only: Python `bool` indicating if only the `shift` term shall be
      computed (i.e. NICE bijector). Default: `False`.
    activation: Activation function (callable). Explicitly setting to `None`
      implies a linear activation.
    name: A name for ops managed by this function. Default:
      "real_nvp_default_template".
    *args: `tf.layers.dense` arguments.
    **kwargs: `tf.layers.dense` keyword arguments.

  Returns:
    shift: `Float`-like `Tensor` of shift terms ("mu" in
      [Papamakarios et al.  (2016)][1]).
    log_scale: `Float`-like `Tensor` of log(scale) terms ("alpha" in
      [Papamakarios et al. (2016)][1]).

  Raises:
    NotImplementedError: if rightmost dimension of `inputs` is unknown prior to
      graph execution.

  #### References

  [1]: George Papamakarios, Theo Pavlakou, and Iain Murray. Masked
       Autoregressive Flow for Density Estimation. In _Neural Information
       Processing Systems_, 2017. https://arxiv.org/abs/1705.07057
  """

  with ops.name_scope(name, "real_nvp_default_template"):
    def _fn(x, output_units):
      """Fully connected MLP parameterized via `real_nvp_template`."""
      for units in hidden_layers:
        x = layers.dense(
            inputs=x,
            units=units,
            activation=activation,
            *args,
            **kwargs)
      x = layers.dense(
          inputs=x,
          units=(1 if shift_only else 2) * output_units,
          activation=None,
          *args,
          **kwargs)
      if shift_only:
        return x, None
      shift, log_scale = array_ops.split(x, 2, axis=-1)
      return shift, log_scale
    return template_ops.make_template(
        "real_nvp_default_template", _fn)
