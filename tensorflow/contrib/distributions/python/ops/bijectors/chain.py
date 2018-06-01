# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Chain bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import bijector


__all__ = [
    "Chain",
]


def _use_static_shape(input_tensor, ndims):
  return input_tensor.shape.is_fully_defined() and isinstance(ndims, int)


def _compute_min_event_ndims(bijector_list, compute_forward=True):
  """Computes the min_event_ndims associated with the give list of bijectors.

  Given a list `bijector_list` of bijectors, compute the min_event_ndims that is
  associated with the composition of bijectors in that list.

  min_event_ndims is the # of right most dimensions for which the bijector has
  done necessary computation on (i.e. the non-broadcastable part of the
  computation).

  We can derive the min_event_ndims for a chain of bijectors as follows:

  In the case where there are no rank changing bijectors, this will simply be
  `max(b.forward_min_event_ndims for b in bijector_list)`. This is because the
  bijector with the most forward_min_event_ndims requires the most dimensions,
  and hence the chain also requires operating on those dimensions.

  However in the case of rank changing, more care is needed in determining the
  exact amount of dimensions. Padding dimensions causes subsequent bijectors to
  operate on the padded dimensions, and Removing dimensions causes bijectors to
  operate more left.

  Args:
    bijector_list: List of bijectors to be composed by chain.
    compute_forward: Boolean. If True, computes the min_event_ndims associated
      with a forward call to Chain, and otherwise computes the min_event_ndims
      associated with an inverse call to Chain. The latter is the same as the
      min_event_ndims associated with a forward call to Invert(Chain(....)).

  Returns:
    min_event_ndims
  """
  min_event_ndims = 0
  # This is a mouthful, but what this encapsulates is that if not for rank
  # changing bijectors, we'd only need to compute the largest of the min
  # required ndims. Hence "max_min". Due to rank changing bijectors, we need to
  # account for synthetic rank growth / synthetic rank decrease from a rank
  # changing bijector.
  rank_changed_adjusted_max_min_event_ndims = 0

  if compute_forward:
    bijector_list = reversed(bijector_list)

  for b in bijector_list:
    if compute_forward:
      current_min_event_ndims = b.forward_min_event_ndims
      current_inverse_min_event_ndims = b.inverse_min_event_ndims
    else:
      current_min_event_ndims = b.inverse_min_event_ndims
      current_inverse_min_event_ndims = b.forward_min_event_ndims

    # New dimensions were touched.
    if rank_changed_adjusted_max_min_event_ndims < current_min_event_ndims:
      min_event_ndims += (
          current_min_event_ndims - rank_changed_adjusted_max_min_event_ndims)
    rank_changed_adjusted_max_min_event_ndims = max(
        current_min_event_ndims, rank_changed_adjusted_max_min_event_ndims)

    # If the number of dimensions has increased via forward, then
    # inverse_min_event_ndims > forward_min_event_ndims, and hence the
    # dimensions we computed on, have moved left (so we have operated
    # on additional dimensions).
    # Conversely, if the number of dimensions has decreased via forward,
    # then we have inverse_min_event_ndims < forward_min_event_ndims,
    # and so we will have operated on fewer right most dimensions.

    number_of_changed_dimensions = (
        current_min_event_ndims - current_inverse_min_event_ndims)
    rank_changed_adjusted_max_min_event_ndims -= number_of_changed_dimensions
  return min_event_ndims


class Chain(bijector.Bijector):
  """Bijector which applies a sequence of bijectors.

  Example Use:

  ```python
  chain = Chain([Exp(), Softplus()], name="one_plus_exp")
  ```

  Results in:

  * Forward:

   ```python
   exp = Exp()
   softplus = Softplus()
   Chain([exp, softplus]).forward(x)
   = exp.forward(softplus.forward(x))
   = tf.exp(tf.log(1. + tf.exp(x)))
   = 1. + tf.exp(x)
   ```

  * Inverse:

   ```python
   exp = Exp()
   softplus = Softplus()
   Chain([exp, softplus]).inverse(y)
   = softplus.inverse(exp.inverse(y))
   = tf.log(tf.exp(tf.log(y)) - 1.)
   = tf.log(y - 1.)
   ```

  """

  def __init__(self, bijectors=None, validate_args=False, name=None):
    """Instantiates `Chain` bijector.

    Args:
      bijectors: Python `list` of bijector instances. An empty list makes this
        bijector equivalent to the `Identity` bijector.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str`, name given to ops managed by this object. Default:
        E.g., `Chain([Exp(), Softplus()]).name == "chain_of_exp_of_softplus"`.

    Raises:
      ValueError: if bijectors have different dtypes.
    """
    if bijectors is None:
      bijectors = ()
    self._bijectors = bijectors

    for a_bijector in bijectors:
      if not a_bijector._is_injective:  # pylint: disable=protected-access
        raise NotImplementedError(
            "Invert is not implemented for non-injective bijector ({})".format(
                a_bijector.name))

    dtype = list(set([b.dtype for b in bijectors]))
    if len(dtype) > 2:
      raise ValueError("incompatible dtypes: %s" % dtype)
    elif len(dtype) == 2:
      dtype = dtype[1] if dtype[0] is None else dtype[0]
    elif len(dtype) == 1:
      dtype = dtype[0]
    else:
      dtype = None

    inverse_min_event_ndims = _compute_min_event_ndims(
        bijectors, compute_forward=False)
    forward_min_event_ndims = _compute_min_event_ndims(
        bijectors, compute_forward=True)

    super(Chain, self).__init__(
        graph_parents=list(itertools.chain.from_iterable(
            b.graph_parents for b in bijectors)),
        forward_min_event_ndims=forward_min_event_ndims,
        inverse_min_event_ndims=inverse_min_event_ndims,
        is_constant_jacobian=all(b.is_constant_jacobian for b in bijectors),
        validate_args=validate_args,
        dtype=dtype,
        name=name or ("identity" if not bijectors else
                      "_of_".join(["chain"] + [b.name for b in bijectors])))

  @property
  def bijectors(self):
    return self._bijectors

  def _shape_helper(self, func_name, input_shape, reverse):
    new_shape = input_shape
    for b in reversed(self.bijectors) if reverse else self.bijectors:
      func = getattr(b, func_name, None)
      if func is None:
        raise ValueError("unable to call %s on bijector %s (%s)" %
                         (func_name, b.name, func))
      new_shape = func(new_shape)
    return new_shape

  def _forward_event_shape(self, input_shape):
    return self._shape_helper("forward_event_shape", input_shape,
                              reverse=True)

  def _forward_event_shape_tensor(self, input_shape):
    return self._shape_helper(
        "forward_event_shape_tensor", input_shape, reverse=True)

  def _inverse_event_shape(self, output_shape):
    return self._shape_helper("inverse_event_shape", output_shape,
                              reverse=False)

  def _inverse_event_shape_tensor(self, output_shape):
    return self._shape_helper("inverse_event_shape_tensor", output_shape,
                              reverse=False)

  def _inverse(self, y, **kwargs):
    for b in self.bijectors:
      y = b.inverse(y, **kwargs.get(b.name, {}))
    return y

  def _inverse_log_det_jacobian(self, y, **kwargs):
    y = ops.convert_to_tensor(y, name="y")
    ildj = math_ops.cast(0., dtype=y.dtype.base_dtype)

    if not self.bijectors:
      return ildj

    event_ndims = self._maybe_get_static_event_ndims(
        self.inverse_min_event_ndims)

    if _use_static_shape(y, event_ndims):
      event_shape = y.shape[y.shape.ndims - event_ndims:]
    else:
      event_shape = array_ops.shape(y)[array_ops.rank(y) - event_ndims:]

    for b in self.bijectors:
      ildj += b.inverse_log_det_jacobian(
          y, event_ndims=event_ndims, **kwargs.get(b.name, {}))

      if _use_static_shape(y, event_ndims):
        event_shape = b.inverse_event_shape(event_shape)
        event_ndims = self._maybe_get_static_event_ndims(
            event_shape.ndims)
      else:
        event_shape = b.inverse_event_shape_tensor(event_shape)
        event_ndims = array_ops.size(event_shape)
        event_ndims_ = self._maybe_get_static_event_ndims(event_ndims)
        if event_ndims_ is not None:
          event_ndims = event_ndims_

      y = b.inverse(y, **kwargs.get(b.name, {}))
    return ildj

  def _forward(self, x, **kwargs):
    for b in reversed(self.bijectors):
      x = b.forward(x, **kwargs.get(b.name, {}))
    return x

  def _forward_log_det_jacobian(self, x, **kwargs):
    x = ops.convert_to_tensor(x, name="x")

    fldj = math_ops.cast(0., dtype=x.dtype.base_dtype)

    if not self.bijectors:
      return fldj

    event_ndims = self._maybe_get_static_event_ndims(
        self.forward_min_event_ndims)

    if _use_static_shape(x, event_ndims):
      event_shape = x.shape[x.shape.ndims - event_ndims:]
    else:
      event_shape = array_ops.shape(x)[array_ops.rank(x) - event_ndims:]

    for b in reversed(self.bijectors):
      fldj += b.forward_log_det_jacobian(
          x, event_ndims=event_ndims, **kwargs.get(b.name, {}))
      if _use_static_shape(x, event_ndims):
        event_shape = b.forward_event_shape(event_shape)
        event_ndims = self._maybe_get_static_event_ndims(event_shape.ndims)
      else:
        event_shape = b.forward_event_shape_tensor(event_shape)
        event_ndims = array_ops.size(event_shape)
        event_ndims_ = self._maybe_get_static_event_ndims(event_ndims)
        if event_ndims_ is not None:
          event_ndims = event_ndims_

      x = b.forward(x, **kwargs.get(b.name, {}))

    return fldj
