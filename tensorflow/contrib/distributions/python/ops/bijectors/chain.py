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

from tensorflow.contrib.distributions.python.ops.bijectors import bijector
from tensorflow.python.framework import constant_op


__all__ = [
    "Chain",
]


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
    dtype = list(set([b.dtype for b in bijectors]))
    if len(dtype) > 2:
      raise ValueError("incompatible dtypes: %s" % dtype)
    elif len(dtype) == 2:
      dtype = dtype[1] if dtype[0] is None else dtype[0]
      event_ndims = bijectors[0].event_ndims
    elif len(dtype) == 1:
      dtype = dtype[0]
      event_ndims = bijectors[0].event_ndims
    else:
      dtype = None
      event_ndims = None

    super(Chain, self).__init__(
        graph_parents=list(itertools.chain.from_iterable(
            b.graph_parents for b in bijectors)),
        is_constant_jacobian=all(b.is_constant_jacobian for b in bijectors),
        validate_args=validate_args,
        dtype=dtype,
        event_ndims=event_ndims,
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
    ildj = constant_op.constant(0., dtype=y.dtype,
                                name="inverse_log_det_jacobian")
    for b in self.bijectors:
      ildj += b.inverse_log_det_jacobian(y, **kwargs.get(b.name, {}))
      y = b.inverse(y, **kwargs.get(b.name, {}))
    return ildj

  def _forward(self, x, **kwargs):
    for b in reversed(self.bijectors):
      x = b.forward(x, **kwargs.get(b.name, {}))
    return x

  def _forward_log_det_jacobian(self, x, **kwargs):
    fldj = constant_op.constant(0., dtype=x.dtype,
                                name="forward_log_det_jacobian")
    for b in reversed(self.bijectors):
      fldj += b.forward_log_det_jacobian(x, **kwargs.get(b.name, {}))
      x = b.forward(x, **kwargs.get(b.name, {}))
    return fldj
