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
"""Utilities for forward-mode automatic differentiation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest


# TODO(allenl): Special-case op gradients and tf.functions to avoid unnecessary
# evaluation of gradient functions.
class ForwardGradientAccumulator(object):
  """Computes Jacobian-vector products using forward-mode autodiff.

  Example:

  ```
  with ForwardGradientAccumulator() as acc:
    x = tf.constant([[2.0, 3.0], [1.0, 4.0]])
    acc.watch(x, tf.constant([[5., 6.], [7., 8.]]))
    y = tf.reduce_sum(tf.sin(x) * tf.tan(x), axis=1)
  jvp = acc.jvp(y)
  ```
  """

  def __init__(self):
    self._accumulator = None
    self._recording = False

  def __enter__(self):
    self._push_accumulator()
    return self

  def __exit__(self, typ, value, traceback):
    if self._recording:
      self._pop_accumulator()

  def _push_accumulator(self):
    if self._recording:
      raise ValueError("Accumulator is already recording.")
    if self._accumulator is None:
      self._accumulator = pywrap_tensorflow.TFE_Py_ForwardAccumulatorNew()
    else:
      # TODO(allenl): Allow reuse
      raise NotImplementedError("Accumulator reuse isn't implemented yet.")
    self._recording = True

  def _pop_accumulator(self):
    if not self._recording:
      raise ValueError("Tape is not recording.")
    pywrap_tensorflow.TFE_Py_ForwardAccumulatorSetRemove(self._accumulator)
    self._recording = False

  def watch(self, tensor, tangents):
    """Ensures that `tensor` is being traced by this tape.

    Mathematically, `tangents` is part of a vector right-multiplying the
    Jacobian matrix (a Jacobian-vector product) for the function computed while
    the tape is active. Since JVPs are computed in forward mode as the
    computation happens, this vector must be supplied before the computation
    takes place.

    Watching a single Tensor multiple times sums each `tangents`. An un-watched
    Tensor has zeros for its tangent vector.

    Args:
      tensor: A Tensor or list of Tensors.
      tangents: A Tensor or list of Tensors matching `tensor`.
    """
    nest.assert_same_structure(tensor, tangents)
    for t, g in zip(nest.flatten(tensor), nest.flatten(tangents)):
      if not t.dtype.is_floating:
        logging.log_first_n(
            logging.WARN, "The dtype of the watched tensor must be "
            "floating (e.g. tf.float32), got %r", 5, t.dtype)
      if hasattr(t, "handle"):
        # TODO(allenl): Handle watching variables.
        raise NotImplementedError("Currently only Tensors may be watched.")
      g = ops.convert_to_tensor(g, dtype=t.dtype)
      pywrap_tensorflow.TFE_Py_ForwardAccumulatorWatch(self._accumulator, t, g)

  def jvp(self, target):
    """Fetches the Jacobian-vector product computed for `target`.

    Note that this function performs no computation, and simply looks up a
    JVP that was already computed (unlike backprop using a
    `tf.GradientTape`, where the computation happens on the call to
    `tape.gradient`).

    Args:
      target: A watched Tensor or structure of Tensors to fetch the JVPs for.

    Returns:
      Tensors with the same shapes and dtypes as `target`, or None if no JVP
      is available.
    """
    if self._accumulator is None:
      raise ValueError("Called jvp() without first tracing anything.")
    return nest.map_structure(
        functools.partial(pywrap_tensorflow.TFE_Py_ForwardAccumulatorJVP,
                          self._accumulator), target)
