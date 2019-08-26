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
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.eager import execute

from tensorflow.python.framework import ops

from tensorflow.python.ops import array_ops
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest


# TODO(allenl): experimental_relax_shapes for gradients which rely on static
# shape information may be underspecialized. We may want hand-written forward
# implementations.
@def_function.function(experimental_relax_shapes=True)
def _forward_gradient(op_name, attr_tuple, inputs, outputs, tangents):
  """Computes a Jacobian-vector product for an op.

  Note that this function would be wasteful if executed eagerly. It runs the
  backward gradient function and throws away the result just to record its
  operations on a GradientTape. These unused ops are pruned away when this
  function is traced.

  Args:
    op_name: A string, the type of operation being executed.
    attr_tuple: Attributes of the operation.
    inputs: A flat list of input Tensors to the operation.
    outputs: A flat list of output Tensors from the operation.
    tangents: A flat list of Tensors, same shape as `inputs`.

  Returns:
    A flat list of tangents corresponding to `outputs`.
  """
  float_inputs = []
  float_indices = []
  nontrivial_tangents = []
  for input_index, tensor in enumerate(inputs):
    if tensor.dtype.is_floating:
      float_inputs.append(tensor)
      float_indices.append(input_index)
      nontrivial_tangents.append(tangents[input_index])

  with backprop.GradientTape() as transpose_tape:
    with backprop.GradientTape() as backfunc_tape:
      backfunc_tape.watch(float_inputs)
      execute.record_gradient(op_name, inputs, attr_tuple, outputs,
                              "forward_op_replay")

    forwardprop_aids = []
    float_outputs = []
    nontrivial_output_indices = []
    for output_index, output in enumerate(outputs):
      if output.dtype.is_floating:
        forwardprop_aids.append(
            array_ops.ones_like(output, name="unused_forwardprop_aid"))
        float_outputs.append(output)
        nontrivial_output_indices.append(output_index)

    transpose_tape.watch(forwardprop_aids)
    grads = backfunc_tape.gradient(
        float_outputs,
        float_inputs,
        forwardprop_aids,
        unconnected_gradients=UnconnectedGradients.ZERO)
  nontrivial_output_tangents = transpose_tape.gradient(
      grads, forwardprop_aids, output_gradients=nontrivial_tangents)
  output_tangents = [None] * len(outputs)
  for index, tangent in zip(nontrivial_output_indices,
                            nontrivial_output_tangents):
    output_tangents[index] = tangent
  return output_tangents


pywrap_tensorflow.TFE_Py_RegisterForwardGradientFunction(_forward_gradient)


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

  Note that `ForwardGradientAccumulator`s are always applied in creation order,
  so inner accumulators may not see JVP computation from outer
  accumulators. Take higher-order gradients from outer accumulators:

  ```
  primal = tf.constant(1.1)
  with ForwardGradientAccumulator() as outer_acc:
    outer_acc.watch(primal, tf.constant(1.))
    with ForwardGradientAccumulator() as acc:
      acc.watch(primal, tf.constant(1.))
      primal_out = primal ** tf.constant(3.5)
  inner_jvp = acc.jvp(primal_out)
  outer_jvp = outer_acc.jvp(inner_jvp)
  ```

  Reversing the collection in the last two lines to instead retrieve
  `acc.jvp(outer_acc.jvp(primal_out))` will not work.
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

  # TODO(allenl): Does this need to be public, or should the constructor instead
  # take all watched Tensors? Write a realistic usage example (e.g. Hessian-free
  # optimization) and decide.
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
