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

import threading

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import backprop
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import def_function
from tensorflow.python.eager import execute
from tensorflow.python.eager import forwardprop_util

from tensorflow.python.framework import ops

from tensorflow.python.ops import array_ops
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest


# Dictionary mapping from op names to special-cased jvp functions. Otherwise
# backward functions are transposed on the tape.
_SPECIAL_CASES = {}


def _identity_jvp(attr_tuple, inputs, outputs, tangents):
  # Special-cased mostly for resource handles, where creating ones Tensors from
  # handle data for transposing the backward function on the tape is error-prone
  # (even if we get good handle data, partially defined shapes are an issue).
  del attr_tuple, inputs, outputs
  return [array_ops.identity(t) for t in tangents]


_SPECIAL_CASES["Identity"] = _identity_jvp


def _read_variable_jvp(attr_tuple, inputs, outputs, tangents):
  # Like for Identity, this special case means we don't need to create
  # variable-shaped Tensors from resource handles.
  del attr_tuple, inputs, outputs
  return [array_ops.identity(t) for t in tangents]


_SPECIAL_CASES["ReadVariableOp"] = _read_variable_jvp


_TRACE_COUNT_CONSISTENCY_LOCK = threading.Lock()
# Map from op names to number of traces of _jvp_helper. Used to cap the number
# of traces due to shape differences while still specializing where possible.
_TRACE_COUNT = {}


def _jvp_helper(op_name, attr_tuple, inputs, outputs, tangents):
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
  with _TRACE_COUNT_CONSISTENCY_LOCK:
    # Just make sure writes don't clobber each other's increments; reads in
    # _jvp_dispatch do not lock.
    _TRACE_COUNT[op_name] = _TRACE_COUNT.get(op_name, 0) + 1

  special_case = _SPECIAL_CASES.get(op_name, None)
  if special_case is not None:
    return special_case(attr_tuple, inputs, outputs, tangents)
  if not outputs:
    # tape.gradients([], inputs) doesn't make much sense
    return []
  # Generally inner GradientTapes won't function while outer accumulators are
  # recording. We temporarily reset forwardprop state to allow GradientTapes to
  # function here.
  with forwardprop_util.push_forwardprop_state():
    trainable_inputs = []
    trainable_indices = []
    nontrivial_tangents = []
    for input_index, tensor in enumerate(inputs):
      if backprop_util.IsTrainable(tensor):
        trainable_inputs.append(tensor)
        trainable_indices.append(input_index)
        nontrivial_tangents.append(tangents[input_index])

    with backprop.GradientTape() as transpose_tape:
      with backprop.GradientTape() as backfunc_tape:
        backfunc_tape.watch(trainable_inputs)
        execute.record_gradient(op_name, inputs, attr_tuple, outputs)

      forwardprop_aids = []
      trainable_outputs = []
      nontrivial_output_indices = []
      for output_index, output in enumerate(outputs):
        if backprop_util.IsTrainable(output):
          forwardprop_aids.append(
              array_ops.ones_like(output, name="unused_forwardprop_aid"))
          trainable_outputs.append(output)
          nontrivial_output_indices.append(output_index)

      transpose_tape.watch(forwardprop_aids)
      grads = backfunc_tape.gradient(
          trainable_outputs,
          trainable_inputs,
          forwardprop_aids,
          unconnected_gradients=UnconnectedGradients.ZERO)
    nontrivial_output_tangents = transpose_tape.gradient(
        grads, forwardprop_aids, output_gradients=nontrivial_tangents)
    output_tangents = [None] * len(outputs)
    for index, tangent in zip(nontrivial_output_indices,
                              nontrivial_output_tangents):
      output_tangents[index] = tangent
    return output_tangents


# TODO(allenl): experimental_relax_shapes for gradients which rely on static
# shape information are underspecialized. We may want hand-written forward
# implementations, or a more satisfying story about how we re-specialize
# gradients which were traced with relaxed shapes (e.g. use conds instead of
# trace-time Python logic).
_jvp_relaxed_shapes = def_function.function(
    _jvp_helper, experimental_relax_shapes=True)
_jvp_exact_shapes = def_function.function(
    _jvp_helper, experimental_relax_shapes=False)

# The maximum number of exact-shape traces to perform for a single op before
# switching to shape relaxation.
_TRACE_COUNT_LIMIT = 32


def _jvp_dispatch(op_name, attr_tuple, inputs, outputs, tangents):
  """Determine which forwardprop function to call."""
  # Note that this _TRACE_COUNT read races with writes. That's fine, it just
  # means we may trace a few more exact shapes before moving on to relaxation.
  if _TRACE_COUNT.get(op_name, 0) < _TRACE_COUNT_LIMIT:
    return _jvp_exact_shapes(
        op_name, attr_tuple, inputs, outputs, tangents)
  else:
    return _jvp_relaxed_shapes(
        op_name, attr_tuple, inputs, outputs, tangents)

pywrap_tensorflow.TFE_Py_RegisterJVPFunction(_jvp_dispatch)


class ForwardAccumulator(object):
  """Computes Jacobian-vector products using forward-mode autodiff.

  Example:

  ```
  with ForwardAccumulator(
      primals=x,
      tangents=tf.constant([[5., 6.], [7., 8.]])) as acc:
    x = tf.constant([[2.0, 3.0], [1.0, 4.0]])
    y = tf.reduce_sum(tf.sin(x) * tf.tan(x), axis=1)
  jvp = acc.jvp(y)
  ```

  Note that `ForwardAccumulator`s are always applied in creation order, so inner
  accumulators will not see JVP computation from outer accumulators. Take
  higher-order jvps from outer accumulators:

  ```
  primal = tf.constant(1.1)
  with ForwardAccumulator(primal, tf.constant(1.)) as outer_acc:
    with ForwardAccumulator(primal, tf.constant(1.)) as acc:
      primal_out = primal ** tf.constant(3.5)
  inner_jvp = acc.jvp(primal_out)
  outer_jvp = outer_acc.jvp(inner_jvp)
  ```

  Reversing the collection in the last two lines to instead retrieve
  `acc.jvp(outer_acc.jvp(primal_out))` will not work.

  Strict nesting also applies to combinations of `ForwardAccumulator` and
  `tf.GradientTape`. More deeply nested `GradientTape` objects will ignore the
  products of outer `ForwardAccumulator` objects. This allows (for example)
  memory-efficient forward-over-backward computation of Hessian-vector products,
  where the inner `GradientTape` would otherwise hold on to all intermediate
  jvps.
  """

  def __init__(self, primals, tangents):
    """Specify tensors to watch and their Jacobian-vector products.

    Mathematically, `tangents` is a vector right-multiplying the Jacobian matrix
    (a Jacobian-vector product) for the function computed while this accumulator
    is active. Since JVPs are computed in forward mode as the computation
    happens, this vector must be supplied before the computation takes place.

    Listing a single Tensor multiple times sums each `tangents`. An un-watched
    Tensor has zeros for its tangent vector.

    Args:
      primals: A Tensor or nested structure of Tensors to watch.
      tangents: A Tensor or list of Tensors matching `primals`.
    """
    self._accumulator = pywrap_tensorflow.TFE_Py_ForwardAccumulatorNew()
    self._recording = False
    self._watch(primals, tangents)

  def __enter__(self):
    self._push_accumulator()
    return self

  def __exit__(self, typ, value, traceback):
    if self._recording:
      self._pop_accumulator()

  def _push_accumulator(self):
    if self._recording:
      raise ValueError("Accumulator is already recording.")
    pywrap_tensorflow.TFE_Py_ForwardAccumulatorSetAdd(self._accumulator)
    self._recording = True

  def _pop_accumulator(self):
    if not self._recording:
      raise ValueError("Accumulator is not recording.")
    pywrap_tensorflow.TFE_Py_ForwardAccumulatorSetRemove(self._accumulator)
    self._recording = False

  def _watch(self, primals, tangents):
    """Ensures that `primals` are being traced by this accumulator.

    Mathematically, `tangents` is a vector right-multiplying the Jacobian matrix
    (a Jacobian-vector product) for the function computed while this accumulator
    is active. Since JVPs are computed in forward mode as the computation
    happens, this vector must be supplied before the computation takes place.

    Watching a single tensor multiple times sums each of its `tangents`. Any
    un-watched tensor has zeros for its tangent vector.

    Args:
      primals: A Tensor or list of Tensors.
      tangents: A Tensor or list of Tensors matching `primals`.
    """
    nest.assert_same_structure(primals, tangents)
    for t, g in zip(nest.flatten(primals), nest.flatten(tangents)):
      if not t.dtype.is_floating:
        logging.log_first_n(
            logging.WARN, "The dtype of the watched primal must be "
            "floating (e.g. tf.float32), got %r", 5, t.dtype)
      g = ops.convert_to_tensor(g, dtype=t.dtype)
      if hasattr(t, "handle"):
        # Run convert_to_tensor to get the captured handle from whichever
        # function we're running if necessary.
        t = ops.convert_to_tensor(t.handle)
      pywrap_tensorflow.TFE_Py_ForwardAccumulatorWatch(self._accumulator, t, g)

  def jvp(self, target, unconnected_gradients=UnconnectedGradients.NONE):
    """Fetches the Jacobian-vector product computed for `target`.

    Note that this function performs no computation, and simply looks up a
    JVP that was already computed (unlike backprop using a
    `tf.GradientTape`, where the computation happens on the call to
    `tape.gradient`).

    Args:
      target: A watched Tensor or structure of Tensors to fetch the JVPs for.
      unconnected_gradients: A value which can either hold 'none' or 'zero' and
        alters the value which will be returned if no JVP was computed for
        `target`. The possible values and effects are detailed in
        'tf.UnconnectedGradients' and it defaults to 'none'.

    Returns:
      Tensors with the same shapes and dtypes as `target`, or None if no JVP
      is available.
    """
    unconnected_gradients = UnconnectedGradients(unconnected_gradients)
    if self._accumulator is None:
      raise ValueError("Called jvp() without first tracing anything.")
    def _fetch_jvp(tensor):
      if hasattr(tensor, "handle"):
        tensor = ops.convert_to_tensor(tensor.handle)
      result = pywrap_tensorflow.TFE_Py_ForwardAccumulatorJVP(
          self._accumulator, tensor)
      if result is None and unconnected_gradients == UnconnectedGradients.ZERO:
        return array_ops.zeros_like(tensor)
      return result
    return nest.map_structure(_fetch_jvp, target)
