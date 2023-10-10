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

import functools
import threading

from tensorflow.core.function.polymorphism import function_cache
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import execute
from tensorflow.python.eager import forwardprop_util
from tensorflow.python.eager.polymorphic_function import tracing_compilation
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.parallel_for import control_flow_ops
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


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


def _jvp_helper_wrapper(op_name, attr_tuple, inputs, outputs, tangents,
                        use_batch):
  """Computes a batch of Jacobian-vector product for an op.

  Args:
    op_name: A string, the type of operation being executed.
    attr_tuple: Attributes of the operation.
    inputs: A flat list of input Tensors to the operation.
    outputs: A flat list of output Tensors from the operation.
    tangents: A flat list of Tensors, compatible with shape `[None] +
      input_shape`.
    use_batch: A bool, True to vetorize over batch of tangents of shape `[None]
      + input_shape`.

  Returns:
    A flat list of tangents compatible with `outputs`
    or `[None] + output_shape`.

  Raises:
    ValueError: if tangent shapes are not compatible with input shapes.
  """
  if use_batch:
    for primal, tangent in zip(inputs, tangents):
      if not tangent.shape.is_compatible_with([None] + primal.shape):
        raise ValueError("Tangent {} was expected to be of shape "
                         "{} but is instead of shape {}".format(
                             tangent, [None] + primal.shape, tangent.shape))

    return control_flow_ops.vectorized_map(
        functools.partial(_jvp_helper, op_name, attr_tuple, inputs, outputs),
        tangents,
    )
  return _jvp_helper(op_name, attr_tuple, inputs, outputs, tangents)


# TODO(allenl): reduce_retracing for gradients which rely on static
# shape information are underspecialized. We may want hand-written forward
# implementations, or a more satisfying story about how we re-specialize
# gradients which were traced with relaxed shapes (e.g. use conds instead of
# trace-time Python logic).
#
# Using function.defun rather than def_function.function avoids
# tf.config.run_functions_eagerly(True). `_jvp_helper` doesn't successfully run
# eagerly (infinite recursion), and even if it did it would use extra memory and
# run unnecessary computation. The function does not create variables, so the
# two symbols are otherwise equivalent.
_jvp_function_cache = function_cache.FunctionCache()
_jvp_relaxed_config = tracing_compilation.TracingOptions(
    _jvp_helper_wrapper,
    name="_jvp_relaxed_shapes",
    reduce_retracing=True,
    function_cache=_jvp_function_cache,
)

_jvp_exact_config = tracing_compilation.TracingOptions(
    _jvp_helper_wrapper,
    name="_jvp_exact_shapes",
    reduce_retracing=False,
    function_cache=_jvp_function_cache,
)

# The maximum number of exact-shape traces to perform for a single op before
# switching to shape relaxation.
_TRACE_COUNT_LIMIT = 32


def _jvp_dispatch(op_name,
                  attr_tuple,
                  inputs,
                  outputs,
                  tangents,
                  use_batch=False):
  """Determine which forwardprop function to call."""
  # Note that this _TRACE_COUNT read races with writes. That's fine, it just
  # means we may trace a few more exact shapes before moving on to relaxation.
  if _TRACE_COUNT.get(op_name, 0) < _TRACE_COUNT_LIMIT:
    config = _jvp_exact_config
  else:
    config = _jvp_relaxed_config
  return tracing_compilation.call_function(
      (op_name, attr_tuple, inputs, outputs, tangents, use_batch),
      tracing_options=config,
  )


pywrap_tfe.TFE_Py_RegisterJVPFunction(_jvp_dispatch)


@tf_export("autodiff.ForwardAccumulator", v1=[])
class ForwardAccumulator():
  """Computes Jacobian-vector products ("JVP"s) using forward-mode autodiff.

  Compare to `tf.GradientTape` which computes vector-Jacobian products ("VJP"s)
  using reverse-mode autodiff (backprop). Reverse mode is more attractive when
  computing gradients of a scalar-valued function with respect to many inputs
  (e.g. a neural network with many parameters and a scalar loss). Forward mode
  works best on functions with many outputs and few inputs. Since it does not
  hold on to intermediate activations, it is much more memory efficient than
  backprop where it is applicable.

  Consider a simple linear regression:

  >>> x = tf.constant([[2.0, 3.0], [1.0, 4.0]])
  >>> targets = tf.constant([[1.], [-1.]])
  >>> dense = tf.keras.layers.Dense(1)
  >>> dense.build([None, 2])
  >>> with tf.autodiff.ForwardAccumulator(
  ...    primals=dense.kernel,
  ...    tangents=tf.constant([[1.], [0.]])) as acc:
  ...   loss = tf.reduce_sum((dense(x) - targets) ** 2.)
  >>> acc.jvp(loss)
  <tf.Tensor: shape=(), dtype=float32, numpy=...>

  The example has two variables containing parameters, `dense.kernel` (2
  parameters) and `dense.bias` (1 parameter). Considering the training data `x`
  as a constant, this means the Jacobian matrix for the function mapping from
  parameters to loss has one row and three columns.

  With forwardprop, we specify a length-three vector in advance which multiplies
  the Jacobian. The `primals` constructor argument is the parameter (a
  `tf.Tensor` or `tf.Variable`) we're specifying a vector for, and the
  `tangents` argument is the "vector" in Jacobian-vector product. If our goal is
  to compute the entire Jacobian matrix, forwardprop computes one column at a
  time while backprop computes one row at a time. Since the Jacobian in the
  linear regression example has only one row, backprop requires fewer
  invocations:

  >>> x = tf.constant([[2.0, 3.0], [1.0, 4.0]])
  >>> targets = tf.constant([[1.], [-1.]])
  >>> dense = tf.keras.layers.Dense(1)
  >>> dense.build([None, 2])
  >>> loss_fn = lambda: tf.reduce_sum((dense(x) - targets) ** 2.)
  >>> kernel_fprop = []
  >>> with tf.autodiff.ForwardAccumulator(
  ...     dense.kernel, tf.constant([[1.], [0.]])) as acc:
  ...   kernel_fprop.append(acc.jvp(loss_fn()))
  >>> with tf.autodiff.ForwardAccumulator(
  ...     dense.kernel, tf.constant([[0.], [1.]])) as acc:
  ...   kernel_fprop.append(acc.jvp(loss_fn()))
  >>> with tf.autodiff.ForwardAccumulator(dense.bias, tf.constant([1.])) as acc:
  ...   bias_fprop = acc.jvp(loss_fn())
  >>> with tf.GradientTape() as tape:
  ...   loss = loss_fn()
  >>> kernel_grad, bias_grad = tape.gradient(loss, (dense.kernel, dense.bias))
  >>> np.testing.assert_allclose(
  ...     kernel_grad, tf.stack(kernel_fprop)[:, tf.newaxis])
  >>> np.testing.assert_allclose(bias_grad, bias_fprop[tf.newaxis])

  Implicit in the `tape.gradient` call is a length-one vector which
  left-multiplies the Jacobian, a vector-Jacobian product.

  `ForwardAccumulator` maintains JVPs corresponding primal tensors it is
  watching, derived from the original `primals` specified in the constructor. As
  soon as a primal tensor is deleted, `ForwardAccumulator` deletes the
  corresponding JVP.

  `acc.jvp(x)` retrieves `acc`'s JVP corresponding to the primal tensor `x`. It
  does not perform any computation. `acc.jvp` calls can be repeated as long as
  `acc` is accessible, whether the context manager is active or not. New JVPs
  are only computed while the context manager is active.

  Note that `ForwardAccumulator`s are always applied in the order their context
  managers were entered, so inner accumulators will not see JVP computation from
  outer accumulators. Take higher-order JVPs from outer accumulators:

  >>> primal = tf.constant(1.1)
  >>> with tf.autodiff.ForwardAccumulator(primal, tf.constant(1.)) as outer:
  ...   with tf.autodiff.ForwardAccumulator(primal, tf.constant(1.)) as inner:
  ...     primal_out = primal ** tf.constant(3.5)
  >>> inner_jvp = inner.jvp(primal_out)
  >>> inner_jvp  # 3.5 * 1.1 ** 2.5
  <tf.Tensor: shape=(), dtype=float32, numpy=4.4417057>
  >>> outer.jvp(inner_jvp)  # 3.5 * 2.5 * 1.1 ** 1.5
  <tf.Tensor: shape=(), dtype=float32, numpy=10.094786>

  Reversing the collection in the last line to instead retrieve
  `inner.jvp(outer.jvp(primal_out))` will not work.

  Strict nesting also applies to combinations of `ForwardAccumulator` and
  `tf.GradientTape`. More deeply nested `GradientTape` objects will ignore the
  products of outer `ForwardAccumulator` objects. This allows (for example)
  memory-efficient forward-over-backward computation of Hessian-vector products,
  where the inner `GradientTape` would otherwise hold on to all intermediate
  JVPs:

  >>> v = tf.Variable([1., 2.])
  >>> with tf.autodiff.ForwardAccumulator(
  ...     v,
  ...     # The "vector" in Hessian-vector product.
  ...     tf.constant([1., 0.])) as acc:
  ...   with tf.GradientTape() as tape:
  ...     y = tf.reduce_sum(v ** 3.)
  ...   backward = tape.gradient(y, v)
  >>> backward  # gradient from backprop
  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([ 3., 12.], dtype=float32)>
  >>> acc.jvp(backward)  # forward-over-backward Hessian-vector product
  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([6., 0.], dtype=float32)>
  """

  def __init__(self, primals, tangents):
    """Specify tensors to watch and their Jacobian-vector products.

    Mathematically, `tangents` is a vector right-multiplying the Jacobian matrix
    (a Jacobian-vector product) for the function computed while this accumulator
    is active. Since JVPs are computed in forward mode as the computation
    happens, this vector must be supplied in advance.

    Listing a single tensor multiple times in `primals` raises an
    exception. Excluding a tensor from `primals` is equivalent to watching it
    with a tangent tensor of zeros.

    Args:
      primals: A tensor or nested structure of tensors to watch.
      tangents: A tensor or nested structure of tensors, with the same nesting
        structure as `primals`, with each element being a vector with the same
        size as the corresponding primal element.

    Raises:
      ValueError: If the same tensor or variable is specified multiple times in
        `primals`.
    """
    self._accumulator = pywrap_tfe.TFE_Py_ForwardAccumulatorNew(False)
    self._recording = False
    primal_ids = set()
    for primal in nest.flatten(primals):
      if id(primal) in primal_ids:
        raise ValueError(
            "Tensor {} was specified as a primal multiple times. This may "
            "indicate an error. If it was intended, please sum the "
            "corresponding tangents.")
      primal_ids.add(id(primal))
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
    pywrap_tfe.TFE_Py_ForwardAccumulatorSetAdd(self._accumulator)
    self._recording = True

  def _pop_accumulator(self):
    if not self._recording:
      raise ValueError("Accumulator is not recording.")
    pywrap_tfe.TFE_Py_ForwardAccumulatorSetRemove(self._accumulator)
    self._recording = False

  def _watch(self, primals, tangents):
    """Ensures that `primals` are being traced by this accumulator.

    Mathematically, `tangents` is a vector right-multiplying the Jacobian matrix
    (a Jacobian-vector product) for the function computed while this accumulator
    is active. Since JVPs are computed in forward mode as the computation
    happens, this vector must be supplied in advance.

    Watching a single tensor multiple times sums each of its `tangents`. Any
    un-watched tensor has zeros for its tangent vector.

    Args:
      primals: A Tensor or list of Tensors.
      tangents: A Tensor or list of Tensors matching `primals`.
    """

    def _watch(primal, tangent):
      if not primal.dtype.is_floating:
        logging.log_first_n(
            logging.WARN, "The dtype of the watched primal must be "
            "floating (e.g. tf.float32), got %r", 5, primal.dtype)
      tangent = ops.convert_to_tensor(tangent, dtype=primal.dtype)
      if hasattr(primal, "handle"):
        # Run convert_to_tensor to get the captured handle from whichever
        # function we're running if necessary.
        primal = ops.convert_to_tensor(primal.handle)
      pywrap_tfe.TFE_Py_ForwardAccumulatorWatch(self._accumulator, primal,
                                                tangent)

    nest.map_structure(_watch, primals, tangents)

  def jvp(self, primals, unconnected_gradients=UnconnectedGradients.NONE):
    """Fetches the Jacobian-vector product computed for `primals`.

    Note that this method performs no computation, and simply looks up a JVP
    that was already computed (unlike backprop using a `tf.GradientTape`, where
    the computation happens on the call to `tape.gradient`).

    Args:
      primals: A watched Tensor or structure of Tensors to fetch the JVPs for.
      unconnected_gradients: A value which can either hold 'none' or 'zero' and
        alters the value which will be returned if no JVP was computed for
        `primals`. The possible values and effects are detailed in
        'tf.UnconnectedGradients' and it defaults to 'none'.

    Returns:
      Tensors with the same shapes and dtypes as `primals`, or None if no JVP
      is available.
    """
    unconnected_gradients = UnconnectedGradients(unconnected_gradients)
    if self._accumulator is None:
      raise ValueError("Called jvp() without first tracing anything.")

    def _fetch_jvp(tensor):
      if hasattr(tensor, "handle"):
        unwrapped_tensor = ops.convert_to_tensor(tensor.handle)
      else:
        unwrapped_tensor = tensor
      result = pywrap_tfe.TFE_Py_ForwardAccumulatorJVP(self._accumulator,
                                                       unwrapped_tensor)
      if result is None and unconnected_gradients == UnconnectedGradients.ZERO:
        result = array_ops.zeros_like(tensor)
      return result

    return nest.map_structure(_fetch_jvp, primals)

  @classmethod
  def _batch_accumulator(cls, primals, tangents):
    """Factory constructor to test accumulator on batches of tangents.

    Args:
      primals: A tensor or nested structure of tensors to watch.
      tangents: A tensor or nested structure of tensors, with the same nesting
        structure as `primals`, with each element being a vector with compatible
        shape `[None] + primal.shape` of the corresponding primal element.

    Returns:
      A batch accumulator object.
    """
    acc = super(ForwardAccumulator, cls).__new__(cls, primals, tangents)
    acc._recording = False
    acc._accumulator = pywrap_tfe.TFE_Py_ForwardAccumulatorNew(True)
    primal_ids = set()
    for primal, tangent in zip(nest.flatten(primals), nest.flatten(tangents)):
      tangent.shape.assert_is_compatible_with(
          tensor_shape.TensorShape([None]) + primal.shape)
      if id(primal) in primal_ids:
        raise ValueError(
            "Tensor {} was specified as a primal multiple times. This may "
            "indicate an error. If it was intended, please sum the "
            "corresponding tangents.")
      primal_ids.add(id(primal))
    acc._watch(primals, tangents)
    return acc
