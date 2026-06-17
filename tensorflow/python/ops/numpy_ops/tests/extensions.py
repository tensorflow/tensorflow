# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Extensions such as `jit`, `grad`, `logsumexp`, etc."""
import bisect
import contextlib
import copy
import functools
import string
import sys
import threading
import numpy as np
import six
from tensorflow.python.compiler.xla import xla
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device_spec
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import gen_bitwise_ops
from tensorflow.python.ops import gen_collective_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_loop
import tensorflow.python.ops.numpy_ops.tests.np_wrapper as tf_np
from tensorflow.python.ops.parallel_for import control_flow_ops as pfor_ops
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util import nest

_int_dtype_lower_bounds = [
    -2**63, -2**31, -2**15, -2**7, 0, 2**7, 2**15, 2**31, 2**64
]
_int_dtypes = [
    dtypes.int64,
    dtypes.int32,
    dtypes.int16,
    dtypes.int8,
    dtypes.uint8,
    dtypes.uint16,
    dtypes.uint32,
    dtypes.uint64,
]
_tf_nn_APIs = {
    1: [nn_ops.conv1d, nn_ops.conv1d_transpose],
    2: [nn_ops.conv2d_v2, nn_ops.conv2d_transpose],
    3: [nn_ops.conv3d_v2, nn_ops.conv3d_transpose],
}


remat = custom_gradient.recompute_grad


def most_precise_int_dtype(x):
  if not isinstance(x, six.integer_types) or isinstance(x, bool):
    return None
  i = bisect.bisect_right(_int_dtype_lower_bounds, x)
  if i in (0, len(_int_dtype_lower_bounds)):
    raise ValueError("Integer %s is out of bounds" % x)
  assert len(_int_dtype_lower_bounds) == len(_int_dtypes) + 1
  return _int_dtypes[i - 1]


def _canonicalize_jit_arg(x):  # pylint: disable=missing-function-docstring
  if isinstance(x, tf_np.ndarray):
    return x
  try:
    # We need to convert `int` to the most precise dtype, otherwise the dtype
    # of the result may be different from numpy's. For example, when a binary
    # op takes in a Python integer 5 and an array of uint32, numpy will pick
    # uint32 as 5's dtype, while tf.convert_to_tensor will choose int32 which
    # will cause the two arguments to be promoted to int64. We pick uint8
    # here, which will be promoted to uint32 by the binary op.
    # Note that we prefer unsigned int to signed int when both are equally
    # precise. For example, for 5, we pick uint8 instead of int8. There is no
    # reason to prefer one to the other, because for each there is a case
    # where the behavior diverges from numpy. If we prefer signed int,
    # consider the case where the first operand is 5 and the second is
    # 2**64-1. Numpy picks uint64 as the result dtype, but because we choose a
    # signed type for 5 such as int8, the result type will be float64. On the
    # other hand, if we prefer unsigned int, consider the case where the first
    # operand is 2**31-1 and the second is -1. Numpy will pick int32, but
    # because we choose uint32 for 2*32-1, the result will be int64. The root
    # of the problem is that `jit` converts `int` to tensors (hence committing
    # to a dtype) too early, when we don't have enough information about the
    # jitted function (e.g. which subset of the arguments should be promoted
    # together using np.result_type). tf.function doesn't have this problem
    # because it doesn't convert `int` to tensors. jax.jit doesn't have this
    # problem because it converts `int` to "int tracer" which doesn't commit
    # to a dtype.
    # TODO(wangpeng): Revisit this design and see whether we can improve `jit`
    #   and tf.function.
    dtype = most_precise_int_dtype(x)
    if dtype is None and isinstance(x, float):
      dtype = tf_np.default_float_type()
    return ops.convert_to_tensor(value=x, dtype=dtype)
  except (TypeError, ValueError):
    return x


def _canonicalize_jit_arguments(inp):
  """Canonicalize arguments to be used for jit.

  Args:
    inp: a nested structure of arguments to be canonicalized (i.e. to be
      converted to Tensors). Only tf_np.ndarray and things accepted by
      `tf.convert_to_tensor` will be converted.

  Returns:
    The canonicalized version.
  """
  return nest.map_structure(_canonicalize_jit_arg, inp)


def _tf_to_np(inp):

  def f(x):
    if isinstance(x, indexed_slices.IndexedSlices):
      return tf_np.asarray(x)
    else:
      return x

  return nest.map_structure(f, inp)


def stop_gradient(x):

  def static_stop_gradient(x):
    # `tf.stop_gradient` is a no-op for non-Tensor. Returning the original type
    # allows it to be used in the conditional without Autograph, if static. For
    # example:
    # `if fastmath.stop_gradient(5) > 4:`
    return array_ops.stop_gradient(x) if tensor_util.is_tensor(x) else x

  return _tf_to_np(nest.map_structure(static_stop_gradient, x))


def custom_grad(f_vjp, f_original=None):
  """Decorator to define a function with a custom gradient.

  This function is very similar to `tf.custom_gradient`. See the documentation
  of `tf.custom_gradient` for detailed usage.

  The differences with `tf.custom_gradient` are:

  - All arguments and results are tf_np.ndarrays instead of tensors.

  - The `grad_fn` returned by `f_vjp` accepts and returns nested structures,
    unlike that in `tf.custom_gradient` which only accepts and returns lists.

  Args:
    f_vjp: the same as the `f` argument of `tf.custom_gradient`. Note that all
      inputs and outputs of `f_vjp` and of the `grad_fn` function it returns can
      be nested structures.
    f_original: (optional) not used.

  Returns:
    The same as `tf.custom_gradient`.
  """
  del f_original

  @custom_gradient.custom_gradient
  def tf_f(*tf_args, **tf_kwargs):
    np_args = _tf_to_np(tf_args)
    np_kwargs = _tf_to_np(tf_kwargs)
    np_y, np_vjp = f_vjp(*np_args, **np_kwargs)
    tf_y = np_y

    def tf_vjp(*flat_tf_dy):
      tf_dy = nest.pack_sequence_as(tf_y, flat_tf_dy)
      np_dy = _tf_to_np(tf_dy)
      np_dx = np_vjp(np_dy)
      return nest.flatten(np_dx)

    return tf_y, tf_vjp

  def np_f(*args, **kwargs):
    return _tf_to_np(tf_f(*args), **kwargs)

  return np_f


def vjp(f, *primals, has_aux=False):
  """Returns the result and the VJP function of `f`.

  This function returns the result and the vector-Jacobian-product (VJP)
  function of `f`.

  Args:
    f: a function from (nested structures of) tf_np.ndarrays to a (nested
      structure of) tf_np.ndarray. If `has_aux` is True, it should return an
      extra output.
    *primals: the inputs to be fed to `f`.
    has_aux: if True, the second output of `f` will be regarded as an auxiliary,
      non-differentiable output that will be ignored by the VJP function.

  Returns:
    A pair `(y, vjpfun)` if `has_aux` is False; a tuple `(y, vjpfun, aux)`
    otherwise. `y` and `aux` are the outputs of `f`, i.e. `y, aux =
    f(*primals)`. `vjpfun` is a function `dx = vjpfun(dy)`, where `dy` is the
    cotengents of `y`, having the same structures, shapes and dtypes as
    `y`. `dx` is the cotengents of `x`, having the same structures, shapes and
    dtypes as `x`.
  """
  with backprop.GradientTape(persistent=True) as tape:
    tape.watch(nest.flatten(primals))
    outputs = f(*primals)
    if has_aux:
      np_out, aux = outputs
    else:
      np_out = outputs

    def _vjp(dy):
      tf_dx = tape.gradient(np_out, primals, output_gradients=dy)
      return _tf_to_np(tf_dx)

  if has_aux:
    ret = (np_out, _vjp, aux)
  else:
    ret = (np_out, _vjp)
  return ret


# TODO(wangpeng): match JAX's handling of kwargs and non-ndarray args
def grad(f, has_aux=False):
  """Returns a function that computes gradient of f.

  Gradients can only be computed through numpy and tensorflow operations and not
  through python float operations and values.

  Args:
    f: a function of type (params, *args) -> scalar. 'params' can be a nested
      structure (made of lists and tuples) of ndarrays and the gradient is
      evaluated against it. `scalar` is a scalar ndarray.
    has_aux: bool, indicates whether fun returns a pair where the first element
      is considered the output of the mathematical function to be differentiated
      and the second element is auxiliary data.

  Returns:
    A gradient function of type (params, *args) -> gradients, where the result
    'gradients' has the same structure and shapes as 'params'.
  """

  def check_loss_shape(np_loss):
    if not isinstance(np_loss, tf_np.ndarray):
      raise ValueError(
          "The result of the function to take gradient must be an ndarray.")
    if not np_loss.shape.is_compatible_with([]):
      raise ValueError(
          "The result of the function to take gradient must be a scalar.")

  def _f(params, *args):
    """The gradient function to be returned."""
    with backprop.GradientTape() as g:
      g.watch(nest.flatten(params))
      outputs = f(params, *args)
      if has_aux:
        np_loss, aux = outputs
      else:
        np_loss = outputs
      check_loss_shape(np_loss)
      tf_grads = g.gradient(np_loss, params)
      if has_aux:
        res = (tf_grads, aux)
      else:
        res = tf_grads
      return _tf_to_np(res)

  return _f


def _record_result_type(recorder, f):
  """A decorator that records some information about the function.

  Args:
    recorder: a function of signature `(args, kwargs, res) -> res`.
    f: the original function.

  Returns:
    A transformed function that calls the original function and then the
    recorder afterwards.
  """
  def wrapper(*args, **kwargs):
    res = f(*args, **kwargs)
    res = recorder(args, kwargs, res)
    return res

  return wrapper


def jit(f,
        static_argnums=(),
        xla_forced_compile=False,
        input_signature=None,
        autograph=False,
        experimental_compile=False):
  """Returns a function that runs a trace-compiled version of `f`.

  A trace-compiled version of a function `f` has the same behavior as `f` (when
  called with the same "static arguments", see below), but runs faster because
  the whole computation is compiled into a computation graph once which is
  reused for subsequent executions.

  The trace compilation happens lazily, when the returned function is called for
  the first time. The compiled function may not be cached implicitly and
  multiple calls to `jit` may not share the compiled function (see below for
  "static" vs "dynamic" arguments).

  Args:
    f: a function that takes any positional arguments `args` and any keyword
      arguments `kwargs`. `ndarray`s and things accepted by
      `tf.convert_to_tensor` in `args` and `kwargs` will be treated as 'dynamic
      arguments' in the sense that calling the function with different values
      for these arguments will not cause retracing. In contrast, arguments of
      other types in `args` and `kwargs` are treated as 'static arguments' and
      calling the function with different values of them will cause
      re-compiling. Positional arguments whose positions are in `static_argnums`
      are always treated as static arguments.
    static_argnums: a tuple of positions of arguments that will be treated as
      static arguments. Note that as aforementioned, any arguments that were not
      convertible to tensor will also be static.
    xla_forced_compile: if true, it will use XLA to force-compile the graph.
      This requires that the function only contain ops that are XLA
      compatible. It will compile the entire function into a single XLA op.
    input_signature: a list of `tf.TensorSpec`, as the input signature to
      control tracing behavior. See the
      [doc](https://www.tensorflow.org/api_docs/python/tf/function]) of
        `tf.function` for details.
    autograph: whether to use autograph to convert Python constructs such as
      `if` and `while` to their TensorFlow counterparts. See the
      [doc](https://www.tensorflow.org/api_docs/python/tf/function]) of
        `tf.function` for details.
    experimental_compile: the `experimental_compile` flag for `tf.function`. See
      the [doc](https://www.tensorflow.org/api_docs/python/tf/function]) of
      `tf.function` for details. This is the recommended way to turn on XLA for
      tf.function, but unlike xla_forced_compile, it doesn't force-compile the
      entire function into a single XLA op.

  Returns:
    A trace-compiled version of f.
  """

  @polymorphic_function.function(
      input_signature=input_signature,
      autograph=autograph,
      experimental_compile=experimental_compile,
  )
  def _tf_f(*args, **kwargs):
    """Accelerated function with tensor inputs/outputs."""
    np_args = _tf_to_np(args)
    kwargs = {k: _tf_to_np(v) for k, v in kwargs.items()}
    if xla_forced_compile:
      # Use list for mutability
      output_is_list = [False]
      output_is_empty = [False]
      output_structure = [None]
      def recorder(args, kwargs, res):
        del args, kwargs
        # Workaround b/121383831
        output_is_list[0] = isinstance(res, list)
        # If outputs are empty, xla.compile returns an `Operation`, which we
        # don't want.
        if nest.flatten(res):
          output_is_empty[0] = False
          output_structure[0] = None
        else:
          output_is_empty[0] = True
          # Without deepcopy, xla.compile will change output_structure[0] to a
          # list of `Operation`.
          output_structure[0] = copy.deepcopy(res)
        return res
      f_ = _record_result_type(recorder, f)
      np_out = xla.compile(lambda: f_(*np_args, **kwargs))
      # Workaround b/121383831
      if output_is_empty[0]:
        np_out = output_structure[0]
      elif (isinstance(np_out, list) and len(np_out) == 1 and
            not output_is_list[0]):
        np_out = np_out[0]
    else:
      np_out = f(*np_args, **kwargs)
    return np_out

  def _f(*args, **kwargs):
    args = [
        _canonicalize_jit_arguments(arg) if i not in static_argnums else arg
        for i, arg in enumerate(args)
    ]
    kwargs = {k: _canonicalize_jit_arguments(v) for k, v in kwargs.items()}
    tf_out = _tf_f(*args, **kwargs)
    return _tf_to_np(tf_out)

  _f.tf_function = _tf_f

  return _f


def eval_on_shapes(f, static_argnums=(), allow_static_outputs=False):
  """Returns a function that evaluates `f` given input shapes and dtypes.

  It transforms function `f` to a function that performs the same computation as
  `f` but only on shapes and dtypes (a.k.a. shape inference).

  Args:
    f: the function to be transformed.
    static_argnums: see documentation of `jit`.
    allow_static_outputs: whether to allow non-array outputs. If True, non-array
      outputs (e.g. Python integers) will be returned as-is; otherwise, they
      will be converted to ndarrays, and then specs of those ndarrays will be
      returned.

  Returns:
    A function whose input arguments can be either the same as `f`'s or only
    their shapes/dtypes represented by `tf.TensorSpec`, and whose return values
    are `tf.TensorSpec`s with the same nested structure as `f`'s return
    values. If `allow_static_outputs` is True, when `f` returns some non-array
    outputs (e.g. Python integers), the converted function will return them
    as-is instead of returning `tf.TensorSpec`s for them.
  """
  def abstractify(args):
    def _abstractify(x):
      x = _canonicalize_jit_arg(x)
      if isinstance(x, (tensor_lib.Tensor, tf_np.ndarray)):
        return tensor_lib.TensorSpec(x.shape, x.dtype)
      else:
        return x
    new_args = []
    for i, arg in enumerate(args):
      if i in static_argnums:
        new_args.append(arg)
      else:
        new_args.append(nest.map_structure(_abstractify, arg))
    return new_args

  if allow_static_outputs:
    # When `tf_f` below is called (via get_concrete_function) with the same
    # arugments (after abstraction), the Python function `f` won't be run, so we
    # need this python_outputs_map to retrieve the Python outputs we've seen
    # before that correspond the arguments.
    python_outputs_map = {}
    def recorder(args, kwargs, res):
      # Since the get_concrete_function below only uses positional args, we also
      # only positional args here.
      del args, kwargs
      def is_tensor_like(x):
        if hasattr(x, "_type_spec"):
          return True  # x is a CompositeTensor
        return isinstance(x, (tf_np.ndarray, tensor_lib.Tensor))
      py_values = nest.map_structure(
          lambda x: None if is_tensor_like(x) else x, res
      )
      key = id(ops.get_default_graph())
      python_outputs_map[key] = py_values
      # Set non-tensor outputs to None to avoid tf.function calling
      # tf.convert_to_tensor on them.
      res = nest.map_structure(
          lambda x: None if not is_tensor_like(x) else x, res
      )
      return res
    f = _record_result_type(recorder, f)

  # TODO(wangpeng): tf.function could add a knob to turn off materializing the
  #   graph, so that we don't waste computation and memory when we just want
  #   shape inference.
  tf_f = jit(f, static_argnums=static_argnums).tf_function

  # pylint: disable=missing-docstring
  def f_return(*args):
    def to_tensor_spec(x):
      if isinstance(x, tensor_lib.Tensor):
        return tensor_lib.TensorSpec(x.shape, x.dtype)
      else:
        return x

    new_args = abstractify(args)
    cfun = tf_f.get_concrete_function(*new_args)
    res = cfun.structured_outputs
    res = nest.map_structure(to_tensor_spec, res)

    if allow_static_outputs:
      key = id(cfun.graph)
      py_values = python_outputs_map[key]
      # We can also call tf.get_static_value on structured_outputs to retrieve
      # the Python values, but since we'll need to use python_outputs_map to
      # record "which outputs are static?" anyway, we choose to directly store
      # the Python values in python_outputs_map.
      res = nest.map_structure(
          lambda x, python_value: x if python_value is None else python_value,
          res,
          py_values,
      )

    return res

  # Provides access to `tf_f` for testing purpose.
  f_return._tf_function = tf_f  # pylint: disable=protected-access
  return f_return


def _index_update_helper(updater, x, idx, y):
  x = tf_np.asarray(x)
  y = tf_np.asarray(y)
  # TODO(b/164251540): Remove this expensive manual broadcasting once
  #   tf.raw_ops.tensor_strided_slice_update and tf.tensor_scatter_nd_update
  #   support broadcasting.
  y = array_ops.broadcast_to(y, array_ops.shape_v2(x[idx]))
  return updater(x, idx, y)


# pylint: disable=protected-access
def index_update(x, idx, y):
  """Pure equivalent of `x[idx] = y`.

  Returns the value of x that would result from the NumPy-style indexed
  assignment `x[idx] = y`. Because it's a pure function, `x` itself won't be
  changed.

  Args:
    x: an array with the values to be updated.
    idx: a Numpy-style index, consisting of `None`, integers, slice objects,
      ellipses, ndarrays with integer dtypes, or a tuple of the above.
    y: the array of updates. `y` must be broadcastable to the shape of the array
      that would be returned by `x[idx]`.

  Returns:
    The updated version of `x`.
  """
  return _index_update_helper(tf_np.ndarray._with_index_update, x, idx, y)


def index_add(x, idx, y):
  """Pure equivalent of `x[idx] += y`.

  Returns the value of x that would result from the NumPy-style indexed
  assignment `x[idx] += y`. Because it's a pure function, `x` itself won't be
  changed.

  Args:
    x: an array with the values to be updated.
    idx: a Numpy-style index, consisting of `None`, integers, slice objects,
      ellipses, ndarrays with integer dtypes, or a tuple of the above.
    y: the array of updates. `y` must be broadcastable to the shape of the array
      that would be returned by `x[idx]`.

  Returns:
    The updated version of `x`.
  """
  return _index_update_helper(tf_np.ndarray._with_index_add, x, idx, y)


def index_min(x, idx, y):
  """Pure equivalent of `x[idx] = minimum(x[idx], y)`.

  Returns the value of x that would result from the NumPy-style indexed
  assignment `x[idx] = minimum(x[idx], y)`. Because it's a pure function, `x`
  itself won't be changed.

  Args:
    x: an array with the values to be updated.
    idx: a Numpy-style index, consisting of `None`, integers, slice objects,
      ellipses, ndarrays with integer dtypes, or a tuple of the above.
    y: the array of updates. `y` must be broadcastable to the shape of the array
      that would be returned by `x[idx]`.

  Returns:
    The updated version of `x`.
  """
  return _index_update_helper(tf_np.ndarray._with_index_min, x, idx, y)


def index_max(x, idx, y):
  """Pure equivalent of `x[idx] = maximum(x[idx], y)`.

  Returns the value of x that would result from the NumPy-style indexed
  assignment `x[idx] = maximum(x[idx], y)`. Because it's a pure function, `x`
  itself won't be changed.

  Args:
    x: an array with the values to be updated.
    idx: a Numpy-style index, consisting of `None`, integers, slice objects,
      ellipses, ndarrays with integer dtypes, or a tuple of the above.
    y: the array of updates. `y` must be broadcastable to the shape of the array
      that would be returned by `x[idx]`.

  Returns:
    The updated version of `x`.
  """
  return _index_update_helper(tf_np.ndarray._with_index_max, x, idx, y)
# pylint: enable=protected-access


def logsumexp(x, axis=None, keepdims=None):
  """Computes log(sum(exp(elements across dimensions of a tensor))).

  Reduces `x` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.
  If `axis` has no entries, all dimensions are reduced, and a
  tensor with a single element is returned.
  This function is more numerically stable than log(sum(exp(input))). It avoids
  overflows caused by taking the exp of large inputs and underflows caused by
  taking the log of small inputs.

  Args:
    x: The tensor to reduce. Should have numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(x), rank(x))`.
    keepdims: If true, retains reduced dimensions with length 1.

  Returns:
    The reduced tensor.
  """
  return tf_np.asarray(
      math_ops.reduce_logsumexp(input_tensor=x, axis=axis, keepdims=keepdims)
  )


def expit(x):
  """Compute 1 / (1 + exp(-x))."""
  return tf_np.asarray(math_ops.sigmoid(x))


def erf(x):
  """Computes the Gauss error function of x element-wise."""
  return tf_np.asarray(math_ops.erf(x))


def _minus(a, b):
  return [x for x in a if x not in b]


def _compose_output_rep(lhs_rep, rhs_rep, lhs_contraction, rhs_contraction,
                        lhs_batch, rhs_batch):
  """Compose the output string representation.

  e.g., ij, jk, (((1,), (0,)), ((), ())) -> ik
        aij, ajk, (((2,), (1,)), ((0,), (0,))) -> aik

  Args:
    lhs_rep: A string representation for the left-hand side input array
    rhs_rep: A string representation for the right-hand side input array
    lhs_contraction: Sequence[int] (the contraction dimensions of lhs)
    rhs_contraction: Sequence[int] (the contraction dimensions of rhs)
    lhs_batch: Sequence[int] (the batch dimensions of lhs)
    rhs_batch: Sequence[int] (the batch dimensions of rhs)

  Returns:
    A string representation of the result array.
  """
  output_rep = []
  for dim in lhs_batch:
    output_rep.append(lhs_rep[dim])

  for i in _minus(range(len(lhs_rep)), lhs_batch + lhs_contraction):
    output_rep.append(lhs_rep[i])
  for i in _minus(range(len(rhs_rep)), rhs_batch + rhs_contraction):
    output_rep.append(rhs_rep[i])
  return "".join(output_rep)


def _non_batched_matmul(lhs, rhs, lhs_contraction, rhs_contraction):
  """Compute the non-batched matrix multiplication.

  If it is the general non-batched/single-batched matrix multiplication,
  use the highly optimized kernel `tf.tensordot` to handle it.

  Args:
    lhs: an array (the left-hand side matrix/vector to be multiplied)
    rhs: an array (the right-hand side matrix/vector to be multiplied)
    lhs_contraction: Sequence[int] (the contraction dimensions of lhs)
    rhs_contraction: Sequence[int] (the contraction dimensions of rhs)

  Returns:
    An array that contains the result.
  """
  return math_ops.tensordot(
      lhs, rhs, axes=(list(lhs_contraction), list(rhs_contraction))
  )


def tf_dot_general(lhs, rhs, dimension_numbers):
  """The general dot operation for TensorFlow.

  An equivalent general dot operation as that in JAX -
     <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.dot_general.html>
  Although there is an implementation in TF XLA, avoid directly using XLA when
  possible.

  e.g., non-batched: ij,jk->ik
        batched: ijk,ikl->ijl

  Args:
    lhs: an array (the left-hand side matrix/vector to be multiplied)
    rhs: an array (the right-hand side matrix/vector to be multiplied)
    dimension_numbers: (Tuple[Tuple[Sequence[int], Sequence[int]],
      Tuple[Sequence[int], Sequence[int]]]) – a tuple of tuples of the form
      ((lhs_contracting_dims, rhs_contracting_dims), (lhs_batch_dims,
      rhs_batch_dims))

  Returns:
    An array that contains the result.
  """
  char_list = list(string.ascii_lowercase)
  char_list = char_list[8:] + char_list[:8]
  lhs_rank, rhs_rank = len(lhs.shape), len(rhs.shape)
  lhs_rep = char_list[:lhs_rank]
  rhs_rep = char_list[lhs_rank:lhs_rank + rhs_rank]
  contraction, batch = dimension_numbers
  lhs_contraction, rhs_contraction = contraction
  if len(lhs_contraction) != len(rhs_contraction):
    raise ValueError(
        "The input matrices are required to have the same number "
        "of contraction dimensions, but got: lhs {}, rhs: {}".format(
            len(lhs_contraction), len(rhs_contraction)))
  lhs_batch, rhs_batch = batch
  if len(lhs_batch) != len(rhs_batch):
    raise ValueError("The input matrices are required to have the same number "
                     "of batch dimensions, but got: lhs {}, rhs: {}".format(
                         len(lhs_batch), len(rhs_batch)))

  if not lhs_batch and not rhs_batch:
    return _non_batched_matmul(lhs, rhs, lhs_contraction, rhs_contraction)

  if (lhs_rank == rhs_rank == 3 and lhs_batch == (0,) and rhs_batch == (0,) and
      lhs_contraction == (2,) and rhs_contraction == (1,)):
    return math_ops.matmul(lhs, rhs)

  for i in range(len(lhs_contraction)):
    rhs_rep[rhs_contraction[i]] = lhs_rep[lhs_contraction[i]]
  for i in range(len(lhs_batch)):
    rhs_rep[rhs_batch[i]] = lhs_rep[lhs_batch[i]]

  output_rep = _compose_output_rep(lhs_rep, rhs_rep, lhs_contraction,
                                   rhs_contraction, lhs_batch, rhs_batch)
  equation = "".join(lhs_rep) + "," + "".join(rhs_rep) + "->" + output_rep
  return special_math_ops.einsum(equation, lhs, rhs)


def _conv_general_param_type_converter(window_strides, lhs_dilation,
                                       rhs_dilation, dim):
  """Convert strides, lhs_dilation, rhs_dilation to match TF convention.

  For example,
   in the 3D case, if lhs_dilation = 2, then convert it to [2, 2, 2]
                   if lhs_dilation = (2, 2, 2), convert it also to [2, 2, 2]

  Args:
    window_strides: window_strides to be converted
    lhs_dilation: lhs_dilation to be converted
    rhs_dilation: rhs_dilation to be converted
    dim: dim to be converted

  Returns:
    The updated window_strides, lhs_dilation and rhs_dilation
  """
  def _as_list_of_size(item, size):
    if item is None:
      return None
    return [item] * size if isinstance(item, int) else list(item)
  return (_as_list_of_size(window_strides, dim),
          _as_list_of_size(lhs_dilation, dim),
          _as_list_of_size(rhs_dilation, dim))


# pylint: disable=g-bad-todo
# TODO(DarrenZhang01): Expand the test cases of general convolution and revise
# the according bugs.
# TODO(DarrenZhang01): Support feature_group_count, batch_group_count and
# precision, and allow lhs_dilation and rhs_dilation to happen at the same time.
# pylint: enable=g-bad-todo
def tf_conv_general_dilated(lhs, rhs, window_strides, padding, output_shape,
                            lhs_dilation=None, rhs_dilation=None,
                            dimension_numbers=None, feature_group_count=1,
                            batch_group_count=1, precision=None):
  """A general conv API for TensorFlow.

  According JAX version:
    https://jax.readthedocs.io/en/stable/_autosummary/jax.lax.conv_general_dilated.html

  Args:
    lhs: a rank n+2 dimensional input array.
    rhs: a rank n+2 dimensional array of kernel weights.
    window_strides: a sequence of n integers, representing the inter-window
                    strides.
    padding: either the string ‘SAME’, the string ‘VALID’, or a sequence of n
             (low, high) integer pairs that give the padding to apply before and
             after each spatial dimension.
    output_shape: the output shape of the convolution (only required for
                  transpose convolution).
    lhs_dilation: None, or a sequence of n integers, giving the dilation factor
                  to apply in each spatial dimension of lhs. LHS dilation is
                  also known as transposed convolution.
    rhs_dilation: None, or a sequence of n integers, giving the dilation factor
                  to apply in each spatial dimension of rhs. RHS dilation is
                  also known as atrous convolution.
    dimension_numbers: either None, a ConvDimensionNumbers object, or a 3-tuple
                       (lhs_spec, rhs_spec, out_spec), where each element is a
                       string of length n+2.
    feature_group_count:  integer, default 1. Changing this is currently not
                          supported.
    batch_group_count: integer, default 1. Changing this is currently not
                       supported.
    precision: Optional. Either None, which means the default precision for the
               backend, or a Precision enum value.

  Returns:
    A TF NumPy array that contains the convolution result.
  """
  dim = None
  lhs_spec, rhs_spec, out_spec = dimension_numbers
  if lhs_spec != out_spec:
    raise ValueError("Current implementation requires the `data_format` of the "
                     "inputs and outputs to be the same.")
  if len(lhs_spec) >= 6:
    raise ValueError("Current implmentation does not support 4 or higher"
                     "dimensional convolution, but got: ", len(lhs_spec) - 2)
  dim = len(lhs_spec) - 2
  if lhs_dilation and rhs_dilation:
    if lhs_dilation == (1,) * dim and rhs_dilation == (1,) * dim:
      lhs_dilation, rhs_dilation = None, None
    else:
      raise ValueError("Current implementation does not support that "
                       "deconvolution and dilation to be performed at the same "
                       "time, but got lhs_dilation: {}, rhs_dilation: {}"
                       .format(lhs_dilation, rhs_dilation))
  if padding not in ["SAME", "VALID"]:
    raise ValueError("Current implementation requires the padding parameter"
                     "to be either 'VALID' or 'SAME', but got: ", padding)
  if batch_group_count != 1 or feature_group_count != 1:
    raise NotImplementedError("batch_group_count and feature_group_count "
                              "other than 1 is currently not supported, but"
                              " got feature_group_count: {}, batch_group_count"
                              ": {}".format(feature_group_count,
                                            batch_group_count))
  if precision is not None:
    raise NotImplementedError("precision other than `None` is currently not "
                              "supported, but got: {}".format(precision))
  # Convert params from int/Sequence[int] to list of ints.
  strides, lhs_dilation, rhs_dilation = _conv_general_param_type_converter(
      window_strides, lhs_dilation, rhs_dilation, dim
  )
  # Preprocess the shapes
  dim_maps = {}
  if isinstance(lhs_spec, str):
    dim_maps["I"] = list(rhs_spec).index("I")
    dim_maps["O"] = list(rhs_spec).index("O")
    dim_maps["N"] = list(lhs_spec).index("N")
    dim_maps["C"] = list(lhs_spec).index("C")
  else:
    dim_maps["I"] = rhs_spec[1]
    dim_maps["O"] = rhs_spec[0]
    dim_maps["N"] = lhs_spec[0]
    dim_maps["C"] = lhs_spec[1]

  lhs = tf_np.moveaxis(lhs, (dim_maps["N"], dim_maps["C"]), (0, dim + 1))
  # Adjust the filters, put the dimension 'I' and 'O' at last.
  rhs = tf_np.moveaxis(rhs, (dim_maps["O"], dim_maps["I"]), (dim + 1, dim))
  spatial_dim_maps = {1: "W", 2: "HW", 3: "DHW"}
  data_format = "N" + spatial_dim_maps[dim] + "C"

  if rhs_dilation or (lhs_dilation is None and rhs_dilation is None):
    output = _tf_nn_APIs[dim][0](lhs, rhs, strides, padding, data_format,
                                 rhs_dilation)
  else:
    output = _tf_nn_APIs[dim][1](
        lhs,
        rhs,
        constant_op.constant(output_shape),
        strides,
        padding,
        data_format,
        lhs_dilation,
    )
  output = tf_np.moveaxis(output, (0, dim + 1), (dim_maps["N"], dim_maps["C"]))
  return output


def conv(inp,
         fltr,
         window_strides,
         padding,
         dimension_numbers,
         filter_dilation=None):
  """Convolution over an N-D array.

  See https://www.tensorflow.org/api_docs/python/tf/nn/convolution and
  https://www.tensorflow.org/xla/operation_semantics#conv_convolution for
  reference.

  Args:
    inp: an (N+2)-D array. The input of the convolution.
    fltr: an (N+2)-D array. The filter (i.e. kernel) of the convolution.
    window_strides: a sequence of N ints, the strides for moving the convolution
      window.
    padding: a string, either "VALID" or "SAME". The padding algorithm.
    dimension_numbers: a tuple of three strings encoding the data format of
      input, filter and output. "I" means input; "O" means output; "C" means
      channel; other characters such as "W", "H" and "D" means spatial
      dimensions.
    filter_dilation: the dilation rates for the filter. Dilating the filter
      means adding "holes" to the filter.

  Returns:
    An (N+2)-D array. The convolution result.
  """
  input_spec, filter_spec, output_spec = dimension_numbers
  if input_spec != output_spec:
    raise ValueError("Input and output data formats must be the same; got %s "
                     "and %s" % (input_spec, output_spec))
  supported_filter_spec = ["WIO", "HWIO", "DHWIO"]
  if filter_spec not in supported_filter_spec:
    raise ValueError("The supported data format for the filter are %s; got %s" %
                     (supported_filter_spec, filter_spec))
  if input_spec[1:-1] != filter_spec[:-2]:
    raise ValueError("Input data format (%s) is not compatible with filter "
                     "data format (%s)" % (input_spec, filter_spec))
  # No type promotion in order to prevent accidentally doing more expensive
  # computation.
  dtype = tf_np.result_type(inp, fltr)
  inp = tf_np.asarray(inp, dtype)
  fltr = tf_np.asarray(fltr, dtype)
  return tf_np.asarray(
      nn_ops.convolution_v2(
          input=inp,
          filters=fltr,
          padding=padding,
          strides=window_strides,
          dilations=filter_dilation,
          data_format=input_spec,
      )
  )


def avg_pool(x, pool_size, strides, padding):
  """Performs an N-D average pooling.

  Args:
    x: ndarray of rank N+2, of shape `[batch_size] + input_spatial_shape +
      [num_channels]`. Pooling happens over the spatial dimensions only.
    pool_size: sequence of N ints.
    strides: sequence of N ints.
    padding: a string, the padding algorithm. Must be "SAME" or "VALID".

  Returns:
    An (N+2)-D array,  of shape
      [batch_size] + output_spatial_shape + [num_channels],
    where `output_spatial_shape` depends on the value of padding:
    If padding = "SAME":
      output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides[i])
    If padding = "VALID":
      output_spatial_shape[i] =
        ceil((input_spatial_shape[i] - (pool_size[i] - 1)) / strides[i]).
  """
  x = tf_np.asarray(x)
  return tf_np.asarray(
      nn_ops.pool(
          input=x,
          window_shape=pool_size,
          pooling_type="AVG",
          strides=strides,
          padding=padding,
      )
  )


def max_pool(x, pool_size, strides, padding):
  """Performs an N-D max pooling.

  Args:
    x: ndarray of rank N+2, of shape `[batch_size] + input_spatial_shape +
      [num_channels]`. Pooling happens over the spatial dimensions only.
    pool_size: sequence of N ints.
    strides: sequence of N ints.
    padding: a string, the padding algorithm. Must be "SAME" or "VALID".

  Returns:
    An (N+2)-D array,  of shape
      [batch_size] + output_spatial_shape + [num_channels],
    where `output_spatial_shape` depends on the value of padding:
    If padding = "SAME":
      output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides[i])
    If padding = "VALID":
      output_spatial_shape[i] =
        ceil((input_spatial_shape[i] - (pool_size[i] - 1)) / strides[i]).
  """
  x = tf_np.asarray(x)
  return tf_np.asarray(
      nn_ops.pool(
          input=x,
          window_shape=pool_size,
          pooling_type="MAX",
          strides=strides,
          padding=padding,
      )
  )


def sort_key_val(keys, values, dimension=-1):
  """Sorts keys along a dimension and applies same permutation to values.

  Args:
    keys: an array. The dtype must be comparable numbers (integers and reals).
    values: an array, with the same shape of `keys`.
    dimension: an `int`. The dimension along which to sort.

  Returns:
    Permuted keys and values.
  """
  keys = tf_np.asarray(keys)
  values = tf_np.asarray(values)
  rank = keys.shape.ndims
  if rank is None:
    rank = values.shape.ndims
  if rank is None:
    # We need to know the rank because tf.gather requires batch_dims to be `int`
    raise ValueError("The rank of either keys or values must be known, but "
                     "both are unknown (i.e. their shapes are both None).")
  if dimension in (-1, rank - 1):

    def maybe_swapaxes(a):
      return a
  else:

    def maybe_swapaxes(a):
      return tf_np.swapaxes(a, dimension, -1)

  # We need to swap axes because tf.gather (and tf.gather_nd) supports
  # batch_dims on the left but not on the right.
  # TODO(wangpeng): Investigate whether we should do swapaxes or moveaxis.
  keys = maybe_swapaxes(keys)
  values = maybe_swapaxes(values)
  idxs = tf_np.argsort(keys)

  # Using tf.gather rather than np.take because the former supports batch_dims
  def gather(a):
    return tf_np.asarray(array_ops.gather_v2(a, idxs, batch_dims=rank - 1))

  keys = gather(keys)
  values = gather(values)
  keys = maybe_swapaxes(keys)
  values = maybe_swapaxes(values)
  return keys, values


def scan(f, init, xs, length=None, reverse=False):
  """Scan a function over leading array axes while carrying along state.

  See the docstring of `jax.lax.scan`
  (https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html) for
  details.

  Args:
    f: a Python function to be scanned of type ``c -> a -> (c, b)``, meaning
      that ``f`` accepts two arguments where the first is a value of the loop
      carry and the second is a slice of ``xs`` along its leading axis, and that
      ``f`` returns a pair where the first element represents a new value for
      the loop carry and the second represents a slice of the output. Note that
      the input and output carry must have the same dtype.
    init: an initial loop carry value of type ``c``, which can be a scalar,
      array, or any pytree (nested Python tuple/list/dict) thereof, representing
      the initial loop carry value. This value must have the same structure as
      the first element of the pair returned by ``f``.
    xs: the value of type ``[a]`` over which to scan along the leading axis,
      where ``[a]`` can be an array or any pytree (nested Python
      tuple/list/dict) thereof with consistent leading axis sizes.
    length: optional integer specifying the number of loop iterations, which
      must agree with the sizes of leading axes of the arrays in ``xs`` (but can
      be used to perform scans where no input ``xs`` are needed).
    reverse: optional boolean specifying whether to run the scan iteration
      forward (the default) or in reverse, equivalent to reversing the leading
      axes of the arrays in both ``xs`` and in ``ys``.

  Returns:
    A pair of type ``(c, [b])`` where the first element represents the final
    loop carry value and the second element represents the stacked outputs of
    the second output of ``f`` when scanned over the leading axis of the inputs.
  """
  init, xs = nest.map_structure(
      lambda x: tf_np.asarray(x) if x is not None else None, (init, xs)
  )
  if length is not None:
    length = int(length)
  def get_length(x):
    if x is None:
      return None
    if x.shape.rank == 0:
      raise ValueError("Some array in `xs` doesn't have a leading dimension")
    return x.shape[0]
  lengths = nest.flatten(nest.map_structure(get_length, xs))
  for l in lengths:
    if l is not None:
      if length is None:
        length = l
      elif length != l:
        raise ValueError("There are two different leading-dimension lengths: "
                         f"{length} and {l}")
  if length is None:
    raise ValueError(
        "Can't determine length. Please set the `length` argument.")
  xs_ta = nest.map_structure(
      lambda t: (  # pylint: disable=g-long-lambda
          tensor_array_ops.TensorArray(  # pylint: disable=g-long-ternary
              t.dtype, size=length, dynamic_size=False
          ).unstack(  # pylint: disable=g-long-lambda
              t
          )
          if t is not None
          else None
      ),
      xs,
  )
  # tf.while_loop doesn't allow None in loop_vars, so we mask them.
  is_init_none = nest.map_structure(lambda x: x is None, init)
  def to_safe(carry):
    return nest.map_structure(
        lambda x, is_none: array_ops.zeros([]) if is_none else x,
        carry,
        is_init_none,
    )
  def from_safe(safe_carry):
    return nest.map_structure(
        lambda x, is_none: None if is_none else x, safe_carry, is_init_none
    )
  def body(i, safe_carry, ys_ta):
    carry = from_safe(safe_carry)
    if reverse:
      i_ = length - 1 - i
    else:
      i_ = i
    xs = nest.map_structure(
        lambda x_ta: x_ta.read(i_) if x_ta is not None else None, xs_ta
    )
    carry, ys = f(*_tf_to_np((carry, xs)))
    ys_ta = nest.map_structure(
        lambda y_ta, y: (y_ta.write(i_, y) if y is not None else y_ta),
        ys_ta,
        ys,
    )
    i = i + 1
    safe_carry = to_safe(carry)
    return i, safe_carry, ys_ta
  xs_spec = nest.map_structure(
      lambda t: tensor_lib.TensorSpec(t.shape[1:], t.dtype)  # pylint: disable=g-long-lambda
      if t is not None
      else None,
      xs,
  )
  _, ys_spec = eval_on_shapes(f)(init, xs_spec)
  # ys_ta can't contain None because tf.while_loop doesn't allow None in
  # loop_vars.
  ys_ta = nest.map_structure(
      lambda y: tensor_array_ops.TensorArray(  # pylint: disable=g-long-lambda
          y.dtype if y is not None else dtypes.float32,
          size=length,
          dynamic_size=False,
      ),
      ys_spec,
  )
  safe_init = to_safe(init)
  _, safe_carry, ys_ta = while_loop.while_loop_v2(
      lambda i, *_: i < length,
      body,
      (0, safe_init, ys_ta),
      maximum_iterations=length,
  )
  carry = from_safe(safe_carry)
  def _stack(a, spec):
    if spec is None:
      return None
    a = a.stack()
    a.set_shape((length,) + a.shape[1:])
    return a
  ys = nest.map_structure(_stack, ys_ta, ys_spec)
  return _tf_to_np((carry, ys))


# named "tf_map" instead of "map" as in JAX to avoid conflict with Python `map`
def tf_map(f, xs):
  """Map a function over leading array axes.

  See the docstring of `jax.lax.map`
  (https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.map.html) for
  details.

  Args:
    f: a Python function to apply element-wise over the first axis or axes of
      `xs`.
    xs: values over which to map along the leading axis.

  Returns:
    Mapped values.
  """
  def g(unused, x):
    return unused, f(x)
  carry = nest.map_structure(lambda _: None, xs)
  return scan(g, carry, xs)[1]


def _get_dynamic_indices(operand, start_indices, slice_sizes):
  """Calcuates the indices for `tf.gather_nd` from slices.

  Args:
    operand: a Tensor to slice.
    start_indices: a vector Tensor of integers, one per dimension. The starts of
      the slice. The vector can be dynamic.
    slice_sizes: a list of integers, one per dimension. The sizes of the slice.

  Returns:
    An index array suitable for `tf.gather_nd` and `tf.scatter_nd`, or `None` if
    `operand` is a scalar.
  """
  rank = len(slice_sizes)
  operand_rank = array_ops.rank(operand)
  control_flow_assert.Assert(operand_rank == rank, [operand_rank, rank])
  starts_rank = array_ops.rank(start_indices)
  control_flow_assert.Assert(starts_rank == 1, [starts_rank])
  num_starts = array_ops.shape_v2(start_indices)[0]
  control_flow_assert.Assert(num_starts == rank, [num_starts, rank])
  operand_shape = array_ops.shape_v2(operand)
  control_flow_assert.Assert(
      math_ops.reduce_all(slice_sizes <= operand_shape),
      [slice_sizes, operand_shape],
  )
  if rank == 0:
    return None
  start_indices = array_ops.where(
      start_indices < 0, start_indices + operand_shape, start_indices
  )
  idx_list = []
  for i in range(rank):
    start = start_indices[i]
    size = slice_sizes[i]
    dim = operand_shape[i]
    start = clip_ops.clip_by_value(start, 0, dim - size)
    # XLA requires tf.range's `start` to be compile-time constant, so we can't
    # do tf.range(start, ...).
    idx = start + math_ops.range(size)
    shape = [1] * rank
    shape[i] = size
    idx = array_ops.reshape(idx, shape)
    idx_list.append(idx)
  slice_sizes_tensor = ops.convert_to_tensor(slice_sizes)
  # tf.stack doesn't support broadcasting, so we need to broadcast manually.
  # TODO(wangpeng): Reduce peak memory by broadcasting one-by-one instead of
  #   all-together.
  idx_list = [array_ops.broadcast_to(x, slice_sizes_tensor) for x in idx_list]
  return array_ops_stack.stack(idx_list, axis=-1)


def dynamic_slice(operand, start_indices, slice_sizes):
  """Slicing operation where the indices can be dynamic vlaues.

  See the docstring of `jax.lax.dynamic_slice`
  (https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.dynamic_slice.html)
  for details.

  Args:
    operand: an array to slice.
    start_indices: a vector of integers, one per dimension. The starts of the
      slice. The vector can be dynamic.
    slice_sizes: a list of integers, one per dimension. The sizes of the slice.

  Returns:
    An array containing the slice, with shape equal to `slice_sizes`.
  """
  # This implementation uses tf.gather_nd to implement dynamic_slice, which is
  # memory inefficient because the size of `indices` given to gather_nd is
  # large.
  operand = tf_np.asarray(operand).data
  start_indices = tf_np.asarray(start_indices, np.int32).data
  idx = _get_dynamic_indices(operand, start_indices, slice_sizes)
  if idx is not None:
    operand = array_ops.gather_nd(operand, idx)
  return tf_np.asarray(operand)


def dynamic_update_slice(operand, update, start_indices):
  """Updates a dynamic slice.

  See the docstring of `jax.lax.dynamic_update_slice`
  (https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.dynamic_update_slice.html)
  for details.

  Args:
    operand: an array to slice.
    update: an array containing the new values to write onto `operand`.
    start_indices: a vector of integers, one per dimension. The starts of the
      slice. The vector can be dynamic.

  Returns:
    The updated version of `operand`.
  """
  operand = tf_np.asarray(operand).data
  update = tf_np.asarray(update).data
  start_indices = tf_np.asarray(start_indices, np.int32).data
  if not update.shape.is_fully_defined():
    raise ValueError("update's shape must be fully defined")
  slice_sizes = update.shape
  idx = _get_dynamic_indices(operand, start_indices, slice_sizes)
  if idx is None:
    # `np.zeros([])[()] = 1.0` will result in a scalar array of 1.0
    return tf_np.asarray(update)
  operand = array_ops.tensor_scatter_nd_update(operand, idx, update)
  return tf_np.asarray(operand)


def dynamic_slice_in_dim(operand, start_index, slice_size, axis=0):
  """Convenience wrapper around dynamic_slice applying to one dimension."""
  operand = tf_np.asarray(operand)
  start_indices = [0] * operand.ndim
  slice_sizes = list(operand.shape)
  axis = int(axis)
  start_indices[axis] = start_index
  slice_sizes[axis] = int(slice_size)
  return dynamic_slice(operand, start_indices, slice_sizes)


def dynamic_update_slice_in_dim(operand, update, start_index, axis):
  """Convenience wrapper around dynamic_update_slice for one dimension."""
  operand = tf_np.asarray(operand)
  axis = int(axis)
  start_indices = [0] * operand.ndim
  start_indices[axis] = start_index
  return dynamic_update_slice(operand, update, start_indices)


# Use int64 instead of int32 to avoid TF's "int32 problem"
_RNG_KEY_DTYPE = np.int64


def _key2seed(a):
  """Converts an RNG key to an RNG seed.

  Args:
    a: an RNG key, an ndarray of shape [] and dtype `np.int64`.

  Returns:
    an RNG seed, a tensor of shape [2] and dtype `tf.int32`.
  """

  def int64_to_int32s(a):
    """Converts an int64 tensor of shape [] to an int32 tensor of shape [2]."""
    a = math_ops.cast(a, dtypes.uint64)
    fst = math_ops.cast(a, dtypes.uint32)
    snd = math_ops.cast(
        gen_bitwise_ops.right_shift(a, constant_op.constant(32, dtypes.uint64)),
        dtypes.uint32,
    )
    a = [fst, snd]
    a = nest.map_structure(lambda x: math_ops.cast(x, dtypes.int32), a)
    a = array_ops_stack.stack(a)
    return a

  return int64_to_int32s(a)


def _seed2key(a):
  """Converts an RNG seed to an RNG key.

  Args:
    a: an RNG seed, a tensor of shape [2] and dtype `tf.int32`.

  Returns:
    an RNG key, an ndarray of shape [] and dtype `np.int64`.
  """

  def int32s_to_int64(a):
    """Converts an int32 tensor of shape [2] to an int64 tensor of shape []."""
    a = math_ops.bitwise_or(
        math_ops.cast(a[0], dtypes.uint64),
        math_ops.left_shift(
            math_ops.cast(a[1], dtypes.uint64),
            constant_op.constant(32, dtypes.uint64),
        ),
    )
    a = math_ops.cast(a, dtypes.int64)
    return a

  return tf_np.asarray(int32s_to_int64(a))


def prng(s):
  """Creates RNG state from seed.

  Args:
    s: the seed, an integer.

  Returns:
    An RNG state, as a scalar array of dtype `np.int64`.
  """
  # TODO(wangpeng): Become bitwise-identical to JAX when TF stateless RNGs get
  #   improved.
  return tf_np.asarray(s, dtype=_RNG_KEY_DTYPE)


def stateless_split(seed, num=2):
  """Splits an RNG seed into `num` new seeds by adding a leading axis.

  Example:

  >>> seed = [1, 2]
  >>> new_seeds = tf.random.experimental.stateless_split(seed, num=3)
  >>> print(new_seeds)
  tf.Tensor(
  [[1105988140 1738052849]
   [-335576002  370444179]
   [  10670227 -246211131]], shape=(3, 2), dtype=int32)
  >>> tf.random.stateless_normal(shape=[3], seed=new_seeds[0, :])
  <tf.Tensor: shape=(3,), dtype=float32, numpy=array([-0.59835213, -0.9578608 ,
  0.9002807 ], dtype=float32)>

  Args:
    seed: an RNG seed (a tensor with shape [2] and dtype `int32` or `int64`).
      (When using XLA, only `int32` is allowed.)
    num: optional, a positive integer or scalar tensor indicating the number of
      seeds to produce (default 2).

  Returns:
    A tensor with shape [num, 2] representing `num` new seeds. It will have the
    same dtype as `seed` (if `seed` doesn't have an explicit dtype, the dtype
    will be determined by `tf.convert_to_tensor`).
  """
  seed = ops.convert_to_tensor(seed)
  return stateless_random_ops.stateless_random_uniform(
      shape=[num, 2], seed=seed, dtype=seed.dtype, minval=None, maxval=None
  )


def split(state, num):
  """Creates new independent RNG states from an existing state.

  Args:
    state: the existing state.
    num: the number of the new states.

  Returns:
    A tuple of new states.
  """
  state = tf_np.asarray(state, dtype=_RNG_KEY_DTYPE)
  state = _key2seed(state)
  try:
    states = stateless_random_ops.stateless_split(state, num)
  except AttributeError as e:  # pylint: disable=unused-variable
    # TODO(afrozm): For TF < 2.3 we need to do this. Delete once 2.3 launches.
    states = stateless_split(state, num)
  states = array_ops_stack.unstack(states, num)
  states = nest.map_structure(_seed2key, states)
  return states


def uniform(key,
            shape,
            dtype=tf_np.random.DEFAULT_RANDN_DTYPE,
            minval=0.,
            maxval=1.):
  """Sample uniform random values in range [`minval`, `maxval`).

  Args:
    key: the RNG key.
    shape: the shape of the result.
    dtype: the dtype of the result.
    minval: the minimal value (inclusive).
    maxval: the maximal value (exclusive).

  Returns:
    An ndarray with shape `shape` and dtype `dtype`. Each value in the ndarray
    is sampled uniformly randomly in range [`minval`, `maxval`).
  """
  minval = math_ops.cast(minval, dtype)
  maxval = math_ops.cast(maxval, dtype)
  key = tf_np.asarray(key, dtype=_RNG_KEY_DTYPE)
  return tf_np.asarray(
      stateless_random_ops.stateless_random_uniform(
          shape, seed=_key2seed(key), dtype=dtype, minval=minval, maxval=maxval
      )
  )


def normal(key, shape, dtype=dtypes.float32):
  """Sample standard-normal random values.

  Args:
    key: the RNG key.
    shape: the shape of the result.
    dtype: the dtype of the result.

  Returns:
    Random values in standard-normal distribution.
  """
  key = tf_np.asarray(key, dtype=_RNG_KEY_DTYPE)
  return tf_np.asarray(
      stateless_random_ops.stateless_random_normal(
          shape, seed=_key2seed(key), dtype=dtype
      )
  )


def bernoulli(key, mean=np.float32(0.5), shape=None):
  """Sample Bernoulli random values with given shape and mean.

  Args:
    key: the RNG key.
    mean: optional, an array_like broadcastable to `shape` for the mean of the
      random variables (default 0.5).
    shape: optional, a tuple of nonnegative integers representing the shape
      (default to `mean`'s shape).

  Returns:
    A random array with the specified shape and boolean dtype.
  """
  mean = tf_np.asarray(mean)
  if shape is None:
    shape = mean.shape
  return uniform(key, shape) < mean


def _eager_dataset_iterator(dataset):
  for item in dataset:
    yield nest.map_structure(tf_np.asarray, item)


def dataset_as_numpy(dataset):
  """Converts a `tf.data.Dataset` to an iterable of ndarrays.

  `dataset_as_numpy` converts a possibly nested structure of `tf.data.Dataset`s
  and `tf.Tensor`s to iterables of ndarrays and ndarrays, respectively. This
  function must be run in eager mode outside tf.function.

  Args:
    dataset: a possibly nested structure of `tf.data.Dataset`s and/or
      `tf.Tensor`s.

  Returns:
    A structure matching `dataset` where `tf.data.Dataset`s are converted to
    generators of ndarrays and `tf.Tensor`s are converted to ndarrays.
  """
  if not context.executing_eagerly():
    raise ValueError(
        "dataset_as_numpy must be run in eager mode outside tf.function")
  nested_ds = dataset
  del dataset

  # Flatten
  flat_ds = nest.flatten(nested_ds)
  flat_np = []

  # Type check for Tensors and Datasets
  for ds_el in flat_ds:
    if not isinstance(ds_el, (tensor_lib.Tensor, dataset_ops.DatasetV2)):
      types = nest.map_structure(type, nested_ds)
      raise ValueError("Arguments to dataset_as_numpy must be (possibly nested "
                       "structure of) tf.Tensors or tf.data.Datasets. Got: %s" %
                       types)

  for ds_el in flat_ds:
    if isinstance(ds_el, tensor_lib.Tensor):
      np_el = tf_np.asarray(ds_el)
    elif isinstance(ds_el, dataset_ops.DatasetV2):
      np_el = _eager_dataset_iterator(ds_el)
    else:
      assert False
    flat_np.append(np_el)

  return nest.pack_sequence_as(nested_ds, flat_np)


# TODO(nareshmodi): Group key should change based on the set of devices that we
# are mapping over. Make it so that we assign a unique group_key for every
# unique set of devices. We don't change it every time to avoid the overhead of
# discovering the full group (though may not be problematic in the local case).
_GROUP_KEY = 1
_INSTANCE_KEY = 0
_INSTANCE_LOCK = threading.Lock()


# TODO(b/142565636): Ensure that multiple concurrent calls to a tf.function
# containing a collective op run reasonably.
def _get_instance_key():
  global _INSTANCE_KEY
  global _INSTANCE_LOCK
  with _INSTANCE_LOCK:
    _INSTANCE_KEY = _INSTANCE_KEY + 1
    return _INSTANCE_KEY


# Don't use a namedtuple since nest considers that a tuple and unflattens and
# flattens it.
class ShardedNdArray(object):
  """Wrapper over ndarray that can contain tensors on multiple devices.

    This is returned by extensions.pmap, and contains the individual tensors on
    different devices.
  """

  def __init__(self, tensors):
    """Initializes the ShardedNdArray.

    Note that the tensors should be ordered in the way the pmap producing these
    tensors is run.

    Args:
      tensors: list or tuple of eager tensors, one for each device.
    """

    if not isinstance(tensors, (list, tuple)) or not tensors:
      raise ValueError(
          "Unable to create a ShardedNdArray without a list of tensors.")
    self.tensors = tensors
    self.n_devices = len(tensors)

  def __getitem__(self, i):
    return tf_np.asarray(self.tensors[i])

  @property
  def shape(self):
    return (self.n_devices,) + self.tensors[0]._shape_tuple()  # pylint: disable=protected-access

  @property
  def dtype(self):
    return self.tensors[0].dtype


def convert_sharded_tensor_to_eager_tensor(value, *args, **kwargs):
  del args, kwargs
  # TODO(nareshmodi): Consider a collective op to gather the tensors from the
  # various devices for performance reasons.
  return array_ops_stack.stack(value.tensors)


tensor_conversion_registry.register_tensor_conversion_function(
    ShardedNdArray, convert_sharded_tensor_to_eager_tensor
)


class _PmapConfig(threading.local):
  """Simple config used to maintain state related to a current pmap call."""

  def __init__(self):
    super(_PmapConfig, self).__init__()
    self._axis_name = None
    self._devices = None

  def axis_name(self):
    return self._axis_name

  def set_axis_name(self, axis_name):
    self._axis_name = axis_name

  def devices(self):
    return self._devices

  def set_devices(self, devices):
    self._devices = devices


_pmap_config = _PmapConfig()


@contextlib.contextmanager
def pmap_config(axis_name, devices):
  """Records axis_name and devices for this context."""
  old_axis_name = _pmap_config.axis_name()
  old_devices = _pmap_config.devices()
  _pmap_config.set_axis_name(axis_name)
  _pmap_config.set_devices(devices)
  try:
    yield
  finally:
    _pmap_config.set_axis_name(old_axis_name)
    _pmap_config.set_devices(old_devices)


def _psum(tensor, axis_name=None):
  """Sum all-reduction.

  Args:
    tensor: A tensor.
    axis_name: The axis name to reduce. Must equal to that of the surrounding
      pmap.

  Returns:
    The sum of the `tensor` replicas on each participating devices.
  """
  if axis_name != _pmap_config.axis_name():
    raise ValueError("axis_name (%s) is not equal to that of the surrounding "
                     "pmap (%s)" % (axis_name, _pmap_config.axis_name()))
  devices = _pmap_config.devices()
  if devices is None:
    raise ValueError("Can't retrieve the device list from the surrounding pmap")
  tensor = tf_np.asarray(tensor)
  if tpu_devices(devices):
    # TODO(b/170895907): Remove this workaround when tpu.cross_replica_sum
    #   supports int64/float64.
    is_int64 = False
    is_float64 = False
    if tensor.dtype == np.int64:
      is_int64 = True
      tensor = tensor.astype(np.int32)
    elif tensor.dtype == np.float64:
      is_float64 = True
      tensor = tensor.astype(np.float32)
    # TODO(wangpeng): Supply the `group_assignment` argument to
    #   tpu.cross_replica_sum, calculated from `devices`.
    tensor = tpu_ops.cross_replica_sum(tensor)
    if is_int64:
      tensor = math_ops.cast(tensor, dtypes.int64)
    elif is_float64:
      tensor = math_ops.cast(tensor, dtypes.float64)
  else:
    tensor = gen_collective_ops.collective_reduce(
        input=tensor,
        group_size=len(devices),
        group_key=_GROUP_KEY,
        instance_key=_get_instance_key(),
        merge_op="Add",
        final_op="Id",
        subdiv_offsets=(0,),
    )
  return tf_np.asarray(tensor)


def psum(tensors, axis_name=None):
  return nest.map_structure(
      functools.partial(_psum, axis_name=axis_name), tensors
  )


# Note this is not available in the jax api, but seemed like a reasonable API
# to have.
def pmean(tensor, axis_name=None):
  """Mean all-reduction.

  Args:
    tensor: A tensor.
    axis_name: The axis name to reduce. Must equal to that of the surrounding
      pmap.

  Returns:
    The mean of the `tensor` replicas on each participating devices.
  """
  if axis_name != _pmap_config.axis_name():
    raise ValueError("axis_name (%s) is not equal to that of the surrounding "
                     "pmap (%s)" % (axis_name, _pmap_config.axis_name()))
  devices = _pmap_config.devices()
  if devices is None:
    raise ValueError("Can't retrieve the device list from the surrounding pmap")
  if tpu_devices(devices):
    # TODO(wangpeng): Implement this.
    raise ValueError("pmean for TPU is not supported yet.")
  else:
    return gen_collective_ops.collective_reduce(
        input=tensor,
        group_size=len(devices),
        group_key=_GROUP_KEY,
        instance_key=_get_instance_key(),
        merge_op="Add",
        final_op="Div",
        subdiv_offsets=(0,),
    )


def _get_pmap_impl(f, devices, has_tpu):
  """This is a helper function to return the pmap impl.

  Args:
    f: a function that takes ndarrays and returns ndarrays.
    devices: a list of strings; the device list.
    has_tpu: boolean; whether `devices` contains TPU devices.

  Returns:
    A function that takes tensors and returns tensors.
  """
  if has_tpu:
    # Workaround b/121383831
    output_is_list = [False]  # Use list for mutability
    def recorder(args, kwargs, res):
      del args, kwargs
      output_is_list[0] = isinstance(res, list)
      return res
    f = _record_result_type(recorder, f)

  def tf_f(*tf_args):
    """A wrapper for `f` that takes/returns tensors."""
    np_args = _tf_to_np(tf_args)
    np_out = f(*np_args)
    return np_out

  if has_tpu:

    @polymorphic_function.function(autograph=False)
    def fn(inputs):
      # TODO(wangpeng): Supply the `device_assignment` argument to
      # tpu.replicate, calculated from `devices`.
      res = tpu.replicate(tf_f, inputs)
      # Workaround b/121383831
      if (res and isinstance(res[0], list) and len(res[0]) == 1 and
          not output_is_list[0]):
        res = [x[0] for x in res]
      return res

    return fn
  else:
    # This is run in a tf.function so that the various underlying functions can
    # be run in parallel.
    # The trace happens on the client, so any devices should not depend on any
    # side effects.

    jit_tf_f = polymorphic_function.function(tf_f, autograph=False)

    @polymorphic_function.function(autograph=False)
    def fn(all_per_device_args):
      """Multi-device function with calls placed on the correct device."""

      results = []
      for per_device_args, device in zip(all_per_device_args, devices):
        with ops.device(device):
          results.append(jit_tf_f(*per_device_args))
      return results

    return fn


def pmap(f, axis_name=None, devices=None):
  """Transforms a function into a multi-device function.

  The semantics are similar to JAX's pmap.

  Args:
    f: The function to be converted.
    axis_name: Used for nested pmap, which is not supported yet.
    devices: The devices over which the returned function will run.

  Returns:
    A function that runs the underlying function `f` on `devices`. Its arguments
    can be `ShardedNdArray`s, tensors or other Python objects, and its return
    values are all `ShardedNdArray`s. If an input is a tensor, the length of its
    first dimension must equal the number of devices, and the tensor will be
    splitted along its first dimension among the devices. If an input is an
    unknown Python object, it will be replicated among the devices.
  """
  if devices is None:
    devices = accelerators()
  if not isinstance(devices, (list, tuple)):
    raise ValueError("Must pass a list or tuple of devices")
  num_devices = len(devices)
  if not num_devices:
    raise ValueError("There must be at least 1 device")
  has_tpu = bool(tpu_devices(devices))

  pmap_fn = _get_pmap_impl(f, devices, has_tpu)

  def wrapper(*args):
    """Wrapper that wraps/unwraps args, retvals, and runs the function."""
    if _pmap_config.devices() is not None:
      raise ValueError("Found a surrounding pmap. Nested pmap is not supported "
                       "yet.")
    # TODO(wangpeng): Maybe we should use `asarray` to convert everything
    # to ndarray first.

    flattened_input_args = nest.flatten(args)
    flattened_per_device_args = [[] for _ in devices]
    for arg in flattened_input_args:
      if isinstance(arg, tensor_lib.Tensor):
        # TODO(nareshmodi): Try and use the dynamic shape instead.
        if (not arg.shape.rank) or arg.shape[0] != len(devices):
          # TODO(nareshmodi): Fix this restriction
          raise ValueError(
              "Input tensors need to have a first dimension equal to "
              "the number of devices; got tensor of shape %s and %s devices" %
              (arg.shape, len(devices)))
        # NOTE: Alternatively use tf.split, and place the split tensors on the
        # appropriate device. The best solution for this is to have an API that
        # splits a tensor across devices.
        for j, device in enumerate(devices):
          updated_arg = array_ops.gather_v2(arg, j)
          # TODO(wangpeng): Investigate whether we need a tf.identity for TPU.
          if not has_tpu:
            with ops.device(device):
              updated_arg = array_ops.identity(updated_arg)
          flattened_per_device_args[j].append(updated_arg)
      elif isinstance(arg, ShardedNdArray):
        for device_args, tensor in zip(flattened_per_device_args, arg.tensors):
          device_args.append(tensor)
      else:
        for device_args in flattened_per_device_args:
          device_args.append(arg)

    all_per_device_args = [
        nest.pack_sequence_as(args, device_args)
        for device_args in flattened_per_device_args
    ]

    with pmap_config(axis_name, devices):
      results = pmap_fn(all_per_device_args)

    # Rewrap things. This can probably be written better.
    flattened_results = [nest.flatten(result) for result in results]
    final_tree = []

    # TODO(nareshmodi): assert all items in flattened_results have the same
    # structures

    for i in range(len(flattened_results[0])):
      tensors = []
      for j, device in enumerate(devices):
        assert isinstance(
            flattened_results[j][i], tensor_lib.Tensor
        ), "currently only tensor return items are supported"
        tensors.append(flattened_results[j][i])
      final_tree.append(ShardedNdArray(tensors))

    return nest.pack_sequence_as(results[0], final_tree)

  return wrapper


def find_devices(device_type, devices=None):
  if not devices:
    devices = [d.name for d in config.list_logical_devices()]
  devices = [(d, device_spec.DeviceSpecV2.from_string(d)) for d in devices]
  results = [name for name, d in devices if d.device_type == device_type]
  return results


def tpu_devices(devices=None):
  """Gets TPU devices out of `devices`.

  Args:
    devices: A device list (as a list of strings). If None, the list of all
      available devices will be used for it.

  Returns:
    Those in `devices` that are TPUs.
  """
  return find_devices("TPU", devices)


def gpu_devices(devices=None):
  """Gets GPU devices out of `devices`.

  Args:
    devices: A device list (as a list of strings). If None, the list of all
      available devices will be used for it.

  Returns:
    Those in `devices` that are GPUs.
  """
  return find_devices("GPU", devices)


def accelerators(devices=None):
  return tpu_devices(devices) or gpu_devices(devices)


def _tree_broadcast(to, s):
  """Broadcasts `s` to the nested structure `to`."""
  if not isinstance(to, (list, tuple, dict)):
    if not isinstance(s, (int, type(None))):
      raise ValueError
    return s
  if isinstance(s, (int, type(None))):
    return nest.map_structure(lambda x: s, to)
  if isinstance(to, (list, tuple)):
    if len(to) != len(s):
      raise ValueError
    new_s = [_tree_broadcast(x, y) for x, y in zip(to, s)]
    if isinstance(to, tuple):
      new_s = tuple(new_s)
    return new_s
  elif isinstance(to, dict):
    return {k: _tree_broadcast(to[k], s[k]) for k in to}
  else:
    raise TypeError("Unsupported type %s" % type(to))


def vmap(f, in_axes=0, out_axes=0):
  """Returns a function that maps `f` over first dimension of inputs."""
  in_axes_flat = nest.flatten(in_axes)
  if not all(isinstance(l, (type(None), int))
             for l in in_axes_flat):
    raise TypeError(
        "vmap in_axes must be an int, None, or (nested) container with "
        "those types as leaves, but got {}.".format(in_axes))
  if all(isinstance(l, type(None)) for l in in_axes_flat):
    raise ValueError("vmap must have at least one non-None value in in_axes")

  out_axes_flat = nest.flatten(out_axes)
  if not all(isinstance(l, (type(None), int))
             for l in out_axes_flat):
    raise TypeError(
        "vmap out_axes must be an int, None, or (nested) container with "
        "those types as leaves, but got {}.".format(out_axes))

  def _f(*args):
    flat_args = nest.flatten(args)
    try:
      f_in_axes = _tree_broadcast(args, in_axes)
    except ValueError:
      six.reraise(
          ValueError,
          ValueError(
              "vmap in_axes specification must be a tree prefix of the "
              r"corresponding value, got specification %s for value tree %s" % (
                  in_axes, args)),
          sys.exc_info()[2])
    f_in_axes_flat = nest.flatten(f_in_axes)

    def tf_f(tf_args):
      """Function passed to tf.vectorized_map call."""
      # Note that unbatched arguments are not passed to tf_f. Here we fill thos
      # arguments back before calling `f`.
      tf_flat_args = []
      j = 0
      for arg, axis in zip(flat_args, f_in_axes_flat):
        if axis is None:
          tf_flat_args.append(arg)
        else:
          tf_flat_args.append(tf_args[j])
          j += 1
      unbatched_args = nest.pack_sequence_as(args, tf_flat_args)
      return f(*unbatched_args)

    # Constructs arguments to pass to `tf_f`.
    # Unbatch arguments are skipped. Arguments with non-zero axis are
    # transposed.
    tf_args = []
    for arg, axis in zip(flat_args, f_in_axes_flat):
      if axis is None:
        continue
      arg = tf_np.asarray(arg)
      if axis != 0:
        arg = tf_np.moveaxis(arg, axis, 0)
      tf_args.append(arg)
    # TODO(agarwal): consider creating a tf.function outside of _f and reusing
    # that to avoid overheads of re-vectorizing the code when running eagerly.
    outputs = pfor_ops.vectorized_map(tf_f, tf_args)
    try:
      f_out_axes = _tree_broadcast(outputs, out_axes)
    except ValueError:
      six.reraise(
          ValueError,
          ValueError(
              "vmap out_axes specification must be a tree prefix of the "
              r"corresponding value, got specification %s for value tree %s" % (
                  out_axes, outputs)),
          sys.exc_info()[2])

    def map_output(x, axis):
      """Maps output of tf.vectorized_map to the final output."""
      x = tf_np.asarray(x)
      if axis is None:
        # Note that `tf.vectorized_map always batches the outputs.
        # Here we unbatch it again.
        return x[0, ...]
      elif axis == 0:
        return x
      else:
        # Need to transpose the output.
        return tf_np.moveaxis(x, 0, axis)
    new_outputs = [
        map_output(output, axis)
        for output, axis in zip(nest.flatten(outputs), nest.flatten(f_out_axes))
    ]
    return nest.pack_sequence_as(outputs, new_outputs)

  return _f
