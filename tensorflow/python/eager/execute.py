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
"""Functions called by the generated code to execute an eager-mode op."""

import six

from google.protobuf import text_format
from tensorflow.core.framework import tensor_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import core
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.util import compat


def quick_execute(op_name, num_outputs, inputs, attrs, ctx, name=None):
  """Execute a TensorFlow operation.

  Args:
    op_name: Name of the TensorFlow operation (see REGISTER_OP in C++ code) to
      execute.
    num_outputs: The number of outputs of the operation to fetch. (Explicitly
      provided instead of being inferred for performance reasons).
    inputs: A list of inputs to the operation. Each entry should be a Tensor, or
      a value which can be passed to the Tensor constructor to create one.
    attrs: A tuple with alternating string attr names and attr values for this
      operation.
    ctx: The value of context.context().
    name: Customized name for the operation.

  Returns:
    List of output Tensor objects. The list is empty if there are no outputs

  Raises:
    An exception on error.
  """
  device_name = ctx.device_name
  # pylint: disable=protected-access
  try:
    ctx.ensure_initialized()
    tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,
                                        inputs, attrs, num_outputs)
  except core._NotOkStatusException as e:
    if name is not None:
      e.message += " name: " + name
    raise core._status_to_exception(e) from None
  except TypeError as e:
    keras_symbolic_tensors = [
        x for x in inputs if ops._is_keras_symbolic_tensor(x)
    ]
    if keras_symbolic_tensors:
      raise core._SymbolicException(
          "Inputs to eager execution function cannot be Keras symbolic "
          "tensors, but found {}".format(keras_symbolic_tensors))
    raise e
  # pylint: enable=protected-access
  return tensors


def execute_with_cancellation(op_name,
                              num_outputs,
                              inputs,
                              attrs,
                              ctx,
                              cancellation_manager,
                              name=None):
  """Execute a TensorFlow operation.

  Args:
    op_name: Name of the TensorFlow operation (see REGISTER_OP in C++ code) to
      execute.
    num_outputs: The number of outputs of the operation to fetch. (Explicitly
      provided instead of being inferred for performance reasons).
    inputs: A list of inputs to the operation. Each entry should be a Tensor, or
      a value which can be passed to the Tensor constructor to create one.
    attrs: A tuple with alternating string attr names and attr values for this
      operation.
    ctx: The value of context.context().
    cancellation_manager: a `CancellationManager` object that can be used to
      cancel the operation.
    name: Customized name for the operation.

  Returns:
    List of output Tensor objects. The list is empty if there are no outputs

  Raises:
    An exception on error.
  """
  device_name = ctx.device_name
  # pylint: disable=protected-access
  try:
    ctx.ensure_initialized()
    tensors = pywrap_tfe.TFE_Py_ExecuteCancelable(ctx._handle, device_name,
                                                  op_name, inputs, attrs,
                                                  cancellation_manager._impl,
                                                  num_outputs)
  except core._NotOkStatusException as e:
    if name is not None:
      e.message += " name: " + name
    raise core._status_to_exception(e) from None
  except TypeError as e:
    keras_symbolic_tensors = [
        x for x in inputs if ops._is_keras_symbolic_tensor(x)
    ]
    if keras_symbolic_tensors:
      raise core._SymbolicException(
          "Inputs to eager execution function cannot be Keras symbolic "
          "tensors, but found {}".format(keras_symbolic_tensors))
    raise e
  # pylint: enable=protected-access
  return tensors


def execute_with_callbacks(op_name, num_outputs, inputs, attrs, ctx, name=None):
  """Monkey-patch to execute to enable execution callbacks."""
  tensors = quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
  for callback in ctx.op_callbacks:
    callback(op_name, tuple(inputs), attrs, tensors, name)

  return tensors


execute = quick_execute


def must_record_gradient():
  """Import backprop if you want gradients recorded."""
  return False


def record_gradient(unused_op_name, unused_inputs, unused_attrs,
                    unused_outputs):
  """Import backprop if you want gradients recorded."""
  pass


def make_float(v, arg_name):
  if not isinstance(v, compat.real_types):
    raise TypeError("Expected float for argument '%s' not %s." %
                    (arg_name, repr(v)))
  return float(v)


def make_int(v, arg_name):
  if isinstance(v, six.string_types):
    raise TypeError("Expected int for argument '%s' not %s." %
                    (arg_name, repr(v)))
  try:
    return int(v)
  except (ValueError, TypeError):
    raise TypeError("Expected int for argument '%s' not %s." %
                    (arg_name, repr(v)))


def make_str(v, arg_name):
  if not isinstance(v, compat.bytes_or_text_types):
    raise TypeError("Expected string for argument '%s' not %s." %
                    (arg_name, repr(v)))
  return compat.as_bytes(v)  # Convert unicode strings to bytes.


def make_bool(v, arg_name):
  if not isinstance(v, bool):
    raise TypeError("Expected bool for argument '%s' not %s." %
                    (arg_name, repr(v)))
  return v


def make_type(v, arg_name):
  try:
    v = dtypes.as_dtype(v).base_dtype
  except TypeError:
    raise TypeError("Expected DataType for argument '%s' not %s." %
                    (arg_name, repr(v)))
  i = v.as_datatype_enum
  return i


def make_shape(v, arg_name):
  """Convert v into a list."""
  # Args:
  #   v: A TensorShapeProto, a list of ints, or a tensor_shape.TensorShape.
  #   arg_name: String, for error messages.

  # Returns:
  #   None if the rank is unknown, otherwise a list of ints (or Nones in the
  #   position where the dimension is unknown).
  try:
    shape = tensor_shape.as_shape(v)
  except TypeError as e:
    raise TypeError("Error converting %s to a TensorShape: %s." % (arg_name, e))
  except ValueError as e:
    raise ValueError("Error converting %s to a TensorShape: %s." %
                     (arg_name, e))
  if shape.ndims is None:
    return None
  else:
    return shape.as_list()


def make_tensor(v, arg_name):
  """Ensure v is a TensorProto."""
  if isinstance(v, tensor_pb2.TensorProto):
    return v
  elif isinstance(v, six.string_types):
    pb = tensor_pb2.TensorProto()
    text_format.Merge(v, pb)
    return pb
  raise TypeError(
      "Don't know how to convert %s to a TensorProto for argument '%s'." %
      (repr(v), arg_name))


def args_to_matching_eager(l, ctx, allowed_dtypes, default_dtype=None):
  """Convert sequence `l` to eager same-type Tensors."""
  if (not l) and (default_dtype is not None):
    return default_dtype, []  # List is empty; assume default dtype.
  EagerTensor = ops.EagerTensor  # pylint: disable=invalid-name
  for x in l:
    if not isinstance(x, EagerTensor):
      break
  else:  # note: intentional for-else
    return l[0]._datatype_enum(), l  # pylint: disable=protected-access

  # Is some input already a Tensor with a dtype?
  dtype = None
  for t in l:
    if isinstance(t, EagerTensor):
      dtype = t.dtype
      break

  if dtype is None:
    # Infer a dtype based on the first value, and use that dtype for the
    # remaining values.

    ret = []
    for t in l:
      tensor = None
      # First see if we can get a valid dtype with the default conversion
      # and see if it matches an allowed dtypes. Some ops like ConcatV2 may
      # not list allowed dtypes, in which case we should skip this.
      if dtype is None and allowed_dtypes:
        tensor = ops.convert_to_tensor(t, ctx=ctx)
        # If we did not match an allowed dtype, try again with the default
        # dtype. This could be because we have an empty tensor and thus we
        # picked the wrong type.
        if tensor.dtype not in allowed_dtypes:
          tensor = None

      if tensor is None:
        tensor = ops.convert_to_tensor(
            t, dtype, preferred_dtype=default_dtype, ctx=ctx)

      ret.append(tensor)
      if dtype is None:
        dtype = tensor.dtype
  else:
    ret = [ops.convert_to_tensor(t, dtype, ctx=ctx) for t in l]

  # TODO(slebedev): consider removing this as it leaks a Keras concept.
  # pylint: disable=protected-access
  keras_symbolic_tensors = [x for x in ret if ops._is_keras_symbolic_tensor(x)]
  if keras_symbolic_tensors:
    raise core._SymbolicException(
        "Using symbolic output of a Keras layer during eager execution "
        "{}".format(keras_symbolic_tensors))
  # pylint: enable=protected-access
  return dtype.as_datatype_enum, ret


def convert_to_mixed_eager_tensors(values, ctx):
  v = [ops.convert_to_tensor(t, ctx=ctx) for t in values]
  types = [t._datatype_enum() for t in v]  # pylint: disable=protected-access
  return types, v


def args_to_mixed_eager_tensors(lists, ctx):
  """Converts a list of same-length lists of values to eager tensors."""
  assert len(lists) > 1

  # Generate an error if len(lists[i]) is not the same for all i.
  lists_ret = [[]]
  for l in lists[1:]:
    if len(l) != len(lists[0]):
      raise ValueError(
          "Expected list arguments to be the same length: %d != %d (%r vs. %r)."
          % (len(lists[0]), len(l), lists[0], l))
    lists_ret.append([])

  # Convert the first element of each list first, then the second element, etc.
  types = []
  for i in range(len(lists[0])):
    dtype = None
    # If any list has a Tensor, use that dtype
    for l in lists:
      if isinstance(l[i], ops.EagerTensor):
        dtype = l[i].dtype
        break
    if dtype is None:
      # Convert the first one and use its dtype.
      lists_ret[0].append(ops.convert_to_tensor(lists[0][i], ctx=ctx))
      dtype = lists_ret[0][i].dtype
      for j in range(1, len(lists)):
        lists_ret[j].append(
            ops.convert_to_tensor(lists[j][i], dtype=dtype, ctx=ctx))
    else:
      # Convert everything to the found dtype.
      for j in range(len(lists)):
        lists_ret[j].append(
            ops.convert_to_tensor(lists[j][i], dtype=dtype, ctx=ctx))
    types.append(dtype.as_datatype_enum)
  return types, lists_ret
