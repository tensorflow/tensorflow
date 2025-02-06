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
"""Binary operator override class for Tensor overrides."""
import numbers
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.util import nest
from tensorflow.python.util import traceback_utils


def _maybe_get_dtype(x):
  """Returns a numpy type if available from x. Skips if x is numpy.ndarray."""
  # Don't put np.ndarray in this list, because np.result_type looks at the
  # value (not just dtype) of np.ndarray to decide the result type.
  if isinstance(x, numbers.Real):
    return x
  if isinstance(x, tensor_lib.Tensor):
    return x.dtype.as_numpy_dtype
  if isinstance(x, dtypes.DType):
    return x.as_numpy_dtype
  if isinstance(x, tensor_shape.TensorShape):
    return np.int32
  if isinstance(x, (list, tuple)):
    raise ValueError(f"Cannot determine dtype.  Got sequence {x}.")
  return x


def maybe_promote_tensors(*tensors, force_same_dtype=False):
  """Promotes tensors if numpy style promotion is enabled.

  This function promotes `tensors` according to numpy promotion rules
  if numpy style promotion is enabled.  Otherwise, if
  `force_same_dtype` is `True`, it force-casts `tensors[1:]` to
  `tensor[0]`'s dtype. Note that this force-cast can be problematic.
  For example, when some `tensors[1:]` elements can be silently
  downcasted.

  Args:
    *tensors: the list of tensors to promote.
    force_same_dtype: bool (optional, default to `False`). When numpy
      style promotion is disabled and `force_same_dtype` is `True`,
      this function will force-casts `tensors[1:]` to `tensor[0]`'s
      dtype (which could be problematic).

  Returns:
    The promoted list of tensors.
  """
  if ops.is_auto_dtype_conversion_enabled():
    return tensors
  if not tensors:
    return tensors
  if not ops.is_numpy_style_type_promotion():
    if not force_same_dtype:
      return tensors
    promoted_tensors = []
    promoted_tensors.append(tensors[0])
    dtype = tensors[0].dtype.base_dtype
    for tensor in tensors[1:]:
      promoted_tensors.append(
          ops.convert_to_tensor(tensor, dtype, name="x"))
    return promoted_tensors
  result_type = np_dtypes._result_type(  # pylint: disable=protected-access
      *[_maybe_get_dtype(x) for x in nest.flatten(tensors)])
  def _promote_or_cast(x):
    if isinstance(x, tensor_lib.Tensor):
      x = gen_math_ops.cast(x, result_type)
    else:
      x = ops.convert_to_tensor(x, result_type)
    return x
  return [_promote_or_cast(x) for x in tensors]


# pylint: disable=protected-access
def override_binary_operator_helper(
    func, op_name, clazz_object=tensor_lib.Tensor):
  """Register operators with different tensor and scalar versions.

  If `clazz_object` is `SparseTensor`, assumes `func` takes `(sp_indices,
  sp_values, sp_shape, dense)` and outputs `(new_sp_values)`.

  Args:
    func: the operator
    op_name: name of the operator being overridden
    clazz_object: class to override for.  Either `Tensor` or `SparseTensor`.
  """

  @traceback_utils.filter_traceback
  def binary_op_wrapper(x, y):
    with ops.name_scope(None, op_name, [x, y]) as name:
      try:
        # force_same_dtype=False to preserve existing TF behavior
        # TODO(b/178860388): Figure out why binary_op_wrapper and
        #   r_binary_op_wrapper use different force_same_dtype values.
        x, y = maybe_promote_tensors(x, y)
        return func(x, y, name=name)
      except (TypeError, ValueError) as e:
        # Even if dispatching the op failed, the RHS may be a tensor aware
        # object that can implement the operator with knowledge of itself
        # and the tensor.
        # If the RHS is not tensor aware we still want to raise the
        # original error from the LHS, because it may be more
        # informative.
        if hasattr(type(y), "__r%s__" % op_name):
          try:
            r_op = getattr(y, "__r%s__" % op_name)
            out = r_op(x)
            if out is NotImplemented:
              raise
            return out
          except (TypeError, ValueError):
            raise e
        else:
          raise

  @traceback_utils.filter_traceback
  def binary_op_wrapper_sparse(sp_x, y):
    with ops.name_scope(None, op_name, [sp_x, y]) as name:
      y = ops.convert_to_tensor(y, dtype=sp_x.dtype.base_dtype, name="y")
      # use the passed-in SparseTensor class to avoid having to import
      # SparseTensor, which would cause a cyclic dep with math_ops
      return clazz_object(
          sp_x.indices,
          func(sp_x.indices, sp_x.values, sp_x.dense_shape, y, name=name),
          sp_x.dense_shape)

  @traceback_utils.filter_traceback
  def r_binary_op_wrapper(y, x):
    with ops.name_scope(None, op_name, [x, y]) as name:
      # TODO(b/178860388): Figure out why binary_op_wrapper and
      #   r_binary_op_wrapper use different force_same_dtype values.
      y, x = maybe_promote_tensors(y, x, force_same_dtype=True)
      return func(x, y, name=name)

  # Propagate func.__doc__ to the wrappers
  try:
    doc = func.__doc__
  except AttributeError:
    doc = None
  binary_op_wrapper.__doc__ = doc
  r_binary_op_wrapper.__doc__ = doc
  binary_op_wrapper_sparse.__doc__ = doc

  if clazz_object is tensor_lib.Tensor:
    clazz_object._override_operator("__%s__" % op_name, binary_op_wrapper)
    del binary_op_wrapper
    clazz_object._override_operator("__r%s__" % op_name, r_binary_op_wrapper)
    del r_binary_op_wrapper
  else:
    clazz_object._override_operator("__%s__" % op_name,
                                    binary_op_wrapper_sparse)
    del binary_op_wrapper_sparse
