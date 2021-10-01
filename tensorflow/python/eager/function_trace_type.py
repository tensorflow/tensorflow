# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Utitiles for Cache Key generation based on Function Trace Type."""

import weakref

import numpy as np

from tensorflow.python import pywrap_tfe
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import resource_variable_ops


def get_arg_spec(inputs, include_tensor_ranks_only,
                 encode_variables_by_resource_id):
  """Returns the trace type specification of a function's arguments.

  Args:
    inputs: Tuple/List/Dict structure containing the function arguments
    include_tensor_ranks_only: If Tensors should be considered by rank
    encode_variables_by_resource_id: If Variables should be considered by
      resource id

  Returns:
    A hashable object representing the function arguments.
  """
  return _make_input_signature_hashable(pywrap_tfe.TFE_Py_EncodeArg(
      inputs, include_tensor_ranks_only, encode_variables_by_resource_id))


# TODO(b/195985838): Cleanup this function once Tensor protocol is implemented.
def _make_input_signature_hashable(elem):
  """Rewrite input signature to be hashable.

  We replace nested variables in the input signature with TensorSpec in order to
  be hashable.

  Args:
    elem: Input signature element

  Returns:
    A hashable object for the requested input signature
  """
  try:
    hash(elem)
  except TypeError:
    # TODO(slebedev): consider using nest.
    if isinstance(elem, tuple):
      return tuple(map(_make_input_signature_hashable, elem))

    # TFE_Py_EncodeArg weakrefs arguments it does not recognize, and we expect
    # all recognized types to be hashable.
    assert isinstance(elem, weakref.ReferenceType)
    v = elem()

    if resource_variable_ops.is_resource_variable(v):
      # We special case variables here to use unique_id as the cache key. This
      # ensures we have to retrace whenever a different variable is passed in.
      # This is needed to support cases where the user may use the id of a
      # variable in the function perhaps as a lookup in a dictionary.
      #
      # This choice leads to more retracing when we could have possibly used the
      # shape and dtype instead. However, we expect the number of variables in a
      # program to be bounded, and correspondingly the number of retraces.
      #
      # Note we also include the class name to avoid collisions with strings.
      return v.__class__, v._unique_id  # pylint: disable=protected-access

    if _is_ndarray(v):
      # Numpy arrays are not hashable, but when calling functions we treat them
      # in the same way as tf.Tensors.
      if not hasattr(v, "shape") or not hasattr(v, "dtype"):
        # TODO(tomhennigan) De-dup with _as_ndarray in _convert_numpy_inputs.
        v = _as_ndarray(v)
      return tensor_spec.TensorSpec(v.shape, v.dtype)

    raise ValueError("Arguments to a tf.function must be a nested structure of "
                     "Tensors, Variables, NumPy arrays, or hashable Python "
                     f"objects, got {type(v)}.")

  return elem


def _as_ndarray(value):
  """Converts value to an ndarray, assumes _is_ndarray(value)."""
  # TODO(tomhennigan) Support __array_interface__ too (including for
  # _convert_numpy_inputs).
  return value.__array__()


def _is_ndarray(value):
  """Tests whether the given value is an ndarray (and not a TF tensor/var)."""
  # TODO(tomhennigan) Support __array_interface__ too.
  return hasattr(value, "__array__") and not (
      isinstance(value, ops.Tensor)
      or isinstance(value, resource_variable_ops.BaseResourceVariable)
      or hasattr(value, "_should_act_as_resource_variable")

      # For legacy reasons we do not automatically promote Numpy strings.
      or isinstance(value, np.str_)
      # NumPy dtypes have __array__ as unbound methods.
      or isinstance(value, type)
      # CompositeTensors should be flattened instead.
      or isinstance(value, composite_tensor.CompositeTensor))
