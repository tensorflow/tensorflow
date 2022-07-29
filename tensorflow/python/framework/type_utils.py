# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Utility functions for types information, incuding full type information."""

from typing import List

from tensorflow.core.framework import full_type_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import type_spec
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensorSpec
from tensorflow.python.ops.structured.structured_tensor import StructuredTensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

# TODO(b/226455884) A python binding for DT_TO_FT or map_dtype_to_tensor() from
# tensorflow/core/framework/types.cc to avoid duplication here
_DT_TO_FT = {
    types_pb2.DT_FLOAT: full_type_pb2.TFT_FLOAT,
    types_pb2.DT_DOUBLE: full_type_pb2.TFT_DOUBLE,
    types_pb2.DT_INT32: full_type_pb2.TFT_INT32,
    types_pb2.DT_UINT8: full_type_pb2.TFT_UINT8,
    types_pb2.DT_INT16: full_type_pb2.TFT_INT16,
    types_pb2.DT_INT8: full_type_pb2.TFT_INT8,
    types_pb2.DT_STRING: full_type_pb2.TFT_STRING,
    types_pb2.DT_COMPLEX64: full_type_pb2.TFT_COMPLEX64,
    types_pb2.DT_INT64: full_type_pb2.TFT_INT64,
    types_pb2.DT_BOOL: full_type_pb2.TFT_BOOL,
    types_pb2.DT_UINT16: full_type_pb2.TFT_UINT16,
    types_pb2.DT_COMPLEX128: full_type_pb2.TFT_COMPLEX128,
    types_pb2.DT_HALF: full_type_pb2.TFT_HALF,
    types_pb2.DT_UINT32: full_type_pb2.TFT_UINT32,
    types_pb2.DT_UINT64: full_type_pb2.TFT_UINT64,
    types_pb2.DT_VARIANT: full_type_pb2.TFT_LEGACY_VARIANT,
}


def _translate_to_fulltype_for_flat_tensors(
    spec: type_spec.TypeSpec) -> List[full_type_pb2.FullTypeDef]:
  """Convert a TypeSec to a list of FullTypeDef.

  The FullTypeDef created corresponds to the encoding used with datasets
  (and map_fn) that uses variants (and not FullTypeDef corresponding to the
  default "component" encoding).

  Currently, the only use of this is for information about the contents of
  ragged tensors, so only ragged tensors return useful full type information
  and other types return TFT_UNSET. While this could be improved in the future,
  this function is intended for temporary use and expected to be removed
  when type inference support is sufficient.

  Args:
    spec: A TypeSpec for one element of a dataset or map_fn.

  Returns:
    A list of FullTypeDef corresponding to SPEC. The length of this list
    is always the same as the length of spec._flat_tensor_specs.
  """
  if isinstance(spec, RaggedTensorSpec):
    dt = spec.dtype
    elem_t = _DT_TO_FT.get(dt)
    if elem_t is None:
      logging.vlog(1, "dtype %s that has no conversion to fulltype.", dt)
    elif elem_t == full_type_pb2.TFT_LEGACY_VARIANT:
      logging.vlog(1, "Ragged tensors containing variants are not supported.",
                   dt)
    else:
      assert len(spec._flat_tensor_specs) == 1  # pylint: disable=protected-access
      return [
          full_type_pb2.FullTypeDef(
              type_id=full_type_pb2.TFT_RAGGED,
              args=[full_type_pb2.FullTypeDef(type_id=elem_t)])
      ]
  return [
      full_type_pb2.FullTypeDef(type_id=full_type_pb2.TFT_UNSET)
      for t in spec._flat_tensor_specs  # pylint: disable=protected-access
  ]


# LINT.IfChange(_specs_for_flat_tensors)
def _specs_for_flat_tensors(element_spec):
  """Return a flat list of type specs for element_spec.

  Note that "flat" in this function and in `_flat_tensor_specs` is a nickname
  for the "batchable tensor list" encoding used by datasets and map_fn
  internally (in C++/graphs). The ability to batch, unbatch and change
  batch size is one important characteristic of this encoding. A second
  important characteristic is that it represets a ragged tensor or sparse
  tensor as a single tensor of type variant (and this encoding uses special
  ops to encode/decode to/from variants).

  (In constrast, the more typical encoding, e.g. the C++/graph
  representation when calling a tf.function, is "component encoding" which
  represents sparse and ragged tensors as multiple dense tensors and does
  not use variants or special ops for encoding/decoding.)

  Args:
    element_spec: A nest of TypeSpec describing the elements of a dataset (or
      map_fn).

  Returns:
    A non-nested list of TypeSpec used by the encoding of tensors by
    datasets and map_fn for ELEMENT_SPEC. The items
    in this list correspond to the items in `_flat_tensor_specs`.
  """
  if isinstance(element_spec, StructuredTensor.Spec):
    specs = []
    for _, field_spec in sorted(
        element_spec._field_specs.items(), key=lambda t: t[0]):  # pylint: disable=protected-access
      specs.extend(_specs_for_flat_tensors(field_spec))
  elif isinstance(element_spec, type_spec.BatchableTypeSpec) and (
      element_spec.__class__._flat_tensor_specs is  # pylint: disable=protected-access
      type_spec.BatchableTypeSpec._flat_tensor_specs):  # pylint: disable=protected-access
    # Classes which use the default `_flat_tensor_specs` from
    # `BatchableTypeSpec` case (i.e. a derived class does not override
    # `_flat_tensor_specs`.) are encoded using `component_specs`.
    specs = nest.flatten(
        element_spec._component_specs,  # pylint: disable=protected-access
        expand_composites=False)
  else:
    # In addition flatting any nesting in Python,
    # this default case covers things that are encoded by one tensor,
    # such as dense tensors which are unchanged by encoding and
    # ragged tensors and sparse tensors which are encoded by a variant tensor.
    specs = nest.flatten(element_spec, expand_composites=False)
  return specs
# LINT.ThenChange()
# Note that _specs_for_flat_tensors must correspond to _flat_tensor_specs


def fulltypes_for_flat_tensors(element_spec):
  """Convert the element_spec for a dataset to a list of FullType Def.

  Note that "flat" in this function and in `_flat_tensor_specs` is a nickname
  for the "batchable tensor list" encoding used by datasets and map_fn.
  The FullTypeDef created corresponds to this encoding (e.g. that uses variants
  and not the FullTypeDef corresponding to the default "component" encoding).

  This is intended for temporary internal use and expected to be removed
  when type inference support is sufficient. See limitations of
  `_translate_to_fulltype_for_flat_tensors`.

  Args:
    element_spec: A nest of TypeSpec describing the elements of a dataset (or
      map_fn).

  Returns:
    A list of FullTypeDef correspoinding to ELEMENT_SPEC. The items
    in this list correspond to the items in `_flat_tensor_specs`.
  """
  specs = _specs_for_flat_tensors(element_spec)
  full_types_lists = [_translate_to_fulltype_for_flat_tensors(s) for s in specs]
  rval = nest.flatten(full_types_lists)  # flattens list-of-list to flat list.
  return rval


def fulltype_list_to_product(fulltype_list):
  """Convert a list of FullType Def into a single FullType Def."""
  return full_type_pb2.FullTypeDef(
      type_id=full_type_pb2.TFT_PRODUCT, args=fulltype_list)
