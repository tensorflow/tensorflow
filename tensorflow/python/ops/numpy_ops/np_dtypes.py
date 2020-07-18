# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Dtypes and dtype utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.ops.numpy_ops import np_export


# We use numpy's dtypes instead of TF's, because the user expects to use them
# with numpy facilities such as `np.dtype(np.int64)` and
# `if x.dtype.type is np.int64`.
bool_ = np_export.np_export_constant(__name__, 'bool_', np.bool_)
complex_ = np_export.np_export_constant(__name__, 'complex_', np.complex_)
complex128 = np_export.np_export_constant(__name__, 'complex128', np.complex128)
complex64 = np_export.np_export_constant(__name__, 'complex64', np.complex64)
float_ = np_export.np_export_constant(__name__, 'float_', np.float_)
float16 = np_export.np_export_constant(__name__, 'float16', np.float16)
float32 = np_export.np_export_constant(__name__, 'float32', np.float32)
float64 = np_export.np_export_constant(__name__, 'float64', np.float64)
inexact = np_export.np_export_constant(__name__, 'inexact', np.inexact)
int_ = np_export.np_export_constant(__name__, 'int_', np.int_)
int16 = np_export.np_export_constant(__name__, 'int16', np.int16)
int32 = np_export.np_export_constant(__name__, 'int32', np.int32)
int64 = np_export.np_export_constant(__name__, 'int64', np.int64)
int8 = np_export.np_export_constant(__name__, 'int8', np.int8)
object_ = np_export.np_export_constant(__name__, 'object_', np.object_)
string_ = np_export.np_export_constant(__name__, 'string_', np.string_)
uint16 = np_export.np_export_constant(__name__, 'uint16', np.uint16)
uint32 = np_export.np_export_constant(__name__, 'uint32', np.uint32)
uint64 = np_export.np_export_constant(__name__, 'uint64', np.uint64)
uint8 = np_export.np_export_constant(__name__, 'uint8', np.uint8)
unicode_ = np_export.np_export_constant(__name__, 'unicode_', np.unicode_)


iinfo = np_export.np_export_constant(__name__, 'iinfo', np.iinfo)


issubdtype = np_export.np_export('issubdtype')(np.issubdtype)


_to_float32 = {
    np.dtype('float64'): np.dtype('float32'),
    np.dtype('complex128'): np.dtype('complex64'),
}


_cached_np_dtypes = {}

_allow_float64 = True


def is_allow_float64():
  return _allow_float64


def set_allow_float64(b):
  global _allow_float64
  _allow_float64 = b


def canonicalize_dtype(dtype):
  if not _allow_float64:
    try:
      return _to_float32[dtype]
    except KeyError:
      pass
  return dtype


def _result_type(*arrays_and_dtypes):
  dtype = np.result_type(*arrays_and_dtypes)
  return canonicalize_dtype(dtype)


def _get_cached_dtype(dtype):
  """Returns an np.dtype for the TensorFlow DType."""
  global _cached_np_dtypes
  try:
    return _cached_np_dtypes[dtype]
  except KeyError:
    pass
  cached_dtype = np.dtype(dtype.as_numpy_dtype)
  _cached_np_dtypes[dtype] = cached_dtype
  return cached_dtype


def default_float_type():
  """Gets the default float type.

  Returns:
    If `is_allow_float64()` is true, returns float64; otherwise returns float32.
  """
  if is_allow_float64():
    return float64
  else:
    return float32
