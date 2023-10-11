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

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.util import tf_export


# We use numpy's dtypes instead of TF's, because the user expects to use them
# with numpy facilities such as `np.dtype(np.int64)` and
# `if x.dtype.type is np.int64`.
bool_ = np.bool_
tf_export.tf_export('experimental.numpy.bool_', v1=[]).export_constant(
    __name__, 'bool_'
)
complex_ = np.complex_
tf_export.tf_export('experimental.numpy.complex_', v1=[]).export_constant(
    __name__, 'complex_'
)
complex128 = np.complex128
tf_export.tf_export('experimental.numpy.complex128', v1=[]).export_constant(
    __name__, 'complex128'
)
complex64 = np.complex64
tf_export.tf_export('experimental.numpy.complex64', v1=[]).export_constant(
    __name__, 'complex64'
)
float_ = np.float_
tf_export.tf_export('experimental.numpy.float_', v1=[]).export_constant(
    __name__, 'float_'
)
float16 = np.float16
tf_export.tf_export('experimental.numpy.float16', v1=[]).export_constant(
    __name__, 'float16'
)
float32 = np.float32
tf_export.tf_export('experimental.numpy.float32', v1=[]).export_constant(
    __name__, 'float32'
)
float64 = np.float64
tf_export.tf_export('experimental.numpy.float64', v1=[]).export_constant(
    __name__, 'float64'
)
inexact = np.inexact
tf_export.tf_export('experimental.numpy.inexact', v1=[]).export_constant(
    __name__, 'inexact'
)
int_ = np.int_
tf_export.tf_export('experimental.numpy.int_', v1=[]).export_constant(
    __name__, 'int_'
)
int16 = np.int16
tf_export.tf_export('experimental.numpy.int16', v1=[]).export_constant(
    __name__, 'int16'
)
int32 = np.int32
tf_export.tf_export('experimental.numpy.int32', v1=[]).export_constant(
    __name__, 'int32'
)
int64 = np.int64
tf_export.tf_export('experimental.numpy.int64', v1=[]).export_constant(
    __name__, 'int64'
)
int8 = np.int8
tf_export.tf_export('experimental.numpy.int8', v1=[]).export_constant(
    __name__, 'int8'
)
object_ = np.object_
tf_export.tf_export('experimental.numpy.object_', v1=[]).export_constant(
    __name__, 'object_'
)
string_ = np.string_
tf_export.tf_export('experimental.numpy.string_', v1=[]).export_constant(
    __name__, 'string_'
)
uint16 = np.uint16
tf_export.tf_export('experimental.numpy.uint16', v1=[]).export_constant(
    __name__, 'uint16'
)
uint32 = np.uint32
tf_export.tf_export('experimental.numpy.uint32', v1=[]).export_constant(
    __name__, 'uint32'
)
uint64 = np.uint64
tf_export.tf_export('experimental.numpy.uint64', v1=[]).export_constant(
    __name__, 'uint64'
)
uint8 = np.uint8
tf_export.tf_export('experimental.numpy.uint8', v1=[]).export_constant(
    __name__, 'uint8'
)
unicode_ = np.unicode_
tf_export.tf_export('experimental.numpy.unicode_', v1=[]).export_constant(
    __name__, 'unicode_'
)


iinfo = np.iinfo
tf_export.tf_export('experimental.numpy.iinfo', v1=[]).export_constant(
    __name__, 'iinfo'
)


issubdtype = tf_export.tf_export('experimental.numpy.issubdtype', v1=[])(
    np.issubdtype
)


_to_float32 = {
    np.dtype('float64'): np.dtype('float32'),
    np.dtype('complex128'): np.dtype('complex64'),
}


_cached_np_dtypes = {}


# Difference between is_prefer_float32 and is_allow_float64: is_prefer_float32
# only decides which dtype to use for Python floats; is_allow_float64 decides
# whether float64 dtypes can ever appear in programs. The latter is more
# restrictive than the former.
_prefer_float32 = False


# TODO(b/178862061): Consider removing this knob
_allow_float64 = True


def is_prefer_float32():
  return _prefer_float32


def set_prefer_float32(b):
  global _prefer_float32
  _prefer_float32 = b


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
  """Returns the resulting type given a set of arrays."""

  def preprocess_float(x):
    if is_prefer_float32():
      if isinstance(x, float):
        return np.float32(x)
      elif isinstance(x, complex):
        return np.complex64(x)
    return x

  arrays_and_dtypes = [preprocess_float(x) for x in arrays_and_dtypes]
  dtype = np.result_type(*arrays_and_dtypes)
  return dtypes.as_dtype(canonicalize_dtype(dtype))


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
    If `is_prefer_float32()` is false and `is_allow_float64()` is true, returns
    float64; otherwise returns float32.
  """
  if not is_prefer_float32() and is_allow_float64():
    return float64
  else:
    return float32
