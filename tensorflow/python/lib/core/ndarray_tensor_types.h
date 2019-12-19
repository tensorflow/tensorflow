/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_PYTHON_LIB_CORE_NDARRAY_TENSOR_TYPES_H_
#define TENSORFLOW_PYTHON_LIB_CORE_NDARRAY_TENSOR_TYPES_H_

// Must be included first.
// clang-format: off
#include "tensorflow/python/lib/core/numpy.h"
// clang-format: on

#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

extern PyArray_Descr* QINT8_DESCR;
extern PyArray_Descr* QINT16_DESCR;
extern PyArray_Descr* QINT32_DESCR;
extern PyArray_Descr* QUINT8_DESCR;
extern PyArray_Descr* QUINT16_DESCR;
extern PyArray_Descr* RESOURCE_DESCR;
extern PyArray_Descr* BFLOAT16_DESCR;

// Register custom NumPy types.
//
// This function must be called in order to be able to map TensorFlow
// data types which do not have a corresponding standard NumPy data type
// (e.g. bfloat16 or qint8).
//
// TODO(b/144230631): The name is slightly misleading, as the function only
// registers bfloat16 and defines structured aliases for other data types
// (e.g. qint8).
void MaybeRegisterCustomNumPyTypes();

// Returns a NumPy data type matching a given tensorflow::DataType. If the
// function call succeeds, the caller is responsible for DECREF'ing the
// resulting PyArray_Descr*.
//
// NumPy does not support quantized integer types, so TensorFlow defines
// structured aliases for them, e.g. tf.qint8 is represented as
// np.dtype([("qint8", np.int8)]). However, for historical reasons this
// function does not use these aliases, and instead returns the *aliased*
// types (np.int8 in the example).
// TODO(b/144230631): Return an alias instead of the aliased type.
Status DataTypeToPyArray_Descr(DataType dt, PyArray_Descr** out_descr);

// Returns a tensorflow::DataType corresponding to a given NumPy data type.
Status PyArray_DescrToDataType(PyArray_Descr* descr, DataType* out_dt);

}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_LIB_CORE_NDARRAY_TENSOR_TYPES_H_
