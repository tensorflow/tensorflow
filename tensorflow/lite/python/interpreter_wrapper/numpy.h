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
#ifndef TENSORFLOW_LITE_PYTHON_INTERPRETER_WRAPPER_NUMPY_H_
#define TENSORFLOW_LITE_PYTHON_INTERPRETER_WRAPPER_NUMPY_H_

#ifdef PyArray_Type
#error "Numpy cannot be included before numpy.h."
#endif

// Disallow Numpy 1.7 deprecated symbols.
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// To handle PyArray_* calles, numpy defines a static lookup table called
// PyArray_API, or PY_ARRAY_UNIQUE_SYMBOL, if defined. This causes the
// PyArray_* pointers to be different for different translation units, unless
// we take care of selectivel defined NO_IMPORT_ARRAY.
//
// Virtually every usage will define NO_IMPORT_ARRAY, and will have access to
// the lookup table via:
//   extern void **PyArray_API;
// In numpy.cc we will define TFLITE_IMPORT_NUMPY, effectively disabling that
// and instead using:
//   void **PyArray_API;
// which is initialized when ImportNumpy() is called.
//
// If we don't define PY_ARRAY_UNIQUE_SYMBOL then PyArray_API is a static
// variable, which causes strange crashes when the pointers are used across
// translation unit boundaries.
//
// For mone info see https://sourceforge.net/p/numpy/mailman/message/5700519
// See also tensorflow/python/lib/core/numpy.h for a similar approach.
#define PY_ARRAY_UNIQUE_SYMBOL _tensorflow_numpy_api
#ifndef TFLITE_IMPORT_NUMPY
#define NO_IMPORT_ARRAY
#endif

#include <Python.h>

#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace python {

void ImportNumpy();

}  // namespace python

namespace python_utils {

int TfLiteTypeToPyArrayType(TfLiteType tf_lite_type);

TfLiteType TfLiteTypeFromPyType(int py_type);

TfLiteType TfLiteTypeFromPyArray(PyArrayObject* array);

bool FillStringBufferWithPyArray(PyObject* value,
                                 DynamicBuffer* dynamic_buffer);

}  // namespace python_utils
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PYTHON_INTERPRETER_WRAPPER_NUMPY_H_
