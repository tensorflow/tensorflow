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

#include "tensorflow/lite/python/interpreter_wrapper/python_utils.h"

namespace tflite {
namespace python_utils {

int TfLiteTypeToPyArrayType(TfLiteType tf_lite_type) {
  switch (tf_lite_type) {
    case kTfLiteFloat32:
      return NPY_FLOAT32;
    case kTfLiteInt32:
      return NPY_INT32;
    case kTfLiteInt16:
      return NPY_INT16;
    case kTfLiteUInt8:
      return NPY_UINT8;
    case kTfLiteInt8:
      return NPY_INT8;
    case kTfLiteInt64:
      return NPY_INT64;
    case kTfLiteString:
      return NPY_OBJECT;
    case kTfLiteBool:
      return NPY_BOOL;
    case kTfLiteComplex64:
      return NPY_COMPLEX64;
    case kTfLiteNoType:
      return NPY_NOTYPE;
      // Avoid default so compiler errors created when new types are made.
  }
  return NPY_NOTYPE;
}

TfLiteType TfLiteTypeFromPyArray(PyArrayObject* array) {
  int pyarray_type = PyArray_TYPE(array);
  switch (pyarray_type) {
    case NPY_FLOAT32:
      return kTfLiteFloat32;
    case NPY_INT32:
      return kTfLiteInt32;
    case NPY_INT16:
      return kTfLiteInt16;
    case NPY_UINT8:
      return kTfLiteUInt8;
    case NPY_INT8:
      return kTfLiteInt8;
    case NPY_INT64:
      return kTfLiteInt64;
    case NPY_BOOL:
      return kTfLiteBool;
    case NPY_OBJECT:
    case NPY_STRING:
    case NPY_UNICODE:
      return kTfLiteString;
    case NPY_COMPLEX64:
      return kTfLiteComplex64;
      // Avoid default so compiler errors created when new types are made.
  }
  return kTfLiteNoType;
}

}  // namespace python_utils
}  // namespace tflite
