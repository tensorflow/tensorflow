/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_PYTHON_LIB_CORE_PY_FUNC_H_
#define TENSORFLOW_PYTHON_LIB_CORE_PY_FUNC_H_

// Must be included first
#include "tensorflow/python/lib/core/numpy.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Called by py code on initialization.
//
// "trampoline" must represent a python function which has the
// following signature:
//   (string, list(ndarray)) -> ndarray | list(ndarray) | python scalar
//
// The trampoline takes two arguments, the first is a string token
// used by the python frontend's dispatching logic; the second is a
// list of numpy ndarrays.
//
// The trampoline can return a single numpy ndarray, a list of numpy
// ndarrays, or a simply python scalar. The C++ runtime converts them,
// if supported, back to Tensor objects.
//
// This is called by script_ops.py during its module initialization.
//
// TODO(zhifengc): Support distributed runtime.
void InitializePyTrampoline(PyObject* trampoline);

// Creates a numpy array in 'ret' and copies the content of tensor 't'
// into 'ret'.
Status ConvertTensorToNdarray(const Tensor& t, PyObject** ret);

}  // end namespace tensorflow

#endif  // TENSORFLOW_PYTHON_LIB_CORE_PY_FUNC_H_
