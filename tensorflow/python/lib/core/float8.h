/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_PYTHON_LIB_CORE_FLOAT8_H_
#define TENSORFLOW_PYTHON_LIB_CORE_FLOAT8_H_

#include <Python.h>

namespace tensorflow {

// Register the float8_e4m3fn numpy type. Returns true on success.
bool RegisterNumpyFloat8e4m3fn();

// Returns a pointer to the float8_e4m3fn dtype object.
PyObject* Float8e4m3fnDtype();

// Returns the id number of the float8_e4m3fn numpy type.
int Float8e4m3fnNumpyType();

// Register the float8_e5m2 numpy type. Returns true on success.
bool RegisterNumpyFloat8e5m2();

// Returns a pointer to the float8_e5m2 dtype object.
PyObject* Float8e5m2Dtype();

// Returns the id number of the float8_e5m2 numpy type.
int Float8e5m2NumpyType();

}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_LIB_CORE_FLOAT8_H_
