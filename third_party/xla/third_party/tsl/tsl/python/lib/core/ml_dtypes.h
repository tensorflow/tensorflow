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

#ifndef TENSORFLOW_TSL_PYTHON_LIB_CORE_ML_DTYPES_H_
#define TENSORFLOW_TSL_PYTHON_LIB_CORE_ML_DTYPES_H_

// Registers all custom types from the python ml_dtypes.py package.
//   https://github.com/jax-ml/ml_dtypes

#include <Python.h>

namespace tsl {
namespace ml_dtypes {

// Register all ml dtypes.
bool RegisterTypes();

// Return a pointer to the numpy dtype objects.
PyObject* GetBfloat16Dtype();
PyObject* GetFloat8E4m3b11fnuzDtype();
PyObject* GetFloat8E4m3fnDtype();
PyObject* GetFloat8E5m2Dtype();

// Returns the type id number of the numpy type.
int GetBfloat16TypeNum();
int GetFloat8E4m3b11fnuzTypeNum();
int GetFloat8E4m3fnTypeNum();
int GetFloat8E5m2TypeNum();

}  // namespace ml_dtypes
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PYTHON_LIB_CORE_ML_DTYPES_H_
