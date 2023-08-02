/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_PYTHON_LIB_CORE_BFLOAT16_H_
#define TENSORFLOW_TSL_PYTHON_LIB_CORE_BFLOAT16_H_

#include <Python.h>

#include "tensorflow/tsl/python/lib/core/ml_dtypes.h"

namespace tsl {

// Deprecated, use underlying methods.
inline PyObject* Bfloat16Dtype() { return ml_dtypes::GetBfloat16Dtype(); }

inline int Bfloat16NumpyType() { return ml_dtypes::GetBfloat16TypeNum(); }

inline bool RegisterNumpyBfloat16() { return ml_dtypes::RegisterTypes(); }

inline PyObject* Float8_E4M3B11Dtype() {
  return tsl::ml_dtypes::GetFloat8E4m3b11fnuzDtype();
}

}  // namespace tsl

#endif  // TENSORFLOW_TSL_PYTHON_LIB_CORE_BFLOAT16_H_
