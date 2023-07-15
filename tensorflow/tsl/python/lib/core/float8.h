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

#ifndef TENSORFLOW_TSL_PYTHON_LIB_CORE_FLOAT8_H_
#define TENSORFLOW_TSL_PYTHON_LIB_CORE_FLOAT8_H_

#include <Python.h>

#include "tensorflow/tsl/python/lib/core/ml_dtypes.h"

namespace tsl {

// Deprecated, use underlying methods.
inline bool RegisterNumpyFloat8e4m3fn() { return ml_dtypes::RegisterTypes(); }

inline PyObject* Float8e4m3fnDtype() {
  return ml_dtypes::GetFloat8E4m3fnDtype();
}

inline int Float8e4m3fnNumpyType() {
  return ml_dtypes::GetFloat8E4m3fnTypeNum();
}

inline bool RegisterNumpyFloat8e5m2() { return ml_dtypes::RegisterTypes(); }

inline PyObject* Float8e5m2Dtype() { return ml_dtypes::GetFloat8E5m2Dtype(); }

inline int Float8e5m2NumpyType() { return ml_dtypes::GetFloat8E5m2TypeNum(); }

}  // namespace tsl

#endif  // TENSORFLOW_TSL_PYTHON_LIB_CORE_FLOAT8_H_
