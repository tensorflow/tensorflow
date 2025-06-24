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

#ifndef XLA_TSL_PYTHON_LIB_CORE_ML_DTYPES_H_
#define XLA_TSL_PYTHON_LIB_CORE_ML_DTYPES_H_

// Registers all custom types from the python ml_dtypes package.
//   https://github.com/jax-ml/ml_dtypes

namespace tsl {
namespace ml_dtypes {

struct NumpyDtypes {
  int bfloat16;
  int float4_e2m1fn;
  int float8_e3m4;
  int float8_e4m3;
  int float8_e4m3fn;
  int float8_e4m3b11fnuz;
  int float8_e4m3fnuz;
  int float8_e5m2;
  int float8_e5m2fnuz;
  int float8_e8m0fnu;
  int int4;
  int uint4;
  int int2;
  int uint2;
};

// RegisterTypes imports the ml_dtypes module. It should be called before using
// the functions below, and it fails (by returning false) if there was an error
// importing that module. If the build system guarantees that the module exists,
// the call can be omitted, since it is implied by the functions below.
bool RegisterTypes();

// Implicitly calls RegisterTypes on first use.
const NumpyDtypes& GetNumpyDtypes();

inline int GetBfloat16TypeNum() { return GetNumpyDtypes().bfloat16; }

}  // namespace ml_dtypes
}  // namespace tsl

#endif  // XLA_TSL_PYTHON_LIB_CORE_ML_DTYPES_H_
