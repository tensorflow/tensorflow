/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_FLOAT8_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_FLOAT8_H_

#include "ml_dtypes/include/float8.h"  // from @ml_dtypes_py

namespace tflite {
namespace float8_internal {

using Float8E4M3FN = ml_dtypes::float8_e4m3fn;
using Float8E5M2 = ml_dtypes::float8_e5m2;

}  // namespace float8_internal
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_FLOAT8_H_
