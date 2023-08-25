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

#ifndef TENSORFLOW_TSL_PLATFORM_FLOAT8_H_
#define TENSORFLOW_TSL_PLATFORM_FLOAT8_H_

#include "include/float8.h"  // from @ml_dtypes

namespace tsl {
using float8_e4m3fn = ml_dtypes::float8_e4m3fn;
using float8_e4m3fnuz = ml_dtypes::float8_e4m3fnuz;
using float8_e4m3b11fnuz = ml_dtypes::float8_e4m3b11fnuz;
// Deprecated: old name for backward-compatibility only.
using float8_e4m3b11 = float8_e4m3b11fnuz;
using float8_e5m2 = ml_dtypes::float8_e5m2;
using float8_e5m2fnuz = ml_dtypes::float8_e5m2fnuz;
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_FLOAT8_H_
