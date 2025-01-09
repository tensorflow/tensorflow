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

#ifndef TENSORFLOW_TSL_PLATFORM_ML_DTYPES_H_
#define TENSORFLOW_TSL_PLATFORM_ML_DTYPES_H_

#include "ml_dtypes/include/float8.h"  // from @ml_dtypes
#include "ml_dtypes/include/intn.h"  // from @ml_dtypes

namespace tsl {
using float8_e3m4 = ::ml_dtypes::float8_e3m4;
using float8_e4m3 = ::ml_dtypes::float8_e4m3;
using float8_e4m3fn = ::ml_dtypes::float8_e4m3fn;
using float8_e4m3fnuz = ::ml_dtypes::float8_e4m3fnuz;
using float8_e4m3b11fnuz = ::ml_dtypes::float8_e4m3b11fnuz;
using float8_e5m2 = ::ml_dtypes::float8_e5m2;
using float8_e5m2fnuz = ::ml_dtypes::float8_e5m2fnuz;

using int1 = ::ml_dtypes::int1;
using uint1 = ::ml_dtypes::uint1;
using int2 = ::ml_dtypes::int2;
using uint2 = ::ml_dtypes::uint2;
using int4 = ::ml_dtypes::int4;
using uint4 = ::ml_dtypes::uint4;
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_ML_DTYPES_H_
