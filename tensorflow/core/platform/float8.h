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

#ifndef TENSORFLOW_CORE_PLATFORM_FLOAT8_H_
#define TENSORFLOW_CORE_PLATFORM_FLOAT8_H_

#include "tsl/platform/ml_dtypes.h"

namespace tensorflow {
typedef tsl::float8_e4m3fn float8_e4m3fn;
typedef tsl::float8_e5m2 float8_e5m2;
typedef tsl::float8_e4m3fnuz float8_e4m3fnuz;
typedef tsl::float8_e4m3b11fnuz float8_e4m3b11fnuz;
typedef tsl::float8_e5m2fnuz float8_e5m2fnuz;
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_FLOAT8_H_
