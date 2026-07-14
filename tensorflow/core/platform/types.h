/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_TYPES_H_
#define TENSORFLOW_CORE_PLATFORM_TYPES_H_

#include <cstdint>
#include <string>

#include "absl/base/macros.h"
#include "xla/tsl/platform/types.h"
#include "tensorflow/core/platform/bfloat16.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/tstring.h"
#include "tsl/platform/bfloat16.h"
#include "tsl/platform/tstring.h"

namespace tensorflow {

// Alias tensorflow::string to std::string.
using string ABSL_DEPRECATE_AND_INLINE() = std::string;

using tsl::uint2;
using tsl::uint4;
using uint8 ABSL_DEPRECATE_AND_INLINE() = uint8_t;
using uint16 ABSL_DEPRECATE_AND_INLINE() = uint16_t;
using uint32 ABSL_DEPRECATE_AND_INLINE() = uint32_t;
using uint64 ABSL_DEPRECATE_AND_INLINE() = uint64_t;

using tsl::int2;
using tsl::int4;
using int8 ABSL_DEPRECATE_AND_INLINE() = int8_t;
using int16 ABSL_DEPRECATE_AND_INLINE() = int16_t;
using int32 ABSL_DEPRECATE_AND_INLINE() = int32_t;
using int64 ABSL_DEPRECATE_AND_INLINE() = int64_t;

using tsl::float4_e2m1fn;
using tsl::float8_e4m3b11fnuz;
using tsl::float8_e4m3fn;
using tsl::float8_e4m3fnuz;
using tsl::float8_e5m2;
using tsl::float8_e5m2fnuz;

using tsl::kint16max;
using tsl::kint16min;
using tsl::kint32max;
using tsl::kint32min;
using tsl::kint64max;
using tsl::kint64min;
using tsl::kint8max;
using tsl::kint8min;
using tsl::kuint16max;
using tsl::kuint32max;
using tsl::kuint64max;
using tsl::kuint8max;

// A typedef for a uint64 used as a short fingerprint.
using tsl::bfloat16;
using tsl::Fprint;
using tsl::tstring;  // NOLINT: suppress 'using decl 'tstring' is unused'
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_TYPES_H_
