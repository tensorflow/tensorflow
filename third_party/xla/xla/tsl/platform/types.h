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

#ifndef XLA_TSL_PLATFORM_TYPES_H_
#define XLA_TSL_PLATFORM_TYPES_H_

#include <cstdint>
#include <limits>
#include <string>

#include "absl/base/const_init.h"
#include "absl/base/macros.h"
#include "tsl/platform/bfloat16.h"  // IWYU pragma: export
#include "tsl/platform/ml_dtypes.h"  // IWYU pragma: export
#include "tsl/platform/tstring.h"

namespace tsl {

// Alias tsl::string to std::string.
using string ABSL_DEPRECATE_AND_INLINE() = std::string;
using uint8 ABSL_DEPRECATE_AND_INLINE() = uint8_t;
using uint16 ABSL_DEPRECATE_AND_INLINE() = uint16_t;
using uint32 ABSL_DEPRECATE_AND_INLINE() = uint32_t;
using uint64 ABSL_DEPRECATE_AND_INLINE() = uint64_t;
using int8 ABSL_DEPRECATE_AND_INLINE() = int8_t;
using int16 ABSL_DEPRECATE_AND_INLINE() = int16_t;
using int32 ABSL_DEPRECATE_AND_INLINE() = int32_t;
using int64 ABSL_DEPRECATE_AND_INLINE() = int64_t;

// A typedef for a uint64 used as a short fingerprint.
using Fprint = uint64_t;

}  // namespace tsl

// Alias namespace ::stream_executor as ::tensorflow::se.
namespace stream_executor {}
namespace tensorflow {
namespace se = ::stream_executor;
}  // namespace tensorflow

#if defined(PLATFORM_WINDOWS)
#include <cstddef>
typedef std::ptrdiff_t ssize_t;
#endif

#endif  // XLA_TSL_PLATFORM_TYPES_H_
