/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_PROFILER_UTILS_FORMAT_UTILS_H_
#define TENSORFLOW_TSL_PROFILER_UTILS_FORMAT_UTILS_H_

#include <stdio.h>

#include <string>

#include "tsl/platform/logging.h"

namespace tsl {
namespace profiler {
namespace internal {

inline std::string FormatDouble(const char* fmt, double d) {
  constexpr int kBufferSize = 32;
  char buffer[kBufferSize];
  int result = snprintf(buffer, kBufferSize, fmt, d);
  DCHECK(result > 0 && result < kBufferSize);
  return std::string(buffer);
}

}  // namespace internal

// Formats d with one digit after the decimal point.
inline std::string OneDigit(double d) {
  return internal::FormatDouble("%.1f", d);
}

// Formats d with 2 digits after the decimal point.
inline std::string TwoDigits(double d) {
  return internal::FormatDouble("%.2f", d);
}

// Formats d with 3 digits after the decimal point.
inline std::string ThreeDigits(double d) {
  return internal::FormatDouble("%.3f", d);
}

// Formats d with maximum precision to allow parsing the result back to the same
// number.
inline std::string MaxPrecision(double d) {
  return internal::FormatDouble("%.17g", d);
}

}  // namespace profiler
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_UTILS_FORMAT_UTILS_H_
