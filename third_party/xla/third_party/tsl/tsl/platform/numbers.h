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

#ifndef TENSORFLOW_TSL_PLATFORM_NUMBERS_H_
#define TENSORFLOW_TSL_PLATFORM_NUMBERS_H_

#include <cstdint>
#include <string>
#include <type_traits>

#include "absl/base/macros.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace tsl {
namespace strings {

// Helper function for legacy google formatting.
template <typename T>
const T& LegacyPrecision(const T& t) {
  return t;
}

// Have to use overloads rather than specialization because specialization can't
// change the function return type.

// Helper function for the old strings::StrCat default "float" format, which was
// either %.6g, %.7g, %.8g, or %.9g, basically the smallest string that would
// round-trip back to the original float. This is fast.
[[deprecated(
    "Use absl::HighPrecision instead")]] ABSL_REFACTOR_INLINE inline auto
LegacyPrecision(float f) {
  return absl::HighPrecision(f);
}

// Helper function for the old strings::StrCat default "double" format, which
// was either %.15g or %.17g, depending on whether the %.15g format would
// round-trip back to the original double.  This is approx. 20-30x slower than
// the others.
[[deprecated(
    "Use absl::HighPrecision instead")]] ABSL_REFACTOR_INLINE inline auto
LegacyPrecision(double d) {
  return absl::HighPrecision(d);
}

// Convert a 64-bit fingerprint value to an ASCII representation.
std::string FpToString(uint64_t fp);

// Attempt to parse a `uint64_t` in the form encoded by
// `absl::StrCat(absl::Hex(*result))`.  If successful, stores the value in
// `result` and returns true.  Otherwise, returns false.
bool HexStringToUint64(absl::string_view s, uint64_t* result);

// Convert strings to 32bit integer values.
// Leading and trailing spaces are allowed.
// Return false with overflow or invalid input.
ABSL_DEPRECATE_AND_INLINE()
inline bool safe_strto32(absl::string_view str, int32_t* value) {
  return absl::SimpleAtoi(str, value);
}

// Convert strings to unsigned 32bit integer values.
// Leading and trailing spaces are allowed.
// Return false with overflow or invalid input.
ABSL_DEPRECATE_AND_INLINE()
inline bool safe_strtou32(absl::string_view str, uint32_t* value) {
  return absl::SimpleAtoi(str, value);
}

// Convert strings to 64bit integer values.
// Leading and trailing spaces are allowed.
// Return false with overflow or invalid input.
ABSL_DEPRECATE_AND_INLINE()
inline bool safe_strto64(absl::string_view str, int64_t* value) {
  return absl::SimpleAtoi(str, value);
}

// Convert strings to unsigned 64bit integer values.
// Leading and trailing spaces are allowed.
// Return false with overflow or invalid input.
ABSL_DEPRECATE_AND_INLINE()
inline bool safe_strtou64(absl::string_view str, uint64_t* value) {
  return absl::SimpleAtoi(str, value);
}

// Convert strings to floating point values.
// Leading and trailing spaces are allowed.
// Values may be rounded on over- and underflow.
// Returns false on invalid input or if `strlen(value) >= kFastToBufferSize`.
ABSL_DEPRECATE_AND_INLINE()
inline bool safe_strtof(absl::string_view str, float* value) {
  return absl::SimpleAtof(str, value);
}

// Convert strings to double precision floating point values.
// Leading and trailing spaces are allowed.
// Values may be rounded on over- and underflow.
// Returns false on invalid input or if `strlen(value) >= kFastToBufferSize`.
ABSL_DEPRECATE_AND_INLINE()
inline bool safe_strtod(absl::string_view str, double* value) {
  return absl::SimpleAtod(str, value);
}

// Convert strings to number of type T.
// Leading and trailing spaces are allowed.
// Values may be rounded on over- and underflow.
template <typename T>
bool SafeStringToNumeric(absl::string_view s, T* value) {
  if constexpr (std::is_integral_v<T>) {
    return absl::SimpleAtoi(s, value);
  } else if constexpr (std::is_same_v<T, float>) {
    return absl::SimpleAtof(s, value);
  } else {
    static_assert(std::is_same_v<T, double>);
    return absl::SimpleAtod(s, value);
  }
}

// Converts from an int64 to a human readable string representing the
// same number, using decimal powers.  e.g. 1200000 -> "1.20M".
std::string HumanReadableNum(int64_t value);

// Converts from an int64 representing a number of bytes to a
// human readable string representing the same number.
// e.g. 12345678 -> "11.77MiB".
std::string HumanReadableNumBytes(int64_t num_bytes);

// Converts a time interval as double to a human readable
// string. For example:
//   0.001       -> "1 ms"
//   10.0        -> "10 s"
//   933120.0    -> "10.8 days"
//   39420000.0  -> "1.25 years"
//   -10         -> "-10 s"
std::string HumanReadableElapsedTime(double seconds);

}  // namespace strings
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_NUMBERS_H_
