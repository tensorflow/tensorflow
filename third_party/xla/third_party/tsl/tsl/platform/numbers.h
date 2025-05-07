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

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/base/macros.h"
#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/types.h"
#include "tsl/platform/stringpiece.h"

namespace tsl {
namespace strings {

// ----------------------------------------------------------------------
// FastIntToBufferLeft()
//    These are intended for speed.
//
//    All functions take the output buffer as an arg.  FastInt() uses
//    at most 22 bytes, FastTime() uses exactly 30 bytes.  They all
//    return a pointer to the beginning of the output, which is the same as
//    the beginning of the input buffer.
//
//    NOTE: In 64-bit land, sizeof(time_t) is 8, so it is possible
//    to pass to FastTimeToBuffer() a time whose year cannot be
//    represented in 4 digits. In this case, the output buffer
//    will contain the string "Invalid:<value>"
// ----------------------------------------------------------------------

// Previously documented minimums -- the buffers provided must be at least this
// long, though these numbers are subject to change:
//     Int32, UInt32:                   12 bytes
//     Int64, UInt64, Int, Uint:        22 bytes
//     Time:                            30 bytes
// Use kFastToBufferSize rather than hardcoding constants.
inline constexpr int kFastToBufferSize = 32;

// ----------------------------------------------------------------------
// FastInt32ToBufferLeft()
// FastUInt32ToBufferLeft()
// FastInt64ToBufferLeft()
// FastUInt64ToBufferLeft()
//
// These functions convert their numeric argument to an ASCII
// representation of the numeric value in base 10, with the
// representation being left-aligned in the buffer.  The caller is
// responsible for ensuring that the buffer has enough space to hold
// the output.  The buffer should typically be at least kFastToBufferSize
// bytes.
//
// Returns the number of characters written.
// ----------------------------------------------------------------------

size_t FastInt32ToBufferLeft(int32_t i, char* buffer);  // at least 12 bytes
size_t FastUInt32ToBufferLeft(uint32_t i, char* buffer);  // at least 12 bytes
size_t FastInt64ToBufferLeft(int64_t i, char* buffer);  // at least 22 bytes
size_t FastUInt64ToBufferLeft(uint64_t i, char* buffer);  // at least 22 bytes

// Required buffer size for DoubleToBuffer is kFastToBufferSize.
// Required buffer size for FloatToBuffer is kFastToBufferSize.
size_t DoubleToBuffer(double value, char* buffer);
size_t FloatToBuffer(float value, char* buffer);

namespace strings_internal {
// AlphaNumBuffer allows a way to pass a string to absl::StrCat without having
// to do memory allocation. It is simply a pair of a fixed-size character
// array, and a size.  Please don't use outside of the "strings" package.
struct AlphaNumBuffer {
  std::array<char, kFastToBufferSize> data;
  size_t size;

  // Support for absl::StrCat() etc.
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const AlphaNumBuffer& buffer) {
    absl::Format(&sink, "%s",
                 absl::string_view(buffer.data.data(), buffer.size));
  }
};
}  // namespace strings_internal

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
strings_internal::AlphaNumBuffer LegacyPrecision(float f);

// Helper function for the old strings::StrCat default "double" format, which
// was either %.15g or %.17g, depending on whether the %.15g format would
// round-trip back to the original double.  This is approx. 20-30x slower than
// the others.
strings_internal::AlphaNumBuffer LegacyPrecision(double d);

// Convert a 64-bit fingerprint value to an ASCII representation.
std::string FpToString(Fprint fp);

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

inline bool ProtoParseNumeric(absl::string_view s, int32_t* value) {
  return absl::SimpleAtoi(s, value);
}

inline bool ProtoParseNumeric(absl::string_view s, uint32_t* value) {
  return absl::SimpleAtoi(s, value);
}

inline bool ProtoParseNumeric(absl::string_view s, int64_t* value) {
  return absl::SimpleAtoi(s, value);
}

inline bool ProtoParseNumeric(absl::string_view s, uint64_t* value) {
  return absl::SimpleAtoi(s, value);
}

inline bool ProtoParseNumeric(absl::string_view s, float* value) {
  return absl::SimpleAtof(s, value);
}

inline bool ProtoParseNumeric(absl::string_view s, double* value) {
  return absl::SimpleAtod(s, value);
}

// Convert strings to number of type T.
// Leading and trailing spaces are allowed.
// Values may be rounded on over- and underflow.
template <typename T>
bool SafeStringToNumeric(absl::string_view s, T* value) {
  return ProtoParseNumeric(s, value);
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
