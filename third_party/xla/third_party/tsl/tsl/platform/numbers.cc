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

#include "tsl/platform/numbers.h"

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <system_error>  // NOLINT
#include <type_traits>

#include "absl/strings/charconv.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/logging.h"

namespace tsl {

namespace {

template <typename T>
std::optional<T> AsciiToFp(absl::string_view str) {
  T value;
  absl::from_chars_result result =
      absl::from_chars(str.data(), str.data() + str.size(), value);
  if (result.ec != std::errc{}) {
    return std::nullopt;
  }
  if (result.ptr != str.data() + str.size()) {
    // Not all characters consumed.
    return std::nullopt;
  }
  return value;
}

template <typename T>
size_t FastUIntToBufferLeft(T i, char* buffer) {
  static_assert(std::is_unsigned_v<T>);
  char* start = buffer;
  do {
    *buffer++ = ((i % 10) + '0');
    i /= 10;
  } while (i > 0);
  *buffer = 0;
  std::reverse(start, buffer);
  return buffer - start;
}

template <typename T>
size_t FastIntToBufferLeft(T i, char* buffer) {
  static_assert(std::is_signed_v<T>);
  std::make_unsigned_t<T> u = i;
  size_t length = 0;
  if (i < 0) {
    *buffer++ = '-';
    ++length;
    // We need to do the negation in modular (i.e., "unsigned")
    // arithmetic; MSVC++ apparently warns for plain "-u", so
    // we write the equivalent expression "0 - u" instead.
    u = 0 - u;
  }
  length += FastUIntToBufferLeft(u, buffer);
  return length;
}
}  // namespace

namespace strings {

size_t FastInt32ToBufferLeft(int32_t i, char* buffer) {
  return FastIntToBufferLeft(i, buffer);
}

size_t FastUInt32ToBufferLeft(uint32_t i, char* buffer) {
  return FastUIntToBufferLeft(i, buffer);
}

size_t FastInt64ToBufferLeft(int64_t i, char* buffer) {
  return FastIntToBufferLeft(i, buffer);
}

size_t FastUInt64ToBufferLeft(uint64_t i, char* buffer) {
  return FastUIntToBufferLeft(i, buffer);
}

std::string FpToString(uint64_t fp) {
  return absl::StrCat(absl::Hex(fp, absl::kZeroPad16));
}

bool HexStringToUint64(absl::string_view s, uint64_t* result) {
  auto end_ptr = s.data() + s.size();
  uint64_t parsed_result;
  auto [ptr, ec] =
      std::from_chars(s.data(), end_ptr, parsed_result, /*base=*/16);
  if (ec != std::errc{}) {
    return false;
  }
  if (ptr != end_ptr) {
    return false;
  }
  *result = parsed_result;
  return true;
}

std::string HumanReadableNum(int64_t signed_value) {
  std::string s;

  uint64_t value = static_cast<uint64_t>(signed_value);
  if (signed_value < 0) {
    s = "-";
    value = -value;
  }
  if (value < 1000) {
    absl::StrAppendFormat(&s, "%d", value);
  } else if (value >= static_cast<int64_t>(1e15)) {
    // Number bigger than 1E15; use that notation.
    absl::StrAppendFormat(&s, "%0.3G", static_cast<double>(value));
  } else {
    static absl::string_view kUnits = "kMBT";
    auto unit = kUnits.begin();
    while (value >= static_cast<int64_t>(1000000)) {
      value /= static_cast<int64_t>(1000);
      ++unit;
      CHECK(unit < kUnits.end());
    }
    absl::StrAppendFormat(&s, "%.2f%c", value / 1000.0, *unit);
  }
  return s;
}

std::string HumanReadableNumBytes(int64_t signed_num_bytes) {
  static absl::string_view kNegSign = "-";
  absl::string_view sign_str;
  uint64_t num_bytes = static_cast<uint64_t>(signed_num_bytes);
  if (signed_num_bytes < 0) {
    num_bytes = -num_bytes;
    sign_str = kNegSign;
  }

  // Special case for bytes.
  if (num_bytes < 1024) {
    // No fractions for bytes.
    return absl::StrCat(sign_str, num_bytes, "B");
  }

  static absl::string_view kUnits = "KMGTPE";  // int64 only goes up to E.
  auto unit = kUnits.begin();
  while (num_bytes >= static_cast<int64_t>(1024) * 1024) {
    num_bytes /= 1024;
    ++unit;
    CHECK(unit < kUnits.end());
  }

  // We use SI prefixes.
  return absl::StrFormat("%s%.*f%ciB", sign_str, *unit == 'K' ? 1 : 2,
                         num_bytes / 1024.0, *unit);
}

std::string HumanReadableElapsedTime(double seconds) {
  std::string human_readable;

  if (seconds < 0) {
    human_readable = "-";
    seconds = -seconds;
  }

  // Start with us and keep going up to years.
  // The comparisons must account for rounding to prevent the format breaking
  // the tested condition and returning, e.g., "1e+03 us" instead of "1 ms".
  const double microseconds = seconds * 1.0e6;
  if (microseconds < 999.5) {
    absl::StrAppendFormat(&human_readable, "%0.3g us", microseconds);
    return human_readable;
  }
  double milliseconds = seconds * 1e3;
  if (milliseconds >= .995 && milliseconds < 1) {
    // Round half to even in StrAppendFormat would convert this to 0.999 ms.
    milliseconds = 1.0;
  }
  if (milliseconds < 999.5) {
    absl::StrAppendFormat(&human_readable, "%0.3g ms", milliseconds);
    return human_readable;
  }
  if (seconds < 60.0) {
    absl::StrAppendFormat(&human_readable, "%0.3g s", seconds);
    return human_readable;
  }
  seconds /= 60.0;
  if (seconds < 60.0) {
    absl::StrAppendFormat(&human_readable, "%0.3g min", seconds);
    return human_readable;
  }
  seconds /= 60.0;
  if (seconds < 24.0) {
    absl::StrAppendFormat(&human_readable, "%0.3g h", seconds);
    return human_readable;
  }
  seconds /= 24.0;
  if (seconds < 30.0) {
    absl::StrAppendFormat(&human_readable, "%0.3g days", seconds);
    return human_readable;
  }
  if (seconds < 365.2425) {
    absl::StrAppendFormat(&human_readable, "%0.3g months", seconds / 30.436875);
    return human_readable;
  }
  seconds /= 365.2425;
  absl::StrAppendFormat(&human_readable, "%0.3g years", seconds);
  return human_readable;
}

}  // namespace strings
}  // namespace tsl
