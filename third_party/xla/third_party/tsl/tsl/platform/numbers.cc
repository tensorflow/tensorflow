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

#include <ctype.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <locale>
#include <string>
#include <system_error>  // NOLINT
#include <unordered_map>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/platform/types.h"
#include "tsl/platform/stringprintf.h"

namespace tsl {

namespace {

template <typename T>
const std::unordered_map<std::string, T>* GetSpecialNumsSingleton() {
  static const std::unordered_map<std::string, T>* special_nums =
      CHECK_NOTNULL((new const std::unordered_map<std::string, T>{
          {"inf", std::numeric_limits<T>::infinity()},
          {"+inf", std::numeric_limits<T>::infinity()},
          {"-inf", -std::numeric_limits<T>::infinity()},
          {"infinity", std::numeric_limits<T>::infinity()},
          {"+infinity", std::numeric_limits<T>::infinity()},
          {"-infinity", -std::numeric_limits<T>::infinity()},
          {"nan", std::numeric_limits<T>::quiet_NaN()},
          {"+nan", std::numeric_limits<T>::quiet_NaN()},
          {"-nan", -std::numeric_limits<T>::quiet_NaN()},
      }));
  return special_nums;
}

template <typename T>
T locale_independent_strtonum(const char* str, const char** endptr) {
  auto special_nums = GetSpecialNumsSingleton<T>();
  std::stringstream s(str);

  // Check if str is one of the special numbers.
  std::string special_num_str;
  s >> special_num_str;

  for (size_t i = 0; i < special_num_str.length(); ++i) {
    special_num_str[i] =
        std::tolower(special_num_str[i], std::locale::classic());
  }

  auto entry = special_nums->find(special_num_str);
  if (entry != special_nums->end()) {
    *endptr = str + (s.eof() ? static_cast<std::iostream::pos_type>(strlen(str))
                             : s.tellg());
    return entry->second;
  } else {
    // Perhaps it's a hex number
    if (special_num_str.compare(0, 2, "0x") == 0 ||
        special_num_str.compare(0, 3, "-0x") == 0) {
      return strtol(str, const_cast<char**>(endptr), 16);
    }
  }
  // Reset the stream
  s.str(str);
  s.clear();
  // Use the "C" locale
  s.imbue(std::locale::classic());

  T result;
  s >> result;

  // Set to result to what strto{f,d} functions would have returned. If the
  // number was outside the range, the stringstream sets the fail flag, but
  // returns the +/-max() value, whereas strto{f,d} functions return +/-INF.
  if (s.fail()) {
    if (result == std::numeric_limits<T>::max() ||
        result == std::numeric_limits<T>::infinity()) {
      result = std::numeric_limits<T>::infinity();
      s.clear(s.rdstate() & ~std::ios::failbit);
    } else if (result == -std::numeric_limits<T>::max() ||
               result == -std::numeric_limits<T>::infinity()) {
      result = -std::numeric_limits<T>::infinity();
      s.clear(s.rdstate() & ~std::ios::failbit);
    }
  }

  if (endptr) {
    *endptr =
        str +
        (s.fail() ? static_cast<std::iostream::pos_type>(0)
                  : (s.eof() ? static_cast<std::iostream::pos_type>(strlen(str))
                             : s.tellg()));
  }
  return result;
}

}  // namespace

namespace strings {

size_t FastInt32ToBufferLeft(int32_t i, char* buffer) {
  uint32_t u = i;
  size_t length = 0;
  if (i < 0) {
    *buffer++ = '-';
    ++length;
    // We need to do the negation in modular (i.e., "unsigned")
    // arithmetic; MSVC++ apparently warns for plain "-u", so
    // we write the equivalent expression "0 - u" instead.
    u = 0 - u;
  }
  length += FastUInt32ToBufferLeft(u, buffer);
  return length;
}

size_t FastUInt32ToBufferLeft(uint32_t i, char* buffer) {
  char* start = buffer;
  do {
    *buffer++ = ((i % 10) + '0');
    i /= 10;
  } while (i > 0);
  *buffer = 0;
  std::reverse(start, buffer);
  return buffer - start;
}

size_t FastInt64ToBufferLeft(int64_t i, char* buffer) {
  uint64_t u = i;
  size_t length = 0;
  if (i < 0) {
    *buffer++ = '-';
    ++length;
    u = 0 - u;
  }
  length += FastUInt64ToBufferLeft(u, buffer);
  return length;
}

size_t FastUInt64ToBufferLeft(uint64_t i, char* buffer) {
  char* start = buffer;
  do {
    *buffer++ = ((i % 10) + '0');
    i /= 10;
  } while (i > 0);
  *buffer = 0;
  std::reverse(start, buffer);
  return buffer - start;
}

static const double kDoublePrecisionCheckMax = DBL_MAX / 1.000000000000001;

size_t DoubleToBuffer(double value, char* buffer) {
  // DBL_DIG is 15 for IEEE-754 doubles, which are used on almost all
  // platforms these days.  Just in case some system exists where DBL_DIG
  // is significantly larger -- and risks overflowing our buffer -- we have
  // this assert.
  static_assert(DBL_DIG < 20, "DBL_DIG is too big");

  if (std::isnan(value)) {
    int snprintf_result = snprintf(buffer, kFastToBufferSize, "%snan",
                                   std::signbit(value) ? "-" : "");
    // Paranoid check to ensure we don't overflow the buffer.
    DCHECK(snprintf_result > 0 && snprintf_result < kFastToBufferSize);
    return snprintf_result;
  }

  if (std::abs(value) <= kDoublePrecisionCheckMax) {
    int snprintf_result =
        snprintf(buffer, kFastToBufferSize, "%.*g", DBL_DIG, value);

    // The snprintf should never overflow because the buffer is significantly
    // larger than the precision we asked for.
    DCHECK(snprintf_result > 0 && snprintf_result < kFastToBufferSize);

    if (locale_independent_strtonum<double>(buffer, nullptr) == value) {
      // Round-tripping the string to double works; we're done.
      return snprintf_result;
    }
    // else: full precision formatting needed. Fall through.
  }

  int snprintf_result =
      snprintf(buffer, kFastToBufferSize, "%.*g", DBL_DIG + 2, value);

  // Should never overflow; see above.
  DCHECK(snprintf_result > 0 && snprintf_result < kFastToBufferSize);

  return snprintf_result;
}

size_t FloatToBuffer(float value, char* buffer) {
  // FLT_DIG is 6 for IEEE-754 floats, which are used on almost all
  // platforms these days.  Just in case some system exists where FLT_DIG
  // is significantly larger -- and risks overflowing our buffer -- we have
  // this assert.
  static_assert(FLT_DIG < 10, "FLT_DIG is too big");

  if (std::isnan(value)) {
    int snprintf_result = snprintf(buffer, kFastToBufferSize, "%snan",
                                   std::signbit(value) ? "-" : "");
    // Paranoid check to ensure we don't overflow the buffer.
    DCHECK(snprintf_result > 0 && snprintf_result < kFastToBufferSize);
    return snprintf_result;
  }

  int snprintf_result =
      snprintf(buffer, kFastToBufferSize, "%.*g", FLT_DIG, value);

  // The snprintf should never overflow because the buffer is significantly
  // larger than the precision we asked for.
  DCHECK(snprintf_result > 0 && snprintf_result < kFastToBufferSize);

  float parsed_value;
  if (!absl::SimpleAtof(buffer, &parsed_value) || parsed_value != value) {
    snprintf_result =
        snprintf(buffer, kFastToBufferSize, "%.*g", FLT_DIG + 3, value);

    // Should never overflow; see above.
    DCHECK(snprintf_result > 0 && snprintf_result < kFastToBufferSize);
  }
  return snprintf_result;
}

strings_internal::AlphaNumBuffer LegacyPrecision(double d) {
  strings_internal::AlphaNumBuffer result;
  result.size = DoubleToBuffer(d, result.data.data());
  return result;
}

strings_internal::AlphaNumBuffer LegacyPrecision(float f) {
  strings_internal::AlphaNumBuffer result;
  result.size = FloatToBuffer(f, result.data.data());
  return result;
}

std::string FpToString(Fprint fp) {
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

std::string HumanReadableNum(int64_t value) {
  std::string s;
  if (value < 0) {
    s += "-";
    value = -value;
  }
  if (value < 1000) {
    Appendf(&s, "%lld", static_cast<long long>(value));
  } else if (value >= static_cast<int64_t>(1e15)) {
    // Number bigger than 1E15; use that notation.
    Appendf(&s, "%0.3G", static_cast<double>(value));
  } else {
    static const char units[] = "kMBT";
    const char* unit = units;
    while (value >= static_cast<int64_t>(1000000)) {
      value /= static_cast<int64_t>(1000);
      ++unit;
      CHECK(unit < units + TF_ARRAYSIZE(units));
    }
    Appendf(&s, "%.2f%c", value / 1000.0, *unit);
  }
  return s;
}

std::string HumanReadableNumBytes(int64_t num_bytes) {
  if (num_bytes == kint64min) {
    // Special case for number with not representable negation.
    return "-8E";
  }

  const char* neg_str = (num_bytes < 0) ? "-" : "";
  if (num_bytes < 0) {
    num_bytes = -num_bytes;
  }

  // Special case for bytes.
  if (num_bytes < 1024) {
    // No fractions for bytes.
    char buf[8];  // Longest possible string is '-XXXXB'
    snprintf(buf, sizeof(buf), "%s%lldB", neg_str,
             static_cast<long long>(num_bytes));
    return std::string(buf);
  }

  static const char units[] = "KMGTPE";  // int64 only goes up to E.
  const char* unit = units;
  while (num_bytes >= static_cast<int64_t>(1024) * 1024) {
    num_bytes /= 1024;
    ++unit;
    CHECK(unit < units + TF_ARRAYSIZE(units));
  }

  // We use SI prefixes.
  char buf[16];
  snprintf(buf, sizeof(buf), ((*unit == 'K') ? "%s%.1f%ciB" : "%s%.2f%ciB"),
           neg_str, num_bytes / 1024.0, *unit);
  return std::string(buf);
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
    strings::Appendf(&human_readable, "%0.3g us", microseconds);
    return human_readable;
  }
  double milliseconds = seconds * 1e3;
  if (milliseconds >= .995 && milliseconds < 1) {
    // Round half to even in Appendf would convert this to 0.999 ms.
    milliseconds = 1.0;
  }
  if (milliseconds < 999.5) {
    strings::Appendf(&human_readable, "%0.3g ms", milliseconds);
    return human_readable;
  }
  if (seconds < 60.0) {
    strings::Appendf(&human_readable, "%0.3g s", seconds);
    return human_readable;
  }
  seconds /= 60.0;
  if (seconds < 60.0) {
    strings::Appendf(&human_readable, "%0.3g min", seconds);
    return human_readable;
  }
  seconds /= 60.0;
  if (seconds < 24.0) {
    strings::Appendf(&human_readable, "%0.3g h", seconds);
    return human_readable;
  }
  seconds /= 24.0;
  if (seconds < 30.0) {
    strings::Appendf(&human_readable, "%0.3g days", seconds);
    return human_readable;
  }
  if (seconds < 365.2425) {
    strings::Appendf(&human_readable, "%0.3g months", seconds / 30.436875);
    return human_readable;
  }
  seconds /= 365.2425;
  strings::Appendf(&human_readable, "%0.3g years", seconds);
  return human_readable;
}

}  // namespace strings
}  // namespace tsl
