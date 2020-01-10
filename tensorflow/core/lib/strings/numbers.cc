#include "tensorflow/core/lib/strings/numbers.h"

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>

#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace strings {

char* FastInt32ToBufferLeft(int32 i, char* buffer) {
  uint32 u = i;
  if (i < 0) {
    *buffer++ = '-';
    // We need to do the negation in modular (i.e., "unsigned")
    // arithmetic; MSVC++ apprently warns for plain "-u", so
    // we write the equivalent expression "0 - u" instead.
    u = 0 - u;
  }
  return FastUInt32ToBufferLeft(u, buffer);
}

char* FastUInt32ToBufferLeft(uint32 i, char* buffer) {
  char* start = buffer;
  do {
    *buffer++ = ((i % 10) + '0');
    i /= 10;
  } while (i > 0);
  *buffer = 0;
  std::reverse(start, buffer);
  return buffer;
}

char* FastInt64ToBufferLeft(int64 i, char* buffer) {
  uint64 u = i;
  if (i < 0) {
    *buffer++ = '-';
    u = 0 - u;
  }
  return FastUInt64ToBufferLeft(u, buffer);
}

char* FastUInt64ToBufferLeft(uint64 i, char* buffer) {
  char* start = buffer;
  do {
    *buffer++ = ((i % 10) + '0');
    i /= 10;
  } while (i > 0);
  *buffer = 0;
  std::reverse(start, buffer);
  return buffer;
}

static const double kDoublePrecisionCheckMax = DBL_MAX / 1.000000000000001;

char* DoubleToBuffer(double value, char* buffer) {
  // DBL_DIG is 15 for IEEE-754 doubles, which are used on almost all
  // platforms these days.  Just in case some system exists where DBL_DIG
  // is significantly larger -- and risks overflowing our buffer -- we have
  // this assert.
  static_assert(DBL_DIG < 20, "DBL_DIG is too big");

  bool full_precision_needed = true;
  if (std::abs(value) <= kDoublePrecisionCheckMax) {
    int snprintf_result =
        snprintf(buffer, kFastToBufferSize, "%.*g", DBL_DIG, value);

    // The snprintf should never overflow because the buffer is significantly
    // larger than the precision we asked for.
    DCHECK(snprintf_result > 0 && snprintf_result < kFastToBufferSize);

    full_precision_needed = strtod(buffer, NULL) != value;
  }

  if (full_precision_needed) {
    int snprintf_result =
        snprintf(buffer, kFastToBufferSize, "%.*g", DBL_DIG + 2, value);

    // Should never overflow; see above.
    DCHECK(snprintf_result > 0 && snprintf_result < kFastToBufferSize);
  }
  return buffer;
}

bool safe_strto64(const char* str, int64* value) {
  if (!str) return false;

  // Skip leading space.
  while (isspace(*str)) ++str;

  int64 vlimit = kint64max;
  int sign = 1;
  if (*str == '-') {
    sign = -1;
    ++str;
    // Different limit for positive and negative integers.
    vlimit = kint64min;
  }

  if (!isdigit(*str)) return false;

  int64 result = 0;
  if (sign == 1) {
    do {
      int digit = *str - '0';
      if ((vlimit - digit) / 10 < result) {
        return false;
      }
      result = result * 10 + digit;
      ++str;
    } while (isdigit(*str));
  } else {
    do {
      int digit = *str - '0';
      if ((vlimit + digit) / 10 > result) {
        return false;
      }
      result = result * 10 - digit;
      ++str;
    } while (isdigit(*str));
  }

  // Skip trailing space.
  while (isspace(*str)) ++str;

  if (*str) return false;

  *value = result;
  return true;
}

bool safe_strto32(const char* str, int32* value) {
  if (!str) return false;

  // Skip leading space.
  while (isspace(*str)) ++str;

  int64 vmax = kint32max;
  int sign = 1;
  if (*str == '-') {
    sign = -1;
    ++str;
    // Different max for positive and negative integers.
    ++vmax;
  }

  if (!isdigit(*str)) return false;

  int64 result = 0;
  do {
    result = result * 10 + *str - '0';
    if (result > vmax) {
      return false;
    }
    ++str;
  } while (isdigit(*str));

  // Skip trailing space.
  while (isspace(*str)) ++str;

  if (*str) return false;

  *value = result * sign;
  return true;
}

bool safe_strtof(const char* str, float* value) {
  char* endptr;
  *value = strtof(str, &endptr);
  while (isspace(*endptr)) ++endptr;
  // Ignore range errors from strtod/strtof.
  // The values it returns on underflow and
  // overflow are the right fallback in a
  // robust setting.
  return *str != '\0' && *endptr == '\0';
}

char* FloatToBuffer(float value, char* buffer) {
  // FLT_DIG is 6 for IEEE-754 floats, which are used on almost all
  // platforms these days.  Just in case some system exists where FLT_DIG
  // is significantly larger -- and risks overflowing our buffer -- we have
  // this assert.
  static_assert(FLT_DIG < 10, "FLT_DIG is too big");

  int snprintf_result =
      snprintf(buffer, kFastToBufferSize, "%.*g", FLT_DIG, value);

  // The snprintf should never overflow because the buffer is significantly
  // larger than the precision we asked for.
  DCHECK(snprintf_result > 0 && snprintf_result < kFastToBufferSize);

  float parsed_value;
  if (!safe_strtof(buffer, &parsed_value) || parsed_value != value) {
    snprintf_result =
        snprintf(buffer, kFastToBufferSize, "%.*g", FLT_DIG + 2, value);

    // Should never overflow; see above.
    DCHECK(snprintf_result > 0 && snprintf_result < kFastToBufferSize);
  }
  return buffer;
}

string FpToString(Fprint fp) {
  char buf[17];
  snprintf(buf, sizeof(buf), "%016llx", static_cast<uint64>(fp));
  return string(buf);
}

bool StringToFp(const string& s, Fprint* fp) {
  char junk;
  uint64 result;
  if (sscanf(s.c_str(), "%llx%c", &result, &junk) == 1) {
    *fp = result;
    return true;
  } else {
    return false;
  }
}

string HumanReadableNumBytes(int64 num_bytes) {
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
             static_cast<int64>(num_bytes));
    return string(buf);
  }

  static const char units[] = "KMGTPE";  // int64 only goes up to E.
  const char* unit = units;
  while (num_bytes >= static_cast<int64>(1024) * 1024) {
    num_bytes /= 1024;
    ++unit;
    CHECK(unit < units + TF_ARRAYSIZE(units));
  }

  // We use SI prefixes.
  char buf[16];
  snprintf(buf, sizeof(buf), ((*unit == 'K') ? "%s%.1f%ciB" : "%s%.2f%ciB"),
           neg_str, num_bytes / 1024.0, *unit);
  return string(buf);
}

}  // namespace strings
}  // namespace tensorflow
