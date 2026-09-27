/* Copyright 2026 The OpenXLA Authors.

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
#ifndef XLA_TSL_BUILDDATA_UTILS_H_
#define XLA_TSL_BUILDDATA_UTILS_H_

namespace tsl::builddata {

constexpr bool StrEq(const char* a, const char* b) {
  while (*a && *b && *a == *b) {
    a++;
    b++;
  }
  return *a == *b;
}

// Converts the changelist string to an integer.
// Returns 0 if the string is "unknown".
// Returns -1 if the string is empty.
// Returns -2 if the string cannot be parsed as an integer (e.g. a git hash)
// We use `long long` instead of `int64_t` to avoid including `<cstdint>` for
// link-time performance.
constexpr long long  // NOLINT(runtime/int) NOLINT(google-runtime-int)
ParseChangelist(const char* str) {
  if (str == nullptr || *str == '\0') {
    return -1;
  }
  if (StrEq(str, "unknown")) {
    return 0;
  }
  long long result = 0;  // NOLINT(runtime/int) NOLINT(google-runtime-int)
  while (*str) {
    if (*str >= '0' && *str <= '9') {
      result = result * 10 + (*str - '0');
    } else {
      return -2;
    }
    str++;
  }
  return result;
}

constexpr int ParseMintStatus(int val) { return val; }

constexpr int ParseMintStatus(const char* str) {
  if (str == nullptr) {
    return -1;
  }
  if (StrEq(str, "mint")) {
    return 1;
  }
  if (StrEq(str, "modified")) {
    return 0;
  }
  return -1;
}

}  // namespace tsl::builddata

#endif  // XLA_TSL_BUILDDATA_UTILS_H_
