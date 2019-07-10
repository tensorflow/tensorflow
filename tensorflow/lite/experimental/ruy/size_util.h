/* Copyright 2019 Google LLC. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_SIZE_UTIL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_SIZE_UTIL_H_

#include "tensorflow/lite/experimental/ruy/check_macros.h"

#ifdef _WIN32
#include <intrin.h>
#endif

namespace ruy {

inline int floor_log2(int n) {
  RUY_DCHECK_GE(n, 1);
#ifdef _WIN32
  unsigned long result;
  _BitScanReverse(&result, n);
  return result;
#else
  return 31 - __builtin_clz(n);
#endif
}

inline int ceil_log2(int n) {
  RUY_DCHECK_GE(n, 1);
  return n == 1 ? 0 : floor_log2(n - 1) + 1;
}

inline bool is_pot(int value) {
  return (value > 0) && ((value & (value - 1)) == 0);
}

inline int round_down_pot(int value) { return 1 << floor_log2(value); }

inline int round_up_pot(int value) { return 1 << ceil_log2(value); }

inline int round_down_pot(int value, int modulo) {
  RUY_DCHECK_EQ(modulo & (modulo - 1), 0);
  return value & ~(modulo - 1);
}

inline int round_up_pot(int value, int modulo) {
  return round_down_pot(value + modulo - 1, modulo);
}

inline int clamp(int x, int lo, int hi) {
  if (x < lo) {
    return lo;
  } else if (x > hi) {
    return hi;
  } else {
    return x;
  }
}

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_SIZE_UTIL_H_
