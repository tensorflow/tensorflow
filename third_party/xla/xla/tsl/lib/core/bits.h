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

#ifndef XLA_TSL_LIB_CORE_BITS_H_
#define XLA_TSL_LIB_CORE_BITS_H_

#include <cstdint>

#include "absl/numeric/bits.h"
#include "xla/tsl/platform/logging.h"

namespace tsl {

// Return floor(log2(n)) for positive integer n.  Returns -1 iff n == 0.
inline int Log2Floor(uint32_t n) { return absl::bit_width(n) - 1; }

// Return floor(log2(n)) for positive integer n.  Returns -1 iff n == 0.
inline int Log2Floor64(uint64_t n) { return absl::bit_width(n) - 1; }

// Return ceiling(log2(n)) for positive integer n.  Returns -1 iff n == 0.
inline int Log2Ceiling(uint32_t n) {
  return n == 0 ? -1 : absl::bit_width(n - 1);
}

// Return ceiling(log2(n)) for positive integer n.  Returns -1 iff n == 0.
inline int Log2Ceiling64(uint64_t n) {
  return n == 0 ? -1 : absl::bit_width(n - 1);
}

inline uint32_t NextPowerOfTwo(uint32_t value) { return absl::bit_ceil(value); }

inline uint64_t NextPowerOfTwo64(uint64_t value) {
  return absl::bit_ceil(value);
}

inline int64_t NextPowerOfTwoS64(int64_t value) {
  constexpr int64_t kMaxRepresentablePowerOfTwo =
      static_cast<int64_t>(uint64_t{1} << 62);
  DCHECK_GE(value, 0);
  DCHECK_LE(value, kMaxRepresentablePowerOfTwo);
  return static_cast<int64_t>(absl::bit_ceil(static_cast<uint64_t>(value)));
}

}  // namespace tsl

#endif  // XLA_TSL_LIB_CORE_BITS_H_
