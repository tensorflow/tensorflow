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

#ifndef TENSORFLOW_CORE_LIB_CORE_BITS_H_
#define TENSORFLOW_CORE_LIB_CORE_BITS_H_

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Return floor(log2(n)) for positive integer n.  Returns -1 iff n == 0.
int Log2Floor(uint32 n);
int Log2Floor64(uint64 n);

// Return ceiling(log2(n)) for positive integer n.  Returns -1 iff n == 0.
int Log2Ceiling(uint32 n);
int Log2Ceiling64(uint64 n);

// ------------------------------------------------------------------------
// Implementation details follow
// ------------------------------------------------------------------------

#if defined(__GNUC__)

// Return floor(log2(n)) for positive integer n.  Returns -1 iff n == 0.
inline int Log2Floor(uint32 n) { return n == 0 ? -1 : 31 ^ __builtin_clz(n); }

// Return floor(log2(n)) for positive integer n.  Returns -1 iff n == 0.
inline int Log2Floor64(uint64 n) {
  return n == 0 ? -1 : 63 ^ __builtin_clzll(n);
}

#else

// Return floor(log2(n)) for positive integer n.  Returns -1 iff n == 0.
inline int Log2Floor(uint32 n) {
  if (n == 0) return -1;
  int log = 0;
  uint32 value = n;
  for (int i = 4; i >= 0; --i) {
    int shift = (1 << i);
    uint32 x = value >> shift;
    if (x != 0) {
      value = x;
      log += shift;
    }
  }
  assert(value == 1);
  return log;
}

// Return floor(log2(n)) for positive integer n.  Returns -1 iff n == 0.
// Log2Floor64() is defined in terms of Log2Floor32()
inline int Log2Floor64(uint64 n) {
  const uint32 topbits = static_cast<uint32>(n >> 32);
  if (topbits == 0) {
    // Top bits are zero, so scan in bottom bits
    return Log2Floor(static_cast<uint32>(n));
  } else {
    return 32 + Log2Floor(topbits);
  }
}

#endif

inline int Log2Ceiling(uint32 n) {
  int floor = Log2Floor(n);
  if (n == (n & ~(n - 1)))  // zero or a power of two
    return floor;
  else
    return floor + 1;
}

inline int Log2Ceiling64(uint64 n) {
  int floor = Log2Floor64(n);
  if (n == (n & ~(n - 1)))  // zero or a power of two
    return floor;
  else
    return floor + 1;
}

inline uint32 NextPowerOfTwo(uint32 value) {
  int exponent = Log2Ceiling(value);
  DCHECK_LT(exponent, std::numeric_limits<uint32>::digits);
  return 1 << exponent;
}

inline uint64 NextPowerOfTwo64(uint64 value) {
  int exponent = Log2Ceiling(value);
  DCHECK_LT(exponent, std::numeric_limits<uint64>::digits);
  return 1LL << exponent;
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_CORE_BITS_H_
