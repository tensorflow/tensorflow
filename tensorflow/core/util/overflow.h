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

#ifndef TENSORFLOW_CORE_UTIL_OVERFLOW_H_
#define TENSORFLOW_CORE_UTIL_OVERFLOW_H_

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Multiply two nonnegative int64's, returning negative for overflow
inline int64 MultiplyWithoutOverflow(const int64 x, const int64 y) {
  // Multiply in uint64 rather than int64 since signed overflow is undefined.
  // Negative values will wrap around to large unsigned values in the casts
  // (see section 4.7 [conv.integral] of the C++14 standard).
  const uint64 ux = x;
  const uint64 uy = y;
  const uint64 uxy = ux * uy;

  // Check if we overflow uint64, using a cheap check if both inputs are small
  if (TF_PREDICT_FALSE((ux | uy) >> 32 != 0)) {
    // Ensure nonnegativity.  Note that negative numbers will appear "large"
    // to the unsigned comparisons above.
    CHECK(x >= 0 && y >= 0);

    // Otherwise, detect overflow using a division
    if (ux != 0 && uxy / ux != uy) return -1;
  }

  // Cast back to signed.  Any negative value will signal an error.
  return static_cast<int64>(uxy);
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_OVERFLOW_H_
