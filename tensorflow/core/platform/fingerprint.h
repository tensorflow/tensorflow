/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_FINGERPRINT_H_
#define TENSORFLOW_CORE_PLATFORM_FINGERPRINT_H_

#include "tensorflow/core/platform/types.h"

namespace tensorflow {

struct Fprint128 {
  uint64 low64;
  uint64 high64;
};

inline bool operator==(const Fprint128& lhs, const Fprint128& rhs) {
  return lhs.low64 == rhs.low64 && lhs.high64 == rhs.high64;
}

struct Fprint128Hasher {
  size_t operator()(const Fprint128& v) const {
    // Low64 should be sufficiently mixed to allow use of it as a Hash.
    return static_cast<size_t>(v.low64);
  }
};

// TODO(sibyl-Mooth6ku): Change these to accept StringPiece (or make them templated
// on any kind of byte array?).

// This is a portable fingerprint interface for strings that will never change.
// However, it is not suitable for cryptography.
uint64 Fingerprint64(const string& s);

// 128-bit variant of Fingerprint64 above (same properties and caveats apply).
Fprint128 Fingerprint128(const string& s);

namespace internal {
// Mixes some of the bits that got propagated to the high bits back into the
// low bits.
inline uint64 ShiftMix(const uint64 val) { return val ^ (val >> 47); }
}  // namespace internal

// This concatenates two 64-bit fingerprints. It is a convenience function to
// get a fingerprint for a combination of already fingerprinted components. For
// example this code is used to concatenate the hashes from each of the features
// on sparse crosses.
//
// One shouldn't expect FingerprintCat64(Fingerprint64(x), Fingerprint64(y))
// to indicate anything about FingerprintCat64(StrCat(x, y)). This operation
// is not commutative.
//
// From a security standpoint, we don't encourage this pattern to be used
// for everything as it is vulnerable to length-extension attacks and it
// is easier to compute multicollisions.
inline uint64 FingerprintCat64(const uint64 fp1, const uint64 fp2) {
  static const uint64 kMul = 0xc6a4a7935bd1e995ULL;
  uint64 result = fp1 ^ kMul;
  result ^= internal::ShiftMix(fp2 * kMul) * kMul;
  result *= kMul;
  result = internal::ShiftMix(result) * kMul;
  result = internal::ShiftMix(result);
  return result;
}

}  // namespace tensorflow

#if defined(PLATFORM_GOOGLE)
#include "tensorflow/core/platform/google/fingerprint.h"
#else
#include "tensorflow/core/platform/default/fingerprint.h"
#endif

#endif  // TENSORFLOW_CORE_PLATFORM_FINGERPRINT_H_
