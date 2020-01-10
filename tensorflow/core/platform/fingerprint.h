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

#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"

// The following line is used by copybara to set or unset the USE_OSS_FARMHASH
// preprocessor symbol as needed. Please do not remove.
#define USE_OSS_FARMHASH

#ifdef USE_OSS_FARMHASH
#include <farmhash.h>
#else
#include "util/hash/farmhash_fingerprint.h"
#endif

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

// This is a portable fingerprint interface for strings that will never change.
// However, it is not suitable for cryptography.
inline uint64 Fingerprint64(const StringPiece s) {
#ifdef USE_OSS_FARMHASH
  return ::util::Fingerprint64(s.data(), s.size());
#else
  // Fingerprint op depends on the fact that Fingerprint64() is implemented by
  // Farmhash. If the implementation ever changes, Fingerprint op should be
  // modified to keep using Farmhash.
  // LINT.IfChange
  return farmhash::Fingerprint64(s.data(), s.size());
  // LINT.ThenChange(//third_party/tensorflow/core/kernels/fingerprint_op.cc)
#endif
}

// 128-bit variant of Fingerprint64 above (same properties and caveats apply).
inline Fprint128 Fingerprint128(const StringPiece s) {
#ifdef USE_OSS_FARMHASH
  const auto fingerprint = ::util::Fingerprint128(s.data(), s.size());
  return {::util::Uint128Low64(fingerprint),
          ::util::Uint128High64(fingerprint)};
#else
  const auto fingerprint = farmhash::Fingerprint128(s.data(), s.size());
  return {absl::Uint128Low64(fingerprint), absl::Uint128High64(fingerprint)};
#endif
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_FINGERPRINT_H_
