/* Copyright 2019 The OpenXLA Authors.

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

// BitCast is an extension of std::bit_cast/absl::bit_cast. Whereas those
// functions require trivially copyable source and destination types, the
// present function template may be specialized for additional types that
// do not satisfy that triviality property, but that have alternative ways
// of accessing their underlying representation.
//
// Concretely, we provide specializations for the "custom floating point types"
// Eigen::half and tsl::bfloat16. Those types are effectively stored as
// a sequence of bits, but the classes are not trivially copyable.

#ifndef XLA_BIT_CAST_H_
#define XLA_BIT_CAST_H_

#include <cstdint>

#include "absl/base/casts.h"
#include "Eigen/Core"
#include "xla/types.h"
#include "tsl/platform/bfloat16.h"

namespace xla {

template <typename T, typename U>
T BitCast(U src) {
  static_assert(sizeof(T) == sizeof(U), "sizes don't match");
  // We would like to check std::is_trivially_copyable here, but there's no
  // reliable implementation of that available to us.
  return absl::bit_cast<T>(src);
}

template <>
inline tsl::bfloat16 BitCast<tsl::bfloat16, uint16_t>(uint16_t src) {
  return Eigen::numext::bit_cast<tsl::bfloat16>(src);
}

template <>
inline uint16_t BitCast<uint16_t, tsl::bfloat16>(tsl::bfloat16 src) {
  return Eigen::numext::bit_cast<uint16_t>(src);
}

template <>
inline Eigen::half BitCast<Eigen::half, uint16_t>(uint16_t src) {
  return Eigen::numext::bit_cast<Eigen::half>(src);
}

template <>
inline uint16_t BitCast<uint16_t, Eigen::half>(Eigen::half src) {
  return Eigen::numext::bit_cast<uint16_t>(src);
}

}  // namespace xla

#endif  // XLA_BIT_CAST_H_
