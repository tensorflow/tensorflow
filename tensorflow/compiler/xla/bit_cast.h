/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
// Eigen::half and tensorflow::bfloat16. Those types are effectively stored as
// a sequence of bits, but the classes are not trivially copyable.

#ifndef TENSORFLOW_COMPILER_XLA_BIT_CAST_H_
#define TENSORFLOW_COMPILER_XLA_BIT_CAST_H_

#include "absl/base/casts.h"
#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/bfloat16.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

template <typename T, typename U>
T BitCast(U src) {
  static_assert(sizeof(T) == sizeof(U), "sizes don't match");
  // We would like to check std::is_trivially_copyable here, but there's no
  // reliable implementation of that available to us.
  return absl::bit_cast<T>(src);
}

template <>
inline tensorflow::bfloat16 BitCast<tensorflow::bfloat16, uint16_t>(
    uint16 src) {
  tensorflow::bfloat16 result;
  result.value = src;
  return result;
}

template <>
inline uint16 BitCast<uint16, tensorflow::bfloat16>(tensorflow::bfloat16 src) {
  return src.value;
}

template <>
inline Eigen::half BitCast<Eigen::half, uint16>(uint16 src) {
  Eigen::half result;
  result.x = src;
  return result;
}

template <>
inline uint16 BitCast<uint16, Eigen::half>(Eigen::half src) {
  return src.x;
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_BIT_CAST_H_
