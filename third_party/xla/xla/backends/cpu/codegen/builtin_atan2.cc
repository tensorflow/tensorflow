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

#include "xla/backends/cpu/codegen/builtin_atan2.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "absl/base/casts.h"

namespace xla::cpu {
namespace {

// Flushes a subnormal input to signed zero. Floating-point comparisons can
// already treat subnormal operands as zero under DAZ, so inspect their bits
// instead. Signed zeros, infinities, and NaNs pass through unchanged, and the
// sign of a flushed input is kept so the quadrant of the result is correct.
template <typename Float>
Float FlushSubnormalToSignedZero(Float value) {
  using Bits = std::conditional_t<sizeof(Float) == 4, uint32_t, uint64_t>;
  constexpr Bits kSignMask = Bits{1} << (sizeof(Float) * 8 - 1);
  constexpr Bits kMagnitudeMask = static_cast<Bits>(~kSignMask);
  // Smallest magnitude with a nonzero exponent field.
  constexpr Bits kMinNormalBits = Bits{1}
                                  << (std::numeric_limits<Float>::digits - 1);
  const Bits bits = absl::bit_cast<Bits>(value);
  const Bits magnitude = bits & kMagnitudeMask;
  if (magnitude != 0 && magnitude < kMinNormalBits) {
    return absl::bit_cast<Float>(bits & kSignMask);
  }
  return value;
}

}  // namespace

double BuiltinAtan2(double y, double x) {
  return std::atan2(FlushSubnormalToSignedZero(y),
                    FlushSubnormalToSignedZero(x));
}

float BuiltinAtan2f(float y, float x) {
  // glibc's atan2f can return NaN for subnormal inputs (even
  // atan2f(denorm_min, 1)), so flush f32 inputs explicitly as well.
  return ::atan2f(FlushSubnormalToSignedZero(y), FlushSubnormalToSignedZero(x));
}

}  // namespace xla::cpu
