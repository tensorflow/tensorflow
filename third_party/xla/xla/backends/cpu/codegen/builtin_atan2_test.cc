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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "tsl/platform/denormal.h"

namespace xla::cpu {
namespace {

// std::fpclassify is unreliable under DAZ: Clang implements it with
// floating-point compares, and under DAZ a subnormal compares equal to zero.
// Check the bits directly instead; integer ops are unaffected by DAZ.
bool IsSubnormal(double value) {
  const uint64_t bits = absl::bit_cast<uint64_t>(value);
  constexpr uint64_t kMagnitudeMask = 0x7fffffffffffffffULL;
  constexpr uint64_t kMinNormalMagnitude = uint64_t{1} << 52;
  const uint64_t magnitude = bits & kMagnitudeMask;
  return magnitude != 0 && magnitude < kMinNormalMagnitude;
}

bool IsSubnormal(float value) {
  const uint32_t bits = absl::bit_cast<uint32_t>(value);
  constexpr uint32_t kMagnitudeMask = 0x7fffffffU;
  constexpr uint32_t kMinNormalMagnitude = uint32_t{1} << 23;
  const uint32_t magnitude = bits & kMagnitudeMask;
  return magnitude != 0 && magnitude < kMinNormalMagnitude;
}

TEST(BuiltinAtan2Test, FlushesF64SubnormalInputs) {
  const double tiny = std::numeric_limits<double>::denorm_min();
  const double largest_subnormal =
      absl::bit_cast<double>(uint64_t{0x000fffffffffffff});
  const double min_normal = std::numeric_limits<double>::min();
  constexpr double kPi = 3.141592653589793;
  constexpr double kHalfPi = 1.5707963267948966;
  // Once subnormal inputs are flushed to signed zero, every result below is
  // one of +/-0, +/-pi/2, or +/-pi, determined by the signs of the flushed
  // inputs and by any normal operand.
  struct Case {
    double y;
    double x;
    double expected;
  };
  const Case cases[] = {
      {tiny, tiny, 0.0},
      {-tiny, tiny, -0.0},
      {tiny, -tiny, kPi},
      {-tiny, -tiny, -kPi},
      {2 * tiny, tiny, 0.0},
      {tiny, 2 * tiny, 0.0},
      {largest_subnormal, -largest_subnormal, kPi},
      {min_normal / 2, 1.0, 0.0},
      {min_normal / 2, -1.0, kPi},
      {1.0, min_normal / 2, kHalfPi},
      {-1.0, min_normal / 2, -kHalfPi},
      {min_normal, min_normal / 2, kHalfPi},
      {min_normal / 2, min_normal, 0.0},
  };
  for (const Case& test_case : cases) {
    const double actual = BuiltinAtan2(test_case.y, test_case.x);
    if (test_case.expected == 0.0) {
      // Bitwise comparison also checks the sign of the zero.
      EXPECT_EQ(absl::bit_cast<uint64_t>(actual),
                absl::bit_cast<uint64_t>(test_case.expected));
    } else {
      EXPECT_NEAR(actual, test_case.expected, 1e-15);
    }
  }
}

TEST(BuiltinAtan2Test, FlushesF32SubnormalInputs) {
  const float tiny = std::numeric_limits<float>::denorm_min();
  const float largest_subnormal = absl::bit_cast<float>(uint32_t{0x007fffff});
  const float min_normal = std::numeric_limits<float>::min();
  constexpr float kPi = 3.1415927f;
  constexpr float kHalfPi = 1.5707964f;
  struct Case {
    float y;
    float x;
    float expected;
  };
  const Case cases[] = {
      {tiny, tiny, 0.0f},
      {-tiny, tiny, -0.0f},
      {tiny, -tiny, kPi},
      {-tiny, -tiny, -kPi},
      {2 * tiny, tiny, 0.0f},
      {tiny, 2 * tiny, 0.0f},
      {largest_subnormal, -largest_subnormal, kPi},
      {min_normal / 2, 1.0f, 0.0f},
      {min_normal / 2, -1.0f, kPi},
      {1.0f, min_normal / 2, kHalfPi},
      {-1.0f, min_normal / 2, -kHalfPi},
      {min_normal, min_normal / 2, kHalfPi},
      {min_normal / 2, min_normal, 0.0f},
  };
  for (const Case& test_case : cases) {
    const float actual = BuiltinAtan2f(test_case.y, test_case.x);
    EXPECT_FALSE(std::isnan(actual));
    if (test_case.expected == 0.0f) {
      // Bitwise comparison also checks the sign of the zero.
      EXPECT_EQ(absl::bit_cast<uint32_t>(actual),
                absl::bit_cast<uint32_t>(test_case.expected));
    } else {
      EXPECT_NEAR(actual, test_case.expected, 1e-6f);
    }
  }
}

TEST(BuiltinAtan2Test, PreservesStateAndPassesNormalF64InputsThrough) {
  const double tiny = std::numeric_limits<double>::denorm_min();
  const double largest_subnormal =
      absl::bit_cast<double>(uint64_t{0x000fffffffffffff});
  const double min_normal = std::numeric_limits<double>::min();
  const double infinity = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<double> values = {tiny,
                                      -tiny,
                                      largest_subnormal,
                                      -largest_subnormal,
                                      min_normal / 2,
                                      -min_normal / 2,
                                      min_normal,
                                      -min_normal,
                                      1.0,
                                      -1.0,
                                      0.0,
                                      -0.0,
                                      infinity,
                                      -infinity,
                                      nan};

  const tsl::port::DenormalState states[] = {
      {true, true}, {true, false}, {false, true}, {false, false}};
  for (const tsl::port::DenormalState& state : states) {
    tsl::port::ScopedRestoreFlushDenormalState restore;
    if (!tsl::port::SetDenormalState(state)) {
      continue;
    }
    size_t index = 0;
    for (double y : values) {
      for (double x : values) {
        SCOPED_TRACE(index);
        const double actual = BuiltinAtan2(y, x);
        EXPECT_EQ(tsl::port::GetDenormalState(), state);
        // Non-subnormal inputs pass through unchanged, so libm on the same
        // operands is the expectation. Subnormal results are covered by the
        // hardcoded cases above.
        if (!IsSubnormal(y) && !IsSubnormal(x)) {
          const double expected = std::atan2(y, x);
          if (std::isnan(expected)) {
            EXPECT_TRUE(std::isnan(actual));
          } else {
            // Bitwise comparison also checks signed zeros.
            EXPECT_EQ(absl::bit_cast<uint64_t>(actual),
                      absl::bit_cast<uint64_t>(expected));
          }
        }
        ++index;
      }
    }
  }
}

TEST(BuiltinAtan2Test, PreservesStateAndPassesNormalF32InputsThrough) {
  const float tiny = std::numeric_limits<float>::denorm_min();
  const float largest_subnormal = absl::bit_cast<float>(uint32_t{0x007fffff});
  const float min_normal = std::numeric_limits<float>::min();
  const float infinity = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const std::vector<float> values = {tiny,
                                     -tiny,
                                     largest_subnormal,
                                     -largest_subnormal,
                                     min_normal / 2,
                                     -min_normal / 2,
                                     min_normal,
                                     -min_normal,
                                     1.0f,
                                     -1.0f,
                                     0.0f,
                                     -0.0f,
                                     infinity,
                                     -infinity,
                                     nan};

  const tsl::port::DenormalState states[] = {
      {true, true}, {true, false}, {false, true}, {false, false}};
  for (const tsl::port::DenormalState& state : states) {
    tsl::port::ScopedRestoreFlushDenormalState restore;
    if (!tsl::port::SetDenormalState(state)) {
      continue;
    }
    size_t index = 0;
    for (float y : values) {
      for (float x : values) {
        SCOPED_TRACE(index);
        const float actual = BuiltinAtan2f(y, x);
        EXPECT_EQ(tsl::port::GetDenormalState(), state);
        if (!IsSubnormal(y) && !IsSubnormal(x)) {
          const float expected = ::atan2f(y, x);
          if (std::isnan(expected)) {
            EXPECT_TRUE(std::isnan(actual));
          } else {
            EXPECT_EQ(absl::bit_cast<uint32_t>(actual),
                      absl::bit_cast<uint32_t>(expected));
          }
        }
        ++index;
      }
    }
  }
}

TEST(BuiltinAtan2Test, GlibcDenormalMisbehaviors) {
  // Under DAZ, glibc computes atan2(min_normal / 2, -1) as if y were zero
  // but picks the wrong quadrant, returning -pi. The flushed result is +pi.
  const double min_normal = std::numeric_limits<double>::min();
  EXPECT_EQ(absl::bit_cast<uint64_t>(BuiltinAtan2(min_normal / 2, -1.0)),
            absl::bit_cast<uint64_t>(std::atan2(0.0, -1.0)));
  // Under DAZ, glibc returns NaN for atan2f(denorm_min, 1). The flushed
  // result is +0.
  const float tiny_f = std::numeric_limits<float>::denorm_min();
  const float f32_actual = BuiltinAtan2f(tiny_f, 1.0f);
  EXPECT_FALSE(std::isnan(f32_actual));
  EXPECT_EQ(absl::bit_cast<uint32_t>(f32_actual),
            absl::bit_cast<uint32_t>(0.0f));
  // Under DAZ, glibc returns +/-pi/2 for subnormal pairs. The flushed
  // (+0, +0) result is +0.
  const double tiny = std::numeric_limits<double>::denorm_min();
  EXPECT_EQ(absl::bit_cast<uint64_t>(BuiltinAtan2(tiny, tiny)),
            absl::bit_cast<uint64_t>(0.0));
  // Signs of flushed inputs still select the quadrant: (-0, -0) gives -pi.
  EXPECT_EQ(absl::bit_cast<uint64_t>(BuiltinAtan2(-tiny, -tiny)),
            absl::bit_cast<uint64_t>(std::atan2(-0.0, -0.0)));
}

}  // namespace
}  // namespace xla::cpu
