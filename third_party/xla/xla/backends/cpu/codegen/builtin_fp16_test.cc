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

#include "xla/backends/cpu/codegen/builtin_fp16.h"

#include <cstdint>
#include <cstring>
#include <limits>

#include <gtest/gtest.h>

namespace xla::cpu {
namespace {

static uint16_t BitcastBF16ToUInt16(XlaBF16ABIType val) {
  uint16_t bits = 0;
  std::memcpy(&bits, &val, sizeof(bits));
  return bits;
}

static uint16_t BitcastF16ToUInt16(XlaF16ABIType val) {
  uint16_t bits = 0;
  std::memcpy(&bits, &val, sizeof(bits));
  return bits;
}

TEST(BuiltinFP16Test, TruncSFBF2) {
  EXPECT_EQ(BitcastBF16ToUInt16(__truncsfbf2(0.0f)), 0x0000);
  EXPECT_EQ(BitcastBF16ToUInt16(__truncsfbf2(-0.0f)), 0x8000);
  EXPECT_EQ(BitcastBF16ToUInt16(__truncsfbf2(1.0f)), 0x3F80);
  EXPECT_EQ(BitcastBF16ToUInt16(__truncsfbf2(-1.0f)), 0xBF80);
  EXPECT_EQ(BitcastBF16ToUInt16(__truncsfbf2(2.0f)), 0x4000);
  EXPECT_EQ(BitcastBF16ToUInt16(__truncsfbf2(0.5f)), 0x3F00);

  // Infinities.
  EXPECT_EQ(
      BitcastBF16ToUInt16(__truncsfbf2(std::numeric_limits<float>::infinity())),
      0x7F80);
  EXPECT_EQ(BitcastBF16ToUInt16(
                __truncsfbf2(-std::numeric_limits<float>::infinity())),
            0xFF80);

  // NaNs.
  uint16_t nan_bits = BitcastBF16ToUInt16(
      __truncsfbf2(std::numeric_limits<float>::quiet_NaN()));
  EXPECT_EQ(nan_bits & 0x7F80, 0x7F80);
  EXPECT_NE(nan_bits & 0x007F, 0);

  uint16_t neg_nan_bits = BitcastBF16ToUInt16(
      __truncsfbf2(-std::numeric_limits<float>::quiet_NaN()));
  EXPECT_EQ(neg_nan_bits & 0xFF80, 0xFF80);
  EXPECT_NE(neg_nan_bits & 0x007F, 0);

  uint16_t sig_nan_bits = BitcastBF16ToUInt16(
      __truncsfbf2(std::numeric_limits<float>::signaling_NaN()));
  EXPECT_EQ(sig_nan_bits & 0x7F80, 0x7F80);
  EXPECT_NE(sig_nan_bits & 0x007F, 0);

  // Overflow to infinity.
  EXPECT_EQ(
      BitcastBF16ToUInt16(__truncsfbf2(std::numeric_limits<float>::max())),
      0x7F80);
  EXPECT_EQ(
      BitcastBF16ToUInt16(__truncsfbf2(-std::numeric_limits<float>::max())),
      0xFF80);

  // Subnormals and denorm min.
  EXPECT_EQ(BitcastBF16ToUInt16(
                __truncsfbf2(std::numeric_limits<float>::denorm_min())),
            0x0000);

  // Rounding: round to nearest even.
  // 1.0f is 0x3F800000.
  // Value slightly below halfway: rounds down.
  float val_down;
  uint32_t val_down_bits = 0x3F807FFF;
  std::memcpy(&val_down, &val_down_bits, sizeof(float));
  EXPECT_EQ(BitcastBF16ToUInt16(__truncsfbf2(val_down)), 0x3F80);

  // Exact halfway with even LSB (LSB = 0): rounds down (to even).
  float val_half_even;
  uint32_t val_half_even_bits = 0x3F808000;
  std::memcpy(&val_half_even, &val_half_even_bits, sizeof(float));
  EXPECT_EQ(BitcastBF16ToUInt16(__truncsfbf2(val_half_even)), 0x3F80);

  // Exact halfway with odd LSB (LSB = 1): rounds up (to even).
  float val_half_odd;
  uint32_t val_half_odd_bits = 0x3F818000;
  std::memcpy(&val_half_odd, &val_half_odd_bits, sizeof(float));
  EXPECT_EQ(BitcastBF16ToUInt16(__truncsfbf2(val_half_odd)), 0x3F82);
}

TEST(BuiltinFP16Test, TruncDFBF2) {
  EXPECT_EQ(BitcastBF16ToUInt16(__truncdfbf2(0.0)), 0x0000);
  EXPECT_EQ(BitcastBF16ToUInt16(__truncdfbf2(-0.0)), 0x8000);
  EXPECT_EQ(BitcastBF16ToUInt16(__truncdfbf2(1.0)), 0x3F80);
  EXPECT_EQ(BitcastBF16ToUInt16(__truncdfbf2(-1.0)), 0xBF80);
  EXPECT_EQ(BitcastBF16ToUInt16(
                __truncdfbf2(std::numeric_limits<double>::infinity())),
            0x7F80);
  EXPECT_EQ(BitcastBF16ToUInt16(
                __truncdfbf2(-std::numeric_limits<double>::infinity())),
            0xFF80);

  uint16_t d_nan_bits = BitcastBF16ToUInt16(
      __truncdfbf2(std::numeric_limits<double>::quiet_NaN()));
  EXPECT_EQ(d_nan_bits & 0x7F80, 0x7F80);
  EXPECT_NE(d_nan_bits & 0x007F, 0);

  uint16_t d_neg_nan_bits = BitcastBF16ToUInt16(
      __truncdfbf2(-std::numeric_limits<double>::quiet_NaN()));
  EXPECT_EQ(d_neg_nan_bits & 0xFF80, 0xFF80);
  EXPECT_NE(d_neg_nan_bits & 0x007F, 0);
}

TEST(BuiltinFP16Test, GnuF2HAndH2F) {
  XlaF16ABIType h_zero = __gnu_f2h_ieee(0.0f);
  EXPECT_EQ(BitcastF16ToUInt16(h_zero), 0x0000);
  EXPECT_FLOAT_EQ(__gnu_h2f_ieee(h_zero), 0.0f);

  XlaF16ABIType h_one = __gnu_f2h_ieee(1.0f);
  EXPECT_EQ(BitcastF16ToUInt16(h_one), 0x3C00);
  EXPECT_FLOAT_EQ(__gnu_h2f_ieee(h_one), 1.0f);

  XlaF16ABIType h_neg_one = __gnu_f2h_ieee(-1.0f);
  EXPECT_EQ(BitcastF16ToUInt16(h_neg_one), 0xBC00);
  EXPECT_FLOAT_EQ(__gnu_h2f_ieee(h_neg_one), -1.0f);

  XlaF16ABIType h_from_d = __truncdfhf2(1.0);
  EXPECT_EQ(BitcastF16ToUInt16(h_from_d), 0x3C00);
}

}  // namespace
}  // namespace xla::cpu
