/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/stream_executor/bit_pattern.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "tsl/platform/test.h"

namespace stream_executor {
namespace {

TEST(BitPatternTest, 8BitPattern) {
  BitPattern bit_pattern{static_cast<uint8_t>(0xAF)};
  EXPECT_EQ(bit_pattern.GetElementSize(), 1);
  EXPECT_EQ(bit_pattern.GetPatternBroadcastedToUint32(), 0xAFAFAFAF);
  EXPECT_EQ(bit_pattern.ToString(), "u8:175");
}

TEST(BitPatternTest, 16BitPattern) {
  BitPattern bit_pattern{static_cast<uint16_t>(0xABCD)};
  EXPECT_EQ(bit_pattern.GetElementSize(), 2);
  EXPECT_EQ(bit_pattern.GetPatternBroadcastedToUint32(), 0xABCDABCD);
  EXPECT_EQ(bit_pattern.ToString(), "u16:43981");
}

TEST(BitPatternTest, 32BitPattern) {
  BitPattern bit_pattern{static_cast<uint32_t>(0x0123ABCD)};
  EXPECT_EQ(bit_pattern.GetElementSize(), 4);
  EXPECT_EQ(bit_pattern.GetPatternBroadcastedToUint32(), 0x0123ABCD);
  EXPECT_EQ(bit_pattern.ToString(), "u32:19114957");
}

}  // namespace
}  // namespace stream_executor
