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

#include <cstdint>
#include <limits>

#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/types.h"

namespace tsl {
namespace {

TEST(IntegralTypes, Basic) {
  EXPECT_EQ(1, sizeof(int8_t));
  EXPECT_EQ(2, sizeof(int16_t));
  EXPECT_EQ(4, sizeof(int32_t));
  EXPECT_EQ(8, sizeof(int64_t));

  EXPECT_EQ(1, sizeof(uint8_t));
  EXPECT_EQ(2, sizeof(uint16_t));
  EXPECT_EQ(4, sizeof(uint32_t));
  EXPECT_EQ(8, sizeof(uint64_t));
}

TEST(IntegralTypes, MinAndMaxConstants) {
  EXPECT_EQ(static_cast<uint8_t>(std::numeric_limits<int8_t>::min()),
            static_cast<uint8_t>(std::numeric_limits<int8_t>::max()) + 1);
  EXPECT_EQ(static_cast<uint16_t>(std::numeric_limits<int16_t>::min()),
            static_cast<uint16_t>(std::numeric_limits<int16_t>::max()) + 1);
  EXPECT_EQ(static_cast<uint32_t>(std::numeric_limits<int32_t>::min()),
            static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) + 1);
  EXPECT_EQ(static_cast<uint64_t>(std::numeric_limits<int64_t>::min()),
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1);

  EXPECT_EQ(0, static_cast<uint8_t>(std::numeric_limits<uint8_t>::max() + 1));
  EXPECT_EQ(0, static_cast<uint16_t>(std::numeric_limits<uint16_t>::max() + 1));
  EXPECT_EQ(0, static_cast<uint32_t>(std::numeric_limits<uint32_t>::max() + 1));
  EXPECT_EQ(0, static_cast<uint64_t>(std::numeric_limits<uint64_t>::max() + 1));
}

}  // namespace
}  // namespace tsl
