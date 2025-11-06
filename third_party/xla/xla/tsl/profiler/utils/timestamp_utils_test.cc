/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/profiler/utils/timestamp_utils.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "xla/tsl/platform/test.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"

namespace tsl {
namespace profiler {
using ::testing::Eq;

TEST(TimestampUtilsTest, StartAndStopTimestampAreAdded) {
  XSpace xspace;

  SetSessionTimestamps(1000, 2000, xspace);

  const std::optional<std::pair<uint64_t, uint64_t>> timestamps =
      GetSessionTimestamps(xspace);

  ASSERT_TRUE(timestamps.has_value());
  EXPECT_THAT(timestamps->first, Eq(1000));
  EXPECT_THAT(timestamps->second, Eq(2000));
}

}  // namespace profiler

}  // namespace tsl
