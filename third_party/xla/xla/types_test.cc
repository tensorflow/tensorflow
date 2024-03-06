/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/types.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "xla/test.h"

namespace xla {
namespace {

TEST(U4Test, NumericLimits) {
  EXPECT_EQ(static_cast<int64_t>(std::numeric_limits<u4>::min()), 0);
  EXPECT_EQ(static_cast<int64_t>(std::numeric_limits<u4>::max()), 15);
  EXPECT_EQ(static_cast<int64_t>(std::numeric_limits<u4>::lowest()), 0);
}

TEST(S4Test, NumericLimits) {
  EXPECT_EQ(static_cast<int64_t>(std::numeric_limits<s4>::min()), -8);
  EXPECT_EQ(static_cast<int64_t>(std::numeric_limits<s4>::max()), 7);
  EXPECT_EQ(static_cast<int64_t>(std::numeric_limits<s4>::lowest()), -8);
}

}  // namespace
}  // namespace xla
