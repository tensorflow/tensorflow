// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/c/litert_common.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace {

using testing::Eq;
using testing::Gt;
using testing::Lt;

TEST(CompareLiteRtApiVersionTest, Works) {
  // Equality case.
  EXPECT_THAT(LiteRtCompareApiVersion({1, 2, 3}, {1, 2, 3}), Eq(0));
  // First is greater than second at patch level.
  EXPECT_THAT(LiteRtCompareApiVersion({1, 1, 2}, {1, 1, 1}), Gt(0));
  // First is greater than second at minor level.
  EXPECT_THAT(LiteRtCompareApiVersion({1, 2, 0}, {1, 1, 2}), Gt(0));
  // First is greater than second at major level.
  EXPECT_THAT(LiteRtCompareApiVersion({2, 0, 0}, {1, 1, 2}), Gt(0));
  // First is smaller than second at patch level.
  EXPECT_THAT(LiteRtCompareApiVersion({1, 1, 1}, {1, 1, 2}), Lt(0));
  // First is smaller than second at minor level.
  EXPECT_THAT(LiteRtCompareApiVersion({1, 1, 2}, {1, 2, 0}), Lt(0));
  // First is smaller than second at major level.
  EXPECT_THAT(LiteRtCompareApiVersion({1, 1, 2}, {2, 0, 0}), Lt(0));
}

}  // namespace
