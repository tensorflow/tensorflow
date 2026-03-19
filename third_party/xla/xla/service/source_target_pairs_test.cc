/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/source_target_pairs.h"

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_format.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

struct Cannonical {
  SourceTargetPairs cycle;
  SourceTargetPairs main_edge;
  SourceTargetPairs back_edge;
};

class CollectivePermuteUtilsTest : public ::testing::Test {
 protected:
  SourceTargetPairs fwd2_ = SourceTargetPairs({{0, 1}, {1, 0}});
  SourceTargetPairs bwd2_ = SourceTargetPairs({{0, 1}, {1, 0}});
  SourceTargetPairs fwd4_ = SourceTargetPairs({{0, 1}, {1, 2}, {2, 3}, {3, 0}});
  SourceTargetPairs bwd4_ = SourceTargetPairs({{0, 3}, {1, 0}, {2, 1}, {3, 2}});
};

TEST_F(CollectivePermuteUtilsTest, FromString) {
  EXPECT_EQ(SourceTargetPairs::FromString("{{0,1},{1,0}}").value(), fwd2_);
  EXPECT_EQ(SourceTargetPairs::FromString("{{0,1}, {1,0}}").value(), bwd2_);
  EXPECT_EQ(SourceTargetPairs::FromString("{{0,1},{1,2},{2,3},{3,0}}").value(),
            fwd4_);
  EXPECT_THAT(SourceTargetPairs::FromString("{{0,1},{1}}"),
              absl_testing::StatusIs(absl::StatusCode::kInternal));
}

TEST_F(CollectivePermuteUtilsTest, AbslStringify) {
  EXPECT_EQ(
      absl::StrFormat("Source Target Pairs: %v",
                      SourceTargetPairs::FromString("{{0,1},{1,0}}").value()),
      "Source Target Pairs: {{0,1},{1,0}}");
}

TEST_F(CollectivePermuteUtilsTest, SourceTargetPairsString) {
  EXPECT_EQ(fwd2_.ToString(), "{{0,1},{1,0}}");
  EXPECT_EQ(bwd2_.ToString(), "{{0,1},{1,0}}");
  EXPECT_EQ(fwd4_.ToString(), "{{0,1},{1,2},{2,3},{3,0}}");
  EXPECT_EQ(bwd4_.ToString(), "{{0,3},{1,0},{2,1},{3,2}}");
}

TEST_F(CollectivePermuteUtilsTest, GetMaxDeviceNum) {
  EXPECT_EQ(fwd2_.GetMaxDeviceNum(), 1);
  EXPECT_EQ(bwd2_.GetMaxDeviceNum(), 1);
  EXPECT_EQ(fwd4_.GetMaxDeviceNum(), 3);
  EXPECT_EQ(bwd4_.GetMaxDeviceNum(), 3);
  EXPECT_EQ(
      SourceTargetPairs({{0, 1}, {1, 2}, {2, 300}, {3, 4}}).GetMaxDeviceNum(),
      300);
}

}  // namespace
}  // namespace xla
