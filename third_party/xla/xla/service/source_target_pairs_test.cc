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
#include "absl/strings/str_format.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/status_matchers.h"

namespace xla {
namespace {

using CycleType = SourceTargetPairs::CycleType;

struct Cannonical {
  SourceTargetPairs cycle;
  SourceTargetPairs main_edge;
  SourceTargetPairs back_edge;
};

class CollectivePermuteUtilsTest : public ::testing::Test {
 protected:
  Cannonical fwd2_ = {.cycle = SourceTargetPairs({{0, 1}, {1, 0}}),
                      .main_edge = SourceTargetPairs({{0, 1}}),
                      .back_edge = SourceTargetPairs({{1, 0}})};

  Cannonical bwd2_ = {.cycle = SourceTargetPairs({{0, 1}, {1, 0}}),
                      .main_edge = SourceTargetPairs({{1, 0}}),
                      .back_edge = SourceTargetPairs({{0, 1}})};

  Cannonical fwd4_ = {
      .cycle = SourceTargetPairs({{0, 1}, {1, 2}, {2, 3}, {3, 0}}),
      .main_edge = SourceTargetPairs({{0, 1}, {1, 2}, {2, 3}}),
      .back_edge = SourceTargetPairs({{3, 0}})};

  Cannonical bwd4_ = {
      .cycle = SourceTargetPairs({{0, 3}, {1, 0}, {2, 1}, {3, 2}}),
      .main_edge = SourceTargetPairs({{1, 0}, {2, 1}, {3, 2}}),
      .back_edge = SourceTargetPairs({{0, 3}})};
  std::unique_ptr<HloInstruction> simple_input_ = HloInstruction::CreateToken();

  HloCollectivePermuteInstruction CreateCollectivePermute(
      const SourceTargetPairs& pairs) {
    return HloCollectivePermuteInstruction(
        HloOpcode::kCollectivePermute, ShapeUtil::MakeShape(U32, {8, 8}),
        {simple_input_.get()}, pairs.expand(), 1);
  }
};

TEST_F(CollectivePermuteUtilsTest, FromString) {
  EXPECT_EQ(SourceTargetPairs::FromString("{{0,1},{1,0}}").value(),
            fwd2_.cycle);
  EXPECT_EQ(SourceTargetPairs::FromString("{{0,1}, {1,0}}").value(),
            bwd2_.cycle);
  EXPECT_EQ(SourceTargetPairs::FromString("{{0,1},{1,2},{2,3},{3,0}}").value(),
            fwd4_.cycle);
  EXPECT_THAT(SourceTargetPairs::FromString("{{0,1},{1}}"),
              ::tsl::testing::StatusIs(absl::StatusCode::kInternal));
}

TEST_F(CollectivePermuteUtilsTest, AbslStringify) {
  EXPECT_EQ(
      absl::StrFormat("Source Target Pairs: %v",
                      SourceTargetPairs::FromString("{{0,1},{1,0}}").value()),
      "Source Target Pairs: {{0,1},{1,0}}");
}

TEST_F(CollectivePermuteUtilsTest, HasCycles) {
  EXPECT_TRUE(fwd2_.cycle.HasCycles());
  EXPECT_TRUE(bwd2_.cycle.HasCycles());
  EXPECT_TRUE(fwd4_.cycle.HasCycles());
  EXPECT_TRUE(bwd4_.cycle.HasCycles());

  EXPECT_FALSE(SourceTargetPairs({{0, 1}, {1, 2}, {2, 3}, {3, 2}}).HasCycles())
      << "Lasso 3->2";
  EXPECT_FALSE(SourceTargetPairs({{0, 1}, {1, 2}, {2, 3}, {3, 1}}).HasCycles())
      << "Lasso 3->1";

  EXPECT_FALSE(SourceTargetPairs({{1, 2}, {2, 3}, {3, 0}}).HasCycles())
      << "Forward only";
  EXPECT_FALSE(SourceTargetPairs({{1, 2}}).HasCycles()) << "Single edge";
}

bool IsForwardCycle(Cannonical& canonical) {
  return SourceTargetPairs::IsForwardCycle(canonical.back_edge,
                                           canonical.main_edge);
}
bool IsBackwardCycle(Cannonical& canonical) {
  return SourceTargetPairs::IsBackwardCycle(canonical.back_edge,
                                            canonical.main_edge);
}

TEST_F(CollectivePermuteUtilsTest, IsForwardCycle) {
  EXPECT_TRUE(IsForwardCycle(fwd2_));
  EXPECT_TRUE(IsForwardCycle(fwd4_));

  EXPECT_FALSE(IsForwardCycle(bwd2_));
  EXPECT_FALSE(IsForwardCycle(bwd4_));

  EXPECT_FALSE(SourceTargetPairs::IsForwardCycle(
      SourceTargetPairs({{3, 0}}), SourceTargetPairs({{0, 2}, {2, 3}, {3, 0}})))
      << "Skip 1";
}

TEST_F(CollectivePermuteUtilsTest, IsBackwardCycle) {
  EXPECT_TRUE(IsBackwardCycle(bwd2_));
  EXPECT_TRUE(IsBackwardCycle(bwd4_));

  EXPECT_FALSE(IsBackwardCycle(fwd2_));
  EXPECT_FALSE(IsBackwardCycle(fwd4_));
}

TEST_F(CollectivePermuteUtilsTest, SourceTargetPairsString) {
  EXPECT_EQ(fwd2_.cycle.ToString(), "{{0,1},{1,0}}");
  EXPECT_EQ(bwd2_.cycle.ToString(), "{{0,1},{1,0}}");
  EXPECT_EQ(fwd4_.cycle.ToString(), "{{0,1},{1,2},{2,3},{3,0}}");
  EXPECT_EQ(bwd4_.cycle.ToString(), "{{0,3},{1,0},{2,1},{3,2}}");
}

TEST_F(CollectivePermuteUtilsTest, SplitEdges) {
  EXPECT_EQ(fwd2_.cycle.SplitEdges(CycleType::kForward),
            std::make_pair(fwd2_.back_edge, fwd2_.main_edge));
  EXPECT_EQ(bwd2_.cycle.SplitEdges(CycleType::kBackward),
            std::make_pair(bwd2_.back_edge, bwd2_.main_edge));

  EXPECT_EQ(fwd4_.cycle.SplitEdges(CycleType::kForward),
            std::make_pair(fwd4_.back_edge, fwd4_.main_edge));
  EXPECT_EQ(bwd4_.cycle.SplitEdges(CycleType::kBackward),
            std::make_pair(bwd4_.back_edge, bwd4_.main_edge));
}

TEST_F(CollectivePermuteUtilsTest, IsForwardCycle2) {
  EXPECT_TRUE(fwd2_.cycle.IsForwardCycle());
  // Two element cycle is can be interpreted as a forward or backward cycle.
  EXPECT_TRUE(bwd2_.cycle.IsForwardCycle());
  EXPECT_TRUE(fwd4_.cycle.IsForwardCycle());
  EXPECT_FALSE(bwd4_.cycle.IsForwardCycle());
  EXPECT_FALSE(fwd4_.main_edge.IsForwardCycle());
  EXPECT_FALSE(bwd4_.main_edge.IsForwardCycle());
}

TEST_F(CollectivePermuteUtilsTest, IsBackwardCycle2) {
  EXPECT_TRUE(fwd2_.cycle.IsBackwardCycle());
  // Two element cycle is can be interpreted as a forward or backward cycle.
  EXPECT_TRUE(bwd2_.cycle.IsBackwardCycle());
  EXPECT_TRUE(bwd4_.cycle.IsBackwardCycle());
  EXPECT_FALSE(fwd4_.cycle.IsBackwardCycle());
  EXPECT_FALSE(fwd4_.main_edge.IsBackwardCycle());
  EXPECT_FALSE(bwd4_.main_edge.IsBackwardCycle());
}

TEST_F(CollectivePermuteUtilsTest, GetMaxDeviceNum) {
  EXPECT_EQ(fwd2_.cycle.GetMaxDeviceNum(), 1);
  EXPECT_EQ(bwd2_.cycle.GetMaxDeviceNum(), 1);
  EXPECT_EQ(fwd4_.cycle.GetMaxDeviceNum(), 3);
  EXPECT_EQ(bwd4_.cycle.GetMaxDeviceNum(), 3);
  EXPECT_EQ(
      SourceTargetPairs({{0, 1}, {1, 2}, {2, 300}, {3, 4}}).GetMaxDeviceNum(),
      300);
}

TEST_F(CollectivePermuteUtilsTest, GetCycleType) {
  EXPECT_EQ(fwd2_.cycle.GetCycleType(), CycleType::kForward);
  // Two element cycle is can be interpreted as a forward or backward cycle.
  // Forward cycle takes precedence.
  EXPECT_EQ(bwd2_.cycle.GetCycleType(), CycleType::kForward);
  EXPECT_EQ(fwd4_.cycle.GetCycleType(), CycleType::kForward);
  EXPECT_EQ(bwd4_.cycle.GetCycleType(), CycleType::kBackward);

  EXPECT_EQ(SourceTargetPairs({{}}).GetCycleType(), CycleType::kUnknown);
  EXPECT_EQ(SourceTargetPairs({{0, 0}}).GetCycleType(), CycleType::kUnknown);
  EXPECT_EQ(SourceTargetPairs({{0, 1}}).GetCycleType(), CycleType::kUnknown);
  EXPECT_EQ(SourceTargetPairs({{1, 0}}).GetCycleType(), CycleType::kUnknown);
  EXPECT_EQ(fwd4_.main_edge.GetCycleType(), CycleType::kUnknown);
  EXPECT_EQ(bwd4_.main_edge.GetCycleType(), CycleType::kUnknown);

  EXPECT_EQ(SourceTargetPairs({{0, 1}, {2, 0}}).GetCycleType(),
            CycleType::kUnknown)
      << "No link from 1 to 2";
  EXPECT_EQ(SourceTargetPairs({{0, 2}, {2, 1}}).GetCycleType(),
            CycleType::kUnknown)
      << "No link from 1 to 0";

  EXPECT_EQ(SourceTargetPairs({{3, 0}, {0, 1}, {1, 2}, {2, 3}}).GetCycleType(),
            CycleType::kUnknown)
      << "misplaced 3->0";
  EXPECT_EQ(SourceTargetPairs({{0, 1}, {1, 2}, {2, 3}, {4, 5}, {3, 0}})
                .GetCycleType(),
            CycleType::kUnknown)
      << "misplaced 4->5";

  EXPECT_EQ(SourceTargetPairs({{3, 2}, {0, 3}, {1, 0}, {2, 1}}).GetCycleType(),
            CycleType::kUnknown)
      << "misplaced 3->2";

  EXPECT_EQ(SourceTargetPairs({{0, 1}, {1, 2}, {2, 3}, {3, 1}}).GetCycleType(),
            CycleType::kUnknown)
      << "Lasso 3->1";
  EXPECT_EQ(SourceTargetPairs({{0, 3}, {1, 0}, {2, 1}, {3, 1}}).GetCycleType(),
            CycleType::kUnknown)
      << "Lasso 3->1";
}

}  // namespace
}  // namespace xla
