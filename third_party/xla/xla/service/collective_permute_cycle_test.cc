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

#include "xla/service/collective_permute_cycle.h"

#include <utility>

#include <gtest/gtest.h>
#include "xla/service/source_target_pairs.h"

namespace xla {
namespace collective_permute_cycle {
namespace {

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
};

TEST_F(CollectivePermuteUtilsTest, HasCycles) {
  EXPECT_TRUE(HasCycles(fwd2_.cycle));
  EXPECT_TRUE(HasCycles(bwd2_.cycle));
  EXPECT_TRUE(HasCycles(fwd4_.cycle));
  EXPECT_TRUE(HasCycles(bwd4_.cycle));

  EXPECT_TRUE(HasCycles(SourceTargetPairs({{0, 1}, {1, 2}, {2, 3}, {3, 2}})));
  EXPECT_TRUE(HasCycles(SourceTargetPairs({{0, 1}, {1, 2}, {2, 3}, {3, 1}})));

  EXPECT_FALSE(HasCycles(SourceTargetPairs({{1, 2}, {2, 3}, {3, 0}})))
      << "Forward only";
  EXPECT_FALSE(HasCycles(SourceTargetPairs({{1, 2}}))) << "Single edge";
}

TEST_F(CollectivePermuteUtilsTest, IsForwardCycle) {
  EXPECT_TRUE(IsForwardCycle(fwd2_.back_edge, fwd2_.main_edge));
  EXPECT_TRUE(IsForwardCycle(fwd4_.back_edge, fwd4_.main_edge));

  EXPECT_FALSE(IsForwardCycle(bwd2_.back_edge, bwd2_.main_edge));
  EXPECT_FALSE(IsForwardCycle(bwd4_.back_edge, bwd4_.main_edge));

  EXPECT_FALSE(IsForwardCycle(SourceTargetPairs({{3, 0}}),
                              SourceTargetPairs({{0, 2}, {2, 3}, {3, 0}})))
      << "Skip 1";
}

TEST_F(CollectivePermuteUtilsTest, IsBackwardCycle) {
  EXPECT_TRUE(IsBackwardCycle(bwd2_.back_edge, bwd2_.main_edge));
  EXPECT_TRUE(IsBackwardCycle(bwd4_.back_edge, bwd4_.main_edge));

  EXPECT_FALSE(IsBackwardCycle(fwd2_.back_edge, fwd2_.main_edge));
  EXPECT_FALSE(IsBackwardCycle(fwd4_.back_edge, fwd4_.main_edge));
}

TEST_F(CollectivePermuteUtilsTest, SplitEdges) {
  EXPECT_EQ(SplitEdges(fwd2_.cycle, CycleType::kForward),
            std::make_pair(fwd2_.back_edge, fwd2_.main_edge));
  EXPECT_EQ(SplitEdges(bwd2_.cycle, CycleType::kBackward),
            std::make_pair(bwd2_.back_edge, bwd2_.main_edge));

  EXPECT_EQ(SplitEdges(fwd4_.cycle, CycleType::kForward),
            std::make_pair(fwd4_.back_edge, fwd4_.main_edge));
  EXPECT_EQ(SplitEdges(bwd4_.cycle, CycleType::kBackward),
            std::make_pair(bwd4_.back_edge, bwd4_.main_edge));
}

TEST_F(CollectivePermuteUtilsTest, IsForwardCycle2) {
  EXPECT_TRUE(IsForwardCycle(fwd2_.cycle));
  // Two element cycle is can be interpreted as a forward or backward cycle.
  EXPECT_TRUE(IsForwardCycle(bwd2_.cycle));
  EXPECT_TRUE(IsForwardCycle(fwd4_.cycle));
  EXPECT_FALSE(IsForwardCycle(bwd4_.cycle));
  EXPECT_FALSE(IsForwardCycle(fwd4_.main_edge));
  EXPECT_FALSE(IsForwardCycle(bwd4_.main_edge));
}

TEST_F(CollectivePermuteUtilsTest, IsBackwardCycle2) {
  EXPECT_TRUE(IsBackwardCycle(fwd2_.cycle));
  // Two element cycle is can be interpreted as a forward or backward cycle.
  EXPECT_TRUE(IsBackwardCycle(bwd2_.cycle));
  EXPECT_TRUE(IsBackwardCycle(bwd4_.cycle));
  EXPECT_FALSE(IsBackwardCycle(fwd4_.cycle));
  EXPECT_FALSE(IsBackwardCycle(fwd4_.main_edge));
  EXPECT_FALSE(IsBackwardCycle(bwd4_.main_edge));
}

TEST_F(CollectivePermuteUtilsTest, GetCycleType) {
  EXPECT_EQ(GetCycleType(fwd2_.cycle), CycleType::kForward);
  // Two element cycle is can be interpreted as a forward or backward cycle.
  // Forward cycle takes precedence.
  EXPECT_EQ(GetCycleType(bwd2_.cycle), CycleType::kForward);
  EXPECT_EQ(GetCycleType(fwd4_.cycle), CycleType::kForward);
  EXPECT_EQ(GetCycleType(bwd4_.cycle), CycleType::kBackward);

  EXPECT_EQ(GetCycleType(SourceTargetPairs()), CycleType::kNone);
  EXPECT_EQ(GetCycleType(SourceTargetPairs({{0, 0}})), CycleType::kNone);
  EXPECT_EQ(GetCycleType(SourceTargetPairs({{0, 1}})), CycleType::kNone);
  EXPECT_EQ(GetCycleType(SourceTargetPairs({{1, 0}})), CycleType::kNone);
  EXPECT_EQ(GetCycleType(fwd4_.main_edge), CycleType::kNone);
  EXPECT_EQ(GetCycleType(bwd4_.main_edge), CycleType::kNone);

  EXPECT_EQ(GetCycleType(SourceTargetPairs({{0, 1}, {2, 0}})), CycleType::kNone)
      << "No link from 1 to 2";
  EXPECT_EQ(GetCycleType(SourceTargetPairs({{0, 2}, {2, 1}})), CycleType::kNone)
      << "No link from 1 to 0";

  EXPECT_EQ(GetCycleType(SourceTargetPairs({{3, 0}, {0, 1}, {1, 2}, {2, 3}})),
            CycleType::kNone)
      << "misplaced 3->0";
  EXPECT_EQ(
      GetCycleType(SourceTargetPairs({{0, 1}, {1, 2}, {2, 3}, {4, 5}, {3, 0}})),
      CycleType::kNone)
      << "misplaced 4->5";

  EXPECT_EQ(GetCycleType(SourceTargetPairs({{3, 2}, {0, 3}, {1, 0}, {2, 1}})),
            CycleType::kNone)
      << "misplaced 3->2";

  EXPECT_EQ(GetCycleType(SourceTargetPairs({{0, 1}, {1, 2}, {2, 3}, {3, 1}})),
            CycleType::kNone)
      << "Lasso 3->1";
  EXPECT_EQ(GetCycleType(SourceTargetPairs({{0, 3}, {1, 0}, {2, 1}, {3, 1}})),
            CycleType::kNone)
      << "Lasso 3->1";
}

TEST_F(CollectivePermuteUtilsTest, HasCyclesTwoCycles) {
  // Cycle: 0->1, 1->2, 2->3, 3->0
  // Cycle: 4->5, 5->6, 6->7, 7->4
  EXPECT_TRUE(HasCycles(SourceTargetPairs(
      {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}})));
}

TEST_F(CollectivePermuteUtilsTest, HasCyclesOneCycleAndOneAlmostCycle) {
  // Not a cycle: 0->1, 1->2, 2->3 (missing: 3->4)
  // Cycle:       4->5, 5->6, 6->7, 7->4
  EXPECT_TRUE(HasCycles(SourceTargetPairs(
      {{0, 1}, {1, 2}, {2, 3}, {4, 5}, {5, 6}, {6, 7}, {7, 4}})));
}

TEST_F(CollectivePermuteUtilsTest, HasCyclesTwoAlmostCycles) {
  // Not a cycle: 0->1, 1->2, 3->0 (missing: 2->3)
  // Not a cycle: 4->5, 5->6, 7->4 (missing: 6->7)
  EXPECT_FALSE(HasCycles(
      SourceTargetPairs({{0, 1}, {1, 2}, {3, 0}, {4, 5}, {5, 6}, {7, 4}})));
}

TEST_F(CollectivePermuteUtilsTest, HasCyclesTwoCyclesInterleaved) {
  // Cycle: 0->2, 2->4, 4->6, 6->0
  // Cycle: 1->3, 3->5, 5->7, 7->1
  EXPECT_TRUE(HasCycles(SourceTargetPairs(
      {{0, 2}, {2, 4}, {4, 6}, {6, 0}, {1, 3}, {3, 5}, {5, 7}, {7, 1}})));
}

TEST_F(CollectivePermuteUtilsTest, HasCyclesSimpleCycle) {
  // Cycle: 0->1, 1->2, 2->3, 3->4, 4->5, 5->6, 6->7, 7->0
  EXPECT_TRUE(HasCycles(SourceTargetPairs(
      {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 0}})));
}

TEST_F(CollectivePermuteUtilsTest, HasCyclesSimpleAlmostCycle) {
  // Not a cycle: 0->1, 1->2, 2->3, 4->5, 5->6, 6->7, 7->0 (missing: 3->4)
  EXPECT_FALSE(HasCycles(SourceTargetPairs(
      {{0, 1}, {1, 2}, {2, 3}, {4, 5}, {5, 6}, {6, 7}, {7, 0}})));
}

TEST_F(CollectivePermuteUtilsTest, HasCyclesSelfCycle) {
  // Self cycle: 0->0
  EXPECT_TRUE(HasCycles(SourceTargetPairs({{0, 0}})));
}

TEST_F(CollectivePermuteUtilsTest, HasCyclesSkippingFirstDeviceCycle) {
  // Cycle: 1->2, 2->3, 3->4, 4->5, 5->6, 6->7, 7->1
  EXPECT_TRUE(HasCycles(SourceTargetPairs(
      {{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 1}})));
}

}  // namespace
}  // namespace collective_permute_cycle
}  // namespace xla
