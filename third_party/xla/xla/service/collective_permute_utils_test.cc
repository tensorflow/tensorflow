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

#include "xla/service/collective_permute_utils.h"

#include <memory>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"

namespace xla {
namespace cp_utils {

struct Cannonical {
  SourceTargetPairs cycle;
  SourceTargetPairs fwd_edge;
  SourceTargetPairs bwd_edge;
};

class CollectivePermuteUtilsTest : public ::testing::Test {
 protected:
  Cannonical fwd2_ = {
      .cycle = {{0, 1}, {1, 0}}, .fwd_edge = {{0, 1}}, .bwd_edge = {{1, 0}}};
  Cannonical bwd2_ = {
      .cycle = {{1, 0}, {0, 1}}, .fwd_edge = {{1, 0}}, .bwd_edge = {{0, 1}}};
  Cannonical fwd4_ = {.cycle = {{0, 1}, {1, 2}, {2, 3}, {3, 0}},
                      .fwd_edge = {{0, 1}, {1, 2}, {2, 3}},
                      .bwd_edge = {{3, 0}}};
  Cannonical bwd4_ = {.cycle = {{0, 3}, {1, 0}, {2, 1}, {3, 2}},
                      .fwd_edge = {{1, 0}, {2, 1}, {3, 2}},
                      .bwd_edge = {{0, 3}}};
  std::unique_ptr<HloInstruction> simple_input_ = HloInstruction::CreateToken();

  HloCollectivePermuteInstruction CreateCollectivePermute(
      const SourceTargetPairs& pairs) {
    return HloCollectivePermuteInstruction(HloOpcode::kCollectivePermute,
                                           ShapeUtil::MakeShape(U32, {8, 8}),
                                           {simple_input_.get()}, pairs, 1);
  }
};

TEST_F(CollectivePermuteUtilsTest, HasCycles) {
  EXPECT_TRUE(HasCycles(fwd2_.cycle));
  EXPECT_TRUE(HasCycles(bwd2_.cycle));
  EXPECT_TRUE(HasCycles(fwd4_.cycle));
  EXPECT_TRUE(HasCycles(bwd4_.cycle));

  EXPECT_TRUE(HasCycles({{0, 1}, {1, 2}, {2, 3}, {3, 2}})) << "Lasso 3->2";
  EXPECT_TRUE(HasCycles({{0, 1}, {1, 2}, {2, 3}, {3, 1}})) << "Lasso 3->1";

  EXPECT_FALSE(HasCycles({{1, 2}, {2, 3}, {3, 0}})) << "Forward only";
  EXPECT_FALSE(HasCycles({{1, 2}})) << "Single edge";
}

bool IsForwardCycle(Cannonical& canonical) {
  return IsForwardCycle(canonical.bwd_edge[0], canonical.fwd_edge);
}
bool IsBackwardCycle(Cannonical& canonical) {
  return IsBackwardCycle(canonical.bwd_edge[0], canonical.fwd_edge);
}

TEST_F(CollectivePermuteUtilsTest, IsForwardCycle) {
  EXPECT_TRUE(IsForwardCycle(fwd2_));
  EXPECT_TRUE(IsForwardCycle(fwd4_));

  EXPECT_FALSE(IsForwardCycle(bwd2_));
  EXPECT_FALSE(IsForwardCycle(bwd4_));

  EXPECT_FALSE(IsForwardCycle({3, 0}, {{0, 2}, {2, 3}, {3, 0}})) << "Skip 1";
}

TEST_F(CollectivePermuteUtilsTest, IsBackwardCycle) {
  EXPECT_TRUE(IsBackwardCycle(bwd2_));
  EXPECT_TRUE(IsBackwardCycle(bwd4_));

  EXPECT_FALSE(IsBackwardCycle(fwd2_));
  EXPECT_FALSE(IsBackwardCycle(fwd4_));
}

TEST_F(CollectivePermuteUtilsTest, SourceTargetPairsString) {
  EXPECT_EQ(SourceTargetPairsString(CreateCollectivePermute(fwd2_.cycle)),
            "{{0,1},{1,0}}");
  EXPECT_EQ(SourceTargetPairsString(CreateCollectivePermute(bwd2_.cycle)),
            "{{1,0},{0,1}}");
  EXPECT_EQ(SourceTargetPairsString(CreateCollectivePermute(fwd4_.cycle)),
            "{{0,1},{1,2},{2,3},{3,0}}");
  EXPECT_EQ(SourceTargetPairsString(CreateCollectivePermute(bwd4_.cycle)),
            "{{0,3},{1,0},{2,1},{3,2}}");
}

}  // namespace cp_utils
}  // namespace xla
