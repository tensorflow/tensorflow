/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/service/all_gather_combiner.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::testing::Matcher;
namespace op = xla::testing::opcode_matchers;
int64_t kMaxCombineCount = 256;

std::vector<HloAllGatherInstruction*> FindAllGathers(const HloModule& module) {
  std::vector<HloAllGatherInstruction*> results;
  for (HloComputation* computation : module.computations()) {
    if (computation->IsFusionComputation()) {
      continue;
    }
    for (HloInstruction* hlo : computation->instructions()) {
      if (auto it = DynCast<HloAllGatherInstruction>(hlo)) {
        results.push_back(it);
      }
    }
  }
  return results;
}

int64_t AllGatherCount(const HloModule& module) {
  return FindAllGathers(module).size();
}

using AllGatherCombinerTest = HloTestBase;

// Tests combination of several AllGather instructions.
TEST_F(AllGatherCombinerTest, CombineAllGathers) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  param0 = f32[32] parameter(0)
  param1 = f32[32] parameter(1)
  allgather0 = f32[128] all-gather(param0), replica_groups={}, dimensions={0}
  allgather1 = f32[128] all-gather(param1), replica_groups={}, dimensions={0}
  ROOT tuple = (f32[128], f32[128]) tuple(allgather0, allgather1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AllGatherCombiner combine(1024 * 1024, kMaxCombineCount,
                            /*combine_by_dim=*/true);
  ASSERT_EQ(AllGatherCount(*module), 2);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_TRUE(changed);

  Matcher<const HloInstruction*> combined_all_gather =
      op::AllGather(op::Parameter(0), op::Parameter(1));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::GetTupleElement(combined_all_gather, 0),
                        op::GetTupleElement(combined_all_gather, 1)));
}

// Tests combination of several cross replica gather instructions with
// different gather dimensions.
TEST_F(AllGatherCombinerTest, CombineAllGathersByAllGatherDimension) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  param0 = f32[2,2] parameter(0)
  param1 = f32[2,2] parameter(1)
  param2 = f32[2,2] parameter(2)
  param3 = f32[2,2] parameter(3)
  param4 = f32[2,2] parameter(4)
  allgather0 = f32[8,2] all-gather(param0), replica_groups={}, dimensions={0}
  allgather1 = f32[8,2] all-gather(param1), replica_groups={}, dimensions={0}
  allgather2 = f32[2,8] all-gather(param2), replica_groups={}, dimensions={1}
  allgather3 = f32[2,8] all-gather(param3), replica_groups={}, dimensions={1}
  allgather4 = f32[8,2] all-gather(param4), replica_groups={}, dimensions={0}
  ROOT tuple = (f32[8,2], f32[8,2], f32[2,8], f32[2,8], f32[8,2])
    tuple(allgather0, allgather1, allgather2, allgather3, allgather4)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AllGatherCombiner combine(1024 * 1024, kMaxCombineCount,
                            /*combine_by_dim=*/true);
  ASSERT_EQ(AllGatherCount(*module), 5);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_TRUE(changed);

  Matcher<const HloInstruction*> combined_all_gather0 =
      op::AllGather(op::Parameter(0), op::Parameter(1), op::Parameter(4));
  Matcher<const HloInstruction*> combined_all_gather1 =
      op::AllGather(op::Parameter(2), op::Parameter(3));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::GetTupleElement(combined_all_gather0, 0),
                        op::GetTupleElement(combined_all_gather0, 1),
                        op::GetTupleElement(combined_all_gather1, 0),
                        op::GetTupleElement(combined_all_gather1, 1),
                        op::GetTupleElement(combined_all_gather0, 2)));
}

// Tests that the combination threshold is respected.
TEST_F(AllGatherCombinerTest, DoNotCombineOverThreshold) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  param0 = f32[8] parameter(0)
  param1 = f32[8] parameter(1)
  allgather0 = f32[32] all-gather(param0), replica_groups={}, dimensions={0}
  allgather1 = f32[32] all-gather(param1), replica_groups={}, dimensions={0}
  ROOT tuple = (f32[32], f32[32]) tuple(allgather0, allgather1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  // Run the AllGather combiner optimization pass with threshold less than
  // the combined size of the all gather ops so that the combination
  // cannot occur.
  AllGatherCombiner combine(255, kMaxCombineCount,
                            /*combine_by_dim=*/true);
  ASSERT_EQ(AllGatherCount(*module), 2);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_EQ(AllGatherCount(*module), 2);
  EXPECT_FALSE(changed);
}

// Tests that the combination threshold is respected.
TEST_F(AllGatherCombinerTest, CombineUpToThreshold) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  param0 = f32[8] parameter(0)
  param1 = f32[8] parameter(1)
  allgather0 = f32[32] all-gather(param0), replica_groups={}, dimensions={0}
  allgather1 = f32[32] all-gather(param1), replica_groups={}, dimensions={0}
  ROOT tuple = (f32[32], f32[32]) tuple(allgather0, allgather1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  // Run the AllGather combiner optimization pass with a threshold just higher
  // than that required such that the combination can occur.
  AllGatherCombiner combine(256, kMaxCombineCount,
                            /*combine_by_dim=*/true);
  ASSERT_EQ(AllGatherCount(*module), 2);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_EQ(AllGatherCount(*module), 1);
  EXPECT_TRUE(changed);
}

// Tests that dependent all gathers are not combined.
TEST_F(AllGatherCombinerTest, NoDependentCombination) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  param = f32[1] parameter(0)
  allgather0 = f32[2] all-gather(param), replica_groups={}, dimensions={0}
  ROOT allgather1 = f32[4] all-gather(allgather0), replica_groups={},
      dimensions={0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AllGatherCombiner combine(1024 * 1024, kMaxCombineCount,
                            /*combine_by_dim=*/true);
  ASSERT_EQ(AllGatherCount(*module), 2);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_EQ(AllGatherCount(*module), 2);
  EXPECT_FALSE(changed);
}

// Tests that AllGather ops with different groups are not combined.
TEST_F(AllGatherCombinerTest, NoDifferentReplicaGroupsCombination) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  param0 = f32[32] parameter(0)
  param1 = f32[32] parameter(1)
  allgather0 = f32[64] all-gather(param0), replica_groups={{0, 1}, {2, 3}},
    dimensions={0}
  allgather1 = f32[64] all-gather(param1), replica_groups={{0, 2}, {1, 3}},
    dimensions={0}
  ROOT tuple = (f32[64], f32[64]) tuple(allgather0, allgather1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AllGatherCombiner combine(1024 * 1024, kMaxCombineCount,
                            /*combine_by_dim=*/true);
  ASSERT_EQ(AllGatherCount(*module), 2);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_EQ(AllGatherCount(*module), 2);
  EXPECT_FALSE(changed);
}

TEST_F(AllGatherCombinerTest, DomainPreventsCombining) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  param0 = f32[32] parameter(0), sharding={maximal device=0}
  param1 = f32[32] parameter(1), sharding={maximal device=1}
  allgather0 = f32[128] all-gather(param0),
    replica_groups={}, dimensions={0}, sharding={maximal device=0}
  allgather1 = f32[128] all-gather(param1),
    replica_groups={}, dimensions={0}, sharding={maximal device=1}
  domain0 = f32[128] domain(allgather0),
    domain={kind="sharding", entry={{maximal device=0}, {maximal device=1}},
    exit={maximal device=0}}
  domain1 = f32[128] domain(allgather1),
    domain={kind="sharding", entry={{maximal device=0}, {maximal device=1}},
    exit={maximal device=1}}
  ROOT tuple = (f32[128], f32[128]) tuple(domain0, domain1),
    sharding={{maximal device=0}, {maximal device=1}}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AllGatherCombiner combine(1024 * 1024, kMaxCombineCount,
                            /*combine_by_dim=*/true);
  ASSERT_EQ(AllGatherCount(*module), 2);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_EQ(AllGatherCount(*module), 2);
  EXPECT_FALSE(changed);
}

// This test checks that two AllGather instructions that are in separate domains
// but with the same domain metadata can be combined.
TEST_F(AllGatherCombinerTest, CombineFromTwoDomainsWithSameMetadata) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  param0 = f32[32] parameter(0), sharding={maximal device=0}
  param1 = f32[32] parameter(1), sharding={maximal device=1}
  param2 = f32[32] parameter(2), sharding={maximal device=1}
  allgather0 = f32[128] all-gather(param0),
    replica_groups={}, dimensions={0}, sharding={maximal device=0}
  allgather1 = f32[128] all-gather(param1),
    replica_groups={}, dimensions={0}, sharding={maximal device=1}
  allgather2 = f32[128] all-gather(param2),
    replica_groups={}, dimensions={0}, sharding={maximal device=0}
  domain0 = f32[128] domain(allgather0),
    domain={kind="sharding", entry={{maximal device=0}, {maximal device=1},
    {maximal device=0}}, exit={maximal device=0}}
  domain1 = f32[128] domain(allgather1),
    domain={kind="sharding", entry={{maximal device=0}, {maximal device=1},
    {maximal device=0}}, exit={maximal device=1}}
  domain2 = f32[128] domain(allgather2),
    domain={kind="sharding", entry={{maximal device=0}, {maximal device=1},
    {maximal device=0}}, exit={maximal device=0}}
  ROOT tuple = (f32[128], f32[128], f32[128]) tuple(domain0, domain1,
  domain2),
    sharding={{maximal device=0}, {maximal device=1}, {maximal device=0}}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AllGatherCombiner combine(1024 * 1024, kMaxCombineCount,
                            /*combine_by_dim=*/true);
  ASSERT_EQ(AllGatherCount(*module), 3);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_EQ(AllGatherCount(*module), 2);
  EXPECT_TRUE(changed);

  // Verify that the sharding is combined correctly.
  const HloInstruction* param0 =
      module->entry_computation()->parameter_instruction(0);
  ASSERT_EQ(param0->user_count(), 1);
  const HloInstruction* combined_ag = param0->users().front();
  ASSERT_EQ(combined_ag->opcode(), HloOpcode::kAllGather);
  EXPECT_THAT(combined_ag,
              op::Sharding("{{maximal device=0}, {maximal device=0}}"));
}

TEST_F(AllGatherCombinerTest, CombineAllGathersDifferentDims) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  param0 = f32[2,3]{1,0} parameter(0)
  param1 = f32[2,3]{0,1} parameter(1)
  allgather0 = f32[8,3]{1,0} all-gather(param0), replica_groups={},
      dimensions={0}
  allgather1 = f32[2,12]{0,1} all-gather(param1), replica_groups={},
      dimensions={1}
  ROOT tuple = (f32[8,3]{1,0}, f32[2,12]{0,1}) tuple(allgather0, allgather1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AllGatherCombiner combine(1024 * 1024, kMaxCombineCount,
                            /*combine_by_dim=*/false);
  ASSERT_EQ(AllGatherCount(*module), 2);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_TRUE(changed);

  Matcher<const HloInstruction*> combined_all_gather =
      op::AllGather(op::Parameter(0), op::Bitcast(op::Parameter(1)));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(op::GetTupleElement(combined_all_gather, 0),
                op::Bitcast(op::GetTupleElement(combined_all_gather, 1))));
}

TEST_F(AllGatherCombinerTest, CombineManyAllGathersDifferentDims) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  param0 = f32[2,7]{1,0} parameter(0)
  param1 = f32[3,8]{1,0} parameter(1)
  param2 = f32[4,9]{0,1} parameter(2)
  param3 = f32[5,10]{0,1} parameter(3)
  param4 = f32[6,11]{1,0} parameter(4)
  allgather0 = f32[8,7]{1,0} all-gather(param0), replica_groups={},
      dimensions={0}
  allgather1 = f32[12,8]{1,0} all-gather(param1), replica_groups={},
      dimensions={0}
  allgather2 = f32[4,36]{0,1} all-gather(param2), replica_groups={},
      dimensions={1}
  allgather3 = f32[5,40]{0,1} all-gather(param3), replica_groups={},
      dimensions={1}
  allgather4 = f32[24,11]{1,0} all-gather(param4), replica_groups={},
      dimensions={0}
  ROOT tuple = (f32[8,7]{1,0}, f32[12,8]{1,0}, f32[4,36]{0,1}, f32[5,40]{0,1},
      f32[24,11]{1,0}) tuple(allgather0, allgather1, allgather2, allgather3,
      allgather4)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AllGatherCombiner combine(1024 * 1024, kMaxCombineCount,
                            /*combine_by_dim=*/false);
  ASSERT_EQ(AllGatherCount(*module), 5);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_TRUE(changed);

  Matcher<const HloInstruction*> combined_all_gather = op::AllGather(
      op::Parameter(0), op::Parameter(1), op::Bitcast(op::Parameter(2)),
      op::Bitcast(op::Parameter(3)), op::Parameter(4));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(op::GetTupleElement(combined_all_gather, 0),
                op::GetTupleElement(combined_all_gather, 1),
                op::Bitcast(op::GetTupleElement(combined_all_gather, 2)),
                op::Bitcast(op::GetTupleElement(combined_all_gather, 3)),
                op::GetTupleElement(combined_all_gather, 4)));
  std::vector<HloAllGatherInstruction*> all_gathers = FindAllGathers(*module);
  ASSERT_EQ(1, all_gathers.size());
  ASSERT_EQ(0, all_gathers.front()->all_gather_dimension());
}

TEST_F(AllGatherCombinerTest, CombineManyAllGathersDifferentDimsRank4) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  param0 = f32[2,7,2,7]{3,2,1,0} parameter(0)
  param1 = f32[3,8,3,8]{3,2,1,0} parameter(1)
  param2 = f32[4,9,4,9]{3,0,1,2} parameter(2)
  param3 = f32[5,10,5,10]{3,0,1,2} parameter(3)
  param4 = f32[6,11,6,11]{3,2,1,0} parameter(4)
  allgather0 = f32[8,7,2,7]{3,2,1,0} all-gather(param0), replica_groups={},
      dimensions={0}
  allgather1 = f32[12,8,3,8]{3,2,1,0} all-gather(param1), replica_groups={},
      dimensions={0}
  allgather2 = f32[4,9,16,9]{3,0,1,2} all-gather(param2), replica_groups={},
      dimensions={2}
  allgather3 = f32[5,10,20,10]{3,0,1,2} all-gather(param3), replica_groups={},
      dimensions={2}
  allgather4 = f32[24,11,6,11]{3,2,1,0} all-gather(param4), replica_groups={},
      dimensions={0}
  ROOT tuple = (f32[8,7,2,7]{3,2,1,0}, f32[12,8,3,8]{3,2,1,0},
      f32[4,9,16,9]{3,0,1,2}, f32[5,10,20,10]{3,0,1,2},
      f32[24,11,6,11]{3,2,1,0}) tuple(allgather0, allgather1, allgather2,
      allgather3, allgather4)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AllGatherCombiner combine(1024 * 1024, kMaxCombineCount,
                            /*combine_by_dim=*/false);
  ASSERT_EQ(AllGatherCount(*module), 5);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_TRUE(changed);

  Matcher<const HloInstruction*> combined_all_gather = op::AllGather(
      op::Parameter(0), op::Parameter(1), op::Bitcast(op::Parameter(2)),
      op::Bitcast(op::Parameter(3)), op::Parameter(4));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(op::GetTupleElement(combined_all_gather, 0),
                op::GetTupleElement(combined_all_gather, 1),
                op::Bitcast(op::GetTupleElement(combined_all_gather, 2)),
                op::Bitcast(op::GetTupleElement(combined_all_gather, 3)),
                op::GetTupleElement(combined_all_gather, 4)));
  std::vector<HloAllGatherInstruction*> all_gathers = FindAllGathers(*module);
  ASSERT_EQ(1, all_gathers.size());
  ASSERT_EQ(0, all_gathers.front()->all_gather_dimension());
}

TEST_F(AllGatherCombinerTest, CombineManyAllGathersDifferentDimsMixedRanks) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  param0 = f32[2,7]{1,0} parameter(0)
  param1 = f32[3,8]{1,0} parameter(1)
  param2 = f32[4,9]{0,1} parameter(2)
  param3 = f32[5,10]{0,1} parameter(3)
  param4 = f32[6]{0} parameter(4)
  allgather0 = f32[2,28]{1,0} all-gather(param0), replica_groups={},
      dimensions={1}
  allgather1 = f32[3,32]{1,0} all-gather(param1), replica_groups={},
      dimensions={1}
  allgather2 = f32[4,36]{0,1} all-gather(param2), replica_groups={},
      dimensions={1}
  allgather3 = f32[5,40]{0,1} all-gather(param3), replica_groups={},
      dimensions={1}
  allgather4 = f32[24]{0} all-gather(param4), replica_groups={},
      dimensions={0}
  ROOT tuple = (f32[2,28]{1,0}, f32[3,32]{1,0}, f32[4,36]{0,1}, f32[5,40]{0,1},
      f32[24]{0}) tuple(allgather0, allgather1, allgather2, allgather3,
      allgather4)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AllGatherCombiner combine(1024 * 1024, kMaxCombineCount,
                            /*combine_by_dim=*/false);
  ASSERT_EQ(AllGatherCount(*module), 5);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_TRUE(changed);

  Matcher<const HloInstruction*> combined_all_gather = op::AllGather(
      op::Bitcast(op::Parameter(0)), op::Bitcast(op::Parameter(1)),
      op::Bitcast(op::Parameter(2)), op::Bitcast(op::Parameter(3)),
      op::Parameter(4));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(op::Bitcast(op::GetTupleElement(combined_all_gather, 0)),
                op::Bitcast(op::GetTupleElement(combined_all_gather, 1)),
                op::Bitcast(op::GetTupleElement(combined_all_gather, 2)),
                op::Bitcast(op::GetTupleElement(combined_all_gather, 3)),
                op::GetTupleElement(combined_all_gather, 4)));
  std::vector<HloAllGatherInstruction*> all_gathers = FindAllGathers(*module);
  ASSERT_EQ(1, all_gathers.size());

  // when using different ranks and the most frequent AG dim (1) is not valid
  // for rank 1 shape, we use default dim 0.
  ASSERT_EQ(0, all_gathers.front()->all_gather_dimension());
}

TEST_F(AllGatherCombinerTest, CombineAllGathersByDim) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  param0 = f32[2,7]{1,0} parameter(0)
  param1 = f32[3,8]{1,0} parameter(1)
  param2 = f32[4,9]{0,1} parameter(2)
  param3 = f32[5,10]{0,1} parameter(3)
  param4 = f32[6,11]{1,0} parameter(4)
  allgather0 = f32[8,7]{1,0} all-gather(param0), replica_groups={},
      dimensions={0}
  allgather1 = f32[12,8]{1,0} all-gather(param1), replica_groups={},
      dimensions={0}
  allgather2 = f32[4,36]{0,1} all-gather(param2), replica_groups={},
      dimensions={1}
  allgather3 = f32[5,40]{0,1} all-gather(param3), replica_groups={},
      dimensions={1}
  allgather4 = f32[24,11]{1,0} all-gather(param4), replica_groups={},
      dimensions={0}
  ROOT tuple = (f32[8,7]{1,0}, f32[12,8]{1,0}, f32[4,36]{0,1}, f32[5,40]{0,1},
      f32[24,11]{1,0}) tuple(allgather0, allgather1, allgather2, allgather3,
      allgather4)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AllGatherCombiner combine(1024 * 1024, kMaxCombineCount,
                            /*combine_by_dim=*/true);
  ASSERT_EQ(AllGatherCount(*module), 5);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_TRUE(changed);

  Matcher<const HloInstruction*> combined_all_gather_0 =
      op::AllGather(op::Parameter(0), op::Parameter(1), op::Parameter(4));
  Matcher<const HloInstruction*> combined_all_gather_1 =
      op::AllGather(op::Parameter(2), op::Parameter(3));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::GetTupleElement(combined_all_gather_0, 0),
                        op::GetTupleElement(combined_all_gather_0, 1),
                        op::GetTupleElement(combined_all_gather_1, 0),
                        op::GetTupleElement(combined_all_gather_1, 1),
                        op::GetTupleElement(combined_all_gather_0, 2)));
  std::vector<HloAllGatherInstruction*> all_gathers = FindAllGathers(*module);
  ASSERT_EQ(2, all_gathers.size());
  ASSERT_EQ(0, all_gathers[0]->all_gather_dimension());
  ASSERT_EQ(1, all_gathers[1]->all_gather_dimension());
}

}  // namespace
}  // namespace xla
