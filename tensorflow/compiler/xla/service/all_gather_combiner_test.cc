/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/all_gather_combiner.h"

#include <memory>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

using ::testing::Matcher;
namespace op = xla::testing::opcode_matchers;
int64_t kMaxCombineCount = 256;

int64_t AllGatherCount(const HloModule& module) {
  int64_t count = 0;
  for (HloComputation* computation : module.computations()) {
    if (computation->IsFusionComputation()) {
      continue;
    }
    for (HloInstruction* hlo : computation->instructions()) {
      if (hlo->opcode() == HloOpcode::kAllGather) {
        ++count;
      }
    }
  }
  return count;
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

  AllGatherCombiner combine(1024 * 1024, kMaxCombineCount);
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

  AllGatherCombiner combine(1024 * 1024, kMaxCombineCount);
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
  AllGatherCombiner combine(255, kMaxCombineCount);
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
  AllGatherCombiner combine(256, kMaxCombineCount);
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
  ROOT allgather1 = f32[4] all-gather(allgather0), replica_groups={}, dimensions={0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AllGatherCombiner combine(1024 * 1024, kMaxCombineCount);
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

  AllGatherCombiner combine(1024 * 1024, kMaxCombineCount);
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

  AllGatherCombiner combine(1024 * 1024, kMaxCombineCount);
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

  AllGatherCombiner combine(1024 * 1024, kMaxCombineCount);
  ASSERT_EQ(AllGatherCount(*module), 3);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_EQ(AllGatherCount(*module), 2);
  EXPECT_TRUE(changed);
}

TEST_F(AllGatherCombinerTest, DoNotCombineCrossShardAndCrossReplicaInSPMD) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  param0 = f32[32] parameter(0), sharding={maximal device=0}
  param1 = f32[32] parameter(1), sharding={maximal device=1}
  cross_shard_ag = f32[128] all-gather(param0),
    replica_groups={{0}}, dimensions={0}, channel_id=1
  cross_replica_ag = f32[128] all-gather(param1),
    replica_groups={{0}}, dimensions={0}, sharding={maximal device=1}
  ROOT tuple = (f32[128], f32[128]) tuple(cross_shard_ag, cross_replica_ag)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AllGatherCombiner combine(1024 * 1024, kMaxCombineCount);
  ASSERT_EQ(AllGatherCount(*module), 2);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_EQ(AllGatherCount(*module), 2);
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace xla
