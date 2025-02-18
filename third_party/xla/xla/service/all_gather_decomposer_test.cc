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

#include "xla/service/all_gather_decomposer.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using ::testing::AllOf;
namespace op = xla::testing::opcode_matchers;
using AllGatherDecomposerTest = HloTestBase;

TEST_F(AllGatherDecomposerTest, CrossReplicaAllGather) {
  const std::string module_str = R"(
HloModule module

ENTRY entry {
  param0 = f32[10,20] parameter(0)
  ROOT ag = f32[10,80] all-gather(param0), replica_groups={}, dimensions={1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((module_str)));
  AllGatherDecomposer decomposer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::AllReduce(op::DynamicUpdateSlice(
          op::Broadcast(op::Constant()), op::Parameter(0), op::Constant(),
          op::Multiply(op::ReplicaId(), op::Constant()))));
}

TEST_F(AllGatherDecomposerTest, CrossReplicaAndPartitionAllGather) {
  const std::string module_str = R"(
HloModule module

ENTRY entry {
  param0 = f32[10,20] parameter(0)
  ROOT ag = f32[10,80] all-gather(param0), replica_groups={{0}}, channel_id=1,
    dimensions={1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((module_str)));
  AllGatherDecomposer decomposer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::AllReduce(op::DynamicUpdateSlice(
          op::Broadcast(op::Constant()), op::Parameter(0), op::Constant(),
          op::Multiply(op::PartitionId(), op::Constant()))));
}

TEST_F(AllGatherDecomposerTest, CrossReplicaAllGatherWithTrivialGroup) {
  const std::string module_str = R"(
HloModule module

ENTRY entry {
  param0 = f32[10,20] parameter(0)
  ROOT ag = f32[10,80] all-gather(param0), replica_groups={{0,1,2,3}},
    dimensions={1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((module_str)));
  AllGatherDecomposer decomposer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::AllReduce(op::DynamicUpdateSlice(
          op::Broadcast(op::Constant()), op::Parameter(0), op::Constant(),
          op::Multiply(op::ReplicaId(), op::Constant()))));
}

TEST_F(AllGatherDecomposerTest, CrossReplicaAllGatherWithSubgroups) {
  const std::string module_str = R"(
HloModule module

ENTRY entry {
  param0 = f32[10,20] parameter(0)
  ROOT ag = f32[10,80] all-gather(param0),
    replica_groups={{2,1,0,3}, {4,6,7,5}}, dimensions={1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((module_str)));
  AllGatherDecomposer decomposer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_TRUE(changed);
  auto id =
      AllOf(op::Shape("u32[]"),
            op::Reshape(op::DynamicSlice(op::Constant(), op::ReplicaId())));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::AllReduce(op::DynamicUpdateSlice(
                  op::Broadcast(op::Constant()), op::Parameter(0),
                  op::Constant(), op::Multiply(id, op::Constant()))));
}

TEST_F(AllGatherDecomposerTest, CrossReplicaAllGatherWithSubgroupsGlobalIds) {
  const std::string module_str = R"(
HloModule module

ENTRY entry {
  param0 = f32[10,20] parameter(0)
  ROOT ag = f32[10,80] all-gather(param0),
    replica_groups={{2,1,0,3}, {4,6,7,5}}, dimensions={1}, channel_id=1,
    use_global_device_ids=true
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((module_str)));
  AllGatherDecomposer decomposer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_TRUE(changed);
  auto global_id =
      op::Add(op::Multiply(op::ReplicaId(), op::Constant()), op::PartitionId());
  auto id = AllOf(op::Shape("u32[]"),
                  op::Reshape(op::DynamicSlice(op::Constant(), global_id)));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::AllReduce(op::DynamicUpdateSlice(
                  op::Broadcast(op::Constant()), op::Parameter(0),
                  op::Constant(), op::Multiply(id, op::Constant()))));
}

TEST_F(AllGatherDecomposerTest, CrossReplicaAllGatherWithTuple) {
  const std::string module_str = R"(
HloModule module

ENTRY entry {
  param0 = f32[10,20] parameter(0)
  param1 = f32[10,16] parameter(1)
  ROOT ag = (f32[10,80], f32[10,64]) all-gather(param0, param1),
    replica_groups={}, dimensions={1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((module_str)));
  AllGatherDecomposer decomposer;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, decomposer.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(
          op::AllReduce(op::DynamicUpdateSlice(
              op::Broadcast(op::Constant()), op::Parameter(0), op::Constant(),
              op::Multiply(op::ReplicaId(), op::Constant()))),
          op::AllReduce(op::DynamicUpdateSlice(
              op::Broadcast(op::Constant()), op::Parameter(1), op::Constant(),
              op::Multiply(op::ReplicaId(), op::Constant())))));
}

}  // namespace
}  // namespace xla
