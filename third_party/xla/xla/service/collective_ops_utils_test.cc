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

#include "xla/service/collective_ops_utils.h"

#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/array2d.h"
#include "xla/hlo/ir/collective_device_list.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal_util.h"
#include "xla/service/collective_permute_cycle.h"
#include "xla/service/computation_placer.h"
#include "xla/service/global_device_id.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using CycleType = collective_permute_cycle::CycleType;

// Creates a container of ReplicaGroups.
std::vector<ReplicaGroup> CreateReplicaGroups(
    const std::vector<std::vector<int64_t>> &replica_groups) {
  std::vector<ReplicaGroup> result;
  result.reserve(replica_groups.size());
  for (const auto &replica_group : replica_groups) {
    ReplicaGroup &group = result.emplace_back();
    for (auto id : replica_group) {
      group.add_replica_ids(id);
    }
  }
  return result;
}

TEST(CollectiveOpsUtilsTest, GetParticipatingIDs_NoReplicaGroups) {
  std::vector<int> actual =
      GetParticipatingIDs(CollectiveOpGroupMode::kFlattenedID,
                          /*current_id=*/0, /*total_participant_count=*/3,
                          /*groups=*/{})
          .value();
  std::vector<int> expected = {0, 1, 2};
  EXPECT_EQ(actual, expected);
}

TEST(CollectiveOpsUtilsTest, GetParticipatingIDs_ReplicaGroups) {
  std::vector<ReplicaGroup> replica_groups(3);
  replica_groups[0].add_replica_ids(0);
  replica_groups[0].add_replica_ids(4);
  replica_groups[1].add_replica_ids(1);
  replica_groups[1].add_replica_ids(5);
  replica_groups[2].add_replica_ids(2);
  replica_groups[2].add_replica_ids(3);

  std::vector<int> actual =
      GetParticipatingIDs(CollectiveOpGroupMode::kFlattenedID,
                          /*current_id=*/1,
                          /*total_participant_count=*/std::nullopt,
                          replica_groups)
          .value();
  std::vector<int> expected = {1, 5};
  EXPECT_EQ(actual, expected);
}

TEST(CollectiveOpsUtilsTest, CollectiveWithChannelId) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY %cluster {
    %param0 = f32[512]{0} parameter(0)
    %copy0 = f32[512]{0} copy(param0)
    %reshape0 = f32[1,1,512]{2,0,1} reshape(f32[512]{0} %copy0)
    %all-gather = f32[1,4,512]{2,0,1} all-gather(f32[1,1,512]{2,0,1} %reshape0),
        channel_id=3621, replica_groups={{0,1,2,3}}, dimensions={1},
        use_global_device_ids=true
    %copy1 = f32[1,4,512]{2,0,1} copy(all-gather)
    ROOT root = f32[1,4,512]{2,1,0} copy(%copy1)
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  HloInstruction *all_gather =
      module->entry_computation()->GetInstructionWithName("all-gather");

  EXPECT_EQ(IsOrHasCollectiveWithChannelId(all_gather), all_gather);
}

TEST(CollectiveOpsUtilsTest, IsNonFusionCollectiveSendRecv) {
  absl::string_view hlo_string = R"(
  HloModule module

  ENTRY entry_computation {
    data = f32[64] parameter(0)
    tok = token[] after-all()
    recv_ctx = (f32[64], u32[], token[]) recv(tok), channel_id=2,
        frontend_attributes={_xla_send_recv_source_target_pairs={{3,0}}}
    send_ctx = (f32[64], u32[], token[]) send(tok, data), channel_id=2,
        frontend_attributes={_xla_send_recv_source_target_pairs={{3,0}}}
    ROOT root = tuple(send_ctx, recv_ctx)
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  HloInstruction *recv_ctx =
      module->entry_computation()->GetInstructionWithName("recv_ctx");
  ASSERT_NE(recv_ctx, nullptr);
  HloInstruction *send_ctx =
      module->entry_computation()->GetInstructionWithName("send_ctx");
  ASSERT_NE(send_ctx, nullptr);

  EXPECT_TRUE(IsNonFusionCollective(recv_ctx));
  EXPECT_TRUE(IsNonFusionCollective(send_ctx));
}

TEST(CollectiveOpsUtilsTest, CollectiveWithChannelId2) {
  ReplicaGroup group;
  for (int64_t i = 0; i < 8; i++) {
    group.add_replica_ids(i);
  }

  auto builder = HloComputation::Builder("CollectiveWithChannelId2");
  TF_ASSERT_OK_AND_ASSIGN(
      HloInstruction * param_0,
      builder.AddParameter(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(BF16, {1, 512, 4096}), "p0")));
  HloInstruction *instr =
      builder.AddInstruction(HloInstruction::CreateAllGather(
          ShapeUtil::MakeShape(BF16, {1, 4096, 4096}), {param_0}, 1,
          CollectiveDeviceList(std::vector<ReplicaGroup>({group})), true, 231,
          true));
  auto computation = builder.Build(
      builder.AddInstruction(HloInstruction::CreateTuple({instr})));
  auto fusion =
      HloInstruction::CreateFusion(ShapeUtil::MakeShape(BF16, {1, 4096, 4096}),
                                   HloInstruction::FusionKind::kOutput,
                                   {param_0}, computation.get(), "fusion");
  EXPECT_EQ(IsOrHasCollectiveWithChannelId(fusion.get()), instr);

  auto builder2 = HloComputation::Builder("CollectiveWithChannelId2");
  TF_ASSERT_OK_AND_ASSIGN(
      HloInstruction * param_1,
      builder2.AddParameter(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(BF16, {1, 512, 4096}), "p1")));
  HloInstruction *instr_without_channel_id =
      builder2.AddInstruction(HloInstruction::CreateAllGather(
          ShapeUtil::MakeShape(BF16, {1, 4096, 4096}), {param_1}, 1, {group},
          true, std::nullopt, true));
  auto computation2 = builder2.Build(builder2.AddInstruction(
      HloInstruction::CreateTuple({instr_without_channel_id})));
  auto fusion2 =
      HloInstruction::CreateFusion(ShapeUtil::MakeShape(BF16, {1, 4096, 4096}),
                                   HloInstruction::FusionKind::kOutput,
                                   {param_1}, computation2.get(), "fusion2");
  EXPECT_EQ(IsOrHasCollectiveWithChannelId(fusion2.get()), nullptr);
}

TEST(CollectiveOpsUtilsTest, GetForwardCycleIndices) {
  auto res_one_cycle = GetCycleTypeAndIndices({{0, 1}, {1, 2}, {2, 3}, {3, 0}});
  EXPECT_EQ(res_one_cycle.first, CycleType::kForward);
  EXPECT_THAT(res_one_cycle.second, testing::UnorderedElementsAreArray({3}));
  auto res_two_cycles =
      GetCycleTypeAndIndices({{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 1}});
  EXPECT_EQ(res_two_cycles.first, CycleType::kForward);
  EXPECT_THAT(res_two_cycles.second,
              testing::UnorderedElementsAreArray({3, 4}));
}

TEST(CollectiveOpsUtilsTest, GetBackwardCycleIndices) {
  auto res_one_cycle = GetCycleTypeAndIndices({{0, 3}, {1, 0}, {2, 1}, {3, 2}});
  EXPECT_EQ(res_one_cycle.first, CycleType::kBackward);
  EXPECT_THAT(res_one_cycle.second, testing::UnorderedElementsAreArray({0}));
  auto res_two_cycles =
      GetCycleTypeAndIndices({{0, 3}, {1, 4}, {2, 1}, {3, 2}, {4, 3}, {3, 0}});
  EXPECT_EQ(res_two_cycles.first, CycleType::kBackward);
  EXPECT_THAT(res_two_cycles.second,
              testing::UnorderedElementsAreArray({0, 1}));
}

TEST(IsExclusivelyCrossModuleTest, CrossReplicaNoChannelSet) {
  int64_t num_replicas = 4;
  int64_t num_partitions = 2;
  DeviceAssignment device_assignment(num_replicas, num_partitions);
  std::vector<ReplicaGroup> replica_groups =
      CreateReplicaGroups({{0, 1}, {2, 3}});
  bool is_exclusively_cross_module =
      IsExclusivelyCrossModule(replica_groups, /*use_global_ids=*/false,
                               /*has_channel_id=*/false, device_assignment);
  EXPECT_FALSE(is_exclusively_cross_module);
}

TEST(IsExclusivelyCrossModuleTest, CrossReplicaAndCrossModuleNoGlobalIds) {
  int64_t num_replicas = 4;
  int64_t num_partitions = 2;
  DeviceAssignment device_assignment(num_replicas, num_partitions);
  std::vector<ReplicaGroup> replica_groups =
      CreateReplicaGroups({{0, 1}, {2, 3}});
  bool is_exclusively_cross_module =
      IsExclusivelyCrossModule(replica_groups, /*use_global_ids=*/false,
                               /*has_channel_id=*/true, device_assignment);
  EXPECT_FALSE(is_exclusively_cross_module);
}

TEST(IsExclusivelyCrossModuleTest, CrossModuleNoGlobalIds) {
  int64_t num_replicas = 4;
  int64_t num_partitions = 2;
  ComputationPlacer placer;
  TF_ASSERT_OK_AND_ASSIGN(DeviceAssignment device_assignment,
                          placer.AssignDevices(num_replicas, num_partitions));
  std::vector<ReplicaGroup> replica_groups =
      CreateReplicaGroups({{0}, {1}, {2}, {3}});
  bool is_exclusively_cross_module =
      IsExclusivelyCrossModule(replica_groups, /*use_global_ids=*/false,
                               /*has_channel_id=*/true, device_assignment);
  EXPECT_TRUE(is_exclusively_cross_module);
}

TEST(IsExclusivelyCrossModuleTest, CrossReplicaWithGlobalIds) {
  int64_t num_replicas = 8;
  int64_t num_partitions = 1;
  ComputationPlacer placer;
  TF_ASSERT_OK_AND_ASSIGN(DeviceAssignment device_assignment,
                          placer.AssignDevices(num_replicas, num_partitions));
  std::vector<ReplicaGroup> replica_groups =
      CreateReplicaGroups({{0, 1, 2, 3, 4, 5, 6, 7}});
  bool is_exclusively_cross_module =
      IsExclusivelyCrossModule(replica_groups, /*use_global_ids=*/true,
                               /*has_channel_id=*/true, device_assignment);
  EXPECT_FALSE(is_exclusively_cross_module);
}

TEST(IsExclusivelyCrossModuleTest, CrossReplicaAndCrossModuleWithGlobalIds) {
  int64_t num_replicas = 4;
  int64_t num_partitions = 2;
  ComputationPlacer placer;
  TF_ASSERT_OK_AND_ASSIGN(DeviceAssignment device_assignment,
                          placer.AssignDevices(num_replicas, num_partitions));
  std::vector<ReplicaGroup> replica_groups =
      CreateReplicaGroups({{0, 1, 2, 3, 4, 5, 6, 7}});
  bool is_exclusively_cross_module =
      IsExclusivelyCrossModule(replica_groups, /*use_global_ids=*/true,
                               /*has_channel_id=*/true, device_assignment);
  EXPECT_FALSE(is_exclusively_cross_module);
}

TEST(IsExclusivelyCrossModuleTest, CrossModuleWithGlobalIds) {
  int64_t num_replicas = 4;
  int64_t num_partitions = 2;

  ComputationPlacer placer;
  TF_ASSERT_OK_AND_ASSIGN(DeviceAssignment device_assignment,
                          placer.AssignDevices(num_replicas, num_partitions));
  std::vector<ReplicaGroup> replica_groups =
      CreateReplicaGroups({{0, 1}, {2, 3}, {4, 5}, {6, 7}});
  bool is_exclusively_cross_module =
      IsExclusivelyCrossModule(replica_groups, /*use_global_ids=*/true,
                               /*has_channel_id=*/true, device_assignment);
  EXPECT_TRUE(is_exclusively_cross_module);
}

TEST(CollectiveOpsUtilsTest, GetReplicaGroups) {
  // Create a module for the test
  HloModule module("GetReplicaGroupsTest", HloModuleConfig());

  // Set up a collective permute start instruction
  auto builder = HloComputation::Builder("GetReplicaGroupsTest");
  auto param_shape = ShapeUtil::MakeShape(F32, {4, 4});
  HloInstruction *param_0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "p0"));

  // Test for CollectivePermuteStart
  std::vector<std::pair<int64_t, int64_t>> source_target_pairs = {
      {0, 1}, {1, 2}, {2, 3}, {3, 0}};

  HloInstruction *permute_start =
      builder.AddInstruction(HloInstruction::CreateCollectivePermuteStart(
          param_shape, param_0, source_target_pairs, /*channel_id=*/1));

  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::vector<int64_t>> permute_groups,
                          GetAsyncReplicaGroups(permute_start));
  EXPECT_EQ(permute_groups.size(), 4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(permute_groups[i].size(), 2);
    EXPECT_EQ(permute_groups[i][0], source_target_pairs[i].first);
    EXPECT_EQ(permute_groups[i][1], source_target_pairs[i].second);
  }

  // Test for AllGatherStart
  std::vector<ReplicaGroup> replica_groups =
      CreateReplicaGroups({{0, 1}, {2, 3}});
  HloInstruction *all_gather_start =
      builder.AddInstruction(HloInstruction::CreateAllGatherStart(
          ShapeUtil::MakeTupleShape({param_shape, param_shape}), {param_0},
          /*all_gather_dimension=*/0, replica_groups,
          /*constrain_layout=*/false,
          /*channel_id=*/1, /*use_global_device_ids=*/false));

  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::vector<int64_t>> all_gather_groups,
                          GetAsyncReplicaGroups(all_gather_start));
  EXPECT_EQ(all_gather_groups.size(), 2);
  EXPECT_THAT(all_gather_groups[0], testing::ElementsAre(0, 1));
  EXPECT_THAT(all_gather_groups[1], testing::ElementsAre(2, 3));

  // Test for AllReduceStart
  // Create a reduction computation
  HloComputation::Builder reducer_builder("add");
  auto reducer_x = reducer_builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeScalarShape(F32), "x"));
  auto reducer_y = reducer_builder.AddInstruction(
      HloInstruction::CreateParameter(1, ShapeUtil::MakeScalarShape(F32), "y"));
  reducer_builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeScalarShape(F32), HloOpcode::kAdd, reducer_x, reducer_y));

  HloComputation *add_computation =
      module.AddEmbeddedComputation(reducer_builder.Build());

  HloInstruction *all_reduce_start =
      builder.AddInstruction(HloInstruction::CreateAllReduceStart(
          ShapeUtil::MakeTupleShape({param_shape, param_shape}), {param_0},
          add_computation, replica_groups, /*constrain_layout=*/false,
          /*channel_id=*/2, /*use_global_device_ids=*/false));

  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::vector<int64_t>> all_reduce_groups,
                          GetAsyncReplicaGroups(all_reduce_start));
  EXPECT_EQ(all_reduce_groups.size(), 2);
  EXPECT_THAT(all_reduce_groups[0], testing::ElementsAre(0, 1));
  EXPECT_THAT(all_reduce_groups[1], testing::ElementsAre(2, 3));
}

TEST(CollectiveOpsUtilsTest, IsAsyncCollective) {
  // Create module and computation
  HloModule module("test_module", HloModuleConfig());
  auto builder = HloComputation::Builder("IsAsyncCollectiveTest");
  auto param_shape = ShapeUtil::MakeShape(F32, {4, 4});
  HloInstruction *param_0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "p0"));

  // Test for CollectivePermuteStart and CollectivePermuteDone
  std::vector<std::pair<int64_t, int64_t>> source_target_pairs = {
      {0, 1}, {1, 2}, {2, 3}, {3, 0}};

  HloInstruction *permute_start =
      builder.AddInstruction(HloInstruction::CreateCollectivePermuteStart(
          param_shape, param_0, source_target_pairs, /*channel_id=*/1));

  auto is_async_status = IsAsyncCollective(permute_start);
  EXPECT_TRUE(is_async_status.ok());
  EXPECT_TRUE(is_async_status.value());

  HloInstruction *permute_done =
      builder.AddInstruction(HloInstruction::CreateUnary(
          param_shape, HloOpcode::kCollectivePermuteDone, permute_start));

  is_async_status = IsAsyncCollective(permute_done);
  EXPECT_TRUE(is_async_status.ok());
  EXPECT_TRUE(is_async_status.value());

  // Test for AllGatherStart and AllGatherDone
  std::vector<ReplicaGroup> replica_groups =
      CreateReplicaGroups({{0, 1}, {2, 3}});

  HloInstruction *all_gather_start =
      builder.AddInstruction(HloInstruction::CreateAllGatherStart(
          ShapeUtil::MakeTupleShape(
              {ShapeUtil::MakeShape(F32, {8, 4}), param_shape}),
          {param_0}, /*all_gather_dimension=*/0, replica_groups,
          /*constrain_layout=*/false,
          /*channel_id=*/2, /*use_global_device_ids=*/false));

  is_async_status = IsAsyncCollective(all_gather_start);
  EXPECT_TRUE(is_async_status.ok());
  EXPECT_TRUE(is_async_status.value());

  HloInstruction *all_gather_done = builder.AddInstruction(
      HloInstruction::CreateUnary(ShapeUtil::MakeShape(F32, {8, 4}),
                                  HloOpcode::kAllGatherDone, all_gather_start));

  is_async_status = IsAsyncCollective(all_gather_done);
  EXPECT_TRUE(is_async_status.ok());
  EXPECT_TRUE(is_async_status.value());

  // Test for AllReduceStart and AllReduceDone
  // First create a reduction computation
  HloComputation::Builder reducer_builder("add");
  HloInstruction *reducer_x = reducer_builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeScalarShape(F32), "x"));
  HloInstruction *reducer_y = reducer_builder.AddInstruction(
      HloInstruction::CreateParameter(1, ShapeUtil::MakeScalarShape(F32), "y"));
  reducer_builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeScalarShape(F32), HloOpcode::kAdd, reducer_x, reducer_y));

  HloComputation *add_computation =
      module.AddEmbeddedComputation(reducer_builder.Build());

  HloInstruction *all_reduce_start =
      builder.AddInstruction(HloInstruction::CreateAllReduceStart(
          ShapeUtil::MakeTupleShape({param_shape, param_shape}), {param_0},
          add_computation, replica_groups, /*constrain_layout=*/false,
          /*channel_id=*/3, /*use_global_device_ids=*/false));

  is_async_status = IsAsyncCollective(all_reduce_start);
  EXPECT_TRUE(is_async_status.ok());
  EXPECT_TRUE(is_async_status.value());

  HloInstruction *all_reduce_done =
      builder.AddInstruction(HloInstruction::CreateUnary(
          param_shape, HloOpcode::kAllReduceDone, all_reduce_start));

  is_async_status = IsAsyncCollective(all_reduce_done);
  EXPECT_TRUE(is_async_status.ok());
  EXPECT_TRUE(is_async_status.value());

  // Test for regular CollectivePermute (non-async)
  HloInstruction *permute =
      builder.AddInstruction(HloInstruction::CreateCollectivePermute(
          param_shape, param_0, source_target_pairs, /*channel_id=*/1));

  is_async_status = IsAsyncCollective(permute);
  EXPECT_TRUE(is_async_status.ok());
  EXPECT_FALSE(is_async_status.value());
}

TEST(IsExclusivelyCrossReplicaTest, CrossReplicaNoChannelSet) {
  int64_t num_replicas = 4;
  int64_t num_partitions = 2;
  DeviceAssignment device_assignment(num_replicas, num_partitions);
  std::vector<ReplicaGroup> replica_groups =
      CreateReplicaGroups({{0, 1}, {2, 3}});
  EXPECT_TRUE(
      IsExclusivelyCrossReplica(replica_groups, /*use_global_ids=*/false,
                                /*has_channel_id=*/false, device_assignment));
}

TEST(IsExclusivelyCrossReplicaTest, CrossReplicaAndCrossModuleNoGlobalIds) {
  int64_t num_replicas = 4;
  int64_t num_partitions = 2;
  DeviceAssignment device_assignment(num_replicas, num_partitions);
  std::vector<ReplicaGroup> replica_groups =
      CreateReplicaGroups({{0, 1}, {2, 3}});

  EXPECT_FALSE(
      IsExclusivelyCrossReplica(replica_groups, /*use_global_ids=*/false,
                                /*has_channel_id=*/true, device_assignment));
}

TEST(IsExclusivelyCrossReplicaTest, CrossModuleNoGlobalIds) {
  int64_t num_replicas = 4;
  int64_t num_partitions = 2;
  ComputationPlacer placer;
  TF_ASSERT_OK_AND_ASSIGN(DeviceAssignment device_assignment,
                          placer.AssignDevices(num_replicas, num_partitions));
  std::vector<ReplicaGroup> replica_groups =
      CreateReplicaGroups({{0}, {1}, {2}, {3}});

  EXPECT_FALSE(
      IsExclusivelyCrossReplica(replica_groups, /*use_global_ids=*/false,
                                /*has_channel_id=*/true, device_assignment));
}

TEST(IsExclusivelyCrossReplicaTest, CrossReplicaWithGlobalIds) {
  int64_t num_replicas = 8;
  int64_t num_partitions = 1;
  ComputationPlacer placer;
  TF_ASSERT_OK_AND_ASSIGN(DeviceAssignment device_assignment,
                          placer.AssignDevices(num_replicas, num_partitions));
  std::vector<ReplicaGroup> replica_groups =
      CreateReplicaGroups({{0, 1, 2, 3, 4, 5, 6, 7}});

  EXPECT_TRUE(IsExclusivelyCrossReplica(replica_groups, /*use_global_ids=*/true,
                                        /*has_channel_id=*/true,
                                        device_assignment));
}

TEST(IsExclusivelyCrossReplicaTest, CrossReplicaAndCrossModuleWithGlobalIds) {
  int64_t num_replicas = 4;
  int64_t num_partitions = 2;
  ComputationPlacer placer;
  TF_ASSERT_OK_AND_ASSIGN(DeviceAssignment device_assignment,
                          placer.AssignDevices(num_replicas, num_partitions));
  std::vector<ReplicaGroup> replica_groups =
      CreateReplicaGroups({{0, 1, 2, 3, 4, 5, 6, 7}});

  EXPECT_FALSE(
      IsExclusivelyCrossReplica(replica_groups, /*use_global_ids=*/true,
                                /*has_channel_id=*/true, device_assignment));
}

TEST(IsExclusivelyCrossReplicaTest, CrossModuleWithGlobalIds) {
  int64_t num_replicas = 4;
  int64_t num_partitions = 2;

  ComputationPlacer placer;
  TF_ASSERT_OK_AND_ASSIGN(DeviceAssignment device_assignment,
                          placer.AssignDevices(num_replicas, num_partitions));
  std::vector<ReplicaGroup> replica_groups =
      CreateReplicaGroups({{0, 1}, {2, 3}, {4, 5}, {6, 7}});

  EXPECT_FALSE(
      IsExclusivelyCrossReplica(replica_groups, /*use_global_ids=*/true,
                                /*has_channel_id=*/true, device_assignment));
}
}  // namespace

// Tests for GetCollectOpGroupMode
namespace GetCollectiveOpGroupModeTest {
struct TestCase {
  bool has_channel_id;
  std::optional<bool> use_global_device_ids;
  std::optional<xla::CollectiveOpGroupMode> expected;

  std::string ToString() const {
    std::ostringstream s;
    s << (has_channel_id ? "chnl" : "nochnl");
    s << "_"
      << (use_global_device_ids
              ? (*use_global_device_ids ? "ugdi_true" : "ugdi_false")
              : "nougdi");
    return s.str();
  }
};

std::vector<TestCase> GetTestCases() {
  const std::vector<TestCase> test_cases = {
      // clang-format off
      // has_channel_id, use_global_device_ids, expected mode
      {false, std::nullopt, CollectiveOpGroupMode::kCrossReplica},
      {false, false,         CollectiveOpGroupMode::kCrossReplica},
      {false, true,          std::nullopt},
      {true,  std::nullopt, CollectiveOpGroupMode::kCrossPartition},
      {true,  false,         CollectiveOpGroupMode::kCrossReplicaAndPartition},
      {true,  true,          CollectiveOpGroupMode::kFlattenedID},
      // clang-format on
  };
  return test_cases;
}

class GetCollectOpGroupModeTest : public testing::TestWithParam<TestCase> {};

TEST_P(GetCollectOpGroupModeTest, Test) {
  const TestCase &tc = GetParam();
  absl::StatusOr<CollectiveOpGroupMode> actual =
      GetCollectiveOpGroupMode(tc.has_channel_id, tc.use_global_device_ids);
  if (tc.expected) {
    TF_ASSERT_OK(actual.status());
    EXPECT_EQ(*actual, *tc.expected);
  } else {
    EXPECT_FALSE(actual.ok());
  }
}

INSTANTIATE_TEST_SUITE_P(GetCollectOpGroupMode, GetCollectOpGroupModeTest,
                         testing::ValuesIn(GetTestCases()));

// Tests for GetCollectiveOpGroupMode(HloInstruction*)
struct TestCaseForInstruction {
  HloOpcode op_code;
  bool has_channel_id;
  std::optional<bool> use_global_device_ids;
  xla::CollectiveOpGroupMode expected_group_mode;
};

std::vector<TestCaseForInstruction> GetTestCasesForInstruction() {
  return std::vector<TestCaseForInstruction>{
      //  opcode, has_channel_id, use_global_device_ids, expected_group_mode
      {HloOpcode::kAllGather, true, true, CollectiveOpGroupMode::kFlattenedID},
      {HloOpcode::kAllGather, true, false,
       CollectiveOpGroupMode::kCrossReplicaAndPartition},
      {HloOpcode::kAllGather, false, false,
       CollectiveOpGroupMode::kCrossReplica},
      {HloOpcode::kAllReduce, true, true, CollectiveOpGroupMode::kFlattenedID},
      {HloOpcode::kAllReduce, true, false,
       CollectiveOpGroupMode::kCrossReplicaAndPartition},
      {HloOpcode::kAllReduce, false, false,
       CollectiveOpGroupMode::kCrossReplica},
      {HloOpcode::kAllToAll, true, std::nullopt,
       CollectiveOpGroupMode::kCrossPartition},
      {HloOpcode::kAllToAll, false, std::nullopt,
       CollectiveOpGroupMode::kCrossReplica},
      {HloOpcode::kCollectiveBroadcast, true, std::nullopt,
       CollectiveOpGroupMode::kCrossPartition},
      {HloOpcode::kCollectiveBroadcast, false, std::nullopt,
       CollectiveOpGroupMode::kCrossReplica},
      {HloOpcode::kCollectivePermute, true, std::nullopt,
       CollectiveOpGroupMode::kCrossPartition},
      {HloOpcode::kCollectivePermute, false, std::nullopt,
       CollectiveOpGroupMode::kCrossReplica},
      {HloOpcode::kRaggedAllToAll, true, std::nullopt,
       CollectiveOpGroupMode::kCrossPartition},
      {HloOpcode::kRaggedAllToAll, false, std::nullopt,
       CollectiveOpGroupMode::kCrossReplica}};
}

class GetCollectOpGroupModeTestForInstruction
    : public testing::TestWithParam<TestCaseForInstruction> {};

absl::StatusOr<std::unique_ptr<HloComputation>> CreateMaxComputation() {
  Shape scalar = ShapeUtil::MakeScalarShape(F32);
  auto builder_max = HloComputation::Builder("max");
  TF_ASSIGN_OR_RETURN(HloInstruction * a,
                      builder_max.AddParameter(
                          HloInstruction::CreateParameter(0, scalar, "a")));
  TF_ASSIGN_OR_RETURN(HloInstruction * b,
                      builder_max.AddParameter(
                          HloInstruction::CreateParameter(1, scalar, "b")));
  HloInstruction *max = builder_max.AddInstruction(
      HloInstruction::CreateBinary(scalar, HloOpcode::kMaximum, a, b), "max");
  return builder_max.Build(max);
}

TEST_P(GetCollectOpGroupModeTestForInstruction, Test) {
  const TestCaseForInstruction &test_case = GetParam();
  ReplicaGroup group;
  for (int k = 0; k < 4; ++k) {
    group.add_replica_ids(k);
  }
  std::vector<std::pair<int64_t, int64_t>> source_target_pairs{{0, 1}, {2, 3}};

  Shape two_elements = ShapeUtil::MakeShape(F32, {2});
  Shape eight_elements = ShapeUtil::MakeShape(F32, {8});

  auto channel_id = [&test_case]() -> std::optional<int64_t> {
    return test_case.has_channel_id ? std::make_optional<int64_t>(1)
                                    : std::nullopt;
  };

  auto use_global_device_ids = [&test_case]() -> bool {
    return test_case.use_global_device_ids.value();
  };

  // Create the entry computation for testing the group mode of the collectives.
  auto builder = HloComputation::Builder("entry");
  TF_ASSERT_OK_AND_ASSIGN(HloInstruction * parameter,
                          builder.AddParameter(HloInstruction::CreateParameter(
                              0, two_elements, "parameter")));

  HloInstruction *collective;
  switch (test_case.op_code) {
    case HloOpcode::kAllGather:
      collective = builder.AddInstruction(HloInstruction::CreateAllGather(
          eight_elements, {parameter}, 1, {group}, /*constrain_layout=*/true,
          channel_id(), use_global_device_ids()));
      break;
    case HloOpcode::kAllReduce: {
      // Create a computation to be applied by the all-reduce instruction.
      TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloComputation> max_computation,
                              CreateMaxComputation());

      collective = builder.AddInstruction(HloInstruction::CreateAllReduce(
          two_elements, {parameter}, max_computation.get(), {group},
          /*constrain_layout=*/true, channel_id(), use_global_device_ids()));
      break;
    }
    case HloOpcode::kAllToAll:
      collective = builder.AddInstruction(HloInstruction::CreateAllToAll(
          eight_elements, {parameter}, {group}, /*constrain_layout=*/true,
          channel_id(), std::nullopt));
      break;
    case HloOpcode::kCollectiveBroadcast:
      collective =
          builder.AddInstruction(HloInstruction::CreateCollectiveBroadcast(
              two_elements, {parameter}, {group}, /*constrain_layout=*/true,
              channel_id()));
      break;
    case HloOpcode::kCollectivePermute:
      collective =
          builder.AddInstruction(HloInstruction::CreateCollectivePermute(
              two_elements, parameter, source_target_pairs, channel_id()));
      break;
    case HloOpcode::kRaggedAllToAll: {
      // Create a parameter with s64 to use a offset and size operands.
      TF_ASSERT_OK_AND_ASSIGN(
          HloInstruction * offset_size_parameter,
          builder.AddParameter(HloInstruction::CreateParameter(
              1, ShapeUtil::MakeShape(S64, {4}), "offset_size_parameter")));

      collective = builder.AddInstruction(HloInstruction::CreateRaggedAllToAll(
          eight_elements,
          {parameter, parameter, offset_size_parameter, offset_size_parameter,
           offset_size_parameter, offset_size_parameter},
          {group}, channel_id()));
      break;
    }
    default:
      LOG(FATAL) << "Unexpected opcode.";
  }
  TF_ASSERT_OK_AND_ASSIGN(auto collective_group_mode,
                          GetCollectiveOpGroupMode(collective));
  EXPECT_EQ(collective_group_mode, test_case.expected_group_mode);
}

INSTANTIATE_TEST_SUITE_P(GetCollectOpGroupModeForInstruction,
                         GetCollectOpGroupModeTestForInstruction,
                         testing::ValuesIn(GetTestCasesForInstruction()));

}  // namespace GetCollectiveOpGroupModeTest

// Tests for GetParticipating* related functions.
namespace GetParticipatingTest {

// Test case for GetParticipating* functions. Describes all the inputs to the
// function and for a given "setup", multiple "current_id" values and the
// expected output corresponding to those values.
struct TestCase {
  xla::Array2D<int> device_assignment;
  std::vector<std::vector<int64_t>> replica_groups;
  bool has_channel_id;
  std::optional<bool> use_global_device_ids;

  // For a given test case, its useful to test multiple 'current_id' inputs.
  struct CurrentIdAndOutput {
    int current_id;
    std::vector<int> expected_output;
  };
  std::vector<CurrentIdAndOutput> subtests;

  // Expected output for GetParticipatingDevicesGroups.
  std::vector<std::vector<int64_t>> participating_device_groups;
  // Expected output for GetParticipatingFlattenedIdGroups.
  std::vector<std::vector<int64_t>> participating_flattened_id_groups;
  // Expected output for GetPariticipantCountsForReplicaGroups.
  std::vector<int64_t> participant_counts_for_replica_groups;
  // Expected output for GetReplicaGroupCountAndSize.
  std::optional<std::pair<int64_t, int64_t>> replica_group_count_and_size;
  bool expected_failure;

  std::string ToString() const;
};

// Please see the comment for GetParticipatingDevices() for a description of
// modes and their behavior.
std::string TestCase::ToString() const {
  std::ostringstream s;
  absl::StatusOr<CollectiveOpGroupMode> group_mode =
      GetCollectiveOpGroupMode(has_channel_id, use_global_device_ids);
  if (group_mode.ok()) {
    s << CollectiveOpGroupModeToString(*group_mode);
  } else {
    s << "Invalid";
  }

  s << "_" << device_assignment.n1() << "x" << device_assignment.n2();
  s << "_" << (replica_groups.empty() ? "NoRG" : "RG");
  s << "_" << subtests.size() << "SubTests";
  return s.str();
}

std::ostream &operator<<(std::ostream &os, const TestCase &tc) {
  os << tc.ToString();
  return os;
}

std::vector<TestCase> GetTestCases() {
  std::vector<TestCase> test_cases;
  // clang-format off
  const std::vector<TestCase> cross_replica_test_cases = {
    // with empty replica groups, 1 partition.
    {
      {{33}, {44}, {55}},     // 3 replicas, 1 partition.
      {},                     // empty replica groups
      false,                  // has_channel_id
      false,                  // use_global_device_ids
      {                       // subtests
        // for empty replica group, any id should return all ids.
        {33, {33, 44, 55}},
        {44, {33, 44, 55}},
      },
      {{33, 44, 55}},         // participating device groups
      {{0, 1, 2}},            // participating flattened id groups
      {3},                    // participant counts for replica groups
      std::optional<std::pair<int64_t, int64_t>>({1, 3}),
                             // replica group count and size
      false                   // expected_failure
    },

    // empty replica groups, > 1 partition
    {
      {{33, 34}, {44, 45}, {55, 56}},  // 3r, 2p
      {},                              // empty replica groups
      false,                           // has_channel_id
      false,                           // use_global_device_ids
      // for empty replica group, any id should return all replicas within that
      // partition.
      {                                // subtests
        {33, {33, 44, 55}},
        {34, {34, 45, 56}},
        {45, {34, 45, 56}},
      },
      {{33, 44, 55}, {34, 45, 56}},    // participating device groups
      {{0, 2, 4}, {1, 3, 5}},          // participating flattened id groups
      {3, 3},                          // participant counts for replica groups
      std::optional<std::pair<int64_t, int64_t>>({2, 3}),
                                      // replica group count and size
      false                            // expected_failure
    },

    // non-empty replica groups, 1 partition.
    {
      {{33}, {44}, {55}},   // 3r, 1p.
      {{0}, {1, 2}},        // replica groups
      false,                // has_channel_id
      false,                // use_global_device_ids
      {                     // subtests
        // 33 is r0, so it's a singleton group.
        {33, {33}},
        // 44 is r1, so it should give {r1, r2}.
        {44, {44, 55}},
      },
      {{ 33 }, {44, 55}},    // participating device groups
      {{0}, {1, 2}},         // participating flattened id groups
      {1, 2},                // participant counts for replica groups
      std::nullopt,          // replica group count and size
      false                  // expected_failure
    },

    // non-empty, > 1 partition
    {
      {{33, 34}, {44, 45}, {55, 56}},   // 3r, 2p
      {{0}, {1, 2}},                    // replica groups
      false,                            // has_channel_id
      false,                            // use_global_device_ids
      {                                 // subtests
        // 33 is r0p0, so should be singleton.
        {33, {33}},
        // 34 is r0p1, so should be singleton.
        {34, {34}},
        // 45 is r1p1, so should get r1p1 and r2p1.
        {45, {45, 56}},
      },
      {{33}, {34}, {44, 55}, {45, 56}},
                                        // participating device groups
      {{0}, {1}, {2, 4}, {3, 5}},       // participating flattened id groups
      {1, 1, 2, 2},                     // participant counts for replica groups
      std::nullopt,                     // replica group count and size
      false                             // expected_failure
    },
  };

  // replica groups contain partition ids.
  const std::vector<TestCase> cross_partition_test_cases = {
    {
      // 3x4 device assignment
      {
        {33, 34, 35, 36}, {44, 45, 46, 47}, {55, 56, 57, 58}
      },
      {{0, 1}, {2, 3}},          // replica groups
      true,                      // has_channel_id
      std::nullopt,              // use_global_device_ids
      {                          // subtests
        // 33 is r0p0, p0 group has p0, p1 so we get r0p0 and r0p1.
        {33, {33, 34}},
        // 35 is r0p2, so we get r0p2 and r0p3
        {35, {35, 36}},
        {45, {44, 45}},
        {47, {46, 47}},
        {58, {57, 58}},
      },
      {{33, 34}, {44, 45}, {55, 56},
       {35, 36}, {46, 47}, {57, 58}},  // participating device groups
      {{0, 1}, {4, 5}, {8, 9},
       {2, 3}, {6, 7}, {10, 11}},      // participating flattened id groups
      {2, 2},                          // participant counts for replica groups
      std::optional<std::pair<int64_t, int64_t>>({2, 8}),
                                       // replica group count and size
      false                            // expected_failure
    }
  };


  const std::vector<TestCase> cross_replica_and_partition_test_cases = {
    {
      {{33, 34}, {44, 45}, {55, 56}},   // 3r, 2p
      {{0}, {1, 2}},                    // replica groups
      true,                             // has_channel_id
      false,                            // use_global_device_ids
      {                                 // subtests
        // 33 is r0p0, so should get r0 from all partitions.
        {33, {33, 34}},
        // 34 is r0p1, so should get r0 from all partitions.
        {34, {33, 34}},
        // 45 is r1p1, so should get r1, r2 from all partitions.
        {45, {44, 45, 55, 56}},
      },
      {{33, 34}, {44, 45, 55, 56}},   // participating device groups
      {{0, 1}, {2, 3, 4, 5}},         // participating flattened id groups
      {2, 4},                         // participant counts for replica groups
      std::nullopt,                   // replica group count and size
      false
    },

    // empty replica group = all replicas, so we should get all devices.
    {
      {{33, 34}, {44, 45}, {55, 56}},   // 3r, 2p
      {},                               // replica groups
      true,                             // has_channel_id
      false,                            // use_global_device_ids
      {                                 // subtests
        {33, {33, 34, 44, 45, 55, 56}},
        {34, {33, 34, 44, 45, 55, 56}},
        {56, {33, 34, 44, 45, 55, 56}},
      },
      {{33, 34, 44, 45, 55, 56}},       // participating device groups
      {{0, 1, 2, 3, 4, 5}},             // participating flattened id groups
      {6},                              // participant counts for replica groups
      std::optional<std::pair<int64_t, int64_t>>({1, 6}),
                                        // replica group count and size
      false                             // expected_failure
    },
  };

  // Replica groups are flattened ids. For a 3x2 device assignment
  // used in these tests, the flattened ID and deviceId correspondence is as
  // follows:
  //   r0p0 = f#0 = d#33
  //   r0p1 = f#1 = d#34
  //   r1p0 = f#2 = d#44
  //   r1p1 = f#3 = d#45
  //   r2p0 = f#4 = d#55
  //   r2p1 = f#5 = d#56
  const std::vector<TestCase> flattened_id_test_cases = {
    {
      {{33, 34}, {44, 45}, {55, 56}},  // 3r, 2p
      {{0}, {1, 2}, {3, 4, 5}},        // replica groups
      true,                            // has_channel_id
      true,                            // use_global_device_ids
      {                                // subtests
        {33, {33}},
        {34, {34, 44}},
        {44, {34, 44}},
        {45, {45, 55, 56}},
        {55, {45, 55, 56}},
        {56, {45, 55, 56}},
      },
      {{33}, {34, 44}, {45, 55, 56}},  // participating device groups
      {{0}, {1, 2}, {3, 4, 5}},        // participating flattened id groups
      {1, 2, 3},                       // participant counts for replica groups
      std::nullopt,                    // replica group count and size
      false                            // expected_failure
    },
    {
      {{33}},
      {},         // empty replica groups not allowed.
      true,       // has_channel_id
      true,       // use_global_device_ids
      {           // subtests
        {33, {33}},
      },
      {{33}},      // participating device groups
      {{0}},       // participating flattened id groups
      {1},         // participant counts for replica groups
      std::optional<std::pair<int64_t, int64_t>>({1, 1}),
                   // replica group count and size
      true         // expected_failure
    },
  };

  const std::vector<TestCase> failure_test_cases = {
    // No channel id, use_global_device_ids = true;
    {
      {{33}, {44}, {55}},   // 3r, 1p
      {},                   // replica groups
      false,                // has_channel_id
      true,                 // use_global_device_ids
      {                     // subtests
        {33, {}},
      },
      {{33, 44, 55}},       // participating device groups
      {{0, 1, 2}},          // participating flattened id groups
      {3},                  // participant counts for replica groups
      std::optional<std::pair<int64_t, int64_t>>({1, 3}),
                            // replica group count and size
      true                  // expected_failure
    },
  };
  // clang-format on

  test_cases.insert(test_cases.end(), cross_replica_test_cases.begin(),
                    cross_replica_test_cases.end());
  // When use_global_device_ids is not present and channel_id is not present,
  // that implies cross replica mode as well.
  for (TestCase tc : cross_replica_test_cases) {
    tc.use_global_device_ids = std::nullopt;
    test_cases.push_back(tc);
  }

  test_cases.insert(test_cases.end(), cross_partition_test_cases.begin(),
                    cross_partition_test_cases.end());
  test_cases.insert(test_cases.end(),
                    cross_replica_and_partition_test_cases.begin(),
                    cross_replica_and_partition_test_cases.end());
  test_cases.insert(test_cases.end(), flattened_id_test_cases.begin(),
                    flattened_id_test_cases.end());
  test_cases.insert(test_cases.end(), failure_test_cases.begin(),
                    failure_test_cases.end());

  return test_cases;
}

class GetParticipatingTest : public testing::TestWithParam<TestCase> {};

TEST_P(GetParticipatingTest, Test) {
  const TestCase &tc = GetParam();

  int64_t num_replicas = tc.device_assignment.n1();
  int64_t num_partitions = tc.device_assignment.n2();
  DeviceAssignment device_assignment(num_replicas, num_partitions);

  for (int64_t replica_id = 0; replica_id < num_replicas; ++replica_id) {
    for (int64_t partition_id = 0; partition_id < num_partitions;
         ++partition_id) {
      device_assignment(replica_id, partition_id) =
          tc.device_assignment(replica_id, partition_id);
    }
  }

  std::vector<ReplicaGroup> replica_groups =
      CreateReplicaGroups(tc.replica_groups);

  absl::StatusOr<CollectiveOpGroupMode> group_mode =
      GetCollectiveOpGroupMode(tc.has_channel_id, tc.use_global_device_ids);

  if (!group_mode.ok()) {
    EXPECT_TRUE(tc.expected_failure);
    return;
  }

  // Test GetParticipatingDevices.
  for (const TestCase::CurrentIdAndOutput &subtest : tc.subtests) {
    absl::StatusOr<std::vector<GlobalDeviceId>> actual =
        GetParticipatingDevices(GlobalDeviceId(subtest.current_id),
                                device_assignment, replica_groups, *group_mode);
    if (!actual.ok()) {
      EXPECT_TRUE(tc.expected_failure);
      continue;
    }
    std::vector<GlobalDeviceId> expected;
    expected.reserve(subtest.expected_output.size());
    absl::c_transform(subtest.expected_output, std::back_inserter(expected),
                      [](int id) { return GlobalDeviceId(id); });
    EXPECT_EQ(*actual, expected);
  }

  // Test GetParticipatingDevicesGroups.
  absl::StatusOr<std::vector<std::vector<GlobalDeviceId>>>
      actual_device_groups = GetParticipatingDevicesGroups(
          device_assignment, replica_groups, *group_mode);

  if (!actual_device_groups.ok()) {
    EXPECT_TRUE(tc.expected_failure);
    return;
  }

  std::vector<std::vector<GlobalDeviceId>> expect_device_groups;
  expect_device_groups.reserve(tc.participating_device_groups.size());

  for (auto subgroup : tc.participating_device_groups) {
    std::vector<GlobalDeviceId> subgroup_device_ids;
    subgroup_device_ids.reserve(subgroup.size());
    absl::c_transform(subgroup, std::back_inserter(subgroup_device_ids),
                      [](int id) { return GlobalDeviceId(id); });

    expect_device_groups.push_back(subgroup_device_ids);
  }

  EXPECT_THAT(*actual_device_groups,
              testing::UnorderedElementsAreArray(expect_device_groups));

  // Test GetParticipatingFlattenedIdGroups.
  absl::StatusOr<CollectiveDeviceList> collective_device_list =
      GetParticipatingFlattenedIdGroups(
          device_assignment, CollectiveDeviceList(replica_groups), *group_mode);
  if (!collective_device_list.ok()) {
    EXPECT_TRUE(tc.expected_failure);
    return;
  }
  const std::vector<ReplicaGroup> &actual_flattened_id_groups =
      collective_device_list.value().replica_groups();

  std::vector<std::vector<int64_t>> actual_flattened_id_groups_int;
  actual_flattened_id_groups_int.reserve(actual_flattened_id_groups.size());

  for (auto subgroup : actual_flattened_id_groups) {
    std::vector<int64_t> replica_group;
    for (int id : subgroup.replica_ids()) {
      replica_group.push_back(id);
    }
    actual_flattened_id_groups_int.push_back(replica_group);
  }

  EXPECT_EQ(actual_flattened_id_groups_int,
            tc.participating_flattened_id_groups);

  // Test GetPariticipantCountsForReplicaGroups.
  absl::StatusOr<std::vector<int64_t>> actual_participant_counts =
      GetPariticipantCountsForReplicaGroups(num_replicas, num_partitions,
                                            replica_groups, *group_mode);
  if (!actual_participant_counts.ok()) {
    EXPECT_TRUE(tc.expected_failure);
    return;
  }
  EXPECT_EQ(*actual_participant_counts,
            tc.participant_counts_for_replica_groups);

  // Test GetReplicaGroupCountAndSize.
  HloModuleConfig config;
  config.set_replica_count(num_replicas);
  config.set_num_partitions(num_partitions);
  config.set_static_device_assignment(device_assignment);
  HloModule hlo_module("AllReduce", config);
  HloComputation::Builder sum_builder("test_reduction");
  auto x = sum_builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, ShapeUtil::MakeShape(F32, {}), "x"));
  auto y = sum_builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, ShapeUtil::MakeShape(F32, {}), "y"));
  sum_builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {}), HloOpcode::kAdd, x, y));
  HloComputation *reduction =
      hlo_module.AddEmbeddedComputation(sum_builder.Build());
  HloComputation::Builder entry_builder("test_entry");
  HloInstruction *operand = entry_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0f)));
  std::optional<int64_t> channel_id = std::nullopt;
  if (tc.has_channel_id) {
    channel_id = 0;
  }
  HloInstruction *ar =
      entry_builder.AddInstruction(HloInstruction::CreateAllReduce(
          operand->shape(), {operand}, reduction, replica_groups,
          /*constrain_layout=*/false,
          /*channel_id=*/channel_id,
          /*use_global_device_ids=*/tc.use_global_device_ids.has_value()
              ? tc.use_global_device_ids.value()
              : false));
  hlo_module.AddEntryComputation(entry_builder.Build());

  absl::StatusOr<std::optional<std::pair<int64_t, int64_t>>>
      actual_replica_group_count_and_size = GetReplicaGroupCountAndSize(ar);
  if (!actual_replica_group_count_and_size.ok()) {
    EXPECT_TRUE(tc.expected_failure);
    return;
  }
  EXPECT_EQ(*actual_replica_group_count_and_size,
            tc.replica_group_count_and_size);
}

INSTANTIATE_TEST_SUITE_P(GetParticipating, GetParticipatingTest,
                         testing::ValuesIn(GetTestCases()));

}  // namespace GetParticipatingTest

namespace GetPariticipantCountsForReplicaGroupsTest {

struct TestCase {
  std::string test_name;
  std::vector<std::vector<int64_t>> replica_groups;
  CollectiveOpGroupMode group_mode;
  int64_t num_replicas;
  int64_t num_partitions;
  std::vector<int64_t> expected;
};

class GetPariticipantCountsForReplicaGroupsTest
    : public testing::TestWithParam<TestCase> {};

TEST_P(GetPariticipantCountsForReplicaGroupsTest, Test) {
  const TestCase &tc = GetParam();

  std::vector<ReplicaGroup> replica_groups =
      CreateReplicaGroups(tc.replica_groups);
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<int64_t> actual,
      GetPariticipantCountsForReplicaGroups(tc.num_replicas, tc.num_partitions,
                                            replica_groups, tc.group_mode));
  EXPECT_THAT(actual, testing::ElementsAreArray(tc.expected));
}

std::vector<TestCase> GetTestCases() {
  return {
      {
          "CrossReplicaEmptyGroup",
          {},
          CollectiveOpGroupMode::kCrossReplica,
          8,
          1,
          {8},
      },
      {
          "CrossReplicaWithPartitions",
          {{0, 1}, {2, 3}},
          CollectiveOpGroupMode::kCrossReplica,
          4,
          2,
          {2, 2, 2, 2},
      },
      {
          "CrossReplicaAndPartition",
          {{0, 1}, {2, 3}},
          CollectiveOpGroupMode::kCrossReplicaAndPartition,
          4,
          2,
          {4, 4},
      },
      {
          "FlattenedID",
          {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}},
          CollectiveOpGroupMode::kFlattenedID,
          4,
          2,
          {1, 1, 1, 1, 1, 1, 1, 1},
      },
  };
}
INSTANTIATE_TEST_SUITE_P(
    GetPariticipantCountsForReplicaGroups,
    GetPariticipantCountsForReplicaGroupsTest,
    testing::ValuesIn(GetTestCases()),
    [](const testing::TestParamInfo<
        GetPariticipantCountsForReplicaGroupsTest::ParamType> &info) {
      return info.param.test_name;
    });

}  // namespace GetPariticipantCountsForReplicaGroupsTest
}  // namespace xla
