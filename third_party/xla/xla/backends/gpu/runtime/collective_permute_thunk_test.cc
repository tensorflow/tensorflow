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

#include "xla/backends/gpu/runtime/collective_permute_thunk.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/async_thunk.h"
#include "xla/backends/gpu/runtime/command_buffer_cmd_emitter.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/execution_stream_id.h"
#include "xla/backends/gpu/runtime/p2p_thunk_common.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/hlo/ir/collective_op_group_mode.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/runtime/device_id.h"
#include "xla/service/backend.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/computation_placer.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tests/hlo_test_base_legacy.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"

namespace xla::gpu {
namespace {

using ::testing::ElementsAre;
using Kind = Thunk::Kind;
using ::tsl::proto_testing::EqualsProto;

using GpuCollectivePermuteTest = HloTestBaseLegacy;

// Test case to verify that a CollectivePermute HLO instruction is correctly
// converted into a sequence of command buffer commands (Start and Done).
TEST_F(GpuCollectivePermuteTest, TestConvertToCommands) {
  // Generate HLO text
  std::string hlo_text = R"(
HloModule test, replica_count=2
ENTRY test_computation {
  p = u32[4] parameter(0)
  ROOT permute = u32[4] collective-permute(p), source_target_pairs={{0,1}, {1,0}}
}
)";

  // Configure module with debug options for command buffer.
  HloModuleConfig config;
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_graph_min_graph_size(1);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::COLLECTIVES);
  config.set_debug_options(debug_options);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));

  // Get CollectivePermute Instruction
  const HloInstruction* root_instr =
      module->entry_computation()->root_instruction();
  ASSERT_EQ(root_instr->opcode(), HloOpcode::kCollectivePermute);
  const HloCollectivePermuteInstruction* cp_instr =
      tensorflow::down_cast<const HloCollectivePermuteInstruction*>(root_instr);
  ASSERT_NE(cp_instr, nullptr);

  // Buffer and Allocation Setup
  using DataT = int32_t;
  constexpr int64_t kNumElements = 4;
  constexpr int64_t kAlignmentBytes = kXlaAllocatedBufferAlignBytes;

  const int64_t kElementSize = sizeof(DataT);
  const int64_t kTotalDataBytes = kNumElements * kElementSize;
  Shape shape = ShapeUtil::MakeShape(S32, {kNumElements});

  // Use RoundUpTo to calculate the actual size needed for one buffer.
  const int64_t kAlignedSliceBytes =
      xla::RoundUpTo<uint64_t>(kTotalDataBytes, kAlignmentBytes);

  // The total buffer size must accommodate input and output slices.
  const int64_t kTotalBufferBytes = 2 * kAlignedSliceBytes;

  BufferAllocation buffer_allocation(/*index=*/0, kTotalBufferBytes,
                                     /*color=*/0);
  BufferAllocation::Slice input_slice(&buffer_allocation, /*offset=*/0,
                                      kAlignedSliceBytes);
  BufferAllocation::Slice output_slice(&buffer_allocation, kAlignedSliceBytes,
                                       kAlignedSliceBytes);

  // Use designated initializers if possible, or format for clarity.
  std::vector<CollectiveThunk::Buffer> buffers = {
      {/*element_count=*/kNumElements,
       /*source_buffer=*/{input_slice, shape},
       /*destination_buffer=*/{output_slice, shape},
       /*source_memory_space=*/0,
       /*destination_memory_space=*/0},
  };

  // ThunkSequence Creation
  auto cp_start_thunk = std::make_unique<CollectivePermuteThunk>(
      Thunk::ThunkInfo{}, cp_instr, /*replica_count=*/2,
      /*partition_count=*/1, std::move(buffers),
      /*collectives_mode=*/DebugOptions::COLLECTIVES_PRIVATE_MEMORY,
      /*connected_components_enabled=*/false);

  ThunkSequence start_sequence;
  start_sequence.push_back(std::move(cp_start_thunk));
  auto async_start = std::make_unique<AsyncStartThunk>(
      Thunk::ThunkInfo(), CommunicationStreamId(0), std::move(start_sequence));
  auto async_done = std::make_unique<AsyncDoneThunk>(
      Thunk::ThunkInfo(), async_start->async_execution());

  ThunkSequence thunk_sequence;
  thunk_sequence.push_back(std::move(async_start));
  thunk_sequence.push_back(std::move(async_done));

  // Convert to Commands and Verification
  ConvertToCommandsOptions conv_options;
  // Use LHS synchronization mode to append Done command
  conv_options.synchronization_mode =
      CommandExecutor::SynchronizationMode::kLHS;
  TF_ASSERT_OK_AND_ASSIGN(CommandExecutor cb_cmd_executor,
                          ConvertToCommands(thunk_sequence, conv_options));

  // AsyncStart inlines its nested thunk as a command, and AsyncDone
  // with no control predecessors is a no-op, so we get 1 command.
  EXPECT_EQ(cb_cmd_executor.size(), 1);
}

TEST_F(GpuCollectivePermuteTest,
       TestCommandBufferThunkContainsCollectivePermute) {
  // Generate HLO text
  std::string hlo_text = R"(
HloModule test, replica_count=2
ENTRY test_computation {
  replica = u32[] replica-id()
  p = u32[4] broadcast(replica), dimensions={}
  ROOT permute = u32[4] collective-permute(p), source_target_pairs={{0,1}, {1,0}}
}
)";

  // Configure module with debug options for command buffer.
  HloModuleConfig config;
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_graph_min_graph_size(1);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::COLLECTIVES);
  config.set_debug_options(debug_options);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));

  se::StreamExecutor* executor = backend().default_stream_executor();

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> compiled_module,
      backend().compiler()->RunHloPasses(module->Clone(), executor,
                                         /*device_allocator=*/nullptr));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Executable> executable,
      backend().compiler()->RunBackend(std::move(compiled_module), executor,
                                       /*device_allocator=*/nullptr));
  // Downcast to GPU executable
  xla::gpu::GpuExecutable* gpu_executable =
      tensorflow::down_cast<xla::gpu::GpuExecutable*>(executable.get());
  ASSERT_NE(gpu_executable, nullptr);

  // Get the thunk sequence and check its size and type
  const ThunkExecutor& seq_thunk = gpu_executable->thunk_executor();
  ASSERT_EQ(seq_thunk.thunks().size(), 1);

  const std::unique_ptr<Thunk>& thunk = seq_thunk.thunks().front();
  ASSERT_EQ(thunk->kind(), Thunk::kCommandBuffer);

  // Downcast to the specific CommandBufferThunk type for inspection.
  CommandBufferThunk* cmd_buffer_thunk =
      tensorflow::down_cast<CommandBufferThunk*>(thunk.get());
  ASSERT_NE(cmd_buffer_thunk, nullptr);

  // Inspect the Thunk kinds
  std::vector<Kind> kinds;
  const auto& inner_thunks = cmd_buffer_thunk->thunks()->thunks();
  kinds.reserve(inner_thunks.size());
  for (const auto& thunk : inner_thunks) {
    kinds.push_back(thunk->kind());
  }
  // Verify that the inner Thunks match the expected sequence from the HLO.
  // The collective is sync (single device), so no AsyncStart/Done wrapping.
  EXPECT_THAT(kinds, ElementsAre(Kind::kReplicaId, Kind::kKernel,
                                 Kind::kCollectivePermute));
}

TEST(CollectiveThunkTest, ProtoRoundTrip) {
  ThunkProto proto = tsl::proto_testing::ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info { profile_annotation: "partition_id_profile_annotation" }
        collective_permute_thunk {
          collective_config {}
          collectives_mode: COLLECTIVES_PEER_MEMORY
          source_target_pairs: { source: 1 target: 2 }
        }
      )pb");

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = proto.thunk_info().profile_annotation();

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0)};

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CollectivePermuteThunk> thunk,
      CollectivePermuteThunk::FromProto(
          thunk_info, proto.collective_permute_thunk(), buffer_allocations));

  ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());

  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(CollectiveThunkTest, SyncCollective) {
  ThunkProto proto = tsl::proto_testing::ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info { profile_annotation: "partition_id_profile_annotation" }
        collective_permute_thunk {
          collective_config {}
          collectives_mode: COLLECTIVES_PEER_MEMORY
          source_target_pairs: { source: 1 target: 2 }
        }
      )pb");

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = proto.thunk_info().profile_annotation();

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0)};

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CollectivePermuteThunk> thunk,
      CollectivePermuteThunk::FromProto(
          thunk_info, proto.collective_permute_thunk(), buffer_allocations));
}

// Helper to extract just the sorted component member lists (ignoring root keys)
// for easier comparison.
std::vector<std::vector<int64_t>> ComponentValues(
    const absl::flat_hash_map<int64_t, std::vector<int64_t>>& components) {
  std::vector<std::vector<int64_t>> result;
  result.reserve(components.size());
  for (const auto& [root, members] : components) {
    result.push_back(members);
  }
  absl::c_sort(result);
  return result;
}

TEST(SourceTargetConnectedComponentsTest, SingleComponent) {
  // Ring pattern: 0->1->2->3->0, all connected.
  std::vector<std::pair<int64_t, int64_t>> pairs = {
      {0, 1}, {1, 2}, {2, 3}, {3, 0}};
  auto components = SourceTargetConnectedComponents(4, pairs);
  EXPECT_THAT(ComponentValues(components),
              ElementsAre(ElementsAre(0, 1, 2, 3)));
}

TEST(SourceTargetConnectedComponentsTest, TwoDisjointRings) {
  // Two 4-device rings on a 2-node setup: {0..3} and {4..7}.
  std::vector<std::pair<int64_t, int64_t>> pairs = {
      {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}};
  auto components = SourceTargetConnectedComponents(8, pairs);
  EXPECT_THAT(ComponentValues(components),
              ElementsAre(ElementsAre(0, 1, 2, 3), ElementsAre(4, 5, 6, 7)));
}

TEST(SourceTargetConnectedComponentsTest, IsolatedDevices) {
  // Only devices 0 and 1 communicate; device 2 is isolated.
  std::vector<std::pair<int64_t, int64_t>> pairs = {{0, 1}};
  auto components = SourceTargetConnectedComponents(3, pairs);
  EXPECT_THAT(ComponentValues(components),
              ElementsAre(ElementsAre(0, 1), ElementsAre(2)));
}

TEST(SourceTargetConnectedComponentsTest, AllIsolated) {
  // No pairs at all — every device is its own singleton.
  std::vector<std::pair<int64_t, int64_t>> pairs = {};
  auto components = SourceTargetConnectedComponents(4, pairs);
  EXPECT_THAT(ComponentValues(components),
              ElementsAre(ElementsAre(0), ElementsAre(1), ElementsAre(2),
                          ElementsAre(3)));
}

TEST(SourceTargetConnectedComponentsTest, ChainNotRing) {
  // Chain: 0->1->2->3 (no wrap). All connected via transitivity.
  std::vector<std::pair<int64_t, int64_t>> pairs = {{0, 1}, {1, 2}, {2, 3}};
  auto components = SourceTargetConnectedComponents(4, pairs);
  EXPECT_THAT(ComponentValues(components),
              ElementsAre(ElementsAre(0, 1, 2, 3)));
}

TEST(SourceTargetConnectedComponentsTest, SinglePairManyDevices) {
  // 16 devices, only 0->1 communicates.
  std::vector<std::pair<int64_t, int64_t>> pairs = {{0, 1}};
  auto components = SourceTargetConnectedComponents(16, pairs);
  // Should have {0,1} and 14 singletons.
  EXPECT_EQ(components.size(), 15);
  // Check the communicating pair is together.
  bool found_pair = false;
  for (const auto& [root, members] : components) {
    if (members.size() == 2) {
      EXPECT_THAT(members, ElementsAre(0, 1));
      found_pair = true;
    }
  }
  EXPECT_TRUE(found_pair);
}

TEST(RemapSourceTargetToCliqueRanksTest, CrossPartitionRemapsToCliqueRanks) {
  // 4 devices, partition IDs 0-3. Clique only has devices {2, 3} (mapped to
  // GlobalDeviceIds {2, 3}). Source-target pair: 2->3.
  DeviceAssignment device_assn(/*replica_count=*/1, /*computation_count=*/4);
  device_assn(0, 0) = 0;
  device_assn(0, 1) = 1;
  device_assn(0, 2) = 2;
  device_assn(0, 3) = 3;

  // Clique with devices {2, 3} — rank 0 = device 2, rank 1 = device 3.
  GpuCliqueKey clique_key({GlobalDeviceId(2), GlobalDeviceId(3)},
                          /*num_local_participants=*/2);

  P2PConfig::SourceTargetMapEntry source_target;
  source_target.source = 2;  // Partition ID 2 sends to us.
  source_target.target = 3;  // We send to partition ID 3.

  ASSERT_OK_AND_ASSIGN(
      auto remapped,
      RemapSourceTargetToCliqueRanks(
          source_target, clique_key, device_assn,
          CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_PARTITION,
          GlobalDeviceId(3)));

  // Partition 2 -> GlobalDeviceId 2 -> clique rank 0.
  EXPECT_EQ(remapped.source, RankId(0));
  // Partition 3 -> GlobalDeviceId 3 -> clique rank 1.
  EXPECT_EQ(remapped.target, RankId(1));
}

TEST(RemapSourceTargetToCliqueRanksTest, NoSourceOrTarget) {
  // Device is isolated — no source or target.
  DeviceAssignment device_assn(/*replica_count=*/1, /*computation_count=*/2);
  device_assn(0, 0) = 0;
  device_assn(0, 1) = 1;

  GpuCliqueKey clique_key({GlobalDeviceId(0)},
                          /*num_local_participants=*/1);

  P2PConfig::SourceTargetMapEntry source_target;  // Both nullopt.

  ASSERT_OK_AND_ASSIGN(
      auto remapped,
      RemapSourceTargetToCliqueRanks(
          source_target, clique_key, device_assn,
          CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_PARTITION,
          GlobalDeviceId(0)));

  EXPECT_EQ(remapped.source, std::nullopt);
  EXPECT_EQ(remapped.target, std::nullopt);
}

TEST(RemapSourceTargetToCliqueRanksTest, DeviceNotInCliqueReturnsError) {
  DeviceAssignment device_assn(/*replica_count=*/1, /*computation_count=*/4);
  device_assn(0, 0) = 0;
  device_assn(0, 1) = 1;
  device_assn(0, 2) = 2;
  device_assn(0, 3) = 3;

  // Clique only has device {0, 1}.
  GpuCliqueKey clique_key({GlobalDeviceId(0), GlobalDeviceId(1)},
                          /*num_local_participants=*/2);

  P2PConfig::SourceTargetMapEntry source_target;
  source_target.target = 3;  // Partition 3 -> GlobalDeviceId 3, not in clique.

  auto result = RemapSourceTargetToCliqueRanks(
      source_target, clique_key, device_assn,
      CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_PARTITION,
      GlobalDeviceId(0));

  EXPECT_FALSE(result.ok());
}

}  // namespace
}  // namespace xla::gpu
