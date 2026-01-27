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

#include "xla/backends/gpu/runtime/collective_broadcast_thunk.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/command_buffer_cmd_emitter.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/backend.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tests/hlo_test_base.h"
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

class GpuCollectiveBroadcastTest : public HloTestBase {};

TEST_F(GpuCollectiveBroadcastTest, TestConvertToCommands) {
  // Generate HLO text with parameters substituted.
  std::string hlo_text = R"(
HloModule test, replica_count=2
ENTRY test_computation {
  p = u32[4] parameter(0)
  ROOT res = u32[4] collective-broadcast(p), replica_groups={{1, 0}}
}
)";

  // Configure module with debug options for command buffer.
  HloModuleConfig config;
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_graph_min_graph_size(1);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::COLLECTIVES);
  config.set_debug_options(debug_options);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_text, config));

  // Get CollectiveBroadcast Instruction
  const HloInstruction* root_instr =
      module->entry_computation()->root_instruction();
  ASSERT_EQ(root_instr->opcode(), HloOpcode::kCollectiveBroadcast);
  const HloCollectiveBroadcastInstruction* cb_instr =
      absl::down_cast<const HloCollectiveBroadcastInstruction*>(root_instr);
  ASSERT_NE(cb_instr, nullptr);

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
  std::shared_ptr<CollectiveThunk::AsyncEvents> async_events =
      std::make_shared<CollectiveThunk::AsyncEvents>();

  auto cb_start_thunk = std::make_unique<CollectiveBroadcastStartThunk>(
      Thunk::ThunkInfo{}, cb_instr, std::move(buffers));

  cb_start_thunk->set_async_events(async_events);

  auto cb_done_thunk = std::make_unique<CollectiveDoneThunk>(
      Kind::kCollectiveBroadcastDone, Thunk::ThunkInfo{}, async_events);

  ThunkSequence thunk_sequence;
  thunk_sequence.push_back(std::move(cb_start_thunk));
  thunk_sequence.push_back(std::move(cb_done_thunk));

  // Convert to Commands and Verification
  ConvertToCommandsOptions conv_options;
  // Use LHS synchronization mode to append Done command
  conv_options.synchronization_mode =
      CommandBufferCmdExecutor::SynchronizationMode::kLHS;
  TF_ASSERT_OK_AND_ASSIGN(CommandBufferCmdExecutor cb_cmd_executor,
                          ConvertToCommands(thunk_sequence, conv_options));

  // Check that we have two commands: start and done.
  EXPECT_EQ(cb_cmd_executor.size(), 2);
}

TEST_F(GpuCollectiveBroadcastTest,
       TestCommandBufferThunkContainsCorrectThunks) {
  // Generate HLO text with parameters substituted.
  std::string hlo_text = R"(
HloModule test, replica_count=2
ENTRY test_computation {
  replica = u32[] replica-id()
  p = u32[4] broadcast(replica), dimensions={}
  ROOT res = u32[4] collective-broadcast(p), replica_groups={{1, 0}}
}
)";

  // Configure module with debug options for command buffer.
  HloModuleConfig config;
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_graph_min_graph_size(1);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::COLLECTIVES);
  config.set_debug_options(debug_options);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
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
      absl::down_cast<GpuExecutable*>(executable.get());
  ASSERT_NE(gpu_executable, nullptr);

  // Get the thunk sequence and check its size and type
  const SequentialThunk& seq_thunk = gpu_executable->GetThunk();
  ASSERT_EQ(seq_thunk.thunks().size(), 1);

  const std::unique_ptr<Thunk>& thunk = seq_thunk.thunks().front();
  ASSERT_EQ(thunk->kind(), Thunk::kCommandBuffer);

  CommandBufferThunk* cmd_buffer_thunk =
      absl::down_cast<CommandBufferThunk*>(thunk.get());
  ASSERT_NE(cmd_buffer_thunk, nullptr);

  std::vector<Kind> kinds;
  const auto& inner_thunks = cmd_buffer_thunk->thunks()->thunks();
  kinds.reserve(inner_thunks.size());
  for (const auto& thunk : inner_thunks) {
    kinds.push_back(thunk->kind());
  }
  EXPECT_THAT(kinds, ElementsAre(Kind::kReplicaId, Kind::kKernel,
                                 Kind::kCollectiveBroadcastStart,
                                 Kind::kCollectiveBroadcastDone));
}

TEST(CollectiveThunkTest, ProtoRoundTrip) {
  ThunkProto proto = tsl::proto_testing::ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info {
          profile_annotation: "partition_id_profile_annotation"
          execution_stream_id: 2
        }
        collective_broadcast_start_thunk {
          async_events_unique_id: 3
          collective_config {}
        }
      )pb");

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = proto.thunk_info().profile_annotation();
  thunk_info.execution_stream_id = xla::gpu::ExecutionStreamId{
      static_cast<xla::gpu::ExecutionStreamId::ValueType>(
          proto.thunk_info().execution_stream_id())};

  CollectiveThunk::AsyncEventsMap async_events_map;
  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0)};

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<CollectiveBroadcastStartThunk> thunk,
                       CollectiveBroadcastStartThunk::FromProto(
                           thunk_info, proto.collective_broadcast_start_thunk(),
                           buffer_allocations, async_events_map));
  ASSERT_NE(thunk->async_events(), nullptr);

  ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());

  // Ids are unique and expected to differ.
  proto.mutable_collective_broadcast_start_thunk()->set_async_events_unique_id(
      round_trip_proto.collective_broadcast_start_thunk()
          .async_events_unique_id());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

}  // namespace
}  // namespace xla::gpu
