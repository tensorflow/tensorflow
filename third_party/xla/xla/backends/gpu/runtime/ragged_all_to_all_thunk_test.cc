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

#include "xla/backends/gpu/runtime/ragged_all_to_all_thunk.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
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

using GpuRaggedAllToAllTest = HloTestBase;

TEST_F(GpuRaggedAllToAllTest, TestConvertToCommands) {
  // Generate HLO text with parameters substituted.
  constexpr absl::string_view hlo_text = R"(
  HloModule module, num_partitions=1, replica_count=2
  ENTRY main {
      p0 = f32[8] parameter(0)
      id = u32[] replica-id()
      output = f32[8] constant({-1, -1, -1, -1, -1, -1, -1, -1})
      send_sizes = s32[2] constant({4, 4})
      recv_sizes = s32[2] constant({4, 4})
      input_offsets = s32[2] constant({0, 4})
      four = u32[] constant(4)
      oof = u32[] multiply(id, four)
      oof2 = s32[] convert(oof)
      output_offsets = s32[2] broadcast(oof2)

      ROOT ra2a = f32[8] ragged-all-to-all(p0, output, input_offsets, send_sizes, output_offsets, recv_sizes), replica_groups={{0,1}}
  }
  )";

  // Configure module with debug options for command buffer.
  HloModuleConfig config;
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_graph_min_graph_size(1);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::COLLECTIVES);
  debug_options.set_xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel(
      true);
  debug_options.set_xla_gpu_experimental_ragged_all_to_all_use_barrier(true);
  config.set_debug_options(debug_options);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_text, config));

  // Get CollectiveBroadcast Instruction
  const HloInstruction* root_instr =
      module->entry_computation()->root_instruction();
  ASSERT_EQ(root_instr->opcode(), HloOpcode::kRaggedAllToAll);
  const HloRaggedAllToAllInstruction* ra2a_instr =
      tensorflow::down_cast<const HloRaggedAllToAllInstruction*>(root_instr);
  ASSERT_NE(ra2a_instr, nullptr);

  // Buffer and Allocation Setup
  // We allocate a generic dummy slice. Because this test only validates command
  // conversion and doesn't execute kernels, reusing the slice is completely
  // safe.
  BufferAllocation buffer_allocation(/*index=*/0, /*size=*/4096, /*color=*/0);
  BufferAllocation::Slice dummy_slice(&buffer_allocation, /*offset=*/0,
                                      /*size=*/4096);

  std::vector<CollectiveThunk::Buffer> buffers;
  buffers.reserve(ra2a_instr->operand_count());

  // RaggedAllToAll expects 6 buffers (data + 5 metadata arrays).
  // Populate them dynamically based on the instruction's operands.
  for (int i = 0; i < ra2a_instr->operand_count(); ++i) {
    const Shape& shape = ra2a_instr->operand(i)->shape();
    buffers.push_back({/*element_count=*/ShapeUtil::ElementsIn(shape),
                       /*source_buffer=*/{dummy_slice, shape},
                       /*destination_buffer=*/{dummy_slice, shape},
                       /*source_memory_space=*/0,
                       /*destination_memory_space=*/0});
  }

  // ThunkSequence Creation
  std::shared_ptr<CollectiveThunk::AsyncEvents> async_events =
      std::make_shared<CollectiveThunk::AsyncEvents>();

  auto ra2a_start_thunk = std::make_unique<RaggedAllToAllStartThunk>(
      Thunk::ThunkInfo{}, ra2a_instr, std::move(buffers),
      /*p2p_memcpy_enabled=*/false);

  ra2a_start_thunk->set_async_events(async_events);

  auto ra2a_done_thunk = std::make_unique<CollectiveDoneThunk>(
      Kind::kRaggedAllToAllDone, Thunk::ThunkInfo{}, async_events);

  ThunkSequence thunk_sequence;
  thunk_sequence.push_back(std::move(ra2a_start_thunk));
  thunk_sequence.push_back(std::move(ra2a_done_thunk));

  // Convert to Commands and Verification
  ConvertToCommandsOptions conv_options;
  // Use LHS synchronization mode to append Done command
  conv_options.synchronization_mode =
      CommandExecutor::SynchronizationMode::kLHS;
  TF_ASSERT_OK_AND_ASSIGN(CommandExecutor cb_cmd_executor,
                          ConvertToCommands(thunk_sequence, conv_options));

  // Check that we have two commands: start and done.
  EXPECT_EQ(cb_cmd_executor.size(), 2);
}

TEST_F(GpuRaggedAllToAllTest, TestCommandBufferThunkContainsCorrectThunks) {
  // Generate HLO text with parameters substituted.
  constexpr absl::string_view hlo_text = R"(
  HloModule module, replica_count=2

  ENTRY entry {
    p0 = f32[8] parameter(0)
    output = f32[8] constant({-1, -1, -1, -1, -1, -1, -1, -1})
    send_sizes = s32[2] constant({4, 4})
    recv_sizes = s32[2] constant({4, 4})
    input_offsets = s32[2] constant({0, 4})
    output_offsets = s32[2] constant({0, 0})
    ROOT ra2a = f32[8] ragged-all-to-all(p0, output, input_offsets, send_sizes, output_offsets, recv_sizes), replica_groups={{0,1}}
  }
  )";

  // Configure module with debug options for command buffer.
  HloModuleConfig config = GetModuleConfigForTest(/*replica_count=*/2);
  DebugOptions& debug_options = config.mutable_debug_options();
  debug_options.set_xla_gpu_graph_min_graph_size(1);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::COLLECTIVES);
  debug_options.set_xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel(
      true);
  debug_options.set_xla_gpu_experimental_ragged_all_to_all_use_barrier(true);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_text, config));

  se::StreamExecutor* executor = backend().default_stream_executor();
  // CHECK_NE(executor, nullptr);

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
  const SequentialThunk& seq_thunk = gpu_executable->GetThunk();
  ASSERT_EQ(seq_thunk.thunks().size(), 1);

  const std::unique_ptr<Thunk>& thunk = seq_thunk.thunks().front();
  ASSERT_EQ(thunk->kind(), Thunk::kCommandBuffer);

  CommandBufferThunk* cmd_buffer_thunk =
      tensorflow::down_cast<CommandBufferThunk*>(thunk.get());
  ASSERT_NE(cmd_buffer_thunk, nullptr);

  std::vector<Kind> kinds;
  const auto& inner_thunks = cmd_buffer_thunk->thunks()->thunks();
  kinds.reserve(inner_thunks.size());
  for (const auto& thunk : inner_thunks) {
    kinds.push_back(thunk->kind());
  }

  EXPECT_THAT(kinds, ElementsAre(Kind::kKernel, Kind::kKernel, Kind::kKernel,
                                 Kind::kKernel, Kind::kRaggedAllToAllStart,
                                 Kind::kRaggedAllToAllDone));
}

TEST(CollectiveThunkTest, ProtoRoundTrip) {
  ThunkProto proto = tsl::proto_testing::ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info {
          profile_annotation: "partition_id_profile_annotation"
          execution_stream_id: 2
        }
        ragged_all_to_all_start_thunk {
          async_events_unique_id: 3
          collective_config {}
          num_total_updates: 10
          num_input_rows: 2
          num_row_elements: 5
          one_shot_kernel_enabled: true
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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<RaggedAllToAllStartThunk> thunk,
                       RaggedAllToAllStartThunk::FromProto(
                           thunk_info, proto.ragged_all_to_all_start_thunk(),
                           buffer_allocations, async_events_map));
  ASSERT_NE(thunk->async_events(), nullptr);

  ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());

  // Ids are unique and expected to differ.
  proto.mutable_ragged_all_to_all_start_thunk()->set_async_events_unique_id(
      round_trip_proto.ragged_all_to_all_start_thunk()
          .async_events_unique_id());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

}  // namespace
}  // namespace xla::gpu
