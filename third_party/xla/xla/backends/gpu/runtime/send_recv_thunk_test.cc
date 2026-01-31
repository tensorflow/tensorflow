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

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/command_buffer_cmd_emitter.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/recv_thunk.h"
#include "xla/backends/gpu/runtime/send_thunk.h"
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
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"

namespace xla::gpu {
namespace {

using ::testing::UnorderedElementsAre;
using Kind = Thunk::Kind;

class GpuSendRecvTest : public HloTestBase {};

// Test case to verify that Send HLO instruction is correctly
// converted into a sequence of command buffer commands (Send and SendDone).
TEST_F(GpuSendRecvTest, TestConvertSendToCommands) {
  // Generate HLO text
  absl::string_view hlo_text = R"(
HloModule test, replica_count=2
ENTRY computation {
  a = u32[8] parameter(0)
  after_all = token[] after-all()
  send = (u32[8], u32[8], token[]) send(a, after_all),
      frontend_attributes={_xla_send_recv_source_target_pairs="{{1,0}}"}
  ROOT send_done = token[] send-done(send)
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

  // Get Send Instruction
  const HloInstruction* root2_instr =
      module->entry_computation()->root_instruction()->operand(0);
  ASSERT_EQ(root2_instr->opcode(), HloOpcode::kSend);
  const HloSendInstruction* send_instr =
      tensorflow::down_cast<const HloSendInstruction*>(root2_instr);
  ASSERT_NE(send_instr, nullptr);

  // Buffer and Allocation Setup
  using DataT = int32_t;
  constexpr int64_t kNumElements = 8;
  constexpr int64_t kAlignmentBytes = kXlaAllocatedBufferAlignBytes;

  const int64_t kElementSize = sizeof(DataT);
  const int64_t kTotalDataBytes = kNumElements * kElementSize;
  Shape shape = ShapeUtil::MakeShape(S32, {kNumElements});

  // Use RoundUpTo to calculate the actual size needed for one buffer.
  const int64_t kAlignedSliceBytes =
      xla::RoundUpTo<uint64_t>(kTotalDataBytes, kAlignmentBytes);

  // The total buffer size must accommodate input and output slices.
  // But output slice is alias of input slice in Send operation,
  // so we only need one copy.
  const int64_t kTotalBufferBytes = 1 * kAlignedSliceBytes;

  BufferAllocation buffer_allocation(/*index=*/0, kTotalBufferBytes,
                                     /*color=*/0);
  BufferAllocation::Slice input_slice(&buffer_allocation, /*offset=*/0,
                                      kAlignedSliceBytes);

  CollectiveThunk::Buffer buffer = {/*element_count=*/kNumElements,
                                    /*source_buffer=*/{input_slice, shape},
                                    /*destination_buffer=*/{input_slice, shape},
                                    /*source_memory_space=*/0,
                                    /*destination_memory_space=*/0};

  // ThunkSequence Creation
  std::shared_ptr<CollectiveThunk::AsyncEvents> async_events =
      std::make_shared<CollectiveThunk::AsyncEvents>();

  auto send_thunk = std::make_unique<SendThunk>(Thunk::ThunkInfo{}, send_instr,
                                                /*replica_count=*/2,
                                                /*partition_count=*/1, buffer);

  send_thunk->set_async_events(async_events);

  auto send_done_thunk = std::make_unique<CollectiveDoneThunk>(
      Kind::kSendDone, Thunk::ThunkInfo{}, async_events);

  ThunkSequence thunk_sequence;
  thunk_sequence.push_back(std::move(send_thunk));
  thunk_sequence.push_back(std::move(send_done_thunk));

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

// Test case to verify that Recv HLO instruction is correctly
// converted into a sequence of command buffer commands (Recv and RecvDone).
TEST_F(GpuSendRecvTest, TestConvertRecvToCommands) {
  // Generate HLO text
  absl::string_view hlo_text = R"(
HloModule test, replica_count=2
ENTRY computation {
  a = u32[8] parameter(0)
  after_all = token[] after-all()
  recv = (u32[8], u32[8], token[]) recv(after_all),
      frontend_attributes={_xla_send_recv_source_target_pairs="{{1,0}}"}
  ROOT recv_done = token[] recv-done(recv)
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

  // Get Recv Instruction
  const HloInstruction* root2_instr =
      module->entry_computation()->root_instruction()->operand(0);
  ASSERT_EQ(root2_instr->opcode(), HloOpcode::kRecv);
  const HloRecvInstruction* recv_instr =
      tensorflow::down_cast<const HloRecvInstruction*>(root2_instr);
  ASSERT_NE(recv_instr, nullptr);

  // Buffer and Allocation Setup
  using DataT = int32_t;
  constexpr int64_t kNumElements = 8;
  constexpr int64_t kAlignmentBytes = kXlaAllocatedBufferAlignBytes;
  Shape shape = ShapeUtil::MakeShape(S32, {kNumElements});

  const int64_t kElementSize = sizeof(DataT);
  const int64_t kTotalDataBytes = kNumElements * kElementSize;

  // Use RoundUpTo to calculate the actual size needed for one buffer.
  const int64_t kAlignedSliceBytes =
      xla::RoundUpTo<uint64_t>(kTotalDataBytes, kAlignmentBytes);

  // The total buffer size must accommodate input and output slices.
  // But output slice is alias of input slice in Recv operation,
  // so we only need one copy.
  const int64_t kTotalBufferBytes = 1 * kAlignedSliceBytes;

  BufferAllocation buffer_allocation(/*index=*/0, kTotalBufferBytes,
                                     /*color=*/0);
  BufferAllocation::Slice input_slice(&buffer_allocation, /*offset=*/0,
                                      kAlignedSliceBytes);

  // Use designated initializers if possible, or format for clarity.
  CollectiveThunk::Buffer buffer = {/*element_count=*/kNumElements,
                                    /*source_buffer=*/{input_slice, shape},
                                    /*destination_buffer=*/{input_slice, shape},
                                    /*source_memory_space=*/0,
                                    /*destination_memory_space=*/0};

  // ThunkSequence Creation
  std::shared_ptr<CollectiveThunk::AsyncEvents> async_events =
      std::make_shared<CollectiveThunk::AsyncEvents>();

  auto recv_thunk = std::make_unique<RecvThunk>(Thunk::ThunkInfo{}, recv_instr,
                                                /*replica_count=*/2,
                                                /*partition_count=*/1, buffer);

  recv_thunk->set_async_events(async_events);

  auto recv_done_thunk = std::make_unique<CollectiveDoneThunk>(
      Kind::kRecvDone, Thunk::ThunkInfo{}, async_events);

  ThunkSequence thunk_sequence;
  thunk_sequence.push_back(std::move(recv_thunk));
  thunk_sequence.push_back(std::move(recv_done_thunk));

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

TEST_F(GpuSendRecvTest, TestCommandBufferThunkContainsSendRecv) {
  // Generate HLO text
  absl::string_view hlo_text = R"(
HloModule test, replica_count=2
ENTRY computation {
  replica = u32[] replica-id()
  a = u32[8] broadcast(replica), dimensions={}
  after_all = token[] after-all()

  send = (u32[8], u32[8], token[]) send(a, after_all),
      frontend_attributes={_xla_send_recv_source_target_pairs="{{1,0}}"}
  recv = (u32[8], u32[8], token[]) recv(after_all),
      frontend_attributes={_xla_send_recv_source_target_pairs="{{1,0}}"}
  send_done = token[] send-done(send)
  ROOT recv_done = (u32[8], token[]) recv-done(recv)
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
  const SequentialThunk& seq_thunk = gpu_executable->GetThunk();
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
  // Verify that the inner Thunks match the expected sequence from the HLO
  EXPECT_THAT(kinds, UnorderedElementsAre(Kind::kReplicaId, Kind::kKernel,
                                          Kind::kSend, Kind::kSendDone,
                                          Kind::kRecv, Kind::kRecvDone));
}

}  // namespace
}  // namespace xla::gpu
