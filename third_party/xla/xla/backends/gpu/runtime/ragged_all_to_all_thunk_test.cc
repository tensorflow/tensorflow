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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/async_thunk.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_buffer_cmd_emitter.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/execution_stream_id.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/runtime/device_id.h"
#include "xla/service/backend.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/computation_placer.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"
#include "xla/tests/restricted/hlo_test_base_legacy.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"

namespace xla::gpu {
namespace {

using ::testing::ElementsAre;
using Kind = Thunk::Kind;
using ::tsl::proto_testing::EqualsProto;

using GpuRaggedAllToAllTest = HloTestBaseLegacy;

static constexpr int64_t kNumOneRankUpdates = 2;
static constexpr int64_t kNumOneRankInputRows = 4;
static constexpr int64_t kNumOneRankRowElements = 1;
static constexpr int64_t kNumOneRankElements =
    kNumOneRankInputRows * kNumOneRankRowElements;
static constexpr int64_t kNumRaggedBuffers = 6;

static bool IsAtLeastCuda12900(const se::StreamExecutor* executor) {
  const auto& desc = executor->GetDeviceDescription();
  const auto* cuda_cc = desc.gpu_compute_capability().cuda_compute_capability();
  if (cuda_cc == nullptr) {
    return false;
  }
  return std::min(desc.driver_version(), desc.compile_time_toolkit_version()) >=
         se::SemanticVersion(12, 9, 0);
}

static se::StreamExecutor* GpuExecutor() {
  auto* platform =
      se::PlatformManager::PlatformWithName(se::GpuPlatformName()).value();
  return platform->ExecutorForDevice(0).value();
}

static RaggedAllToAllConfig MakeOneRankConfig() {
  ReplicaGroup replica_group;
  replica_group.add_replica_ids(0);

  RaggedAllToAllConfig config;
  config.config.operand_element_type = {F32, F32, S64, S64, S64, S64};
  config.config.replica_groups = {replica_group};
  config.config.group_mode = COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA;
  config.num_total_updates = kNumOneRankUpdates;
  config.num_input_rows = kNumOneRankInputRows;
  config.num_row_elements = kNumOneRankRowElements;
  config.one_shot_kernel_enabled = true;
  return config;
}

static CollectiveThunk::Buffer MakeBuffer(const BufferAllocation& allocation,
                                          PrimitiveType element_type,
                                          int64_t element_count) {
  Shape shape = ShapeUtil::MakeShape(element_type, {element_count});
  ShapedSlice slice{BufferAllocation::Slice(
                        &allocation, 0, ShapeUtil::ByteSizeOfElements(shape)),
                    shape};
  return CollectiveThunk::Buffer{.element_count = element_count,
                                 .source_buffer = slice,
                                 .destination_buffer = slice,
                                 .source_memory_space = 0,
                                 .destination_memory_space = 0};
}

static std::vector<BufferAllocation> MakeOneRankBufferAllocations() {
  std::vector<BufferAllocation> allocations;
  allocations.reserve(kNumRaggedBuffers);
  allocations.emplace_back(/*index=*/0, sizeof(float) * kNumOneRankElements,
                           /*color=*/0);
  allocations.emplace_back(/*index=*/1, sizeof(float) * kNumOneRankElements,
                           /*color=*/0);
  for (int i = 2; i < kNumRaggedBuffers; ++i) {
    allocations.emplace_back(i, sizeof(int64_t) * kNumOneRankUpdates,
                             /*color=*/0);
  }
  return allocations;
}

static RaggedAllToAllThunk MakeOneRankThunk(
    const std::vector<BufferAllocation>& allocations) {
  std::vector<CollectiveThunk::Buffer> buffers;
  buffers.reserve(kNumRaggedBuffers);
  buffers.push_back(MakeBuffer(allocations[0], F32, kNumOneRankElements));
  buffers.push_back(MakeBuffer(allocations[1], F32, kNumOneRankElements));
  for (int i = 2; i < kNumRaggedBuffers; ++i) {
    buffers.push_back(MakeBuffer(allocations[i], S64, kNumOneRankUpdates));
  }
  return RaggedAllToAllThunk(Thunk::ThunkInfo(), MakeOneRankConfig(),
                             std::move(buffers));
}

static std::vector<se::DeviceAddressBase> AllocateOneRankDeviceBuffers(
    se::StreamExecutor* executor) {
  std::vector<se::DeviceAddressBase> buffers;
  buffers.reserve(kNumRaggedBuffers);
  buffers.push_back(executor->AllocateArray<float>(kNumOneRankElements, 0));
  buffers.push_back(executor->AllocateArray<float>(kNumOneRankElements, 0));
  for (int i = 2; i < kNumRaggedBuffers; ++i) {
    buffers.push_back(executor->AllocateArray<int64_t>(kNumOneRankUpdates, 0));
  }
  return buffers;
}

static std::vector<float> OneRankInputValues(int phase) {
  std::vector<float> values(kNumOneRankElements);
  for (int i = 0; i < values.size(); ++i) {
    values[i] = static_cast<float>(phase * 100 + i);
  }
  return values;
}

static std::vector<float> OneRankExpectedValues(int phase) {
  std::vector<float> input = OneRankInputValues(phase);
  return {-1.0f, input[0], input[2], input[3]};
}

static absl::Status WriteBuffer(se::Stream& stream,
                                se::DeviceAddressBase buffer,
                                const std::vector<float>& data) {
  RETURN_IF_ERROR(
      stream.Memcpy(&buffer, data.data(), data.size() * sizeof(float)));
  return absl::OkStatus();
}

static absl::Status WriteBuffer(se::Stream& stream,
                                se::DeviceAddressBase buffer,
                                const std::vector<int64_t>& data) {
  RETURN_IF_ERROR(
      stream.Memcpy(&buffer, data.data(), data.size() * sizeof(int64_t)));
  return absl::OkStatus();
}

static absl::Status PrepareOneRankInputs(
    se::Stream& stream, absl::Span<const se::DeviceAddressBase> buffers,
    int phase) {
  RETURN_IF_ERROR(WriteBuffer(stream, buffers[0], OneRankInputValues(phase)));
  RETURN_IF_ERROR(WriteBuffer(stream, buffers[1],
                              std::vector<float>(kNumOneRankElements, -1.0f)));
  RETURN_IF_ERROR(WriteBuffer(stream, buffers[2], std::vector<int64_t>{0, 2}));
  RETURN_IF_ERROR(WriteBuffer(stream, buffers[3], std::vector<int64_t>{1, 2}));
  RETURN_IF_ERROR(WriteBuffer(stream, buffers[4], std::vector<int64_t>{1, 2}));
  RETURN_IF_ERROR(WriteBuffer(stream, buffers[5], std::vector<int64_t>{1, 2}));
  return stream.BlockHostUntilDone();
}

static absl::StatusOr<std::vector<float>> ReadOneRankOutput(
    se::Stream& stream, se::DeviceAddressBase buffer) {
  std::vector<float> output(kNumOneRankElements);
  RETURN_IF_ERROR(
      stream.Memcpy(output.data(), buffer, output.size() * sizeof(float)));
  RETURN_IF_ERROR(stream.BlockHostUntilDone());
  return output;
}

static absl::Status VerifyOneRankOutput(se::Stream& stream,
                                        se::DeviceAddressBase buffer,
                                        int phase) {
  ASSIGN_OR_RETURN(std::vector<float> output,
                   ReadOneRankOutput(stream, buffer));
  std::vector<float> expected = OneRankExpectedValues(phase);
  for (int i = 0; i < output.size(); ++i) {
    if (output[i] != expected[i]) {
      return absl::InternalError(absl::StrFormat("output[%d] = %g, expected %g",
                                                 i, output[i], expected[i]));
    }
  }
  return absl::OkStatus();
}

TEST(RaggedAllToAllThunkTest, RecordCommandBufferCreateAndUpdate) {
  se::StreamExecutor* executor = GpuExecutor();
  if (!IsAtLeastCuda12900(executor)) {
    GTEST_SKIP() << "Child command nodes require CUDA 12.9+";
  }

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                       executor->CreateStream());
  auto allocator =
      std::make_unique<se::StreamExecutorAddressAllocator>(executor);

  DeviceAssignment device_assignment(/*replica_count=*/1,
                                     /*computation_count=*/1);
  device_assignment(0, 0) = 0;

  GpuExecutableRunOptions::DeviceIdMap id_map;
  id_map[LocalDeviceId(0)] = GlobalDeviceId(0);
  GpuExecutableRunOptions gpu_run_options;
  gpu_run_options.set_gpu_global_device_ids(std::move(id_map));

  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  run_options.mutable_run_options()->set_device_assignment(&device_assignment);
  run_options.mutable_run_options()->set_local_device_count(1);
  run_options.mutable_run_options()->set_gpu_executable_run_options(
      &gpu_run_options);

  ASSERT_OK_AND_ASSIGN(
      CollectiveParams collective_params,
      CollectiveParams::Create(run_options, /*async_streams=*/{},
                               LocalDeviceId(0)));

  std::vector<BufferAllocation> buffer_allocations =
      MakeOneRankBufferAllocations();
  RaggedAllToAllThunk thunk = MakeOneRankThunk(buffer_allocations);

  std::vector<se::DeviceAddressBase> phase1_buffers =
      AllocateOneRankDeviceBuffers(executor);
  std::vector<se::DeviceAddressBase> phase2_buffers =
      AllocateOneRankDeviceBuffers(executor);

  BufferAllocations allocations1(phase1_buffers, /*device_ordinal=*/0,
                                 allocator.get());
  CollectiveCliqueRequests clique_requests;
  CollectiveMemoryRequests memory_requests(allocations1);
  Thunk::PrepareParams prepare_params{&collective_params, &clique_requests,
                                      &memory_requests, executor,
                                      &allocations1};
  ASSERT_OK(thunk.Prepare(prepare_params));
  ASSERT_OK_AND_ASSIGN(
      CollectiveCliques collective_cliques,
      AcquireCollectiveCliques(collective_params, clique_requests));

  Thunk::InitializeParams init_params;
  init_params.executor = executor;
  init_params.stream = stream.get();
  init_params.buffer_allocations = &allocations1;
  init_params.collective_params = &collective_params;
  init_params.collective_cliques = &collective_cliques;
  init_params.local_device_count = 1;
  ASSERT_OK(thunk.Initialize(init_params));

  ASSERT_OK(PrepareOneRankInputs(*stream, phase1_buffers, /*phase=*/1));
  Thunk::ExecuteParams execute_params1 = Thunk::ExecuteParams::Create(
      run_options, allocations1, stream.get(),
      /*command_buffer_trace_stream=*/stream.get(), &collective_params,
      &collective_cliques, /*collective_memory=*/nullptr);

  ASSERT_OK(thunk.ExecuteOnStream(execute_params1));
  ASSERT_OK(stream->BlockHostUntilDone());
  ASSERT_OK(PrepareOneRankInputs(*stream, phase1_buffers, /*phase=*/1));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<se::CommandBuffer> command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  CommandStateManager state_manager;
  Command::RecordParams record_params = {state_manager};
  ASSERT_OK_AND_ASSIGN(const se::CommandBuffer::Command* command,
                       thunk.Record(execute_params1, record_params,
                                    Command::RecordCreate{/*dependencies=*/{}},
                                    command_buffer.get()));
  ASSERT_NE(command, nullptr);
  ASSERT_OK(command_buffer->Finalize());
  ASSERT_OK(command_buffer->Submit(stream.get()));
  ASSERT_OK(stream->BlockHostUntilDone());
  ASSERT_OK(VerifyOneRankOutput(*stream, phase1_buffers[1], /*phase=*/1));

  ASSERT_OK(PrepareOneRankInputs(*stream, phase2_buffers, /*phase=*/2));
  BufferAllocations allocations2(phase2_buffers, /*device_ordinal=*/0,
                                 allocator.get());
  Thunk::ExecuteParams execute_params2 = Thunk::ExecuteParams::Create(
      run_options, allocations2, stream.get(),
      /*command_buffer_trace_stream=*/stream.get(), &collective_params,
      &collective_cliques, /*collective_memory=*/nullptr);

  std::vector<BufferAllocation::Index> updated_allocs;
  updated_allocs.reserve(kNumRaggedBuffers);
  for (int i = 0; i < kNumRaggedBuffers; ++i) {
    updated_allocs.push_back(i);
  }
  Command::RecordParams update_params = {state_manager,
                                         std::move(updated_allocs)};

  ASSERT_OK(command_buffer->Update());
  ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* updated_command,
      thunk.Record(execute_params2, update_params,
                   Command::RecordUpdate{command}, command_buffer.get()));
  EXPECT_EQ(updated_command, command);
  ASSERT_OK(command_buffer->Finalize());
  ASSERT_OK(command_buffer->Submit(stream.get()));
  ASSERT_OK(stream->BlockHostUntilDone());
  ASSERT_OK(VerifyOneRankOutput(*stream, phase2_buffers[1], /*phase=*/2));
}

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
  config.set_debug_options(debug_options);

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(hlo_text, config));

  // Get CollectiveBroadcast Instruction
  const HloInstruction* root_instr =
      module->entry_computation()->root_instruction();
  ASSERT_EQ(root_instr->opcode(), HloOpcode::kRaggedAllToAll);
  const HloRaggedAllToAllInstruction* ra2a_instr =
      absl::down_cast<const HloRaggedAllToAllInstruction*>(root_instr);
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
  auto ra2a_start_thunk = std::make_unique<RaggedAllToAllThunk>(
      Thunk::ThunkInfo{}, ra2a_instr, std::move(buffers),
      /*p2p_memcpy_enabled=*/false);

  ThunkSequence start_sequence;
  start_sequence.push_back(std::move(ra2a_start_thunk));
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
  ASSERT_OK_AND_ASSIGN(CommandExecutor cb_cmd_executor,
                       ConvertToCommands(thunk_sequence, conv_options));

  // AsyncStart inlines its nested thunk as a command, and AsyncDone
  // with no control predecessors is a no-op, so we get 1 command.
  EXPECT_EQ(cb_cmd_executor.size(), 1);
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

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(hlo_text, config));

  se::StreamExecutor* executor = backend().default_stream_executor();
  // TODO(Intel-tf): To remove this check once command buffer is implemented.
  if (executor->GetPlatform()->id() == se::sycl::kSyclPlatformId) {
    GTEST_SKIP() << "Command buffer is not supported on SYCL platform yet.";
  }
  // CHECK_NE(executor, nullptr);

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> compiled_module,
      backend().compiler()->RunHloPasses(module->Clone(), executor,
                                         /*device_allocator=*/nullptr));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Executable> executable,
      backend().compiler()->RunBackend(std::move(compiled_module), executor,
                                       /*device_allocator=*/nullptr));

  // Downcast to GPU executable
  xla::gpu::GpuExecutable* gpu_executable =
      absl::down_cast<GpuExecutable*>(executable.get());
  ASSERT_NE(gpu_executable, nullptr);

  // Get the thunk sequence and check its size and type
  const ThunkExecutor& seq_thunk = gpu_executable->thunk_executor();
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

  // The collective is sync (single device), so no AsyncStart/Done wrapping.
  EXPECT_THAT(kinds, ElementsAre(Kind::kCustomKernel, Kind::kCustomKernel,
                                 Kind::kCustomKernel, Kind::kCustomKernel,
                                 Kind::kRaggedAllToAll));
}

TEST(CollectiveThunkTest, ProtoRoundTrip) {
  ThunkProto proto = tsl::proto_testing::ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info { profile_annotation: "partition_id_profile_annotation" }
        ragged_all_to_all_thunk {
          collective_config {}
          num_total_updates: 10
          num_input_rows: 2
          num_row_elements: 5
          one_shot_kernel_enabled: true
        }
      )pb");

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = proto.thunk_info().profile_annotation();

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0)};

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<RaggedAllToAllThunk> thunk,
      RaggedAllToAllThunk::FromProto(
          thunk_info, proto.ragged_all_to_all_thunk(), buffer_allocations));

  // We're not setting the fast interconnect slice size override in the
  // proto, so it should be nullopt in the thunk.
  EXPECT_EQ(
      thunk->ragged_all_to_all_config().fast_interconnect_slice_size_override,
      std::nullopt);

  ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());

  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

}  // namespace
}  // namespace xla::gpu
