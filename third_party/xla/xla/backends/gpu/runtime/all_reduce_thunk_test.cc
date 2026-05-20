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

#include "xla/backends/gpu/runtime/all_reduce_thunk.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/stream_executor/trace_command_buffer_factory.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::tsl::proto_testing::EqualsProto;

static se::StreamExecutor* GpuExecutor() {
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  auto* platform = se::PlatformManager::PlatformWithName(name).value();
  return platform->ExecutorForDevice(0).value();
}

// Child command nodes (CreateChildCommand / UpdateChildCommand) require
// CUDA 12.9+ driver and toolkit.
static bool IsAtLeastCuda12900(const se::StreamExecutor* executor) {
  const auto& desc = executor->GetDeviceDescription();
  const auto* cuda_cc = desc.gpu_compute_capability().cuda_compute_capability();
  if (cuda_cc == nullptr) {
    return false;
  }
  return std::min(desc.driver_version(), desc.compile_time_toolkit_version()) >=
         se::SemanticVersion(12, 9, 0);
}

// Test-only subclass whose ExecuteOnStream and Record both bypass NCCL so the
// command-buffer wiring can be exercised without a live communicator. Record
// traces a trivial memset into a nested command buffer and attaches it as a
// child command, mirroring the structure produced by the production Record.
static absl::StatusOr<const se::CommandBuffer::Command*> RecordNoOpCollective(
    const CollectiveThunk& thunk, const Thunk::ExecuteParams& execute_params,
    const Command::RecordParams&, Command::RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase dst =
      execute_params.buffer_allocations->GetDeviceAddress(
          thunk.buffers()[0].destination_buffer.slice);
  ASSIGN_OR_RETURN(
      std::unique_ptr<se::CommandBuffer> nested_cmd,
      se::TraceCommandBufferFactory::Create(
          execute_params.stream->parent(),
          execute_params.command_buffer_trace_stream,
          [&](se::Stream* stream) { return stream->MemZero(&dst, 4); }));

  if (auto* create = std::get_if<Command::RecordCreate>(&record_action)) {
    return command_buffer->CreateChildCommand(*nested_cmd,
                                              create->dependencies);
  }
  if (auto* update = std::get_if<Command::RecordUpdate>(&record_action)) {
    RETURN_IF_ERROR(
        command_buffer->UpdateChildCommand(update->command, *nested_cmd));
    return update->command;
  }
  return absl::InternalError("Invalid record action");
}

class NoOpAllReduceThunk : public AllReduceThunk {
 public:
  NoOpAllReduceThunk(Thunk::ThunkInfo thunk_info, AllReduceConfig config,
                     std::vector<CollectiveThunk::Buffer> buffers)
      : AllReduceThunk(std::move(thunk_info), std::move(config),
                       std::move(buffers)) {}

  absl::Status ExecuteOnStream(const ExecuteParams& params) override {
    return absl::OkStatus();
  }

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const ExecuteParams& execute_params, const RecordParams& record_params,
      RecordAction record_action, se::CommandBuffer* command_buffer) override {
    return RecordNoOpCollective(*this, execute_params, record_params,
                                std::move(record_action), command_buffer);
  }
};

class NoOpReduceScatterThunk : public ReduceScatterThunk {
 public:
  NoOpReduceScatterThunk(Thunk::ThunkInfo thunk_info, AllReduceConfig config,
                         std::vector<CollectiveThunk::Buffer> buffers)
      : ReduceScatterThunk(std::move(thunk_info), std::move(config),
                           std::move(buffers)) {}

  absl::Status ExecuteOnStream(const ExecuteParams& params) override {
    return absl::OkStatus();
  }

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const ExecuteParams& execute_params, const RecordParams& record_params,
      RecordAction record_action, se::CommandBuffer* command_buffer) override {
    return RecordNoOpCollective(*this, execute_params, record_params,
                                std::move(record_action), command_buffer);
  }
};

//===----------------------------------------------------------------------===//
// Proto round-trip tests
//===----------------------------------------------------------------------===//

TEST(CollectiveThunkTest, ProtoRoundTrip) {
  ThunkProto proto = tsl::proto_testing::ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info { profile_annotation: "partition_id_profile_annotation" }
        all_reduce_thunk {
          collective_config {}
          reduction_kind: 1
        }
      )pb");

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = proto.thunk_info().profile_annotation();

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0)};

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<AllReduceThunk> thunk,
      AllReduceThunk::FromProto(thunk_info, proto.all_reduce_thunk(),
                                buffer_allocations));

  ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());

  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(CollectiveThunkTest, SyncCollective) {
  ThunkProto proto = tsl::proto_testing::ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info { profile_annotation: "partition_id_profile_annotation" }
        all_reduce_thunk {
          collective_config {}
          reduction_kind: 1
        }
      )pb");

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = proto.thunk_info().profile_annotation();

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0)};

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<AllReduceThunk> thunk,
      AllReduceThunk::FromProto(thunk_info, proto.all_reduce_thunk(),
                                buffer_allocations));
}

TEST(ReduceScatterThunkTest, ProtoRoundTrip) {
  ThunkProto proto = tsl::proto_testing::ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info { profile_annotation: "partition_id_profile_annotation" }
        reduce_scatter_thunk {
          collective_config {}
          reduction_kind: 1
        }
      )pb");

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = proto.thunk_info().profile_annotation();

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0)};

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ReduceScatterThunk> thunk,
      ReduceScatterThunk::FromProto(thunk_info, proto.reduce_scatter_thunk(),
                                    buffer_allocations));

  ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());

  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

//===----------------------------------------------------------------------===//
// Command buffer tests (Record)
//===----------------------------------------------------------------------===//

// Builds a NoOpAllReduceThunk with one F32[length] src->dst buffer pair.
static CollectiveThunk::Buffer MakeNoOpBuffer(const BufferAllocation& alloc_src,
                                              const BufferAllocation& alloc_dst,
                                              int64_t length) {
  int64_t byte_length = sizeof(float) * length;
  ShapedSlice src_slice{BufferAllocation::Slice(&alloc_src, 0, byte_length),
                        ShapeUtil::MakeShape(F32, {length})};
  ShapedSlice dst_slice{BufferAllocation::Slice(&alloc_dst, 0, byte_length),
                        ShapeUtil::MakeShape(F32, {length})};
  CollectiveThunk::Buffer buffer{.element_count = length,
                                 .source_buffer = src_slice,
                                 .destination_buffer = dst_slice,
                                 .source_memory_space = 0,
                                 .destination_memory_space = 0};
  return buffer;
}

static AllReduceConfig MakeSumConfig() {
  AllReduceConfig config;
  config.config.operand_element_type = {F32};
  config.reduction_kind = ReductionKind::SUM;
  return config;
}

static NoOpAllReduceThunk MakeNoOpThunk(const BufferAllocation& alloc_src,
                                        const BufferAllocation& alloc_dst,
                                        int64_t length) {
  return NoOpAllReduceThunk(Thunk::ThunkInfo(), MakeSumConfig(),
                            {MakeNoOpBuffer(alloc_src, alloc_dst, length)});
}

static NoOpReduceScatterThunk MakeNoOpReduceScatterThunk(
    const BufferAllocation& alloc_src, const BufferAllocation& alloc_dst,
    int64_t length) {
  return NoOpReduceScatterThunk(Thunk::ThunkInfo(), MakeSumConfig(),
                                {MakeNoOpBuffer(alloc_src, alloc_dst, length)});
}

// Records AllReduceThunk into a primary command buffer (create phase) and
// verifies that a non-null command node is returned.
TEST(AllReduceThunkTest, RecordCommandBufferCreate) {
  se::StreamExecutor* executor = GpuExecutor();
  if (!IsAtLeastCuda12900(executor)) {
    GTEST_SKIP() << "Child command nodes require CUDA 12.9+";
  }

  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(float) * length;

  se::DeviceAddress<float> src = executor->AllocateArray<float>(length, 0);
  se::DeviceAddress<float> dst = executor->AllocateArray<float>(length, 0);

  BufferAllocation alloc_src(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation alloc_dst(/*index=*/1, byte_length, /*color=*/0);

  NoOpAllReduceThunk thunk = MakeNoOpThunk(alloc_src, alloc_dst, length);

  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations allocations({src, dst}, 0, &allocator);

  ServiceExecutableRunOptions run_options;
  Thunk::ExecuteParams execute_params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   /*command_buffer_trace_stream=*/stream.get(),
                                   /*collective_params=*/nullptr,
                                   /*collective_cliques=*/nullptr,
                                   /*collective_memory=*/nullptr);

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  ASSERT_OK_AND_ASSIGN(const se::CommandBuffer::Command* cmd,
                       thunk.Record(execute_params, record_params,
                                    Command::RecordCreate{/*dependencies=*/{}},
                                    command_buffer.get()));
  EXPECT_NE(cmd, nullptr);

  ASSERT_OK(command_buffer->Finalize());
  ASSERT_OK(command_buffer->Submit(stream.get()));
  ASSERT_OK(stream->BlockHostUntilDone());
}

// Records AllReduceThunk twice into the same command buffer: first as a create,
// then as an update with different buffer allocations. Verifies that the same
// command node pointer is returned on update.
TEST(AllReduceThunkTest, RecordCommandBufferUpdate) {
  se::StreamExecutor* executor = GpuExecutor();
  if (!IsAtLeastCuda12900(executor)) {
    GTEST_SKIP() << "Child command nodes require CUDA 12.9+";
  }

  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(float) * length;

  // First set of device buffers (used in the create phase).
  se::DeviceAddress<float> src1 = executor->AllocateArray<float>(length, 0);
  se::DeviceAddress<float> dst1 = executor->AllocateArray<float>(length, 0);

  // Second set of device buffers (used in the update phase).
  se::DeviceAddress<float> src2 = executor->AllocateArray<float>(length, 0);
  se::DeviceAddress<float> dst2 = executor->AllocateArray<float>(length, 0);

  BufferAllocation alloc_src(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation alloc_dst(/*index=*/1, byte_length, /*color=*/0);

  NoOpAllReduceThunk thunk = MakeNoOpThunk(alloc_src, alloc_dst, length);

  se::StreamExecutorAddressAllocator allocator(executor);
  ServiceExecutableRunOptions run_options;

  // Create phase: record into a fresh primary command buffer.
  BufferAllocations allocations1({src1, dst1}, 0, &allocator);
  Thunk::ExecuteParams params1 =
      Thunk::ExecuteParams::Create(run_options, allocations1, stream.get(),
                                   /*command_buffer_trace_stream=*/stream.get(),
                                   /*collective_params=*/nullptr,
                                   /*collective_cliques=*/nullptr,
                                   /*collective_memory=*/nullptr);

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  ASSERT_OK_AND_ASSIGN(const se::CommandBuffer::Command* cmd,
                       thunk.Record(params1, record_params,
                                    Command::RecordCreate{/*dependencies=*/{}},
                                    command_buffer.get()));
  ASSERT_NE(cmd, nullptr);

  ASSERT_OK(command_buffer->Finalize());
  ASSERT_OK(command_buffer->Submit(stream.get()));
  ASSERT_OK(stream->BlockHostUntilDone());

  // Update phase: transition the command buffer to update state and re-record
  // with new buffer allocations. The same command node must be reused.
  BufferAllocations allocations2({src2, dst2}, 0, &allocator);
  Thunk::ExecuteParams params2 =
      Thunk::ExecuteParams::Create(run_options, allocations2, stream.get(),
                                   /*command_buffer_trace_stream=*/stream.get(),
                                   /*collective_params=*/nullptr,
                                   /*collective_cliques=*/nullptr,
                                   /*collective_memory=*/nullptr);

  std::vector<BufferAllocation::Index> updated_allocs = {0, 1};
  Command::RecordParams record_params2 = {state, std::move(updated_allocs)};

  ASSERT_OK(command_buffer->Update());
  ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* updated_cmd,
      thunk.Record(params2, record_params2, Command::RecordUpdate{cmd},
                   command_buffer.get()));
  EXPECT_EQ(updated_cmd, cmd);

  ASSERT_OK(command_buffer->Finalize());
  ASSERT_OK(command_buffer->Submit(stream.get()));
  ASSERT_OK(stream->BlockHostUntilDone());
}

// Records ReduceScatterThunk into a primary command buffer (create phase) and
// verifies that a non-null command node is returned.
TEST(ReduceScatterThunkTest, RecordCommandBufferCreate) {
  se::StreamExecutor* executor = GpuExecutor();
  if (!IsAtLeastCuda12900(executor)) {
    GTEST_SKIP() << "Child command nodes require CUDA 12.9+";
  }

  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(float) * length;

  se::DeviceAddress<float> src = executor->AllocateArray<float>(length, 0);
  se::DeviceAddress<float> dst = executor->AllocateArray<float>(length, 0);

  BufferAllocation alloc_src(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation alloc_dst(/*index=*/1, byte_length, /*color=*/0);

  NoOpReduceScatterThunk thunk =
      MakeNoOpReduceScatterThunk(alloc_src, alloc_dst, length);

  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations allocations({src, dst}, 0, &allocator);

  ServiceExecutableRunOptions run_options;
  Thunk::ExecuteParams execute_params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   /*command_buffer_trace_stream=*/stream.get(),
                                   /*collective_params=*/nullptr,
                                   /*collective_cliques=*/nullptr,
                                   /*collective_memory=*/nullptr);

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  ASSERT_OK_AND_ASSIGN(const se::CommandBuffer::Command* cmd,
                       thunk.Record(execute_params, record_params,
                                    Command::RecordCreate{/*dependencies=*/{}},
                                    command_buffer.get()));
  EXPECT_NE(cmd, nullptr);

  ASSERT_OK(command_buffer->Finalize());
  ASSERT_OK(command_buffer->Submit(stream.get()));
  ASSERT_OK(stream->BlockHostUntilDone());
}

// Records ReduceScatterThunk twice into the same command buffer: first as a
// create, then as an update with different buffer allocations. Verifies that
// the same command node pointer is returned on update.
TEST(ReduceScatterThunkTest, RecordCommandBufferUpdate) {
  se::StreamExecutor* executor = GpuExecutor();
  if (!IsAtLeastCuda12900(executor)) {
    GTEST_SKIP() << "Child command nodes require CUDA 12.9+";
  }

  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(float) * length;

  se::DeviceAddress<float> src1 = executor->AllocateArray<float>(length, 0);
  se::DeviceAddress<float> dst1 = executor->AllocateArray<float>(length, 0);

  se::DeviceAddress<float> src2 = executor->AllocateArray<float>(length, 0);
  se::DeviceAddress<float> dst2 = executor->AllocateArray<float>(length, 0);

  BufferAllocation alloc_src(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation alloc_dst(/*index=*/1, byte_length, /*color=*/0);

  NoOpReduceScatterThunk thunk =
      MakeNoOpReduceScatterThunk(alloc_src, alloc_dst, length);

  se::StreamExecutorAddressAllocator allocator(executor);
  ServiceExecutableRunOptions run_options;

  BufferAllocations allocations1({src1, dst1}, 0, &allocator);
  Thunk::ExecuteParams params1 =
      Thunk::ExecuteParams::Create(run_options, allocations1, stream.get(),
                                   /*command_buffer_trace_stream=*/stream.get(),
                                   /*collective_params=*/nullptr,
                                   /*collective_cliques=*/nullptr,
                                   /*collective_memory=*/nullptr);

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  ASSERT_OK_AND_ASSIGN(const se::CommandBuffer::Command* cmd,
                       thunk.Record(params1, record_params,
                                    Command::RecordCreate{/*dependencies=*/{}},
                                    command_buffer.get()));
  ASSERT_NE(cmd, nullptr);

  ASSERT_OK(command_buffer->Finalize());
  ASSERT_OK(command_buffer->Submit(stream.get()));
  ASSERT_OK(stream->BlockHostUntilDone());

  BufferAllocations allocations2({src2, dst2}, 0, &allocator);
  Thunk::ExecuteParams params2 =
      Thunk::ExecuteParams::Create(run_options, allocations2, stream.get(),
                                   /*command_buffer_trace_stream=*/stream.get(),
                                   /*collective_params=*/nullptr,
                                   /*collective_cliques=*/nullptr,
                                   /*collective_memory=*/nullptr);

  std::vector<BufferAllocation::Index> updated_allocs = {0, 1};
  Command::RecordParams record_params2 = {state, std::move(updated_allocs)};

  ASSERT_OK(command_buffer->Update());
  ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* updated_cmd,
      thunk.Record(params2, record_params2, Command::RecordUpdate{cmd},
                   command_buffer.get()));
  EXPECT_EQ(updated_cmd, cmd);

  ASSERT_OK(command_buffer->Finalize());
  ASSERT_OK(command_buffer->Submit(stream.get()));
  ASSERT_OK(stream->BlockHostUntilDone());
}

}  // namespace
}  // namespace xla::gpu
