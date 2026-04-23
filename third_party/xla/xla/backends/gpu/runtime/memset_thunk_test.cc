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

#include "xla/backends/gpu/runtime/memset_thunk.h"

#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/ascii.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
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
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {

using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;

static se::StreamExecutor* GpuExecutor() {
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  auto* platform = se::PlatformManager::PlatformWithName(name).value();
  return platform->ExecutorForDevice(0).value();
}

//===----------------------------------------------------------------------===//
// Proto round-trip tests
//===----------------------------------------------------------------------===//

TEST(MemzeroThunkTest, ProtoRoundTrip) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info { profile_annotation: "partition_id_profile_annotation" }
        memzero_thunk {
          dest_buffer {
            slice { offset: 0 size: 4 buffer_allocation_index: 0 }
            shape {
              dimensions: 1
              element_type: F32
              is_dynamic_dimension: false
            }
          }
        }
      )pb");
  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0)};

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = proto.thunk_info().profile_annotation();
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MemzeroThunk> thunk,
      MemzeroThunk::FromProto(thunk_info, proto.memzero_thunk(),
                              buffer_allocations));

  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(Memset32BitValueThunkTest, ProtoRoundTrip) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info { profile_annotation: "partition_id_profile_annotation" }
        memset32bit_value_thunk {
          dest_buffer { offset: 0 size: 4 buffer_allocation_index: 0 }
          value: 123
        }
      )pb");
  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0)};

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = proto.thunk_info().profile_annotation();
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Memset32BitValueThunk> thunk,
      Memset32BitValueThunk::FromProto(
          thunk_info, proto.memset32bit_value_thunk(), buffer_allocations));

  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

//===----------------------------------------------------------------------===//
// Command buffer tests (Record)
//===----------------------------------------------------------------------===//

TEST(MemzeroThunkTest, RecordCommandBuffer) {
  se::StreamExecutor* executor = GpuExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(uint32_t) * length;

  se::DeviceAddress<uint32_t> dest =
      executor->AllocateArray<uint32_t>(length, 0);
  TF_ASSERT_OK(stream->Memset32(&dest, 0xFFFFFFFF, byte_length));

  BufferAllocation alloc(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation::Slice slice(&alloc, 0, byte_length);
  ShapedSlice dest_slice{slice, ShapeUtil::MakeShape(F32, {length})};

  MemzeroThunk thunk(Thunk::ThunkInfo(), dest_slice);

  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations buffer_allocations({dest}, 0, &allocator);

  ServiceExecutableRunOptions run_options;
  Thunk::ExecuteParams execute_params = Thunk::ExecuteParams::Create(
      run_options, buffer_allocations, stream.get(),
      /*command_buffer_trace_stream=*/nullptr,
      /*collective_params=*/nullptr,
      /*collective_cliques=*/nullptr,
      /*collective_memory=*/nullptr);

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  TF_ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  TF_ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* cmd,
      thunk.Record(execute_params, record_params,
                   Command::RecordCreate{/*dependencies=*/{}},
                   command_buffer.get()));
  ASSERT_NE(cmd, nullptr);
  TF_ASSERT_OK(command_buffer->Finalize());
  TF_ASSERT_OK(command_buffer->Submit(stream.get()));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  std::vector<uint32_t> result(length, 0xFFFFFFFF);
  TF_ASSERT_OK(stream->Memcpy(result.data(), dest, byte_length));
  EXPECT_EQ(result, std::vector<uint32_t>(length, 0));
}

// Records into a command buffer, submits, then updates the buffer allocation
// and re-submits to verify the same command node is reused.
TEST(MemzeroThunkTest, RecordCommandBufferUpdate) {
  se::StreamExecutor* executor = GpuExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(uint32_t) * length;

  se::DeviceAddress<uint32_t> dest_first =
      executor->AllocateArray<uint32_t>(length, 0);
  se::DeviceAddress<uint32_t> dest_second =
      executor->AllocateArray<uint32_t>(length, 0);
  TF_ASSERT_OK(stream->Memset32(&dest_first, 0xFFFFFFFF, byte_length));
  TF_ASSERT_OK(stream->Memset32(&dest_second, 0xFFFFFFFF, byte_length));

  BufferAllocation alloc(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation::Slice slice(&alloc, 0, byte_length);
  ShapedSlice dest_slice{slice, ShapeUtil::MakeShape(F32, {length})};

  MemzeroThunk thunk(Thunk::ThunkInfo(), dest_slice);

  se::StreamExecutorAddressAllocator allocator(executor);

  // First recording: RecordCreate pointing at dest_first.
  BufferAllocations allocs_first({dest_first}, 0, &allocator);
  ServiceExecutableRunOptions run_options;
  Thunk::ExecuteParams params_first =
      Thunk::ExecuteParams::Create(run_options, allocs_first, stream.get(),
                                   /*command_buffer_trace_stream=*/nullptr,
                                   /*collective_params=*/nullptr,
                                   /*collective_cliques=*/nullptr,
                                   /*collective_memory=*/nullptr);

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  TF_ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  TF_ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* cmd,
      thunk.Record(params_first, record_params,
                   Command::RecordCreate{/*dependencies=*/{}},
                   command_buffer.get()));
  ASSERT_NE(cmd, nullptr);
  TF_ASSERT_OK(command_buffer->Finalize());
  TF_ASSERT_OK(command_buffer->Submit(stream.get()));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Verify dest_first was zeroed.
  std::vector<uint32_t> result_first(length, 0xFFFFFFFF);
  TF_ASSERT_OK(stream->Memcpy(result_first.data(), dest_first, byte_length));
  EXPECT_EQ(result_first, std::vector<uint32_t>(length, 0));

  // Second recording: RecordUpdate pointing at dest_second.
  BufferAllocations allocs_second({dest_second}, 0, &allocator);
  Thunk::ExecuteParams params_second =
      Thunk::ExecuteParams::Create(run_options, allocs_second, stream.get(),
                                   /*command_buffer_trace_stream=*/nullptr,
                                   /*collective_params=*/nullptr,
                                   /*collective_cliques=*/nullptr,
                                   /*collective_memory=*/nullptr);

  TF_ASSERT_OK(command_buffer->Update());
  TF_ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* updated_cmd,
      thunk.Record(params_second, record_params, Command::RecordUpdate{cmd},
                   command_buffer.get()));
  EXPECT_EQ(updated_cmd, cmd);  // same command node is reused
  TF_ASSERT_OK(command_buffer->Finalize());
  TF_ASSERT_OK(command_buffer->Submit(stream.get()));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Verify dest_second was zeroed.
  std::vector<uint32_t> result_second(length, 0xFFFFFFFF);
  TF_ASSERT_OK(stream->Memcpy(result_second.data(), dest_second, byte_length));
  EXPECT_EQ(result_second, std::vector<uint32_t>(length, 0));
}

TEST(Memset32BitValueThunkTest, RecordCommandBuffer) {
  se::StreamExecutor* executor = GpuExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(uint32_t) * length;

  se::DeviceAddress<uint32_t> dest =
      executor->AllocateArray<uint32_t>(length, 0);
  TF_ASSERT_OK(stream->MemZero(&dest, byte_length));

  BufferAllocation alloc(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation::Slice slice(&alloc, 0, byte_length);

  Memset32BitValueThunk thunk(Thunk::ThunkInfo(), /*value=*/42, slice);

  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations buffer_allocations({dest}, 0, &allocator);

  ServiceExecutableRunOptions run_options;
  Thunk::ExecuteParams execute_params = Thunk::ExecuteParams::Create(
      run_options, buffer_allocations, stream.get(),
      /*command_buffer_trace_stream=*/nullptr,
      /*collective_params=*/nullptr,
      /*collective_cliques=*/nullptr,
      /*collective_memory=*/nullptr);

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  TF_ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  TF_ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* cmd,
      thunk.Record(execute_params, record_params,
                   Command::RecordCreate{/*dependencies=*/{}},
                   command_buffer.get()));
  ASSERT_NE(cmd, nullptr);
  TF_ASSERT_OK(command_buffer->Finalize());
  TF_ASSERT_OK(command_buffer->Submit(stream.get()));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  std::vector<uint32_t> result(length, 0);
  TF_ASSERT_OK(stream->Memcpy(result.data(), dest, byte_length));
  EXPECT_EQ(result, std::vector<uint32_t>(length, 42));
}

// Records into a command buffer, submits, then updates the buffer allocation
// and re-submits to verify the same command node is reused.
TEST(Memset32BitValueThunkTest, RecordCommandBufferUpdate) {
  se::StreamExecutor* executor = GpuExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(uint32_t) * length;

  se::DeviceAddress<uint32_t> dest_first =
      executor->AllocateArray<uint32_t>(length, 0);
  se::DeviceAddress<uint32_t> dest_second =
      executor->AllocateArray<uint32_t>(length, 0);
  TF_ASSERT_OK(stream->MemZero(&dest_first, byte_length));
  TF_ASSERT_OK(stream->MemZero(&dest_second, byte_length));

  BufferAllocation alloc(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation::Slice slice(&alloc, 0, byte_length);

  Memset32BitValueThunk thunk(Thunk::ThunkInfo(), /*value=*/42, slice);

  se::StreamExecutorAddressAllocator allocator(executor);

  // First recording: RecordCreate pointing at dest_first.
  BufferAllocations allocs_first({dest_first}, 0, &allocator);
  ServiceExecutableRunOptions run_options;
  Thunk::ExecuteParams params_first =
      Thunk::ExecuteParams::Create(run_options, allocs_first, stream.get(),
                                   /*command_buffer_trace_stream=*/nullptr,
                                   /*collective_params=*/nullptr,
                                   /*collective_cliques=*/nullptr,
                                   /*collective_memory=*/nullptr);

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  TF_ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  TF_ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* cmd,
      thunk.Record(params_first, record_params,
                   Command::RecordCreate{/*dependencies=*/{}},
                   command_buffer.get()));
  ASSERT_NE(cmd, nullptr);
  TF_ASSERT_OK(command_buffer->Finalize());
  TF_ASSERT_OK(command_buffer->Submit(stream.get()));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Verify dest_first was set.
  std::vector<uint32_t> result_first(length, 0);
  TF_ASSERT_OK(stream->Memcpy(result_first.data(), dest_first, byte_length));
  EXPECT_EQ(result_first, std::vector<uint32_t>(length, 42));

  // Second recording: RecordUpdate pointing at dest_second.
  BufferAllocations allocs_second({dest_second}, 0, &allocator);
  Thunk::ExecuteParams params_second =
      Thunk::ExecuteParams::Create(run_options, allocs_second, stream.get(),
                                   /*command_buffer_trace_stream=*/nullptr,
                                   /*collective_params=*/nullptr,
                                   /*collective_cliques=*/nullptr,
                                   /*collective_memory=*/nullptr);

  TF_ASSERT_OK(command_buffer->Update());
  TF_ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* updated_cmd,
      thunk.Record(params_second, record_params, Command::RecordUpdate{cmd},
                   command_buffer.get()));
  EXPECT_EQ(updated_cmd, cmd);  // same command node is reused
  TF_ASSERT_OK(command_buffer->Finalize());
  TF_ASSERT_OK(command_buffer->Submit(stream.get()));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Verify dest_second was set.
  std::vector<uint32_t> result_second(length, 0);
  TF_ASSERT_OK(stream->Memcpy(result_second.data(), dest_second, byte_length));
  EXPECT_EQ(result_second, std::vector<uint32_t>(length, 42));
}

}  // namespace
}  // namespace xla::gpu
