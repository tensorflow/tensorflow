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

#include "xla/backends/gpu/runtime/replica_id_thunk.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/strings/ascii.h"
#include "google/protobuf/text_format.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {

using ::tsl::proto_testing::EqualsProto;

static se::StreamExecutor* GpuExecutor() {
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  auto* platform = se::PlatformManager::PlatformWithName(name).value();
  return platform->ExecutorForDevice(0).value();
}

// ===========================================================================
// Proto round-trip tests
// ===========================================================================

TEST(ReplicaIdThunkTest, ProtoRoundTrip) {
  ThunkProto proto;
  CHECK(tsl::protobuf::TextFormat::ParseFromString(
      R"pb(
        thunk_info { profile_annotation: "replica_id_profile_annotation" }
        replica_id_thunk {
          dest_buffer { offset: 0 size: 4 buffer_allocation_index: 0 }
        }
      )pb",
      &proto));
  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0)};

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = proto.thunk_info().profile_annotation();
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ReplicaIdThunk> thunk,
      ReplicaIdThunk::FromProto(thunk_info, proto.replica_id_thunk(),
                                buffer_allocations));

  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

TEST(PartitionIdThunkTest, ProtoRoundTrip) {
  ThunkProto proto;
  CHECK(tsl::protobuf::TextFormat::ParseFromString(
      R"pb(
        thunk_info { profile_annotation: "partition_id_profile_annotation" }
        partition_id_thunk {
          dest_buffer { offset: 0 size: 4 buffer_allocation_index: 0 }
        }
      )pb",
      &proto));
  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0)};

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = proto.thunk_info().profile_annotation();
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PartitionIdThunk> thunk,
      PartitionIdThunk::FromProto(thunk_info, proto.partition_id_thunk(),
                                  buffer_allocations));

  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());
  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

// ===========================================================================
// Normal execution tests (ExecuteOnStream)
// ===========================================================================

// Uses a 2-replica, 1-partition device assignment where device 0 is replica 1
// (replica 0 -> device 1, replica 1 -> device 0), so the expected value is 1.
TEST(ReplicaIdThunkTest, ExecuteOnStream) {
  se::StreamExecutor* executor = GpuExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  se::DeviceAddress<uint32_t> dest = executor->AllocateArray<uint32_t>(1, 0);
  TF_ASSERT_OK(stream->MemZero(&dest, sizeof(uint32_t)));

  BufferAllocation alloc(/*index=*/0, sizeof(uint32_t), /*color=*/0);
  BufferAllocation::Slice slice(&alloc, 0, sizeof(uint32_t));

  DeviceAssignment device_assignment(/*replica_count=*/2,
                                     /*computation_count=*/1);
  device_assignment(0, 0) = 1;  // replica 0 -> device 1
  device_assignment(1, 0) = 0;  // replica 1 -> device 0

  GpuExecutableRunOptions gpu_options;
  gpu_options.set_gpu_global_device_ids(GpuExecutableRunOptions::DeviceIdMap{
      std::make_pair(LocalDeviceId(0), GlobalDeviceId(0))});

  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  run_options.mutable_run_options()->set_device_assignment(&device_assignment);
  run_options.mutable_run_options()->set_gpu_executable_run_options(
      &gpu_options);

  TF_ASSERT_OK_AND_ASSIGN(
      CollectiveParams collective_params,
      CollectiveParams::Create(run_options, /*async_streams=*/{},
                               LocalDeviceId(executor->device_ordinal())));

  ReplicaIdThunk thunk(Thunk::ThunkInfo{}, slice);

  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations buffer_allocations({dest}, 0, &allocator);

  Thunk::ExecuteParams execute_params = Thunk::ExecuteParams::Create(
      run_options, buffer_allocations, stream.get(),
      /*command_buffer_trace_stream=*/nullptr, &collective_params,
      /*collective_cliques=*/nullptr, /*collective_memory=*/nullptr);

  TF_ASSERT_OK(thunk.ExecuteOnStream(execute_params));

  uint32_t result = 0;
  TF_ASSERT_OK(stream->Memcpy(&result, dest, sizeof(uint32_t)));
  EXPECT_EQ(result, 1u);  // device 0 is replica 1
}

// Uses a 1-replica, 2-partition device assignment where device 0 is partition 1
// ((r0,p0) -> device 1, (r0,p1) -> device 0), so the expected value is 1.
TEST(PartitionIdThunkTest, ExecuteOnStream) {
  se::StreamExecutor* executor = GpuExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  se::DeviceAddress<uint32_t> dest = executor->AllocateArray<uint32_t>(1, 0);
  TF_ASSERT_OK(stream->MemZero(&dest, sizeof(uint32_t)));

  BufferAllocation alloc(/*index=*/0, sizeof(uint32_t), /*color=*/0);
  BufferAllocation::Slice slice(&alloc, 0, sizeof(uint32_t));

  DeviceAssignment device_assignment(/*replica_count=*/1,
                                     /*computation_count=*/2);
  device_assignment(0, 0) = 1;  // (replica 0, partition 0) -> device 1
  device_assignment(0, 1) = 0;  // (replica 0, partition 1) -> device 0

  GpuExecutableRunOptions gpu_options;
  gpu_options.set_gpu_global_device_ids(GpuExecutableRunOptions::DeviceIdMap{
      std::make_pair(LocalDeviceId(0), GlobalDeviceId(0))});

  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  run_options.mutable_run_options()->set_device_assignment(&device_assignment);
  run_options.mutable_run_options()->set_gpu_executable_run_options(
      &gpu_options);

  TF_ASSERT_OK_AND_ASSIGN(
      CollectiveParams collective_params,
      CollectiveParams::Create(run_options, /*async_streams=*/{},
                               LocalDeviceId(executor->device_ordinal())));

  PartitionIdThunk thunk(Thunk::ThunkInfo{}, slice);

  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations buffer_allocations({dest}, 0, &allocator);

  Thunk::ExecuteParams execute_params = Thunk::ExecuteParams::Create(
      run_options, buffer_allocations, stream.get(),
      /*command_buffer_trace_stream=*/nullptr, &collective_params,
      /*collective_cliques=*/nullptr, /*collective_memory=*/nullptr);

  TF_ASSERT_OK(thunk.ExecuteOnStream(execute_params));

  uint32_t result = 0;
  TF_ASSERT_OK(stream->Memcpy(&result, dest, sizeof(uint32_t)));
  EXPECT_EQ(result, 1u);  // device 0 is partition 1
}

// ===========================================================================
// Command buffer tests (Record)
// ===========================================================================

TEST(ReplicaIdThunkTest, RecordCommandBuffer) {
  se::StreamExecutor* executor = GpuExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  se::DeviceAddress<uint32_t> dest = executor->AllocateArray<uint32_t>(1, 0);
  TF_ASSERT_OK(stream->MemZero(&dest, sizeof(uint32_t)));

  BufferAllocation alloc(/*index=*/0, sizeof(uint32_t), /*color=*/0);
  BufferAllocation::Slice slice(&alloc, 0, sizeof(uint32_t));

  DeviceAssignment device_assignment(/*replica_count=*/2,
                                     /*computation_count=*/1);
  device_assignment(0, 0) = 1;
  device_assignment(1, 0) = 0;

  GpuExecutableRunOptions gpu_options;
  gpu_options.set_gpu_global_device_ids(GpuExecutableRunOptions::DeviceIdMap{
      std::make_pair(LocalDeviceId(0), GlobalDeviceId(0))});

  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  run_options.mutable_run_options()->set_device_assignment(&device_assignment);
  run_options.mutable_run_options()->set_gpu_executable_run_options(
      &gpu_options);

  TF_ASSERT_OK_AND_ASSIGN(
      CollectiveParams collective_params,
      CollectiveParams::Create(run_options, /*async_streams=*/{},
                               LocalDeviceId(executor->device_ordinal())));

  ReplicaIdThunk thunk(Thunk::ThunkInfo{}, slice);

  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations buffer_allocations({dest}, 0, &allocator);

  Thunk::ExecuteParams execute_params = Thunk::ExecuteParams::Create(
      run_options, buffer_allocations, stream.get(),
      /*command_buffer_trace_stream=*/nullptr, &collective_params,
      /*collective_cliques=*/nullptr, /*collective_memory=*/nullptr);

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

  uint32_t result = 0;
  TF_ASSERT_OK(stream->Memcpy(&result, dest, sizeof(uint32_t)));
  EXPECT_EQ(result, 1u);  // device 0 is replica 1
}

// Records into a command buffer, submits, then updates and re-submits.
TEST(ReplicaIdThunkTest, RecordCommandBufferUpdate) {
  se::StreamExecutor* executor = GpuExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  se::DeviceAddress<uint32_t> dest = executor->AllocateArray<uint32_t>(1, 0);
  TF_ASSERT_OK(stream->MemZero(&dest, sizeof(uint32_t)));

  BufferAllocation alloc(/*index=*/0, sizeof(uint32_t), /*color=*/0);
  BufferAllocation::Slice slice(&alloc, 0, sizeof(uint32_t));

  DeviceAssignment device_assignment(/*replica_count=*/2,
                                     /*computation_count=*/1);
  device_assignment(0, 0) = 1;
  device_assignment(1, 0) = 0;

  GpuExecutableRunOptions gpu_options;
  gpu_options.set_gpu_global_device_ids(GpuExecutableRunOptions::DeviceIdMap{
      std::make_pair(LocalDeviceId(0), GlobalDeviceId(0))});

  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  run_options.mutable_run_options()->set_device_assignment(&device_assignment);
  run_options.mutable_run_options()->set_gpu_executable_run_options(
      &gpu_options);

  TF_ASSERT_OK_AND_ASSIGN(
      CollectiveParams collective_params,
      CollectiveParams::Create(run_options, /*async_streams=*/{},
                               LocalDeviceId(executor->device_ordinal())));

  ReplicaIdThunk thunk(Thunk::ThunkInfo{}, slice);

  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations buffer_allocations({dest}, 0, &allocator);

  Thunk::ExecuteParams execute_params = Thunk::ExecuteParams::Create(
      run_options, buffer_allocations, stream.get(),
      /*command_buffer_trace_stream=*/nullptr, &collective_params,
      /*collective_cliques=*/nullptr, /*collective_memory=*/nullptr);

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  // First recording: RecordCreate.
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

  // Transition to update state and re-record with RecordUpdate.
  TF_ASSERT_OK(command_buffer->Update());
  TF_ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* updated_cmd,
      thunk.Record(execute_params, record_params, Command::RecordUpdate{cmd},
                   command_buffer.get()));
  EXPECT_EQ(updated_cmd, cmd);  // same command node is reused
  TF_ASSERT_OK(command_buffer->Finalize());
  TF_ASSERT_OK(command_buffer->Submit(stream.get()));

  uint32_t result = 0;
  TF_ASSERT_OK(stream->Memcpy(&result, dest, sizeof(uint32_t)));
  EXPECT_EQ(result, 1u);  // device 0 is replica 1
}

TEST(PartitionIdThunkTest, RecordCommandBuffer) {
  se::StreamExecutor* executor = GpuExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  se::DeviceAddress<uint32_t> dest = executor->AllocateArray<uint32_t>(1, 0);
  TF_ASSERT_OK(stream->MemZero(&dest, sizeof(uint32_t)));

  BufferAllocation alloc(/*index=*/0, sizeof(uint32_t), /*color=*/0);
  BufferAllocation::Slice slice(&alloc, 0, sizeof(uint32_t));

  DeviceAssignment device_assignment(/*replica_count=*/1,
                                     /*computation_count=*/2);
  device_assignment(0, 0) = 1;
  device_assignment(0, 1) = 0;

  GpuExecutableRunOptions gpu_options;
  gpu_options.set_gpu_global_device_ids(GpuExecutableRunOptions::DeviceIdMap{
      std::make_pair(LocalDeviceId(0), GlobalDeviceId(0))});

  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  run_options.mutable_run_options()->set_device_assignment(&device_assignment);
  run_options.mutable_run_options()->set_gpu_executable_run_options(
      &gpu_options);

  TF_ASSERT_OK_AND_ASSIGN(
      CollectiveParams collective_params,
      CollectiveParams::Create(run_options, /*async_streams=*/{},
                               LocalDeviceId(executor->device_ordinal())));

  PartitionIdThunk thunk(Thunk::ThunkInfo{}, slice);

  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations buffer_allocations({dest}, 0, &allocator);

  Thunk::ExecuteParams execute_params = Thunk::ExecuteParams::Create(
      run_options, buffer_allocations, stream.get(),
      /*command_buffer_trace_stream=*/nullptr, &collective_params,
      /*collective_cliques=*/nullptr, /*collective_memory=*/nullptr);

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

  uint32_t result = 0;
  TF_ASSERT_OK(stream->Memcpy(&result, dest, sizeof(uint32_t)));
  EXPECT_EQ(result, 1u);  // device 0 is partition 1
}

}  // namespace
}  // namespace xla::gpu
