/* Copyright 2026 The OpenXLA Authors.

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

// Multi-GPU integration tests for SendThunk and RecvThunk.
// Requires two GPUs. Command-buffer tests also require CUDA 12.9+ for child
// command create/update support.

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "absl/status/status_macros.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk_multigpu_test_utils.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/p2p_thunk_common.h"
#include "xla/backends/gpu/runtime/recv_thunk.h"
#include "xla/backends/gpu/runtime/send_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

static constexpr int kNumDevices = 2;
static constexpr int64_t kLength = 4;
static constexpr int64_t kByteLength = sizeof(float) * kLength;

static P2PConfig MakeSendRecvConfig() {
  ReplicaGroup replica_group;
  for (int i = 0; i < kNumDevices; ++i) {
    replica_group.add_replica_ids(i);
  }

  P2PConfig config;
  config.config.operand_element_type = {F32};
  config.config.replica_groups = {replica_group};
  config.config.group_mode = COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA;
  config.id_to_source_target[0].target = 1;
  config.id_to_source_target[1].source = 0;
  return config;
}

static CollectiveThunk::Buffer MakeBuffer(const BufferAllocation& allocation) {
  ShapedSlice slice{BufferAllocation::Slice(&allocation, 0, kByteLength),
                    ShapeUtil::MakeShape(F32, {kLength})};
  return CollectiveThunk::Buffer{.element_count = kLength,
                                 .source_buffer = slice,
                                 .destination_buffer = slice,
                                 .source_memory_space = 0,
                                 .destination_memory_space = 0};
}

static SendThunk MakeSendThunk(const BufferAllocation& send_allocation) {
  return SendThunk(Thunk::ThunkInfo(), MakeSendRecvConfig(),
                   MakeBuffer(send_allocation), "send");
}

static RecvThunk MakeRecvThunk(const BufferAllocation& recv_allocation) {
  return RecvThunk(Thunk::ThunkInfo(), MakeSendRecvConfig(),
                   MakeBuffer(recv_allocation), "recv");
}

struct DeviceTestSlot : CollectiveThunkMultiGpuTestState {
  const se::CommandBuffer::Command* send_command = nullptr;
  const se::CommandBuffer::Command* recv_command = nullptr;
};

static std::vector<int64_t> DeviceBufferSizes() {
  return {kByteLength, kByteLength};
}

static std::vector<float> SourceValues(int device_ordinal, int phase) {
  std::vector<float> data(kLength);
  for (int i = 0; i < kLength; ++i) {
    data[i] = static_cast<float>(phase * 100 + device_ordinal * 10 + i);
  }
  return data;
}

static std::vector<float> ExpectedRecvValues(int device_ordinal, int phase) {
  if (device_ordinal == 1) {
    return SourceValues(/*device_ordinal=*/0, phase);
  }
  return std::vector<float>(kLength, 0.0f);
}

static absl::Status FillDeviceBufferWithValue(se::Stream& stream,
                                              se::DeviceAddressBase buffer,
                                              float value) {
  return FillDeviceBuffer(stream, buffer, std::vector<float>(kLength, value));
}

static absl::Status PreparePhaseInputs(
    DeviceTestSlot& slot, int device_ordinal, int phase,
    absl::Span<const se::DeviceAddressBase> buffers) {
  RETURN_IF_ERROR(FillDeviceBuffer(*slot.stream, buffers[0],
                                   SourceValues(device_ordinal, phase)));
  return FillDeviceBufferWithValue(*slot.stream, buffers[1], -1.0f);
}

static absl::Status VerifyRecvOutput(DeviceTestSlot& slot, int device_ordinal,
                                     int phase,
                                     se::DeviceAddressBase recv_dst) {
  ASSIGN_OR_RETURN(std::vector<float> output,
                   ReadDeviceBuffer(*slot.stream, recv_dst, kLength));
  std::vector<float> expected = ExpectedRecvValues(device_ordinal, phase);
  for (int i = 0; i < kLength; ++i) {
    if (output[i] != expected[i]) {
      return absl::InternalError(
          absl::StrFormat("device %d output[%d] = %g, expected %g",
                          device_ordinal, i, output[i], expected[i]));
    }
  }
  return absl::OkStatus();
}

static absl::Status SetupDeviceSlot(int device_ordinal, DeviceTestSlot& slot,
                                    SendThunk& send_thunk,
                                    RecvThunk& recv_thunk,
                                    const DeviceAssignment& device_assignment) {
  std::vector<int64_t> buffer_sizes = DeviceBufferSizes();
  return SetupCollectiveThunksDevice(device_ordinal, kNumDevices, buffer_sizes,
                                     {&send_thunk, &recv_thunk},
                                     device_assignment, slot);
}

static absl::Status ExecuteSendRecv(DeviceTestSlot& slot, SendThunk& send_thunk,
                                    RecvThunk& recv_thunk,
                                    BufferAllocations& allocations) {
  Thunk::ExecuteParams execute_params = MakeExecuteParams(slot, allocations);
  RETURN_IF_ERROR(send_thunk.ExecuteOnStream(execute_params));
  RETURN_IF_ERROR(recv_thunk.ExecuteOnStream(execute_params));
  return slot.stream->BlockHostUntilDone();
}

static absl::Status RunExecuteOnStreamPhase(int device_ordinal,
                                            DeviceTestSlot& slot,
                                            SendThunk& send_thunk,
                                            RecvThunk& recv_thunk) {
  constexpr int kPhase = 1;
  RETURN_IF_ERROR(
      PreparePhaseInputs(slot, device_ordinal, kPhase, slot.create_buffers));
  BufferAllocations allocations =
      MakeBufferAllocations(slot, slot.create_buffers);
  RETURN_IF_ERROR(ExecuteSendRecv(slot, send_thunk, recv_thunk, allocations));
  return VerifyRecvOutput(slot, device_ordinal, kPhase, slot.create_buffers[1]);
}

static absl::Status RunCreatePhase(int device_ordinal, DeviceTestSlot& slot,
                                   SendThunk& send_thunk,
                                   RecvThunk& recv_thunk) {
  constexpr int kPhase = 1;
  RETURN_IF_ERROR(
      PreparePhaseInputs(slot, device_ordinal, kPhase, slot.create_buffers));

  BufferAllocations allocations =
      MakeBufferAllocations(slot, slot.create_buffers);
  RETURN_IF_ERROR(ExecuteSendRecv(slot, send_thunk, recv_thunk, allocations));
  RETURN_IF_ERROR(
      FillDeviceBufferWithValue(*slot.stream, slot.create_buffers[1], -1.0f));

  Thunk::ExecuteParams execute_params = MakeExecuteParams(slot, allocations);
  ASSIGN_OR_RETURN(slot.command_buffer, slot.executor->CreateCommandBuffer(
                                            se::CommandBuffer::Mode::kPrimary));

  Command::RecordParams record_params = {slot.state_manager};
  ASSIGN_OR_RETURN(slot.send_command,
                   send_thunk.Record(execute_params, record_params,
                                     Command::RecordCreate{/*dependencies=*/{}},
                                     slot.command_buffer.get()));

  std::vector<const se::CommandBuffer::Command*> dependencies;
  if (slot.send_command != nullptr) {
    dependencies.push_back(slot.send_command);
  }
  ASSIGN_OR_RETURN(slot.recv_command,
                   recv_thunk.Record(execute_params, record_params,
                                     Command::RecordCreate{dependencies},
                                     slot.command_buffer.get()));
  if (slot.recv_command == nullptr) {
    return absl::InternalError("RecvThunk returned null command node");
  }

  RETURN_IF_ERROR(slot.command_buffer->Finalize());
  RETURN_IF_ERROR(SubmitCommandBuffer(slot));
  return VerifyRecvOutput(slot, device_ordinal, kPhase, slot.create_buffers[1]);
}

static absl::Status RunUpdatePhase(int device_ordinal, DeviceTestSlot& slot,
                                   SendThunk& send_thunk,
                                   RecvThunk& recv_thunk) {
  constexpr int kPhase = 2;
  RETURN_IF_ERROR(
      PreparePhaseInputs(slot, device_ordinal, kPhase, slot.update_buffers));

  BufferAllocations allocations =
      MakeBufferAllocations(slot, slot.update_buffers);
  Thunk::ExecuteParams execute_params = MakeExecuteParams(slot, allocations);
  Command::RecordParams record_params = {
      slot.state_manager,
      /*updated_allocs=*/std::vector<BufferAllocation::Index>{0, 1}};

  RETURN_IF_ERROR(slot.command_buffer->Update());
  ASSIGN_OR_RETURN(const se::CommandBuffer::Command* updated_send_command,
                   send_thunk.Record(execute_params, record_params,
                                     Command::RecordUpdate{slot.send_command},
                                     slot.command_buffer.get()));
  if (updated_send_command != slot.send_command) {
    return absl::InternalError("SendThunk update returned a new command node");
  }

  ASSIGN_OR_RETURN(const se::CommandBuffer::Command* updated_recv_command,
                   recv_thunk.Record(execute_params, record_params,
                                     Command::RecordUpdate{slot.recv_command},
                                     slot.command_buffer.get()));
  if (updated_recv_command != slot.recv_command) {
    return absl::InternalError("RecvThunk update returned a new command node");
  }

  RETURN_IF_ERROR(slot.command_buffer->Finalize());
  RETURN_IF_ERROR(SubmitCommandBuffer(slot));
  return VerifyRecvOutput(slot, device_ordinal, kPhase, slot.update_buffers[1]);
}

TEST(SendRecvThunkMultiGpuTest, ExecuteOnStream) {
  if (!HasEnoughGpus(kNumDevices)) {
    GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
  }

  DeviceAssignment device_assignment = MakeDeviceAssignment(kNumDevices);
  BufferAllocation send_alloc(/*index=*/0, kByteLength, /*color=*/0);
  BufferAllocation recv_alloc(/*index=*/1, kByteLength, /*color=*/0);
  SendThunk send_thunk = MakeSendThunk(send_alloc);
  RecvThunk recv_thunk = MakeRecvThunk(recv_alloc);
  std::vector<DeviceTestSlot> slots(kNumDevices);

  ASSERT_OK(
      RunOnDevices(kNumDevices, "sendrecv_execute", [&](int d) -> absl::Status {
        RETURN_IF_ERROR(SetupDeviceSlot(d, slots[d], send_thunk, recv_thunk,
                                        device_assignment));
        return RunExecuteOnStreamPhase(d, slots[d], send_thunk, recv_thunk);
      }));
}

TEST(SendRecvThunkMultiGpuTest, RecordCommandBufferCreate) {
  if (!HasEnoughGpus(kNumDevices)) {
    GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
  }
  if (!IsAtLeastCuda12900(GetGpuExecutor(0))) {
    GTEST_SKIP() << "Child command nodes require CUDA 12.9+";
  }

  DeviceAssignment device_assignment = MakeDeviceAssignment(kNumDevices);
  BufferAllocation send_alloc(/*index=*/0, kByteLength, /*color=*/0);
  BufferAllocation recv_alloc(/*index=*/1, kByteLength, /*color=*/0);
  SendThunk send_thunk = MakeSendThunk(send_alloc);
  RecvThunk recv_thunk = MakeRecvThunk(recv_alloc);
  std::vector<DeviceTestSlot> slots(kNumDevices);

  ASSERT_OK(
      RunOnDevices(kNumDevices, "sendrecv_create", [&](int d) -> absl::Status {
        RETURN_IF_ERROR(SetupDeviceSlot(d, slots[d], send_thunk, recv_thunk,
                                        device_assignment));
        return RunCreatePhase(d, slots[d], send_thunk, recv_thunk);
      }));
}

TEST(SendRecvThunkMultiGpuTest, RecordCommandBufferUpdate) {
  if (!HasEnoughGpus(kNumDevices)) {
    GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
  }
  if (!IsAtLeastCuda12900(GetGpuExecutor(0))) {
    GTEST_SKIP() << "Child command nodes require CUDA 12.9+";
  }

  DeviceAssignment device_assignment = MakeDeviceAssignment(kNumDevices);
  BufferAllocation send_alloc(/*index=*/0, kByteLength, /*color=*/0);
  BufferAllocation recv_alloc(/*index=*/1, kByteLength, /*color=*/0);
  SendThunk send_thunk = MakeSendThunk(send_alloc);
  RecvThunk recv_thunk = MakeRecvThunk(recv_alloc);
  std::vector<DeviceTestSlot> slots(kNumDevices);

  ASSERT_OK(
      RunOnDevices(kNumDevices, "sendrecv_create", [&](int d) -> absl::Status {
        RETURN_IF_ERROR(SetupDeviceSlot(d, slots[d], send_thunk, recv_thunk,
                                        device_assignment));
        return RunCreatePhase(d, slots[d], send_thunk, recv_thunk);
      }));

  ASSERT_OK(RunOnDevices(kNumDevices, "sendrecv_update", [&](int d) {
    return RunUpdatePhase(d, slots[d], send_thunk, recv_thunk);
  }));
}

}  // namespace
}  // namespace xla::gpu
