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

// Multi-GPU integration tests for AllToAllThunk through NCCL. Requires at
// least kNumDevices GPUs. Command-buffer tests additionally require CUDA 12.9+
// driver/toolkit for CreateChildCommand / UpdateChildCommand support.

#include <cstdint>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "absl/status/status_macros.h"
#include "xla/backends/gpu/runtime/all_to_all_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk_multigpu_test_utils.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

static constexpr int kNumDevices = 2;
static constexpr int64_t kLength = 4;
static constexpr int64_t kByteLength = sizeof(float) * kLength;
static constexpr int kNumBuffers = kNumDevices;

static AllToAllConfig MakeAllToAllConfig() {
  ReplicaGroup replica_group;
  for (int i = 0; i < kNumDevices; ++i) {
    replica_group.add_replica_ids(i);
  }

  AllToAllConfig config;
  config.config.operand_element_type =
      std::vector<PrimitiveType>(kNumBuffers, F32);
  config.config.replica_groups = {replica_group};
  config.config.group_mode = COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA;
  config.has_split_dimension = false;
  return config;
}

static std::vector<BufferAllocation> MakeThunkBufferAllocations() {
  std::vector<BufferAllocation> allocations;
  allocations.reserve(2 * kNumBuffers);
  for (int i = 0; i < 2 * kNumBuffers; ++i) {
    allocations.emplace_back(/*index=*/i, kByteLength, /*color=*/0);
  }
  return allocations;
}

static AllToAllThunk MakeThunk(absl::Span<const BufferAllocation> allocations) {
  std::vector<CollectiveThunk::Buffer> buffers;
  buffers.reserve(kNumBuffers);
  for (int i = 0; i < kNumBuffers; ++i) {
    ShapedSlice src_slice{
        BufferAllocation::Slice(&allocations[i], 0, kByteLength),
        ShapeUtil::MakeShape(F32, {kLength})};
    ShapedSlice dst_slice{
        BufferAllocation::Slice(&allocations[kNumBuffers + i], 0, kByteLength),
        ShapeUtil::MakeShape(F32, {kLength})};
    buffers.push_back(CollectiveThunk::Buffer{.element_count = kLength,
                                              .source_buffer = src_slice,
                                              .destination_buffer = dst_slice,
                                              .source_memory_space = 0,
                                              .destination_memory_space = 0});
  }

  return AllToAllThunk(Thunk::ThunkInfo(), MakeAllToAllConfig(),
                       std::move(buffers),
                       /*p2p_memcpy_enabled=*/false);
}

using DeviceTestSlot = CollectiveThunkMultiGpuTestState;

static std::vector<int64_t> DeviceBufferSizes() {
  return std::vector<int64_t>(2 * kNumBuffers, kByteLength);
}

static std::vector<BufferAllocation::Index> AllAllocationIndices() {
  std::vector<BufferAllocation::Index> indices;
  indices.reserve(2 * kNumBuffers);
  for (int i = 0; i < 2 * kNumBuffers; ++i) {
    indices.push_back(i);
  }
  return indices;
}

static absl::Span<const se::DeviceAddressBase> SourceBuffers(
    absl::Span<const se::DeviceAddressBase> buffers) {
  return buffers.subspan(0, kNumBuffers);
}

static absl::Span<const se::DeviceAddressBase> DestinationBuffers(
    absl::Span<const se::DeviceAddressBase> buffers) {
  return buffers.subspan(kNumBuffers, kNumBuffers);
}

static float SourceValue(int source_rank, int target_rank, int phase) {
  return static_cast<float>(phase * 100 + source_rank * 10 + target_rank);
}

static absl::Status FillDeviceBufferWithValue(se::Stream& stream,
                                              se::DeviceAddressBase buffer,
                                              float value) {
  std::vector<float> data(kLength, value);
  return FillDeviceBuffer(stream, buffer, data);
}

static absl::Status FillSourceBuffers(
    se::Stream& stream, absl::Span<const se::DeviceAddressBase> src,
    int device_ordinal, int phase) {
  for (int target_rank = 0; target_rank < kNumBuffers; ++target_rank) {
    RETURN_IF_ERROR(FillDeviceBufferWithValue(
        stream, src[target_rank],
        SourceValue(/*source_rank=*/device_ordinal, target_rank, phase)));
  }
  return absl::OkStatus();
}

static absl::Status FillDestinationBuffers(
    se::Stream& stream, absl::Span<const se::DeviceAddressBase> dst,
    float value) {
  for (se::DeviceAddressBase buffer : dst) {
    RETURN_IF_ERROR(FillDeviceBufferWithValue(stream, buffer, value));
  }
  return absl::OkStatus();
}

static absl::Status PrepareInputs(
    se::Stream& stream, absl::Span<const se::DeviceAddressBase> buffers,
    int device_ordinal, int phase) {
  RETURN_IF_ERROR(
      FillSourceBuffers(stream, SourceBuffers(buffers), device_ordinal, phase));
  return FillDestinationBuffers(stream, DestinationBuffers(buffers), -1.0f);
}

static absl::Status VerifyOutput(se::Stream& stream,
                                 absl::Span<const se::DeviceAddressBase> dst,
                                 int device_ordinal, int phase) {
  for (int source_rank = 0; source_rank < kNumBuffers; ++source_rank) {
    float expected =
        SourceValue(source_rank, /*target_rank=*/device_ordinal, phase);
    ASSIGN_OR_RETURN(std::vector<float> output,
                     ReadDeviceBuffer(stream, dst[source_rank], kLength));
    for (int i = 0; i < kLength; ++i) {
      if (output[i] != expected) {
        return absl::InternalError(absl::StrFormat(
            "dst[%d][%d] on device %d = %g, expected %g", source_rank, i,
            device_ordinal, output[i], expected));
      }
    }
  }
  return absl::OkStatus();
}

static absl::Status SetupDeviceSlot(int device_ordinal, DeviceTestSlot& slot,
                                    AllToAllThunk& thunk,
                                    const DeviceAssignment& device_assignment) {
  std::vector<int64_t> buffer_sizes = DeviceBufferSizes();
  return SetupCollectiveThunkDevice(device_ordinal, kNumDevices, buffer_sizes,
                                    thunk, device_assignment, slot);
}

static absl::Status RunExecuteOnStreamPhase(DeviceTestSlot& slot,
                                            AllToAllThunk& thunk,
                                            int device_ordinal, int phase) {
  RETURN_IF_ERROR(
      PrepareInputs(*slot.stream, slot.create_buffers, device_ordinal, phase));

  BufferAllocations allocations =
      MakeBufferAllocations(slot, slot.create_buffers);
  Thunk::ExecuteParams execute_params = MakeExecuteParams(slot, allocations);

  RETURN_IF_ERROR(ExecuteOnStreamAndBlock(thunk, execute_params));
  return VerifyOutput(*slot.stream, DestinationBuffers(slot.create_buffers),
                      device_ordinal, phase);
}

static absl::Status RunCreatePhase(DeviceTestSlot& slot, AllToAllThunk& thunk,
                                   int device_ordinal, int phase) {
  RETURN_IF_ERROR(
      PrepareInputs(*slot.stream, slot.create_buffers, device_ordinal, phase));

  BufferAllocations allocations =
      MakeBufferAllocations(slot, slot.create_buffers);
  Thunk::ExecuteParams execute_params = MakeExecuteParams(slot, allocations);

  // Warm up NCCL outside stream capture. Reset destination buffers afterward so
  // correctness is verified from command-buffer execution, not from warm-up.
  RETURN_IF_ERROR(ExecuteOnStreamAndBlock(thunk, execute_params));
  RETURN_IF_ERROR(FillDestinationBuffers(
      *slot.stream, DestinationBuffers(slot.create_buffers), -1.0f));

  RETURN_IF_ERROR(RecordCommandBufferCreate(slot, thunk, execute_params));
  RETURN_IF_ERROR(SubmitCommandBuffer(slot));
  return VerifyOutput(*slot.stream, DestinationBuffers(slot.create_buffers),
                      device_ordinal, phase);
}

static absl::Status RunUpdatePhase(DeviceTestSlot& slot, AllToAllThunk& thunk,
                                   int device_ordinal, int phase) {
  RETURN_IF_ERROR(
      PrepareInputs(*slot.stream, slot.update_buffers, device_ordinal, phase));

  BufferAllocations allocations =
      MakeBufferAllocations(slot, slot.update_buffers);
  Thunk::ExecuteParams execute_params = MakeExecuteParams(slot, allocations);

  RETURN_IF_ERROR(RecordCommandBufferUpdate(slot, thunk, execute_params,
                                            AllAllocationIndices()));
  RETURN_IF_ERROR(SubmitCommandBuffer(slot));
  return VerifyOutput(*slot.stream, DestinationBuffers(slot.update_buffers),
                      device_ordinal, phase);
}

TEST(AllToAllThunkMultiGpuTest, ExecuteOnStream) {
  if (!HasEnoughGpus(kNumDevices)) {
    GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
  }

  DeviceAssignment device_assignment = MakeDeviceAssignment(kNumDevices);
  std::vector<BufferAllocation> buffer_allocations =
      MakeThunkBufferAllocations();
  AllToAllThunk thunk = MakeThunk(buffer_allocations);
  std::vector<DeviceTestSlot> slots(kNumDevices);

  ASSERT_OK(
      RunOnDevices(kNumDevices, "alltoall_execute", [&](int d) -> absl::Status {
        RETURN_IF_ERROR(SetupDeviceSlot(d, slots[d], thunk, device_assignment));
        return RunExecuteOnStreamPhase(slots[d], thunk, d,
                                       /*phase=*/1);
      }));
}

TEST(AllToAllThunkMultiGpuTest, RecordCommandBufferCreate) {
  if (!HasEnoughGpus(kNumDevices)) {
    GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
  }
  if (!IsAtLeastCuda12900(GetGpuExecutor(0))) {
    GTEST_SKIP() << "Child command nodes require CUDA 12.9+";
  }

  DeviceAssignment device_assignment = MakeDeviceAssignment(kNumDevices);
  std::vector<BufferAllocation> buffer_allocations =
      MakeThunkBufferAllocations();
  AllToAllThunk thunk = MakeThunk(buffer_allocations);
  std::vector<DeviceTestSlot> slots(kNumDevices);

  ASSERT_OK(
      RunOnDevices(kNumDevices, "alltoall_create", [&](int d) -> absl::Status {
        RETURN_IF_ERROR(SetupDeviceSlot(d, slots[d], thunk, device_assignment));
        return RunCreatePhase(slots[d], thunk, d,
                              /*phase=*/2);
      }));
}

TEST(AllToAllThunkMultiGpuTest, RecordCommandBufferUpdate) {
  if (!HasEnoughGpus(kNumDevices)) {
    GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
  }
  if (!IsAtLeastCuda12900(GetGpuExecutor(0))) {
    GTEST_SKIP() << "Child command nodes require CUDA 12.9+";
  }

  DeviceAssignment device_assignment = MakeDeviceAssignment(kNumDevices);
  std::vector<BufferAllocation> buffer_allocations =
      MakeThunkBufferAllocations();
  AllToAllThunk thunk = MakeThunk(buffer_allocations);
  std::vector<DeviceTestSlot> slots(kNumDevices);

  ASSERT_OK(
      RunOnDevices(kNumDevices, "alltoall_create", [&](int d) -> absl::Status {
        RETURN_IF_ERROR(SetupDeviceSlot(d, slots[d], thunk, device_assignment));
        return RunCreatePhase(slots[d], thunk, d,
                              /*phase=*/2);
      }));

  ASSERT_OK(RunOnDevices(kNumDevices, "alltoall_update", [&](int d) {
    return RunUpdatePhase(slots[d], thunk, d, /*phase=*/3);
  }));
}

}  // namespace
}  // namespace xla::gpu
