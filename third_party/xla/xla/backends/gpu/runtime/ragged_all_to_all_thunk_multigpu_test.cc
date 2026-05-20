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

// Multi-GPU integration tests for RaggedAllToAllThunk command-buffer Record().
// Requires two GPUs and CUDA 12.9+ driver/toolkit for CreateChildCommand /
// UpdateChildCommand support.

#include <cstdint>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk_multigpu_test_utils.h"
#include "xla/backends/gpu/runtime/ragged_all_to_all_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

static constexpr int kNumDevices = 2;
static constexpr int64_t kNumUpdates = 4;
static constexpr int64_t kNumInputRows = 4;
static constexpr int64_t kNumRowElements = 1;
static constexpr int64_t kNumElements = kNumInputRows * kNumRowElements;
static constexpr int64_t kNumBuffers = 6;

static RaggedAllToAllConfig MakeConfig() {
  ReplicaGroup replica_group;
  for (int i = 0; i < kNumDevices; ++i) {
    replica_group.add_replica_ids(i);
  }

  RaggedAllToAllConfig config;
  config.config.operand_element_type = {F32, F32, S64, S64, S64, S64};
  config.config.replica_groups = {replica_group};
  config.config.group_mode = COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA;
  config.num_total_updates = kNumUpdates;
  config.num_input_rows = kNumInputRows;
  config.num_row_elements = kNumRowElements;
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

static std::vector<BufferAllocation> MakeThunkBufferAllocations() {
  std::vector<BufferAllocation> allocations;
  allocations.reserve(kNumBuffers);
  allocations.emplace_back(/*index=*/0, sizeof(float) * kNumElements,
                           /*color=*/0);
  allocations.emplace_back(/*index=*/1, sizeof(float) * kNumElements,
                           /*color=*/0);
  for (int i = 2; i < kNumBuffers; ++i) {
    allocations.emplace_back(i, sizeof(int64_t) * kNumUpdates, /*color=*/0);
  }
  return allocations;
}

static RaggedAllToAllThunk MakeThunk(
    const std::vector<BufferAllocation>& allocations) {
  std::vector<CollectiveThunk::Buffer> buffers;
  buffers.reserve(kNumBuffers);
  buffers.push_back(MakeBuffer(allocations[0], F32, kNumElements));
  buffers.push_back(MakeBuffer(allocations[1], F32, kNumElements));
  for (int i = 2; i < kNumBuffers; ++i) {
    buffers.push_back(MakeBuffer(allocations[i], S64, kNumUpdates));
  }
  return RaggedAllToAllThunk(Thunk::ThunkInfo(), MakeConfig(),
                             std::move(buffers));
}

using DeviceTestSlot = CollectiveThunkMultiGpuTestState;

static std::vector<int64_t> DeviceBufferSizes() {
  std::vector<int64_t> buffer_sizes;
  buffer_sizes.reserve(kNumBuffers);
  buffer_sizes.push_back(static_cast<int64_t>(sizeof(float)) * kNumElements);
  buffer_sizes.push_back(static_cast<int64_t>(sizeof(float)) * kNumElements);
  for (int i = 2; i < kNumBuffers; ++i) {
    buffer_sizes.push_back(static_cast<int64_t>(sizeof(int64_t)) * kNumUpdates);
  }
  return buffer_sizes;
}

static std::vector<BufferAllocation::Index> AllAllocationIndices() {
  std::vector<BufferAllocation::Index> indices;
  indices.reserve(kNumBuffers);
  for (int i = 0; i < kNumBuffers; ++i) {
    indices.push_back(i);
  }
  return indices;
}

static std::vector<float> InputValues(int device_ordinal, int phase) {
  std::vector<float> values(kNumElements);
  for (int i = 0; i < values.size(); ++i) {
    values[i] = static_cast<float>(phase * 100 + device_ordinal * 10 + i);
  }
  return values;
}

static std::vector<int64_t> OutputOffsets(int device_ordinal) {
  int64_t base = 2 * device_ordinal;
  return {base, base + 1, base, base + 1};
}

static std::vector<float> ExpectedValues(int device_ordinal, int phase) {
  std::vector<float> rank0 = InputValues(/*device_ordinal=*/0, phase);
  std::vector<float> rank1 = InputValues(/*device_ordinal=*/1, phase);
  if (device_ordinal == 0) {
    return {rank0[0], rank0[1], rank1[0], rank1[1]};
  }
  return {rank0[2], rank0[3], rank1[2], rank1[3]};
}

static absl::Status WriteBuffer(se::Stream& stream,
                                se::DeviceAddressBase buffer,
                                const std::vector<int64_t>& data) {
  RETURN_IF_ERROR(
      stream.Memcpy(&buffer, data.data(), data.size() * sizeof(int64_t)));
  return absl::OkStatus();
}

static absl::Status PrepareInputs(
    se::Stream& stream, absl::Span<const se::DeviceAddressBase> buffers,
    int device_ordinal, int phase) {
  RETURN_IF_ERROR(
      FillDeviceBuffer(stream, buffers[0], InputValues(device_ordinal, phase)));
  RETURN_IF_ERROR(FillDeviceBuffer(stream, buffers[1],
                                   std::vector<float>(kNumElements, -1.0f)));
  RETURN_IF_ERROR(
      WriteBuffer(stream, buffers[2], std::vector<int64_t>{0, 1, 2, 3}));
  RETURN_IF_ERROR(
      WriteBuffer(stream, buffers[3], std::vector<int64_t>{1, 1, 1, 1}));
  RETURN_IF_ERROR(
      WriteBuffer(stream, buffers[4], OutputOffsets(device_ordinal)));
  RETURN_IF_ERROR(
      WriteBuffer(stream, buffers[5], std::vector<int64_t>{1, 1, 1, 1}));
  return stream.BlockHostUntilDone();
}

static absl::Status VerifyOutput(se::Stream& stream,
                                 se::DeviceAddressBase buffer,
                                 int device_ordinal, int phase) {
  ASSIGN_OR_RETURN(std::vector<float> output,
                   ReadDeviceBuffer(stream, buffer, kNumElements));
  std::vector<float> expected = ExpectedValues(device_ordinal, phase);
  for (int i = 0; i < output.size(); ++i) {
    if (output[i] != expected[i]) {
      return absl::InternalError(
          absl::StrFormat("device %d output[%d] = %g, expected %g",
                          device_ordinal, i, output[i], expected[i]));
    }
  }
  return absl::OkStatus();
}

static absl::Status SetupDeviceSlot(int device_ordinal, DeviceTestSlot& slot,
                                    RaggedAllToAllThunk& thunk,
                                    const DeviceAssignment& device_assignment) {
  std::vector<int64_t> buffer_sizes = DeviceBufferSizes();
  return SetupCollectiveThunkDevice(device_ordinal, kNumDevices, buffer_sizes,
                                    thunk, device_assignment, slot);
}

static absl::Status RunExecuteOnStreamPhase(DeviceTestSlot& slot,
                                            RaggedAllToAllThunk& thunk,
                                            int device_ordinal, int phase) {
  RETURN_IF_ERROR(
      PrepareInputs(*slot.stream, slot.create_buffers, device_ordinal, phase));

  BufferAllocations allocations =
      MakeBufferAllocations(slot, slot.create_buffers);
  Thunk::ExecuteParams execute_params = MakeExecuteParams(slot, allocations);

  RETURN_IF_ERROR(ExecuteOnStreamAndBlock(thunk, execute_params));
  return VerifyOutput(*slot.stream, slot.create_buffers[1], device_ordinal,
                      phase);
}

static absl::Status RunCreatePhase(DeviceTestSlot& slot,
                                   RaggedAllToAllThunk& thunk,
                                   int device_ordinal, int phase) {
  RETURN_IF_ERROR(
      PrepareInputs(*slot.stream, slot.create_buffers, device_ordinal, phase));

  BufferAllocations allocations =
      MakeBufferAllocations(slot, slot.create_buffers);
  Thunk::ExecuteParams execute_params = MakeExecuteParams(slot, allocations);

  RETURN_IF_ERROR(ExecuteOnStreamAndBlock(thunk, execute_params));
  RETURN_IF_ERROR(
      PrepareInputs(*slot.stream, slot.create_buffers, device_ordinal, phase));

  RETURN_IF_ERROR(RecordCommandBufferCreate(slot, thunk, execute_params));
  RETURN_IF_ERROR(SubmitCommandBuffer(slot));
  return VerifyOutput(*slot.stream, slot.create_buffers[1], device_ordinal,
                      phase);
}

static absl::Status RunUpdatePhase(DeviceTestSlot& slot,
                                   RaggedAllToAllThunk& thunk,
                                   int device_ordinal, int phase) {
  RETURN_IF_ERROR(
      PrepareInputs(*slot.stream, slot.update_buffers, device_ordinal, phase));

  BufferAllocations allocations =
      MakeBufferAllocations(slot, slot.update_buffers);
  Thunk::ExecuteParams execute_params = MakeExecuteParams(slot, allocations);

  RETURN_IF_ERROR(RecordCommandBufferUpdate(slot, thunk, execute_params,
                                            AllAllocationIndices()));
  RETURN_IF_ERROR(SubmitCommandBuffer(slot));
  return VerifyOutput(*slot.stream, slot.update_buffers[1], device_ordinal,
                      phase);
}

TEST(RaggedAllToAllThunkMultiGpuTest, ExecuteOnStream) {
  if (!HasEnoughGpus(kNumDevices)) {
    GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
  }

  DeviceAssignment device_assignment = MakeDeviceAssignment(kNumDevices);
  std::vector<BufferAllocation> buffer_allocations =
      MakeThunkBufferAllocations();
  RaggedAllToAllThunk thunk = MakeThunk(buffer_allocations);
  std::vector<DeviceTestSlot> slots(kNumDevices);

  ASSERT_OK(
      RunOnDevices(kNumDevices, "ragged_execute", [&](int d) -> absl::Status {
        RETURN_IF_ERROR(SetupDeviceSlot(d, slots[d], thunk, device_assignment));
        return RunExecuteOnStreamPhase(slots[d], thunk, d, /*phase=*/1);
      }));
}

TEST(RaggedAllToAllThunkMultiGpuTest, RecordCommandBufferCreate) {
  if (!HasEnoughGpus(kNumDevices)) {
    GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
  }
  if (!IsAtLeastCuda12900(GetGpuExecutor(0))) {
    GTEST_SKIP() << "Child command nodes require CUDA 12.9+";
  }

  DeviceAssignment device_assignment = MakeDeviceAssignment(kNumDevices);
  std::vector<BufferAllocation> buffer_allocations =
      MakeThunkBufferAllocations();
  RaggedAllToAllThunk thunk = MakeThunk(buffer_allocations);
  std::vector<DeviceTestSlot> slots(kNumDevices);

  ASSERT_OK(
      RunOnDevices(kNumDevices, "ragged_create", [&](int d) -> absl::Status {
        RETURN_IF_ERROR(SetupDeviceSlot(d, slots[d], thunk, device_assignment));
        return RunCreatePhase(slots[d], thunk, d, /*phase=*/1);
      }));
}

TEST(RaggedAllToAllThunkMultiGpuTest, RecordCommandBufferUpdate) {
  if (!HasEnoughGpus(kNumDevices)) {
    GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
  }
  if (!IsAtLeastCuda12900(GetGpuExecutor(0))) {
    GTEST_SKIP() << "Child command nodes require CUDA 12.9+";
  }

  DeviceAssignment device_assignment = MakeDeviceAssignment(kNumDevices);
  std::vector<BufferAllocation> buffer_allocations =
      MakeThunkBufferAllocations();
  RaggedAllToAllThunk thunk = MakeThunk(buffer_allocations);
  std::vector<DeviceTestSlot> slots(kNumDevices);

  ASSERT_OK(
      RunOnDevices(kNumDevices, "ragged_create", [&](int d) -> absl::Status {
        RETURN_IF_ERROR(SetupDeviceSlot(d, slots[d], thunk, device_assignment));
        return RunCreatePhase(slots[d], thunk, d, /*phase=*/1);
      }));

  ASSERT_OK(RunOnDevices(kNumDevices, "ragged_update", [&](int d) {
    return RunUpdatePhase(slots[d], thunk, d, /*phase=*/2);
  }));
}

}  // namespace
}  // namespace xla::gpu
