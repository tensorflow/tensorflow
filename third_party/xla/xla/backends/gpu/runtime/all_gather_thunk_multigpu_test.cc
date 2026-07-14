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

// Multi-GPU integration tests for AllGatherThunk through NCCL. Requires at
// least kNumDevices GPUs. Command-buffer tests additionally require CUDA 12.9+
// driver/toolkit for CreateChildCommand / UpdateChildCommand support.

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/all_gather_thunk.h"
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
static constexpr int64_t kGatheredLength = kLength * kNumDevices;
static constexpr int64_t kGatheredByteLength = sizeof(float) * kGatheredLength;

static CollectiveConfig MakeAllGatherConfig() {
  ReplicaGroup replica_group;
  for (int i = 0; i < kNumDevices; ++i) {
    replica_group.add_replica_ids(i);
  }

  CollectiveConfig config;
  config.operand_element_type = {F32};
  config.replica_groups = {replica_group};
  config.group_mode = COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA;
  return config;
}

static AllGatherThunk MakeThunk(const BufferAllocation& alloc_src,
                                const BufferAllocation& alloc_dst) {
  ShapedSlice src_slice{BufferAllocation::Slice(&alloc_src, 0, kByteLength),
                        ShapeUtil::MakeShape(F32, {kLength})};
  ShapedSlice dst_slice{
      BufferAllocation::Slice(&alloc_dst, 0, kGatheredByteLength),
      ShapeUtil::MakeShape(F32, {kGatheredLength})};
  CollectiveThunk::Buffer buffer{.element_count = kLength,
                                 .source_buffer = src_slice,
                                 .destination_buffer = dst_slice,
                                 .source_memory_space = 0,
                                 .destination_memory_space = 0};
  return AllGatherThunk(Thunk::ThunkInfo(), MakeAllGatherConfig(), {buffer});
}

using DeviceTestSlot = CollectiveThunkMultiGpuTestState;

static std::vector<int64_t> DeviceBufferSizes() {
  return {kByteLength, kGatheredByteLength};
}

static std::vector<float> SourceValues(int source_rank, int phase) {
  std::vector<float> values(kLength);
  for (int i = 0; i < kLength; ++i) {
    values[i] = static_cast<float>(phase * 100 + source_rank * 10 + i);
  }
  return values;
}

static absl::Status FillSourceBuffer(se::Stream& stream,
                                     se::DeviceAddressBase buffer,
                                     int device_ordinal, int phase) {
  return FillDeviceBuffer(stream, buffer, SourceValues(device_ordinal, phase));
}

static absl::Status FillDestinationBuffer(se::Stream& stream,
                                          se::DeviceAddressBase buffer,
                                          float value) {
  return FillDeviceBuffer(stream, buffer,
                          std::vector<float>(kGatheredLength, value));
}

static absl::Status PrepareInputs(
    se::Stream& stream, absl::Span<const se::DeviceAddressBase> buffers,
    int device_ordinal, int phase) {
  RETURN_IF_ERROR(FillSourceBuffer(stream, buffers[0], device_ordinal, phase));
  return FillDestinationBuffer(stream, buffers[1], -1.0f);
}

static absl::Status VerifyOutput(se::Stream& stream, se::DeviceAddressBase dst,
                                 int phase) {
  ASSIGN_OR_RETURN(std::vector<float> output,
                   ReadDeviceBuffer(stream, dst, kGatheredLength));
  for (int source_rank = 0; source_rank < kNumDevices; ++source_rank) {
    std::vector<float> expected = SourceValues(source_rank, phase);
    for (int i = 0; i < kLength; ++i) {
      int output_index = source_rank * kLength + i;
      if (output[output_index] != expected[i]) {
        return absl::InternalError(
            absl::StrFormat("output[%d] = %g, expected %g", output_index,
                            output[output_index], expected[i]));
      }
    }
  }
  return absl::OkStatus();
}

static absl::Status SetupDeviceSlot(int device_ordinal, DeviceTestSlot& slot,
                                    AllGatherThunk& thunk,
                                    const DeviceAssignment& device_assignment) {
  std::vector<int64_t> buffer_sizes = DeviceBufferSizes();
  return SetupCollectiveThunkDevice(device_ordinal, kNumDevices, buffer_sizes,
                                    thunk, device_assignment, slot);
}

static absl::Status RunExecuteOnStreamPhase(DeviceTestSlot& slot,
                                            AllGatherThunk& thunk,
                                            int device_ordinal, int phase) {
  RETURN_IF_ERROR(
      PrepareInputs(*slot.stream, slot.create_buffers, device_ordinal, phase));

  BufferAllocations allocations =
      MakeBufferAllocations(slot, slot.create_buffers);
  Thunk::ExecuteParams execute_params = MakeExecuteParams(slot, allocations);

  RETURN_IF_ERROR(ExecuteOnStreamAndBlock(thunk, execute_params));
  return VerifyOutput(*slot.stream, slot.create_buffers[1], phase);
}

static absl::Status RunCreatePhase(DeviceTestSlot& slot, AllGatherThunk& thunk,
                                   int device_ordinal, int phase) {
  RETURN_IF_ERROR(
      PrepareInputs(*slot.stream, slot.create_buffers, device_ordinal, phase));

  BufferAllocations allocations =
      MakeBufferAllocations(slot, slot.create_buffers);
  Thunk::ExecuteParams execute_params = MakeExecuteParams(slot, allocations);

  // Warm up NCCL outside stream capture. Reset destination buffers afterward so
  // correctness is verified from command-buffer execution, not from warm-up.
  RETURN_IF_ERROR(ExecuteOnStreamAndBlock(thunk, execute_params));
  RETURN_IF_ERROR(
      FillDestinationBuffer(*slot.stream, slot.create_buffers[1], -1.0f));

  RETURN_IF_ERROR(RecordCommandBufferCreate(slot, thunk, execute_params));
  RETURN_IF_ERROR(SubmitCommandBuffer(slot));
  return VerifyOutput(*slot.stream, slot.create_buffers[1], phase);
}

static absl::Status RunUpdatePhase(DeviceTestSlot& slot, AllGatherThunk& thunk,
                                   int device_ordinal, int phase) {
  RETURN_IF_ERROR(
      PrepareInputs(*slot.stream, slot.update_buffers, device_ordinal, phase));

  BufferAllocations allocations =
      MakeBufferAllocations(slot, slot.update_buffers);
  Thunk::ExecuteParams execute_params = MakeExecuteParams(slot, allocations);

  RETURN_IF_ERROR(
      RecordCommandBufferUpdate(slot, thunk, execute_params, {0, 1}));
  RETURN_IF_ERROR(SubmitCommandBuffer(slot));
  return VerifyOutput(*slot.stream, slot.update_buffers[1], phase);
}

TEST(AllGatherThunkMultiGpuTest, ExecuteOnStream) {
  if (!HasEnoughGpus(kNumDevices)) {
    GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
  }

  DeviceAssignment device_assignment = MakeDeviceAssignment(kNumDevices);
  BufferAllocation alloc_src(/*index=*/0, kByteLength, /*color=*/0);
  BufferAllocation alloc_dst(/*index=*/1, kGatheredByteLength, /*color=*/0);
  AllGatherThunk thunk = MakeThunk(alloc_src, alloc_dst);
  std::vector<DeviceTestSlot> slots(kNumDevices);

  ASSERT_OK(RunOnDevices(
      kNumDevices, "allgather_execute", [&](int d) -> absl::Status {
        RETURN_IF_ERROR(SetupDeviceSlot(d, slots[d], thunk, device_assignment));
        return RunExecuteOnStreamPhase(slots[d], thunk, d, /*phase=*/1);
      }));
}

TEST(AllGatherThunkMultiGpuTest, RecordCommandBufferCreate) {
  if (!HasEnoughGpus(kNumDevices)) {
    GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
  }
  if (!IsAtLeastCuda12900(GetGpuExecutor(0))) {
    GTEST_SKIP() << "Child command nodes require CUDA 12.9+";
  }

  DeviceAssignment device_assignment = MakeDeviceAssignment(kNumDevices);
  BufferAllocation alloc_src(/*index=*/0, kByteLength, /*color=*/0);
  BufferAllocation alloc_dst(/*index=*/1, kGatheredByteLength, /*color=*/0);
  AllGatherThunk thunk = MakeThunk(alloc_src, alloc_dst);
  std::vector<DeviceTestSlot> slots(kNumDevices);

  ASSERT_OK(
      RunOnDevices(kNumDevices, "allgather_create", [&](int d) -> absl::Status {
        RETURN_IF_ERROR(SetupDeviceSlot(d, slots[d], thunk, device_assignment));
        return RunCreatePhase(slots[d], thunk, d, /*phase=*/2);
      }));
}

TEST(AllGatherThunkMultiGpuTest, RecordCommandBufferUpdate) {
  if (!HasEnoughGpus(kNumDevices)) {
    GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
  }
  if (!IsAtLeastCuda12900(GetGpuExecutor(0))) {
    GTEST_SKIP() << "Child command nodes require CUDA 12.9+";
  }

  DeviceAssignment device_assignment = MakeDeviceAssignment(kNumDevices);
  BufferAllocation alloc_src(/*index=*/0, kByteLength, /*color=*/0);
  BufferAllocation alloc_dst(/*index=*/1, kGatheredByteLength, /*color=*/0);
  AllGatherThunk thunk = MakeThunk(alloc_src, alloc_dst);
  std::vector<DeviceTestSlot> slots(kNumDevices);

  ASSERT_OK(
      RunOnDevices(kNumDevices, "allgather_create", [&](int d) -> absl::Status {
        RETURN_IF_ERROR(SetupDeviceSlot(d, slots[d], thunk, device_assignment));
        return RunCreatePhase(slots[d], thunk, d, /*phase=*/2);
      }));

  ASSERT_OK(RunOnDevices(kNumDevices, "allgather_update", [&](int d) {
    return RunUpdatePhase(slots[d], thunk, d, /*phase=*/3);
  }));
}

}  // namespace
}  // namespace xla::gpu
