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

// Multi-GPU integration tests for AllReduceThunk and ReduceScatterThunk
// command-buffer Record().
// Requires exactly kNumDevices GPUs (>= 2) and CUDA 12.9+ driver/toolkit for
// CreateChildCommand / UpdateChildCommand support.

#include <cstdint>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/all_reduce_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk_multigpu_test_utils.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

static constexpr int kNumDevices = 2;
static constexpr int64_t kLength = 4;
static constexpr int64_t kByteLength = sizeof(float) * kLength;
static_assert(kLength % kNumDevices == 0);
static constexpr int64_t kReduceScatterLength = kLength / kNumDevices;
static constexpr int64_t kReduceScatterByteLength =
    sizeof(float) * kReduceScatterLength;

// AllReduceThunk variant that bypasses CollectiveKernelThunk and calls NCCL
// directly via RunAllReduce, allowing multi-GPU tests without a compiled PTX
// kernel.
class DirectAllReduceThunk : public AllReduceReduceScatterThunkBase {
 public:
  DirectAllReduceThunk(Thunk::ThunkInfo thunk_info, AllReduceConfig config,
                       std::vector<Buffer> buffers)
      : AllReduceReduceScatterThunkBase(Thunk::kAllReduce,
                                        std::move(thunk_info),
                                        std::move(config), std::move(buffers)) {
  }

  bool RequiresRendezvous() const override { return true; }

 protected:
  absl::Status RunCollective(const ExecuteParams& params,
                             const GpuCliqueKey& /*clique_key*/,
                             se::Stream& stream, Communicator& comm) override {
    ASSIGN_OR_RETURN(
        std::vector<DeviceBufferPair> device_buffers,
        ConvertToDeviceBuffers(params.buffer_allocations, buffers(),
                               config_.config.operand_element_type));
    return RunAllReduce(config_.reduction_kind, device_buffers, stream, comm,
                        /*use_symmetric_buffer=*/false);
  }
};

static AllReduceConfig MakeSumConfig() {
  ReplicaGroup replica_group;
  for (int i = 0; i < kNumDevices; ++i) {
    replica_group.add_replica_ids(i);
  }
  AllReduceConfig config;
  config.config.operand_element_type = {F32};
  config.config.replica_groups = {replica_group};
  config.config.group_mode = COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA;
  config.reduction_kind = ReductionKind::SUM;
  return config;
}

static CollectiveThunk::Buffer MakeBuffer(const BufferAllocation& alloc_src,
                                          const BufferAllocation& alloc_dst,
                                          int64_t source_length,
                                          int64_t destination_length) {
  int64_t source_byte_length = sizeof(float) * source_length;
  int64_t destination_byte_length = sizeof(float) * destination_length;
  ShapedSlice src_slice{
      BufferAllocation::Slice(&alloc_src, 0, source_byte_length),
      ShapeUtil::MakeShape(F32, {source_length})};
  ShapedSlice dst_slice{
      BufferAllocation::Slice(&alloc_dst, 0, destination_byte_length),
      ShapeUtil::MakeShape(F32, {destination_length})};
  return CollectiveThunk::Buffer{.element_count = source_length,
                                 .source_buffer = src_slice,
                                 .destination_buffer = dst_slice,
                                 .source_memory_space = 0,
                                 .destination_memory_space = 0};
}

static DirectAllReduceThunk MakeAllReduceThunk(
    const BufferAllocation& alloc_src, const BufferAllocation& alloc_dst) {
  return DirectAllReduceThunk(
      Thunk::ThunkInfo(), MakeSumConfig(),
      {MakeBuffer(alloc_src, alloc_dst, kLength, kLength)});
}

static ReduceScatterThunk MakeReduceScatterThunk(
    const BufferAllocation& alloc_src, const BufferAllocation& alloc_dst) {
  return ReduceScatterThunk(
      Thunk::ThunkInfo(), MakeSumConfig(),
      {MakeBuffer(alloc_src, alloc_dst, kLength, kReduceScatterLength)});
}

using DeviceTestSlot = CollectiveThunkMultiGpuTestState;

enum class CollectiveTestKind {
  kAllReduce,
  kReduceScatter,
};

static float DeviceScale(int device_ordinal) {
  float scale = 1.0f;
  for (int i = 0; i < device_ordinal; ++i) {
    scale *= 10.0f;
  }
  return scale;
}

static float DeviceScaleSum() {
  float sum = 0.0f;
  for (int d = 0; d < kNumDevices; ++d) {
    sum += DeviceScale(d);
  }
  return sum;
}

static std::vector<float> AllReduceInput(int device_ordinal,
                                         float phase_scale) {
  return std::vector<float>(
      kLength, static_cast<float>(device_ordinal + 1) * phase_scale);
}

static std::vector<float> AllReduceExpected(float phase_scale) {
  float sum = 0.0f;
  for (int d = 0; d < kNumDevices; ++d) {
    sum += static_cast<float>(d + 1) * phase_scale;
  }
  return std::vector<float>(kLength, sum);
}

static std::vector<float> ReduceScatterInput(int device_ordinal,
                                             float phase_scale) {
  std::vector<float> data;
  data.reserve(kLength);
  float scale = DeviceScale(device_ordinal) * phase_scale;
  for (int i = 0; i < kLength; ++i) {
    data.push_back(static_cast<float>(i + 1) * scale);
  }
  return data;
}

static std::vector<float> ReduceScatterExpected(int device_ordinal,
                                                float phase_scale) {
  std::vector<float> data;
  data.reserve(kReduceScatterLength);
  float scale_sum = DeviceScaleSum() * phase_scale;
  int64_t offset = device_ordinal * kReduceScatterLength;
  for (int i = 0; i < kReduceScatterLength; ++i) {
    data.push_back(static_cast<float>(offset + i + 1) * scale_sum);
  }
  return data;
}

static std::vector<float> InputValues(CollectiveTestKind kind,
                                      int device_ordinal, float phase_scale) {
  switch (kind) {
    case CollectiveTestKind::kAllReduce:
      return AllReduceInput(device_ordinal, phase_scale);
    case CollectiveTestKind::kReduceScatter:
      return ReduceScatterInput(device_ordinal, phase_scale);
  }
  return {};
}

static std::vector<float> ExpectedValues(CollectiveTestKind kind,
                                         int device_ordinal,
                                         float phase_scale) {
  switch (kind) {
    case CollectiveTestKind::kAllReduce:
      return AllReduceExpected(phase_scale);
    case CollectiveTestKind::kReduceScatter:
      return ReduceScatterExpected(device_ordinal, phase_scale);
  }
  return {};
}

static int64_t DestinationByteLength(CollectiveTestKind kind) {
  switch (kind) {
    case CollectiveTestKind::kAllReduce:
      return kByteLength;
    case CollectiveTestKind::kReduceScatter:
      return kReduceScatterByteLength;
  }
  return 0;
}

// Runs per-device setup: allocate buffers, Prepare the thunk, acquire
// collective cliques (collective - must run on all devices concurrently),
// and Initialize the thunk.
static absl::Status SetupDeviceSlot(int device_ordinal, DeviceTestSlot& slot,
                                    AllReduceReduceScatterThunkBase& thunk,
                                    const DeviceAssignment& device_assignment,
                                    CollectiveTestKind kind) {
  std::vector<int64_t> buffer_sizes = {kByteLength,
                                       DestinationByteLength(kind)};
  return SetupCollectiveThunkDevice(device_ordinal, kNumDevices, buffer_sizes,
                                    thunk, device_assignment, slot);
}

// Records the thunk into a new primary command buffer (create phase), submits
// it, waits, and verifies the collective output against expected_dst_values.
static absl::Status RunCreatePhase(DeviceTestSlot& slot,
                                   AllReduceReduceScatterThunkBase& thunk,
                                   const std::vector<float>& src_values,
                                   const std::vector<float>& expected_values) {
  RETURN_IF_ERROR(
      FillDeviceBuffer(*slot.stream, slot.create_buffers[0], src_values));

  BufferAllocations allocations =
      MakeBufferAllocations(slot, slot.create_buffers);
  Thunk::ExecuteParams execute_params = MakeExecuteParams(slot, allocations);

  // Warm-up: execute the collective once eagerly so NCCL completes its lazy
  // initialization outside of the stream capture performed by Record below.
  // CUDA graph capture rejects the NCCL bootstrap's sync primitives, which
  // would otherwise cause CUDA_ERROR_STREAM_CAPTURE_INVALIDATED.
  RETURN_IF_ERROR(ExecuteOnStreamAndBlock(thunk, execute_params));
  RETURN_IF_ERROR(RecordCommandBufferCreate(slot, thunk, execute_params));
  RETURN_IF_ERROR(FillDeviceBuffer(*slot.stream, slot.create_buffers[1],
                                   SentinelValues(expected_values.size())));
  RETURN_IF_ERROR(SubmitCommandBuffer(slot));
  return VerifyDeviceBuffer(*slot.stream, slot.create_buffers[1],
                            expected_values);
}

// Transitions the command buffer to update mode, re-records the thunk with new
// buffer allocations (update phase), submits, waits, and verifies output.
// Also asserts that the same command node pointer is returned.
static absl::Status RunUpdatePhase(DeviceTestSlot& slot,
                                   AllReduceReduceScatterThunkBase& thunk,
                                   const std::vector<float>& src_values,
                                   const std::vector<float>& expected_values) {
  RETURN_IF_ERROR(
      FillDeviceBuffer(*slot.stream, slot.update_buffers[0], src_values));

  BufferAllocations allocations =
      MakeBufferAllocations(slot, slot.update_buffers);
  Thunk::ExecuteParams execute_params = MakeExecuteParams(slot, allocations);

  RETURN_IF_ERROR(
      RecordCommandBufferUpdate(slot, thunk, execute_params, {0, 1}));
  RETURN_IF_ERROR(FillDeviceBuffer(*slot.stream, slot.update_buffers[1],
                                   SentinelValues(expected_values.size())));
  RETURN_IF_ERROR(SubmitCommandBuffer(slot));
  return VerifyDeviceBuffer(*slot.stream, slot.update_buffers[1],
                            expected_values);
}

// Runs setup + create phase for one device from a thread-pool worker.
static absl::Status SetupAndCreate(int d, DeviceTestSlot* slots,
                                   AllReduceReduceScatterThunkBase* thunk,
                                   const DeviceAssignment* device_assignment,
                                   CollectiveTestKind kind) {
  RETURN_IF_ERROR(
      SetupDeviceSlot(d, slots[d], *thunk, *device_assignment, kind));
  RETURN_IF_ERROR(RunCreatePhase(slots[d], *thunk,
                                 InputValues(kind, d, /*phase_scale=*/1.0f),
                                 ExpectedValues(kind, d,
                                                /*phase_scale=*/1.0f)));
  return absl::OkStatus();
}

// Runs update phase for one device from a thread-pool worker.
static absl::Status RunUpdate(int d, DeviceTestSlot* slots,
                              AllReduceReduceScatterThunkBase* thunk,
                              CollectiveTestKind kind) {
  RETURN_IF_ERROR(RunUpdatePhase(slots[d], *thunk,
                                 InputValues(kind, d, /*phase_scale=*/100.0f),
                                 ExpectedValues(kind, d,
                                                /*phase_scale=*/100.0f)));
  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//
// Tests
//===----------------------------------------------------------------------===//

// Records AllReduceThunk into a command buffer on two GPUs, submits it, and
// verifies that each device's output buffer contains the SUM of all inputs.
TEST(AllReduceThunkMultiGpuTest, RecordCommandBufferCreate) {
  if (!HasEnoughGpus(kNumDevices)) {
    GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
  }
  if (!IsAtLeastCuda12900(GetGpuExecutor(0))) {
    GTEST_SKIP() << "Child command nodes require CUDA 12.9+";
  }

  DeviceAssignment device_assignment = MakeDeviceAssignment(kNumDevices);
  BufferAllocation alloc_src(/*index=*/0, kByteLength, /*color=*/0);
  BufferAllocation alloc_dst(/*index=*/1, kByteLength, /*color=*/0);
  DirectAllReduceThunk thunk = MakeAllReduceThunk(alloc_src, alloc_dst);

  std::vector<DeviceTestSlot> slots(kNumDevices);

  ASSERT_OK(RunOnDevices(kNumDevices, "allreduce_create", [&](int d) {
    return SetupAndCreate(d, slots.data(), &thunk, &device_assignment,
                          CollectiveTestKind::kAllReduce);
  }));
}

// Records AllReduceThunk into a command buffer on two GPUs (create phase),
// then updates the command buffer to point at different buffers (update phase).
// Verifies that both phases produce the correct SUM output and that the update
// reuses the original command node.
TEST(AllReduceThunkMultiGpuTest, RecordCommandBufferUpdate) {
  if (!HasEnoughGpus(kNumDevices)) {
    GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
  }
  if (!IsAtLeastCuda12900(GetGpuExecutor(0))) {
    GTEST_SKIP() << "Child command nodes require CUDA 12.9+";
  }

  DeviceAssignment device_assignment = MakeDeviceAssignment(kNumDevices);
  BufferAllocation alloc_src(/*index=*/0, kByteLength, /*color=*/0);
  BufferAllocation alloc_dst(/*index=*/1, kByteLength, /*color=*/0);
  DirectAllReduceThunk thunk = MakeAllReduceThunk(alloc_src, alloc_dst);

  std::vector<DeviceTestSlot> slots(kNumDevices);

  // Phase 1: setup + create - all devices run concurrently so that NCCL
  // bootstrap and the traced AllReduce complete on every rank.
  {
    ASSERT_OK(RunOnDevices(kNumDevices, "allreduce_create", [&](int d) {
      return SetupAndCreate(d, slots.data(), &thunk, &device_assignment,
                            CollectiveTestKind::kAllReduce);
    }));
  }

  // Phase 2: update - all devices must re-enter the NCCL AllReduce in the
  // trace concurrently, so use a fresh thread pool.
  {
    ASSERT_OK(RunOnDevices(kNumDevices, "allreduce_update", [&](int d) {
      return RunUpdate(d, slots.data(), &thunk, CollectiveTestKind::kAllReduce);
    }));
  }
}

// Records ReduceScatterThunk into a command buffer on two GPUs, submits it,
// and verifies that each device's output buffer contains its rank's chunk of
// the reduced input.
TEST(ReduceScatterThunkMultiGpuTest, RecordCommandBufferCreate) {
  if (!HasEnoughGpus(kNumDevices)) {
    GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
  }
  if (!IsAtLeastCuda12900(GetGpuExecutor(0))) {
    GTEST_SKIP() << "Child command nodes require CUDA 12.9+";
  }

  DeviceAssignment device_assignment = MakeDeviceAssignment(kNumDevices);
  BufferAllocation alloc_src(/*index=*/0, kByteLength, /*color=*/0);
  BufferAllocation alloc_dst(/*index=*/1, kReduceScatterByteLength,
                             /*color=*/0);
  ReduceScatterThunk thunk = MakeReduceScatterThunk(alloc_src, alloc_dst);

  std::vector<DeviceTestSlot> slots(kNumDevices);

  ASSERT_OK(RunOnDevices(kNumDevices, "reducescatter_create", [&](int d) {
    return SetupAndCreate(d, slots.data(), &thunk, &device_assignment,
                          CollectiveTestKind::kReduceScatter);
  }));
}

// Records ReduceScatterThunk into a command buffer on two GPUs (create phase),
// then updates the command buffer to point at different buffers (update phase).
// Verifies that both phases produce each rank's correct reduced chunk and that
// the update reuses the original command node.
TEST(ReduceScatterThunkMultiGpuTest, RecordCommandBufferUpdate) {
  if (!HasEnoughGpus(kNumDevices)) {
    GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
  }
  if (!IsAtLeastCuda12900(GetGpuExecutor(0))) {
    GTEST_SKIP() << "Child command nodes require CUDA 12.9+";
  }

  DeviceAssignment device_assignment = MakeDeviceAssignment(kNumDevices);
  BufferAllocation alloc_src(/*index=*/0, kByteLength, /*color=*/0);
  BufferAllocation alloc_dst(/*index=*/1, kReduceScatterByteLength,
                             /*color=*/0);
  ReduceScatterThunk thunk = MakeReduceScatterThunk(alloc_src, alloc_dst);

  std::vector<DeviceTestSlot> slots(kNumDevices);

  // Phase 1: setup + create - all devices run concurrently so that NCCL
  // bootstrap and the traced ReduceScatter complete on every rank.
  {
    ASSERT_OK(RunOnDevices(kNumDevices, "reducescatter_create", [&](int d) {
      return SetupAndCreate(d, slots.data(), &thunk, &device_assignment,
                            CollectiveTestKind::kReduceScatter);
    }));
  }

  // Phase 2: update - all devices must re-enter the NCCL ReduceScatter in the
  // trace concurrently, so use a fresh thread pool.
  {
    ASSERT_OK(RunOnDevices(kNumDevices, "reducescatter_update", [&](int d) {
      return RunUpdate(d, slots.data(), &thunk,
                       CollectiveTestKind::kReduceScatter);
    }));
  }
}

}  // namespace
}  // namespace xla::gpu
