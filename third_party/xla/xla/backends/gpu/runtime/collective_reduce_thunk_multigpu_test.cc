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

// Multi-GPU integration tests for CollectiveReduceThunk through NCCL
// (ncclReduce). Requires at least kNumDevices GPUs.

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/all_reduce_thunk.h"
#include "xla/backends/gpu/runtime/collective_reduce_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk_multigpu_test_utils.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/reduction_kind.h"
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

static AllReduceConfig MakeCollectiveReduceConfig(bool has_dynamic_root) {
  CollectiveConfig config;
  config.operand_element_type = {F32};
  if (has_dynamic_root) {
    config.operand_element_type.push_back(S32);
  }
  config.group_mode = COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA;

  ReplicaGroup replica_group;
  for (int i = 0; i < kNumDevices; ++i) {
    replica_group.add_replica_ids(i);
  }
  config.replica_groups = {replica_group};
  return AllReduceConfig{config, ReductionKind::SUM};
}

// Builds a sum-reduce thunk. When `alloc_root` is provided the reduce root is
// selected at run time from that S32 buffer, otherwise it defaults to rank 0.
static CollectiveReduceThunk MakeThunk(
    const BufferAllocation& alloc_src, const BufferAllocation& alloc_dst,
    const BufferAllocation* alloc_root = nullptr) {
  ShapedSlice src_slice{
      BufferAllocation::Slice(&alloc_src, 0, kFloatByteLength),
      ShapeUtil::MakeShape(F32, {kNumElements})};
  ShapedSlice dst_slice{
      BufferAllocation::Slice(&alloc_dst, 0, kFloatByteLength),
      ShapeUtil::MakeShape(F32, {kNumElements})};
  CollectiveThunk::Buffer buffer{.element_count = kNumElements,
                                 .source_buffer = src_slice,
                                 .destination_buffer = dst_slice,
                                 .source_memory_space = 0,
                                 .destination_memory_space = 0};
  std::vector<CollectiveThunk::Buffer> buffers = {buffer};
  if (alloc_root != nullptr) {
    ShapedSlice root_slice{
        BufferAllocation::Slice(alloc_root, 0, sizeof(int32_t)),
        ShapeUtil::MakeShape(S32, {1})};
    buffers.push_back(CollectiveThunk::Buffer{.element_count = 1,
                                              .source_buffer = root_slice,
                                              .destination_buffer = root_slice,
                                              .source_memory_space = 0,
                                              .destination_memory_space = 0});
  }
  const bool has_dynamic_root = alloc_root != nullptr;
  return CollectiveReduceThunk(Thunk::ThunkInfo(),
                               MakeCollectiveReduceConfig(has_dynamic_root),
                               buffers, has_dynamic_root);
}

using DeviceTestSlot = CollectiveThunkMultiGpuTestState;

static std::vector<float> SourceValues(int device_ordinal) {
  std::vector<float> values(kNumElements);
  for (int i = 0; i < kNumElements; ++i) {
    values[i] = static_cast<float>(device_ordinal * 10 + i);
  }
  return values;
}

// The reduced (summed) value expected on the root rank.
static std::vector<float> ExpectedSum() {
  std::vector<float> expected(kNumElements, 0.0f);
  for (int d = 0; d < kNumDevices; ++d) {
    std::vector<float> values = SourceValues(d);
    for (int i = 0; i < kNumElements; ++i) {
      expected[i] += values[i];
    }
  }
  return expected;
}

static absl::Status VerifyRootOutput(se::Stream& stream,
                                     se::DeviceAddressBase dst,
                                     int device_ordinal) {
  ABSL_ASSIGN_OR_RETURN(std::vector<float> output,
                   ReadDeviceBuffer(stream, dst, kNumElements));
  std::vector<float> expected = ExpectedSum();
  for (int i = 0; i < kNumElements; ++i) {
    if (output[i] != expected[i]) {
      return absl::InternalError(
          absl::StrFormat("root device %d output[%d] = %g, expected %g",
                          device_ordinal, i, output[i], expected[i]));
    }
  }
  return absl::OkStatus();
}

// Runs the reduce on each device and verifies the sum landed on `root_rank`.
static absl::Status RunReduce(std::vector<DeviceTestSlot>& slots,
                              CollectiveReduceThunk& thunk,
                              const DeviceAssignment& device_assignment,
                              int64_t root_buffer_size, int root_rank) {
  return RunOnDevices(
      kNumDevices, "collective_reduce_execute", [&](int d) -> absl::Status {
        std::vector<int64_t> buffer_sizes = {kFloatByteLength,
                                             kFloatByteLength};
        if (root_buffer_size > 0) {
          buffer_sizes.push_back(root_buffer_size);
        }
        ABSL_RETURN_IF_ERROR(SetupCollectiveThunkDevice(
            d, kNumDevices, buffer_sizes, thunk, device_assignment, slots[d]));

        DeviceTestSlot& slot = slots[d];
        ABSL_RETURN_IF_ERROR(FillDeviceBuffer(*slot.stream, slot.create_buffers[0],
                                         SourceValues(d)));
        ABSL_RETURN_IF_ERROR(
            FillDeviceBuffer(*slot.stream, slot.create_buffers[1],
                             std::vector<float>(kNumElements, -1.0f)));
        int32_t root_value = static_cast<int32_t>(root_rank);
        if (root_buffer_size > 0) {
          se::DeviceAddressBase root_buf = slot.create_buffers[2];
          ABSL_RETURN_IF_ERROR(
              slot.stream->Memcpy(&root_buf, &root_value, sizeof(int32_t)));
          // Block so `root_value` outlives the asynchronous host-to-device
          // copy.
          ABSL_RETURN_IF_ERROR(slot.stream->BlockHostUntilDone());
        }

        BufferAllocations allocations =
            MakeBufferAllocations(slot, slot.create_buffers);
        Thunk::ExecuteParams execute_params =
            MakeExecuteParams(slot, allocations);
        ABSL_RETURN_IF_ERROR(ExecuteOnStreamAndBlock(thunk, execute_params));

        // The reduce writes the result only to the root rank; other ranks'
        // destination buffers are left undefined.
        if (d == root_rank) {
          return VerifyRootOutput(*slot.stream, slot.create_buffers[1], d);
        }
        return absl::OkStatus();
      });
}

TEST(CollectiveReduceThunkMultiGpuTest, ExecuteOnStream) {
  if (!HasEnoughGpus(kNumDevices)) {
    GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
  }

  DeviceAssignment device_assignment = MakeDeviceAssignment(kNumDevices);
  BufferAllocation alloc_src(/*index=*/0, kFloatByteLength, /*color=*/0);
  BufferAllocation alloc_dst(/*index=*/1, kFloatByteLength, /*color=*/0);
  CollectiveReduceThunk thunk = MakeThunk(alloc_src, alloc_dst);
  std::vector<DeviceTestSlot> slots(kNumDevices);

  ASSERT_OK(RunReduce(slots, thunk, device_assignment,
                      /*root_buffer_size=*/0, /*root_rank=*/0));
}

TEST(CollectiveReduceThunkMultiGpuTest, ExecuteOnStreamWithDynamicRoot) {
  if (!HasEnoughGpus(kNumDevices)) {
    GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
  }

  DeviceAssignment device_assignment = MakeDeviceAssignment(kNumDevices);
  BufferAllocation alloc_src(/*index=*/0, kFloatByteLength, /*color=*/0);
  BufferAllocation alloc_dst(/*index=*/1, kFloatByteLength, /*color=*/0);
  BufferAllocation alloc_root(/*index=*/2, sizeof(int32_t), /*color=*/0);
  CollectiveReduceThunk thunk = MakeThunk(alloc_src, alloc_dst, &alloc_root);
  std::vector<DeviceTestSlot> slots(kNumDevices);

  // Select rank 1 as the root at run time; the sum must land on device 1.
  ASSERT_OK(RunReduce(slots, thunk, device_assignment,
                      /*root_buffer_size=*/alloc_root.size(), /*root_rank=*/1));
}

}  // namespace
}  // namespace xla::gpu
