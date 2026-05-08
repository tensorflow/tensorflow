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

// Multi-GPU integration tests for AllReduceThunk command-buffer Record().
// Requires exactly kNumDevices GPUs (≥ 2) and CUDA 12.9+ driver/toolkit for
// CreateChildCommand / UpdateChildCommand support.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/all_reduce_thunk.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/scratch_memory_requests.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/future.h"
#include "xla/hlo/ir/collective_op_group_mode.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

static constexpr int kNumDevices = 2;
static constexpr int64_t kLength = 4;
static constexpr int64_t kByteLength = sizeof(float) * kLength;

se::StreamExecutor* GetGpuExecutor(int ordinal) {
  auto* platform =
      se::PlatformManager::PlatformWithName(se::GpuPlatformName()).value();
  return platform->ExecutorForDevice(ordinal).value();
}

static bool IsAtLeastCuda12900(const se::StreamExecutor* executor) {
  const auto& desc = executor->GetDeviceDescription();
  const auto* cuda_cc = desc.gpu_compute_capability().cuda_compute_capability();
  if (cuda_cc == nullptr) {
    return false;
  }
  return std::min(desc.driver_version(), desc.compile_time_toolkit_version()) >=
         se::SemanticVersion(12, 9, 0);
}

static bool HasEnoughGpus() {
  auto platform = se::PlatformManager::PlatformWithName(se::GpuPlatformName());
  if (!platform.ok()) {
    return false;
  }
  return (*platform)->VisibleDeviceCount() >= kNumDevices;
}

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

static AllReduceConfig MakeSumAllReduceConfig() {
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

static DirectAllReduceThunk MakeThunk(const BufferAllocation& alloc_src,
                                      const BufferAllocation& alloc_dst) {
  ShapedSlice src_slice{BufferAllocation::Slice(&alloc_src, 0, kByteLength),
                        ShapeUtil::MakeShape(F32, {kLength})};
  ShapedSlice dst_slice{BufferAllocation::Slice(&alloc_dst, 0, kByteLength),
                        ShapeUtil::MakeShape(F32, {kLength})};
  CollectiveThunk::Buffer buffer{.element_count = kLength,
                                 .source_buffer = src_slice,
                                 .destination_buffer = dst_slice,
                                 .source_memory_space = 0,
                                 .destination_memory_space = 0};
  return DirectAllReduceThunk(Thunk::ThunkInfo(), MakeSumAllReduceConfig(),
                              {buffer});
}

// Holds per-device state that must survive across both the create and update
// phases of the command-buffer tests.
struct DeviceTestSlot {
  se::StreamExecutor* executor = nullptr;
  std::unique_ptr<se::Stream> stream;
  std::unique_ptr<se::StreamExecutorAddressAllocator> allocator;

  // Device buffers for the create phase (phase 1).
  se::DeviceAddressBase src1, dst1;
  // Device buffers for the update phase (phase 2).
  se::DeviceAddressBase src2, dst2;

  // GpuExecutableRunOptions must outlive collective_params because it owns the
  // global-device-ID map that collective_params points into.
  GpuExecutableRunOptions gpu_run_options;
  ServiceExecutableRunOptions run_options;
  std::optional<CollectiveParams> collective_params;
  CollectiveCliques collective_cliques;

  // Command-buffer state (written in the create phase, reused in update).
  CommandStateManager state_manager;
  std::unique_ptr<se::CommandBuffer> command_buffer;
  const se::CommandBuffer::Command* cmd = nullptr;
};

// Fills a device buffer with a uniform float value.
static absl::Status FillDeviceBuffer(se::Stream& stream,
                                     se::DeviceAddressBase buf, float value) {
  std::vector<float> data(kLength, value);
  RETURN_IF_ERROR(stream.Memcpy(&buf, data.data(), kByteLength));
  return stream.BlockHostUntilDone();
}

// Reads a device float buffer back to the host.
static absl::StatusOr<std::vector<float>> ReadDeviceBuffer(
    se::Stream& stream, se::DeviceAddressBase buf) {
  std::vector<float> data(kLength);
  RETURN_IF_ERROR(stream.Memcpy(data.data(), buf, kByteLength));
  RETURN_IF_ERROR(stream.BlockHostUntilDone());
  return data;
}

// Verifies that all elements of a device buffer equal expected_value.
static absl::Status VerifyOutput(se::Stream& stream, se::DeviceAddressBase buf,
                                 float expected_value) {
  ASSIGN_OR_RETURN(std::vector<float> output, ReadDeviceBuffer(stream, buf));
  for (int i = 0; i < kLength; ++i) {
    if (output[i] != expected_value) {
      return absl::InternalError(absl::StrFormat("output[%d] = %g, expected %g",
                                                 i, output[i], expected_value));
    }
  }
  return absl::OkStatus();
}

// Runs per-device setup: allocate buffers, Prepare the thunk, acquire
// collective cliques (collective — must run on all devices concurrently),
// and Initialize the thunk.
static absl::Status SetupDeviceSlot(int device_ordinal, DeviceTestSlot& slot,
                                    DirectAllReduceThunk& thunk,
                                    const DeviceAssignment& device_assignment) {
  slot.executor = GetGpuExecutor(device_ordinal);
  ASSIGN_OR_RETURN(slot.stream, slot.executor->CreateStream());
  slot.allocator =
      std::make_unique<se::StreamExecutorAddressAllocator>(slot.executor);

  // Allocate separate device memory for the create and update phases so the
  // update can point the child command at different physical buffers.
  slot.src1 = slot.executor->AllocateArray<float>(kLength, /*memory_space=*/0);
  slot.dst1 = slot.executor->AllocateArray<float>(kLength, 0);
  slot.src2 = slot.executor->AllocateArray<float>(kLength, 0);
  slot.dst2 = slot.executor->AllocateArray<float>(kLength, 0);

  // Build run options: all devices share the same global-ID map so the
  // NCCL bootstrap can find every peer.
  GpuExecutableRunOptions::DeviceIdMap id_map;
  for (int i = 0; i < kNumDevices; ++i) {
    id_map[LocalDeviceId(i)] = GlobalDeviceId(i);
  }
  slot.gpu_run_options.set_gpu_global_device_ids(std::move(id_map));
  slot.run_options.mutable_run_options()->set_stream(slot.stream.get());
  slot.run_options.mutable_run_options()->set_device_assignment(
      &device_assignment);
  slot.run_options.mutable_run_options()->set_gpu_executable_run_options(
      &slot.gpu_run_options);

  // CollectiveParams holds raw pointers into slot.gpu_run_options and
  // device_assignment, both of which outlive slot.
  ASSIGN_OR_RETURN(
      CollectiveParams params,
      CollectiveParams::Create(slot.run_options, /*async_streams=*/{},
                               LocalDeviceId(device_ordinal)));

  // Prepare: registers this device's clique requests.
  BufferAllocations allocations1({slot.src1, slot.dst1}, 0,
                                 slot.allocator.get());
  CollectiveCliqueRequests clique_requests;
  CollectiveMemoryRequests memory_requests(allocations1);
  ScratchMemoryRequests scratch_requests;
  Thunk::PrepareParams prepare_params{&params,          &clique_requests,
                                      &memory_requests, &scratch_requests,
                                      slot.executor,    &allocations1};
  RETURN_IF_ERROR(thunk.Prepare(prepare_params));

  // AcquireCollectiveCliques: collective — all device threads must reach here
  // simultaneously for the NCCL bootstrap rendezvous to complete.
  ASSIGN_OR_RETURN(slot.collective_cliques,
                   AcquireCollectiveCliques(params, clique_requests));

  // Initialize the thunk (no-op for DirectAllReduceThunk beyond base class).
  Thunk::InitializeParams init_params;
  init_params.executor = slot.executor;
  init_params.stream = slot.stream.get();
  init_params.buffer_allocations = &allocations1;
  init_params.collective_params = &params;
  init_params.collective_cliques = &slot.collective_cliques;
  RETURN_IF_ERROR(thunk.Initialize(init_params));

  slot.collective_params = std::move(params);
  return absl::OkStatus();
}

// Records the thunk into a new primary command buffer (create phase), submits
// it, waits, and verifies the AllReduce output against expected_dst_value.
static absl::Status RunCreatePhase(DeviceTestSlot& slot,
                                   DirectAllReduceThunk& thunk, float src_value,
                                   float expected_dst_value) {
  RETURN_IF_ERROR(FillDeviceBuffer(*slot.stream, slot.src1, src_value));

  BufferAllocations allocations({slot.src1, slot.dst1}, 0,
                                slot.allocator.get());
  Thunk::ExecuteParams execute_params = Thunk::ExecuteParams::Create(
      slot.run_options, allocations, slot.stream.get(),
      /*command_buffer_trace_stream=*/slot.stream.get(),
      &*slot.collective_params, &slot.collective_cliques,
      /*collective_memory=*/nullptr);

  // Warm-up: execute the collective once eagerly so NCCL completes its lazy
  // initialization outside of the stream capture performed by Record below.
  // CUDA graph capture rejects the NCCL bootstrap's sync primitives, which
  // would otherwise cause CUDA_ERROR_STREAM_CAPTURE_INVALIDATED.
  RETURN_IF_ERROR(thunk.ExecuteOnStream(execute_params));
  RETURN_IF_ERROR(slot.stream->BlockHostUntilDone());

  ASSIGN_OR_RETURN(slot.command_buffer, slot.executor->CreateCommandBuffer(
                                            se::CommandBuffer::Mode::kPrimary));

  Command::RecordParams record_params = {slot.state_manager};
  ASSIGN_OR_RETURN(slot.cmd,
                   thunk.Record(execute_params, record_params,
                                Command::RecordCreate{/*dependencies=*/{}},
                                slot.command_buffer.get()));
  if (slot.cmd == nullptr) {
    return absl::InternalError("Record(create) returned null command node");
  }

  RETURN_IF_ERROR(slot.command_buffer->Finalize());
  RETURN_IF_ERROR(slot.command_buffer->Submit(slot.stream.get()));
  RETURN_IF_ERROR(slot.stream->BlockHostUntilDone());
  return VerifyOutput(*slot.stream, slot.dst1, expected_dst_value);
}

// Transitions the command buffer to update mode, re-records the thunk with new
// buffer allocations (update phase), submits, waits, and verifies output.
// Also asserts that the same command node pointer is returned.
static absl::Status RunUpdatePhase(DeviceTestSlot& slot,
                                   DirectAllReduceThunk& thunk, float src_value,
                                   float expected_dst_value) {
  RETURN_IF_ERROR(FillDeviceBuffer(*slot.stream, slot.src2, src_value));

  BufferAllocations allocations2({slot.src2, slot.dst2}, 0,
                                 slot.allocator.get());
  Thunk::ExecuteParams execute_params2 = Thunk::ExecuteParams::Create(
      slot.run_options, allocations2, slot.stream.get(),
      /*command_buffer_trace_stream=*/slot.stream.get(),
      &*slot.collective_params, &slot.collective_cliques,
      /*collective_memory=*/nullptr);

  std::vector<BufferAllocation::Index> updated_allocs = {0, 1};
  Command::RecordParams record_params2 = {slot.state_manager,
                                          std::move(updated_allocs)};

  RETURN_IF_ERROR(slot.command_buffer->Update());
  ASSIGN_OR_RETURN(
      const se::CommandBuffer::Command* updated_cmd,
      thunk.Record(execute_params2, record_params2,
                   Command::RecordUpdate{slot.cmd}, slot.command_buffer.get()));

  if (updated_cmd != slot.cmd) {
    return absl::InternalError(
        "Update returned a different command node — expected the original to "
        "be "
        "reused");
  }

  RETURN_IF_ERROR(slot.command_buffer->Finalize());
  RETURN_IF_ERROR(slot.command_buffer->Submit(slot.stream.get()));
  RETURN_IF_ERROR(slot.stream->BlockHostUntilDone());
  return VerifyOutput(*slot.stream, slot.dst2, expected_dst_value);
}

// Runs setup + create phase for one device from a thread-pool worker.
static absl::StatusOr<int> SetupAndCreate(
    int d, DeviceTestSlot* slots, DirectAllReduceThunk* thunk,
    const DeviceAssignment* device_assignment) {
  // Device d contributes src value = d+1 (device 0 → 1.0, device 1 → 2.0).
  float src_value = static_cast<float>(d + 1);
  // Expected SUM output: 1 + 2 = 3.
  float expected = 3.0f;
  RETURN_IF_ERROR(SetupDeviceSlot(d, slots[d], *thunk, *device_assignment));
  RETURN_IF_ERROR(RunCreatePhase(slots[d], *thunk, src_value, expected));
  return d;
}

// Runs update phase for one device from a thread-pool worker.
static absl::StatusOr<int> RunUpdate(int d, DeviceTestSlot* slots,
                                     DirectAllReduceThunk* thunk) {
  // Device d contributes src value = (d+1)*10 (device 0 → 10.0, device 1
  // → 20.0).
  float src_value = static_cast<float>((d + 1) * 10);
  // Expected SUM output: 10 + 20 = 30.
  float expected = 30.0f;
  RETURN_IF_ERROR(RunUpdatePhase(slots[d], *thunk, src_value, expected));
  return d;
}

//===----------------------------------------------------------------------===//
// Tests
//===----------------------------------------------------------------------===//

// Records AllReduceThunk into a command buffer on two GPUs, submits it, and
// verifies that each device's output buffer contains the SUM of all inputs.
TEST(AllReduceThunkMultiGpuTest, RecordCommandBufferCreate) {
  if (!HasEnoughGpus()) {
    GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
  }
  if (!IsAtLeastCuda12900(GetGpuExecutor(0))) {
    GTEST_SKIP() << "Child command nodes require CUDA 12.9+";
  }

  DeviceAssignment device_assignment(kNumDevices, /*computation_count=*/1);
  for (int i = 0; i < kNumDevices; ++i) {
    device_assignment(i, 0) = i;
  }
  BufferAllocation alloc_src(/*index=*/0, kByteLength, /*color=*/0);
  BufferAllocation alloc_dst(/*index=*/1, kByteLength, /*color=*/0);
  DirectAllReduceThunk thunk = MakeThunk(alloc_src, alloc_dst);

  std::vector<DeviceTestSlot> slots(kNumDevices);

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "allreduce_create",
                               kNumDevices);
  std::vector<tsl::Future<int>> futures(kNumDevices);
  for (int d = 0; d < kNumDevices; ++d) {
    futures[d] = tsl::MakeFutureOn<int>(
        *pool.AsExecutor(), [d, &slots, &thunk, &device_assignment]() {
          return SetupAndCreate(d, slots.data(), &thunk, &device_assignment);
        });
  }
  TF_ASSERT_OK(JoinFutures<int>(futures).Await());
}

// Records AllReduceThunk into a command buffer on two GPUs (create phase),
// then updates the command buffer to point at different buffers (update phase).
// Verifies that both phases produce the correct SUM output and that the update
// reuses the original command node.
TEST(AllReduceThunkMultiGpuTest, RecordCommandBufferUpdate) {
  if (!HasEnoughGpus()) {
    GTEST_SKIP() << "Test requires at least " << kNumDevices << " GPUs";
  }
  if (!IsAtLeastCuda12900(GetGpuExecutor(0))) {
    GTEST_SKIP() << "Child command nodes require CUDA 12.9+";
  }

  DeviceAssignment device_assignment(kNumDevices, /*computation_count=*/1);
  for (int i = 0; i < kNumDevices; ++i) {
    device_assignment(i, 0) = i;
  }
  BufferAllocation alloc_src(/*index=*/0, kByteLength, /*color=*/0);
  BufferAllocation alloc_dst(/*index=*/1, kByteLength, /*color=*/0);
  DirectAllReduceThunk thunk = MakeThunk(alloc_src, alloc_dst);

  std::vector<DeviceTestSlot> slots(kNumDevices);

  // Phase 1: setup + create — all devices run concurrently so that NCCL
  // bootstrap and the traced AllReduce complete on every rank.
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "allreduce_create",
                                 kNumDevices);
    std::vector<tsl::Future<int>> futures(kNumDevices);
    for (int d = 0; d < kNumDevices; ++d) {
      futures[d] = tsl::MakeFutureOn<int>(
          *pool.AsExecutor(), [d, &slots, &thunk, &device_assignment]() {
            return SetupAndCreate(d, slots.data(), &thunk, &device_assignment);
          });
    }
    TF_ASSERT_OK(JoinFutures<int>(futures).Await());
  }

  // Phase 2: update — all devices must re-enter the NCCL AllReduce in the
  // trace concurrently, so use a fresh thread pool.
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "allreduce_update",
                                 kNumDevices);
    std::vector<tsl::Future<int>> futures(kNumDevices);
    for (int d = 0; d < kNumDevices; ++d) {
      futures[d] = tsl::MakeFutureOn<int>(
          *pool.AsExecutor(),
          [d, &slots, &thunk]() { return RunUpdate(d, slots.data(), &thunk); });
    }
    TF_ASSERT_OK(JoinFutures<int>(futures).Await());
  }
}

}  // namespace
}  // namespace xla::gpu
