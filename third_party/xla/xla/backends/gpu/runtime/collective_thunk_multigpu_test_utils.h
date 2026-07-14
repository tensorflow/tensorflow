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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_THUNK_MULTIGPU_TEST_UTILS_H_
#define XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_THUNK_MULTIGPU_TEST_UTILS_H_

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/future.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla::gpu {

struct CollectiveThunkMultiGpuTestState {
  int device_ordinal = 0;
  se::StreamExecutor* executor = nullptr;
  std::unique_ptr<se::Stream> stream;
  std::unique_ptr<se::StreamExecutorAddressAllocator> allocator;

  std::vector<se::DeviceAddressBase> create_buffers;
  std::vector<se::DeviceAddressBase> update_buffers;

  GpuExecutableRunOptions gpu_run_options;
  ServiceExecutableRunOptions run_options;
  std::optional<CollectiveParams> collective_params;
  CollectiveCliques collective_cliques;

  CommandStateManager state_manager;
  std::unique_ptr<se::CommandBuffer> command_buffer;
  const se::CommandBuffer::Command* command = nullptr;
};

se::StreamExecutor* GetGpuExecutor(int ordinal);

bool IsAtLeastCuda12900(const se::StreamExecutor* executor);

bool HasEnoughGpus(int num_devices);

DeviceAssignment MakeDeviceAssignment(int num_devices);

absl::Status SetupCollectiveThunkDevice(
    int device_ordinal, int num_devices, absl::Span<const int64_t> buffer_sizes,
    CollectiveThunk& thunk, const DeviceAssignment& device_assignment,
    CollectiveThunkMultiGpuTestState& state);

absl::Status SetupCollectiveThunksDevice(
    int device_ordinal, int num_devices, absl::Span<const int64_t> buffer_sizes,
    std::initializer_list<CollectiveThunk*> thunks,
    const DeviceAssignment& device_assignment,
    CollectiveThunkMultiGpuTestState& state);

BufferAllocations MakeBufferAllocations(
    const CollectiveThunkMultiGpuTestState& state,
    absl::Span<const se::DeviceAddressBase> buffers);

Thunk::ExecuteParams MakeExecuteParams(
    CollectiveThunkMultiGpuTestState& state,
    const BufferAllocations& buffer_allocations);

absl::Status ExecuteOnStreamAndBlock(CollectiveThunk& thunk,
                                     const Thunk::ExecuteParams& params);

absl::Status RecordCommandBufferCreate(
    CollectiveThunkMultiGpuTestState& state, CollectiveThunk& thunk,
    const Thunk::ExecuteParams& execute_params);

absl::Status RecordCommandBufferUpdate(
    CollectiveThunkMultiGpuTestState& state, CollectiveThunk& thunk,
    const Thunk::ExecuteParams& execute_params,
    std::vector<BufferAllocation::Index> updated_allocations);

absl::Status SubmitCommandBuffer(CollectiveThunkMultiGpuTestState& state);

absl::Status FillDeviceBuffer(se::Stream& stream, se::DeviceAddressBase buffer,
                              absl::Span<const float> data);

absl::StatusOr<std::vector<float>> ReadDeviceBuffer(
    se::Stream& stream, se::DeviceAddressBase buffer, int64_t length);

absl::Status VerifyDeviceBuffer(se::Stream& stream,
                                se::DeviceAddressBase buffer,
                                absl::Span<const float> expected_values);

std::vector<float> SentinelValues(int64_t length);

template <typename Fn>
absl::Status RunOnDevices(int num_devices, absl::string_view pool_name, Fn fn) {
  tsl::thread::ThreadPool pool(tsl::Env::Default(),
                               std::string(pool_name.data(), pool_name.size()),
                               num_devices);
  std::vector<Future<int>> futures(num_devices);
  for (int d = 0; d < num_devices; ++d) {
    futures[d] = MakeFutureOn<int>(*pool.AsExecutor(),
                                   [d, &fn]() -> absl::StatusOr<int> {
                                     absl::Status status = fn(d);
                                     if (!status.ok()) {
                                       return status;
                                     }
                                     return d;
                                   });
  }
  return JoinFutures<int>(futures).Await().status();
}

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_THUNK_MULTIGPU_TEST_UTILS_H_
