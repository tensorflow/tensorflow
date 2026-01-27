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

#ifndef XLA_PJRT_GPU_TFRT_UTILS_H_
#define XLA_PJRT_GPU_TFRT_UTILS_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "xla/client/local_client.h"
#include "xla/executable_run_options.h"
#include "xla/future.h"
#include "xla/layout.h"
#include "xla/maybe_owning.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/gpu/se_gpu_topology_description.h"
#include "xla/pjrt/gpu/tfrt/gpu_event.h"
#include "xla/pjrt/gpu/tfrt/tfrt_gpu_buffer.h"
#include "xla/pjrt/gpu/tfrt/tfrt_gpu_device.h"
#include "xla/pjrt/gpu/tfrt/tracked_gpu_device_buffer.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h"
#include "xla/runtime/device_id.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/gpu_topology.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Invokes `f` on the given async work runner and returns the result. Blocks the
// current thread until the work is done.
template <typename F>
std::invoke_result_t<F> RunOnAsyncWorkRunner(AsyncWorkRunner* runner, F&& f) {
  std::invoke_result_t<F> result;
  absl::Notification done;
  runner->Schedule([&]() {
    result = std::forward<F>(f)();
    done.Notify();
  });
  done.WaitForNotification();
  return result;
}

std::unique_ptr<se::Stream> MaybeCreateStream(se::StreamExecutor* executor);

absl::Status WaitForEventOnStream(se::Stream* stream, se::Event* event);

absl::StatusOr<std::shared_ptr<se::Event>> CreateCudaEvent(
    TfrtGpuDevice* device);

Future<> CreateFutureForEvent(tsl::AsyncValueRef<xla::GpuEvent> event);

absl::StatusOr<Shape> GetDestinationDeviceShape(const Shape& host_shape,
                                                TfrtGpuDevice* device,
                                                TfrtGpuClient* client,
                                                PjRtMemorySpace* memory_space);

absl::StatusOr<std::unique_ptr<TfrtGpuBuffer>> AllocateTfrtGpuDestinationBuffer(
    const Shape& on_host_shape, tsl::AsyncValueRef<GpuEvent> definition_event,
    TfrtGpuDevice* device, TfrtGpuClient* client, PjRtMemorySpace* memory_space,
    int64_t pack_size = 0);

bool IsAllZeros(const DeviceAssignment& assignment);

std::vector<tsl::RCReference<tsl::AsyncValue>> CopyAsyncValues(
    absl::Span<const tsl::RCReference<tsl::AsyncValue>> events);

// Checks that the input buffers passed in by the user have the correct size
// on device for the compiled program.
absl::Status CheckBufferCompatibilities(
    absl::Span<int64_t const> input_buffer_sizes_in_bytes,
    absl::Span<TrackedGpuDeviceBuffer* const> input_buffers);

template <typename MemorySpaceKind>
bool IsMemorySpaceKind(const PjRtMemorySpace* memory_space) {
  return memory_space->kind_id() == MemorySpaceKind::kKindId;
}

std::optional<stream_executor::GpuTargetConfigProto> GetTargetConfigForDevices(
    absl::Span<PjRtDevice* const> devices);

absl::flat_hash_map<std::string, PjRtDeviceAttribute> GetAttrsForDevices(
    std::optional<stream_executor::GpuTargetConfigProto> target_config);

template <typename T>
const T* FindCallback(int channel_id, absl::Span<const T> callbacks) {
  // TODO(ezhulenev): Can we use binary search here assuming that callbacks
  // are sorted by channel id? Are they always sorted?
  auto it = absl::c_find_if(callbacks, [&](const T& callback) {
    return callback.channel_id == channel_id;
  });
  return it == callbacks.end() ? nullptr : &*it;
}

// Converts PjRt SendCallbacks to an XLA StreamExecutor send function.
SendDeviceMemoryFunction ConvertSendCallbacksToSendFunction(
    int replica, const ExecuteOptions& options, AsyncWorkRunner* runner);

RecvDeviceMemoryFunction ConvertRecvCallbacksToRecvFunction(
    int replica, const ExecuteOptions& options);

std::vector<PjRtMemorySpace*> GetMemorySpacePointers(
    const std::vector<std::unique_ptr<PjRtMemorySpace>>& memory_spaces);

std::vector<PjRtDevice*> InitializeDevices(
    TfrtGpuClient* client,
    const std::vector<std::unique_ptr<TfrtGpuDevice>>& owned_devices);

absl::flat_hash_map<PjRtGlobalDeviceId, TfrtGpuDevice*> GetIdToDeviceMap(
    absl::Span<const std::unique_ptr<TfrtGpuDevice>> devices);

std::vector<PjRtDevice*> GetAddressableDevicePointers(
    absl::Span<const std::unique_ptr<TfrtGpuDevice>> devices);

StreamExecutorGpuTopologyDescription GetTopology(
    absl::string_view platform_name,
    std::shared_ptr<const GpuTopology> gpu_topology,
    absl::Span<PjRtDevice* const> devices);

std::vector<std::unique_ptr<PjRtMemorySpace>> InitializeMemorySpaces(
    int global_device_count, absl::Span<PjRtDevice* const> addressable_devices);

absl::StatusOr<std::unique_ptr<tsl::Allocator>> CreateAllocatorForDevice(
    se::StreamExecutor* executor, const GpuAllocatorConfig& allocator_config);

absl::StatusOr<MaybeOwning<se::DeviceAddressAllocator>> CreateDeviceAllocator(
    LocalClient* xla_client, const GpuAllocatorConfig& allocator_config,
    const std::vector<std::unique_ptr<TfrtGpuDevice>>& devices);

using DeviceTopologyPair =
    std::pair<std::vector<std::unique_ptr<TfrtGpuDevice>>, GpuTopologyProto>;

absl::StatusOr<DeviceTopologyPair> BuildDistributedDevices(
    absl::string_view platform_name, LocalClient* xla_client, int node_id,
    int num_nodes, int max_inflight_computations,
    gpu::GpuExecutableRunOptions* gpu_executable_run_options,
    std::shared_ptr<KeyValueStoreInterface> kv_store, bool enable_mock_nccl,
    std::optional<absl::string_view> mock_gpu_topology,
    std::optional<int> partition_index,
    absl::Duration get_local_topology_timeout,
    absl::Duration get_global_topology_timeout);

absl::StatusOr<std::vector<absl::string_view>> MemoryKindsFromShape(
    const Shape& shape, absl::string_view default_memory_kind);

absl::flat_hash_map<GlobalDeviceId, IncarnationId> GetLatestIncarnations(
    absl::Span<PjRtDevice* const> devices,
    const absl::flat_hash_map<int, IncarnationId>& incarnations);

absl::Status BlockHostUntilDoneWithHostCallback(se::Stream* stream);

}  // namespace xla

#endif  // XLA_PJRT_GPU_TFRT_UTILS_H_
