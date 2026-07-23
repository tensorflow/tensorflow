/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_PJRT_GPU_SE_GPU_PJRT_CLIENT_H_
#define XLA_PJRT_GPU_SE_GPU_PJRT_CLIENT_H_

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/allocator_memory_registration.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_cliques.h"
#include "xla/client/local_client.h"
#include "xla/executable_run_options.h"
#include "xla/future.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/layout.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/distributed/coordination/coordination_service.pb.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/gpu/se_gpu_topology_description.h"
#include "xla/pjrt/host_memory_allocator.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/pjrt_abi_version.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/se/buffer_sequencing_event.h"
#include "xla/pjrt/se/local_device_state.h"
#include "xla/pjrt/se/pjrt_stream_executor_client.h"
#include "xla/pjrt/se/se_raw_buffer.h"
#include "xla/runtime/device_id.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/gpu_topology.pb.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/numa.h"

namespace xla {
using DeviceTopologyPair =
    std::pair<std::vector<std::unique_ptr<PjRtStreamExecutorDevice>>,
              GpuTopologyProto>;

class StreamExecutorGpuDevice : public PjRtStreamExecutorDevice {
 public:
  StreamExecutorGpuDevice(int id,
                          std::unique_ptr<LocalDeviceState> local_device_state,
                          std::string device_kind, std::string device_vendor,
                          std::string compute_capability, int core_count,
                          int64_t device_memory_bytes_limit,
                          int64_t shared_memory_per_block_optin,
                          int local_device_id, int process_index,
                          int process_index_in_partition, int partition_index,
                          int numa_node, std::string fabric_uuid);

  absl::string_view device_vendor() const;

  absl::StatusOr<tsl::AllocatorStats> GetAllocatorStats() const override;

  absl::Span<int const> coords() const;

  absl::StatusOr<PjRtMemorySpace*> default_memory_space() const override;

  absl::Status ClearMemoryStats() override;

 private:
  std::string device_vendor_;
};

class StreamExecutorGpuHbmMemorySpace : public PjRtStreamExecutorMemorySpace {
 public:
  static constexpr absl::string_view kKind = "device";
  static const int kKindId;

  StreamExecutorGpuHbmMemorySpace(int id, PjRtDevice* device);
};

// A custom PjRtClient that overrides the device assignment method.
class StreamExecutorGpuClient : public xla::PjRtStreamExecutorClient {
 public:
  StreamExecutorGpuClient(
      std::string platform_name, LocalClient* client,
      std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices,
      int process_index, std::unique_ptr<se::DeviceAddressAllocator> allocator,
      std::unique_ptr<HostMemoryAllocator> host_memory_allocator,
      bool should_stage_host_to_device_transfers,
      std::unique_ptr<gpu::GpuExecutableRunOptions> gpu_run_options,
      std::shared_ptr<KeyValueStoreInterface> kv_store,
      bool abort_collectives_on_failure,
      std::shared_ptr<xla::StreamExecutorGpuTopologyDescription> topology,
      std::optional<int> num_processes,
      std::shared_ptr<gpu::AllocatorMemoryRegistration> memory_registration =
          nullptr);

  absl::StatusOr<xla::DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;

  absl::string_view platform_version() const override;

  std::optional<PjRtPluginAttributes> plugin_attributes() const override;

  void UpdateGlobalProcessInfo(
      absl::Span<xla::coordination::TaskInfo> infos) override;

  // ScheduleRemoteSend and MakeCrossHostReceiveBuffers are methods implemented
  // to support the legacy cross-host transfers API.
  void ScheduleRemoteSend(PjRtMemorySpace* memory_space,
                          PjRtRawBufferRef raw_buffer,
                          PjRtDeviceEventRefVector definition_events,
                          PjRtDeviceEventPromiseRef usage_event_promise,
                          Future<std::string> serialized_descriptor,
                          PjRtBuffer::RemoteSendCallback on_done) override;

  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
  MakeCrossHostReceiveBuffers(absl::Span<const Shape> shapes,
                              PjRtDevice* device,
                              PjRtCrossHostRecvNotifier notifier) override;

  void RecordMemoryStats();

  absl::StatusOr<PjRtStreamExecutorExecutionOutput> RunAsync(
      LocalExecutable& exec, PjRtDevice* device,
      absl::Span<const PjRtRawBufferRef> flat_arguments,
      absl::Span<const PjRtRawBufferRef> results,
      ExecutableRunOptions run_options_inp, bool parameter_is_tupled_arguments,
      absl::Span<const Shape> executable_parameter_shapes) override;

  absl::Status UpdateCompileOptionsInternal(
      CompileOptions* options, ExecutableExtras* returned_extras,
      bool lookup_addressable_devices) override;

  absl::StatusOr<std::unique_ptr<PjRtRuntimeAbiVersion>> RuntimeAbiVersion()
      const override;

 private:
  const bool abort_collectives_on_failure_ = false;
  std::shared_ptr<gpu::AllocatorMemoryRegistration> memory_registration_;

  absl::StatusOr<PjRtDeviceEventRefVector> CrossHostTransferBuffers(
      PjRtDeviceEventRefVector transfer_dependencies,
      std::vector<CrossHostTransferSpec> transfer_specs) override;

  void ScheduleTransfersOnLocalDevice(
      LocalDeviceState* local_device_state, GlobalDeviceId device_id,
      tsl::AsyncValueRef<BufferSequencingEvent> transfer_event,
      PjRtDeviceEventRefVector transfer_dependencies,
      std::vector<CrossHostTransferSpec> transfer_specs);

  struct PrepareReceiveBufferResult {
    std::unique_ptr<PjRtBuffer> buffer;
    PjRtRawBufferRef raw_buffer;
    LocalDeviceState* local_device;
    se::Stream* stream;
    BufferSequencingEventRef definition_event;
  };

  absl::StatusOr<PrepareReceiveBufferResult> PrepareReceiveBuffer(
      PjRtDevice* device, Shape shape);
};

absl::StatusOr<std::unique_ptr<PjRtClient>> GetStreamExecutorGpuClient(
    const GpuClientOptions& options);

// Constructs a StreamExecutorGpuClient which is intended to be used by
// tensorflow. Don't use this for anything because it has tensorflow specific
// quirks.
absl::StatusOr<std::unique_ptr<PjRtClient>> GetSharedStreamExecutorGpuClient(
    const GpuClientOptions& options, LocalClient* local_client,
    std::map<int, std::unique_ptr<LocalDeviceState>> local_device_states,
    std::unique_ptr<se::DeviceAddressAllocator> allocator,
    std::unique_ptr<HostMemoryAllocator> host_memory_allocator);

// Tensorflow specific API for exchanging an empty topology. Tensorflow
// has some processes which don't have any hardware on them but still exchanges
// topologies for these devices for some reason.
absl::Status ExchangeEmptyStreamExecutorGpuTopology(
    int process_id, int num_nodes,
    std::shared_ptr<KeyValueStoreInterface> kv_store,
    absl::Duration get_local_topology_timeout = absl::Minutes(2),
    absl::Duration get_global_topology_timeout = absl::Minutes(5));

// Creates allocator memory registration and adds the required suballocator
// visitors to `allocator_config`.
std::shared_ptr<gpu::AllocatorMemoryRegistration>
CreateAllocatorMemoryRegistration(GpuAllocatorConfig* allocator_config);

}  // namespace xla

#endif  // XLA_PJRT_GPU_SE_GPU_PJRT_CLIENT_H_
