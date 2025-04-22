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

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/client/local_client.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/layout.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/gpu/gpu_topology.h"
#include "xla/pjrt/gpu/gpu_topology.pb.h"
#include "xla/pjrt/gpu/se_gpu_topology_description.h"
#include "xla/pjrt/local_device_state.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/tsl/framework/allocator.h"
#include "tsl/platform/casts.h"

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
                          int node_id, int slice_index = 0);

  int slice_index() const;

  absl::string_view device_vendor() const;

  absl::StatusOr<tsl::AllocatorStats> GetAllocatorStats() const override;

  absl::Span<int const> coords() const;

  absl::StatusOr<PjRtMemorySpace*> default_memory_space() const override;

 private:
  std::string device_vendor_;
  int slice_index_;
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
  using xla::PjRtStreamExecutorClient::PjRtStreamExecutorClient;

  StreamExecutorGpuClient(
      std::string platform_name, LocalClient* client,
      std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices,
      int process_index, std::unique_ptr<se::DeviceMemoryAllocator> allocator,
      std::unique_ptr<tsl::Allocator> host_memory_allocator,
      bool should_stage_host_to_device_transfers,
      std::unique_ptr<gpu::GpuExecutableRunOptions> gpu_run_options,
      std::shared_ptr<KeyValueStoreInterface> kv_store,
      std::shared_ptr<const GpuTopology> gpu_topology);

  std::optional<std::shared_ptr<KeyValueStoreInterface>> key_value_store()
      const override {
    return kv_store_;
  }

  absl::StatusOr<xla::DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;

  absl::string_view platform_version() const override;

  using PjRtStreamExecutorClient::CreateBuffersForAsyncHostToDevice;
  absl::StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
  CreateBuffersForAsyncHostToDevice(
      absl::Span<const PjRtClient::ShapeSpec> shape_specs,
      std::optional<absl::Span<const std::optional<Layout>>> device_layouts,
      PjRtMemorySpace* memory_space) override;

  PjRtFuture<> CopyRawSubBufferToHost(PjRtBuffer* buffer, PjRtFuture<void*> dst,
                                      int64_t offset,
                                      int64_t transfer_size) override;

  PjRtFuture<> CopyRawHostToDevice(
      LocalDeviceState* local_device,
      tsl::RCReference<RawSEDeviceMemory> device_buffer, const void* src,
      int64_t offset, int64_t transfer_size) override;

  PjRtFuture<> CopyRawDeviceToHost(
      LocalDeviceState* local_device,
      tsl::RCReference<RawSEDeviceMemory> device_buffer, void* dst,
      int64_t offset, int64_t transfer_size) override;

  absl::StatusOr<const xla::PjRtTopologyDescription*> GetTopologyDescription()
      const override {
    return &topology_;
  }

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> LoadSerialized(
      absl::string_view serialized, std::optional<CompileOptions> options,
      const LoadOptions& load_options);

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAndLoad(
      const XlaComputation& computation, CompileOptions options) override;

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAndLoad(
      mlir::ModuleOp module, CompileOptions options) override;

  absl::StatusOr<PjRtStreamExecutorExecutionOutput> RunAsync(
      LocalExecutable& exec, PjRtDevice* device,
      std::vector<ShapeTree<PjRtStreamExecutorExecutionInput>> arguments,
      ExecutableRunOptions run_options) override;

 private:
  xla::StreamExecutorGpuTopologyDescription topology_;
  std::shared_ptr<KeyValueStoreInterface> kv_store_;
};

std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> BuildLocalDevices(
    std::map<int, std::unique_ptr<LocalDeviceState>> local_device_states,
    int node_id);

std::string MakeComputeCapabilityString(const se::DeviceDescription* desc);

absl::StatusOr<DeviceTopologyPair> BuildDistributedDevices(
    absl::string_view platform_name,
    std::map<int, std::unique_ptr<LocalDeviceState>> local_device_states,
    int node_id, int num_nodes,
    gpu::GpuExecutableRunOptions* gpu_executable_run_options,
    std::shared_ptr<KeyValueStoreInterface> kv_store, bool enable_mock_nccl,
    std::optional<absl::string_view> mock_gpu_topology = std::nullopt,
    std::optional<int> slice_index = std::nullopt,
    absl::Duration get_local_topology_timeout = absl::Minutes(2),
    absl::Duration get_global_topology_timeout = absl::Minutes(5));

absl::StatusOr<std::unique_ptr<PjRtClient>> GetStreamExecutorGpuClient(
    const GpuClientOptions& options);

// Get the fabric info of a local device ordinal in the format of
// "clusterUuid/cliqueId". Empty on SM90 or lower.
absl::StatusOr<std::string> GetDeviceFabricInfo(int device_ordinal);

}  // namespace xla

#endif  // XLA_PJRT_GPU_SE_GPU_PJRT_CLIENT_H_
