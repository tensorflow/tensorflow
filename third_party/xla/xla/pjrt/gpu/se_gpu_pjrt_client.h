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
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/client/local_client.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/layout.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/gpu/gpu_helpers.h"
#include "xla/pjrt/gpu/gpu_topology.h"
#include "xla/pjrt/gpu/gpu_topology.pb.h"
#include "xla/pjrt/local_device_state.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/tsl/framework/allocator.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/fingerprint.h"

namespace stream_executor {

class MultiDeviceAdapter;

}

namespace xla {
using DeviceTopologyPair =
    std::pair<std::vector<std::unique_ptr<PjRtStreamExecutorDevice>>,
              GpuTopologyProto>;

class StreamExecutorGpuTopologyDescription : public PjRtTopologyDescription {
 public:
  static StreamExecutorGpuTopologyDescription Create(
      const PjRtPlatformId platform_id, const absl::string_view platform_name,
      std::shared_ptr<const GpuTopology> gpu_topology) {
    return StreamExecutorGpuTopologyDescription(platform_id, platform_name,
                                                gpu_topology);
  }

  StreamExecutorGpuTopologyDescription(
      const PjRtPlatformId platform_id, const absl::string_view platform_name,
      std::shared_ptr<const GpuTopology> gpu_topology,
      const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& attributes =
          {})
      : platform_id_(platform_id),
        platform_name_(platform_name),
        gpu_topology_(std::move(gpu_topology)),
        attributes_(attributes) {}

  bool operator==(const StreamExecutorGpuTopologyDescription& other) const {
    return this->platform_id() == other.platform_id() &&
           this->platform_name() == other.platform_name() &&
           this->platform_version() == other.platform_version() &&
           this->gpu_topology() == other.gpu_topology();
  }

  PjRtPlatformId platform_id() const override { return platform_id_; }

  absl::string_view platform_name() const override { return platform_name_; }

  absl::string_view platform_version() const override {
    return gpu_topology_->platform_version();
  }

  std::vector<std::unique_ptr<const PjRtDeviceDescription>> DeviceDescriptions()
      const override {
    std::vector<std::unique_ptr<const PjRtDeviceDescription>> devices;
    devices.reserve(gpu_topology_->number_of_devices());
    for (const int device_id : gpu_topology_->device_ids()) {
      devices.push_back(std::make_unique<PjRtStreamExecutorDeviceDescription>(
          device_id, std::string(platform_version())));
    }
    return devices;
  }

  const GpuTopology& gpu_topology() const { return *gpu_topology_; }
  const GpuTopology* gpu_topology_ptr() const { return gpu_topology_.get(); }

  // No subslice is supported.
  bool is_subslice_topology() const override { return false; }

  absl::StatusOr<int> ProcessCount() const override {
    return gpu_topology_->number_of_hosts();
  }

  absl::StatusOr<int> CoreCountOfDefaultType() const override {
    return gpu_topology_->number_of_devices();
  }

  absl::StatusOr<int> LogicalDeviceCountOfDefaultType() const override {
    return gpu_topology_->number_of_devices();
  }

  absl::StatusOr<int> CoreCountOfDefaultTypePerProcess() const override {
    return gpu_topology_->number_of_devices();
  }

  absl::StatusOr<int> CoreCountOfDefaultTypePerChip() const override {
    return 1;
  }

  absl::StatusOr<std::string> Serialize() const override;

  // Returns vendor specific attributes about the topology.
  const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& Attributes()
      const override {
    return attributes_;
  }

  absl::StatusOr<Layout> GetDefaultLayout(
      PrimitiveType element_type,
      absl::Span<const int64_t> dims) const override;

 private:
  const PjRtPlatformId platform_id_;
  const std::string platform_name_;
  std::shared_ptr<const GpuTopology> gpu_topology_;
  absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute> attributes_;
};

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

  absl::StatusOr<xla::DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;

  absl::string_view platform_version() const override;
  absl::StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
  CreateBuffersForAsyncHostToDevice(
      absl::Span<const PjRtClient::ShapeSpec> shape_specs,
      std::optional<absl::Span<const Layout>> device_layouts,
      PjRtDevice* device) override;

  absl::StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
  CreateBuffersForAsyncHostToDevice(absl::Span<const Shape> shapes,
                                    PjRtDevice* device) override;

  absl::StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
  CreateBuffersForAsyncHostToDevice(
      absl::Span<const PjRtClient::ShapeSpec> shape_specs,
      std::optional<absl::Span<const Layout>> device_layouts,
      PjRtMemorySpace* memory_space) override;

  absl::StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
  CreateBuffersForAsyncHostToDevice(absl::Span<const Shape> shapes,
                                    PjRtMemorySpace* memory_space) override;

  PjRtFuture<> CopyRawSubBufferToHost(PjRtBuffer* buffer, PjRtFuture<void*> dst,
                                      int64_t offset,
                                      int64_t transfer_size) override;

  absl::StatusOr<const xla::PjRtTopologyDescription*> GetTopologyDescription()
      const override {
    return &topology_;
  }

  // TODO(b/285385306): Enable loading a non-loaded PjRtExecutable.
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Load(
      std::unique_ptr<PjRtExecutable> executable,
      const LoadOptions& load_options) override {
    return absl::WrapUnique<PjRtLoadedExecutable>(
        tensorflow::down_cast<PjRtLoadedExecutable*>(executable.release()));
  }

  // TODO(b/296466237): Unify `Load` method after (de)serialization and tests on
  // existing use cases are done.
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Load(
      std::unique_ptr<PjRtExecutable> executable);

  // TODO(b/296466237): Unify `LoadSerializedExecutable` after fixing existing
  // tests.
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> LoadSerialized(
      absl::string_view serialized, std::optional<CompileOptions> options,
      const LoadOptions& load_options);

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      const XlaComputation& computation, CompileOptions options) override;

 private:
  xla::StreamExecutorGpuTopologyDescription topology_;
  std::shared_ptr<KeyValueStoreInterface> kv_store_;
};

std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> BuildLocalDevices(
    std::map<int, std::unique_ptr<LocalDeviceState>> local_device_states,
    int node_id);

std::string MakeComputeCapabilityString(const se::DeviceDescription* desc);

absl::StatusOr<DeviceTopologyPair> BuildDistributedDevices(
    std::string_view platform_name,
    std::map<int, std::unique_ptr<LocalDeviceState>> local_device_states,
    int node_id, int num_nodes,
    gpu::GpuExecutableRunOptions* gpu_executable_run_options,
    std::shared_ptr<KeyValueStoreInterface> kv_store, bool enable_mock_nccl,
    absl::Duration get_local_topology_timeout = absl::Minutes(2),
    absl::Duration get_global_topology_timeout = absl::Minutes(5));

struct GpuClientOptions {
  GpuAllocatorConfig allocator_config;

  int node_id = 0;

  int num_nodes = 1;

  std::optional<std::set<int>> allowed_devices = std::nullopt;

  std::optional<std::string> platform_name = std::nullopt;

  bool should_stage_host_to_device_transfers = true;

  // kv_store must be non-null if num_nodes > 1.
  std::shared_ptr<KeyValueStoreInterface> kv_store = nullptr;

  bool enable_mock_nccl = false;
};

absl::StatusOr<std::unique_ptr<PjRtClient>> GetStreamExecutorGpuClient(
    const GpuClientOptions& options);

}  // namespace xla

#endif  // XLA_PJRT_GPU_SE_GPU_PJRT_CLIENT_H_
