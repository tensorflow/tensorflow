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

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/gpu/gpu_helpers.h"
#include "xla/pjrt/gpu/gpu_topology.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/statusor.h"
#include "tsl/platform/fingerprint.h"

namespace stream_executor {

class MultiDeviceAdapter;

}

namespace xla {

class StreamExecutorGpuTopologyDescription : public PjRtTopologyDescription {
 public:
  static StreamExecutorGpuTopologyDescription Create(
      const PjRtPlatformId platform_id, const absl::string_view platform_name,
      const absl::string_view platform_version,
      const std::vector<PjRtDevice*>& devices) {
    std::vector<int> device_ids;
    device_ids.reserve(devices.size());
    for (PjRtDevice* device : devices) {
      device_ids.push_back(device->id());
    }
    return StreamExecutorGpuTopologyDescription(platform_id, platform_name,
                                                platform_version, device_ids);
  }
  // `gpu_device_ids` is the list of logical device ids for the GPU devices and
  // will be used to initialize the GPU topology.
  StreamExecutorGpuTopologyDescription(
      const PjRtPlatformId platform_id, const absl::string_view platform_name,
      const absl::string_view platform_version,
      const std::vector<int>& gpu_device_ids,
      const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& attributes =
          {})
      : platform_id_(platform_id),
        platform_name_(platform_name),
        platform_version_(platform_version),
        gpu_topology_(gpu_device_ids),
        attributes_(attributes) {}

  bool operator==(const StreamExecutorGpuTopologyDescription& other) const {
    return this->platform_id() == other.platform_id() &&
           this->platform_name() == other.platform_name() &&
           this->platform_version() == other.platform_version() &&
           this->gpu_topology().device_ids() ==
               other.gpu_topology().device_ids();
  }

  PjRtPlatformId platform_id() const override { return platform_id_; }

  absl::string_view platform_name() const override { return platform_name_; }

  absl::string_view platform_version() const override {
    return platform_version_;
  }

  std::vector<std::unique_ptr<const PjRtDeviceDescription>> DeviceDescriptions()
      const override {
    std::vector<std::unique_ptr<const PjRtDeviceDescription>> devices;
    devices.reserve(gpu_topology_.number_of_devices());
    for (const int device_id : gpu_topology_.device_ids()) {
      devices.push_back(std::make_unique<PjRtStreamExecutorDeviceDescription>(
          device_id, platform_version_));
    }
    return devices;
  }

  const GpuTopology& gpu_topology() const { return gpu_topology_; }
  const GpuTopology* gpu_topology_ptr() const { return &gpu_topology_; }

  // No subslice is supported.
  bool is_subslice_topology() const override { return false; }

  // The topology support only single host now.
  absl::StatusOr<int> ProcessCount() const override { return 1; }

  absl::StatusOr<int> CoreCountOfDefaultType() const override {
    return gpu_topology_.number_of_devices();
  }

  absl::StatusOr<int> LogicalDeviceCountOfDefaultType() const override {
    return gpu_topology_.number_of_devices();
  }

  absl::StatusOr<int> CoreCountOfDefaultTypePerProcess() const override {
    return gpu_topology_.number_of_devices();
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

  StatusOr<Layout> GetDefaultLayout(
      PrimitiveType element_type,
      absl::Span<const int64_t> dims) const override;

 private:
  const PjRtPlatformId platform_id_;
  const std::string platform_name_;
  const std::string platform_version_;
  const GpuTopology gpu_topology_;
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

  int core_on_chip() const;

 private:
  std::string device_vendor_;
  int slice_index_;
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
      std::unique_ptr<gpu::GpuExecutableRunOptions> gpu_run_options)
      : xla::PjRtStreamExecutorClient(
            platform_name, client, std::move(devices), process_index,
            std::move(allocator), std::move(host_memory_allocator),
            should_stage_host_to_device_transfers, std::move(gpu_run_options)),
        topology_(xla::StreamExecutorGpuTopologyDescription::Create(
            tsl::Fingerprint64(platform_name), platform_name,
            devices_.back()->device_kind(), devices_)) {}

  absl::StatusOr<xla::DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;

  absl::string_view platform_version() const override;

  absl::StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
  CreateBuffersForAsyncHostToDevice(absl::Span<const Shape> shapes,
                                    PjRtDevice* device) override;

  PjRtFuture<> CopyRawSubBufferToHost(PjRtBuffer* buffer, void* dst,
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

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> DeserializeExecutable(
      absl::string_view serialized,
      std::optional<CompileOptions> options) override;

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      const XlaComputation& computation, CompileOptions options) override;

 private:
  xla::StreamExecutorGpuTopologyDescription topology_;
};

std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> BuildLocalDevices(
    std::map<int, std::unique_ptr<LocalDeviceState>> local_device_states,
    int node_id);

std::string MakeComputeCapabilityString(const se::DeviceDescription* desc);

Status BuildDistributedDevices(
    std::string_view platform_name,
    std::map<int, std::unique_ptr<LocalDeviceState>> local_device_states,
    int node_id, int num_nodes,
    std::vector<std::unique_ptr<PjRtStreamExecutorDevice>>* devices,
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
