/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/pjrt/nvidia_gpu_device.h"

#ifdef NCCL_ENABLED
#include "third_party/nccl/nccl.h"
#endif  // NCCL_ENABLED
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/common_runtime/gpu/gpu_host_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_mem_allocator.h"
#include "tensorflow/stream_executor/tf_allocator_adapter.h"

namespace xla {
namespace {

static const char kGpuPlatformName[] = "gpu";

// A custom PjRtClient that overrides the device assignment method.
class GpuClient : public xla::PjRtClient {
 public:
  using xla::PjRtClient::PjRtClient;

  xla::StatusOr<xla::DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;
};

xla::StatusOr<xla::DeviceAssignment> GpuClient::GetDefaultDeviceAssignment(
    int num_replicas, int num_partitions) const {
  // XLA:GPU does not support multiple partitions yet.
  TF_RET_CHECK(num_partitions == 1) << num_partitions;
  if (num_replicas <= local_devices().size()) {
    xla::DeviceAssignment assignment(num_replicas, 1);
    for (int i = 0; i < num_replicas; ++i) {
      assignment(i, 0) = local_devices().at(i)->id();
    }
    return assignment;
  }
  // Fallback to default global device assignment if we can't run locally.
  return PjRtClient::GetDefaultDeviceAssignment(num_replicas, num_partitions);
}

// Builds an xla::LocalClient for the GPU platform.
StatusOr<LocalClient*> GetGpuXlaClient() {
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      PlatformUtil::GetPlatform("CUDA"));
  if (platform->VisibleDeviceCount() <= 0) {
    return FailedPrecondition("No visible NVidia GPU devices.");
  }
  LocalClientOptions options;
  options.set_platform(platform);
  return ClientLibrary::GetOrCreateLocalClient(options);
}

// Builds a LocalDeviceState for each GPU present.
StatusOr<std::vector<std::unique_ptr<LocalDeviceState>>> BuildLocalDeviceStates(
    LocalClient* xla_client, bool asynchronous) {
  std::vector<std::unique_ptr<LocalDeviceState>> local_devices;
  for (int i = 0; i < xla_client->device_count(); ++i) {
    se::StreamExecutor* executor =
        xla_client->backend().stream_executor(i).ValueOrDie();
    local_devices.push_back(absl::make_unique<LocalDeviceState>(
        executor, xla_client, LocalDeviceState::kComputeSynchronized,
        asynchronous,
        /*allow_event_reuse=*/true));
  }
  return std::move(local_devices);
}

// Builds a BFCAllocator for all local GPUs.
StatusOr<std::unique_ptr<se::MultiDeviceAdapter>> CreateBFCAllocator(
    absl::Span<std::unique_ptr<LocalDeviceState> const> local_devices,
    double memory_fraction, bool preallocate) {
  CHECK_GT(local_devices.size(), 0);
  const se::Platform* platform = local_devices.front()->executor()->platform();
  std::vector<se::MultiDeviceAdapter::AllocatorWithStream> allocators;
  for (auto& local_device : local_devices) {
    se::StreamExecutor* executor = local_device->executor();
    int device_ordinal = executor->device_ordinal();
    auto sub_allocator = absl::make_unique<tensorflow::GPUMemAllocator>(
        executor, tensorflow::PlatformGpuId(device_ordinal),
        /*use_unified_memory=*/false,
        /*alloc_visitors=*/std::vector<tensorflow::SubAllocator::Visitor>(),
        /*free_visitors=*/std::vector<tensorflow::SubAllocator::Visitor>());

    int64 free_memory;
    int64 total_memory;
    if (!executor->DeviceMemoryUsage(&free_memory, &total_memory)) {
      return Unavailable("Failed to query available memory from device %i",
                         device_ordinal);
    }
    size_t allocator_memory = free_memory * memory_fraction;
    if (preallocate) {
      LOG(INFO) << "XLA backend allocating " << allocator_memory
                << " bytes on device " << device_ordinal
                << " for BFCAllocator.";
    } else {
      LOG(INFO) << "XLA backend will use up to " << allocator_memory
                << " bytes on device " << device_ordinal
                << " for BFCAllocator.";
    }
    auto gpu_bfc_allocator = absl::make_unique<tensorflow::BFCAllocator>(
        sub_allocator.release(), allocator_memory,
        /*allow_growth=*/!preallocate,
        absl::StrCat("GPU_", device_ordinal, "_bfc"));
    allocators.emplace_back(std::move(gpu_bfc_allocator),
                            local_device->compute_stream());
  }
  return absl::make_unique<se::MultiDeviceAdapter>(platform,
                                                   std::move(allocators));
}

// Constructs a GPU device memory allocator to use, according to the allocator
// configuration the client requested.
StatusOr<std::unique_ptr<se::DeviceMemoryAllocator>> GetGpuDeviceAllocator(
    const GpuAllocatorConfig& allocator_config,
    absl::Span<std::unique_ptr<LocalDeviceState> const> local_devices) {
  std::unique_ptr<se::DeviceMemoryAllocator> allocator;
  if (allocator_config.kind != GpuAllocatorConfig::Kind::kPlatform) {
    TF_ASSIGN_OR_RETURN(
        allocator,
        CreateBFCAllocator(local_devices, allocator_config.memory_fraction,
                           allocator_config.preallocate));
  }
  return std::move(allocator);
}

// Returns a GPU pinned host memory allocator to use when staging host->GPU
// transfers. We use a fixed 64MB pool of pinned memory.
std::unique_ptr<tensorflow::BFCAllocator> GetGpuHostAllocator(
    se::StreamExecutor* executor) {
  tensorflow::SubAllocator* sub_allocator = new tensorflow::GpuHostAllocator(
      executor, /*numa_node=*/0, /*alloc_visitors=*/{}, /*free_visitors=*/{});
  // TODO(phawkins): allow the user to tune this.
  const int64 kGpuHostMemoryLimitBytes = 64 * (1LL << 30);
  return absl::make_unique<tensorflow::BFCAllocator>(
      sub_allocator, kGpuHostMemoryLimitBytes, /*allow_growth=*/true,
      /*name=*/"xla_gpu_host_bfc");
}

// A table mapping NcclCliqueKeys to ncclUniqueId values encoded as strings.
// In a distributed setup the table of NCCL IDs is kept on the master node
// (node 0). Currently node 0 is the only node that generates ncclUniqueIds;
// see the TODO below.
class NcclIdStore {
 public:
  NcclIdStore(int node_id, std::shared_ptr<DistributedRuntimeClient> client)
      : node_id_(node_id), client_(std::move(client)) {}

  StatusOr<std::string> GetNcclUniqueId(const NcclCliqueKey& key);

 private:
  const int node_id_;
  const std::shared_ptr<DistributedRuntimeClient> client_;

  absl::Mutex mu_;
  absl::flat_hash_map<std::string, std::string> cache_ GUARDED_BY(mu_);
};

StatusOr<std::string> NcclIdStore::GetNcclUniqueId(const NcclCliqueKey& key) {
  std::string key_string = GlobalDeviceIdsToString(key.devices());
  {
    absl::MutexLock lock(&mu_);
    auto it = cache_.find(key_string);
    if (it != cache_.end()) {
      return it->second;
    }
  }
  auto result = [&]() -> StatusOr<std::string> {
    // TODO(phawkins): this will deadlock if node 0 is not involved in the
    // computation. Add support for computations that only use a subset of
    // replicas.
    if (node_id_ == 0) {
#ifdef NCCL_ENABLED
      ncclUniqueId id;
      ncclResult_t r = ncclGetUniqueId(&id);
      TF_RET_CHECK(r == ncclSuccess);
      std::string value(id.internal, NCCL_UNIQUE_ID_BYTES);
      TF_RETURN_IF_ERROR(client_->KeyValueSet(key_string, value));
      return value;
#else
      return FailedPrecondition("NCCL support was not built into XLA binary.");
#endif
    } else {
      return client_->BlockingKeyValueGet(key_string, absl::Minutes(5));
    }
  }();
  if (!result.ok()) {
    return result.status();
  }
  absl::MutexLock lock(&mu_);
  return cache_.emplace(key_string, result.ValueOrDie()).first->second;
}

std::vector<std::unique_ptr<Device>> BuildLocalDevices(
    std::vector<std::unique_ptr<LocalDeviceState>> local_device_states) {
  std::vector<std::unique_ptr<Device>> devices;
  for (auto& local_device : local_device_states) {
    int device_ordinal = local_device->device_ordinal();
    const se::DeviceDescription& description =
        local_device->executor()->GetDeviceDescription();
    auto device = absl::make_unique<GpuDevice>(
        device_ordinal, std::move(local_device), description.name(),
        /*node_id=*/0);
    devices.push_back(std::move(device));
  }
  return devices;
}

Status BuildDistributedDevices(
    std::vector<std::unique_ptr<LocalDeviceState>> local_device_states,
    std::shared_ptr<DistributedRuntimeClient> distributed_client, int node_id,
    std::vector<std::unique_ptr<Device>>* devices,
    GpuExecutableRunOptions* gpu_executable_run_options) {
  LocalTopologyProto local_topology;
  local_topology.set_node_id(node_id);
  for (const auto& local_device : local_device_states) {
    const se::Platform* platform = local_device->executor()->platform();
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<xla::se::DeviceDescription> desc,
        platform->DescriptionForDevice(local_device->device_ordinal()));
    TF_RET_CHECK(local_device->device_ordinal() ==
                 local_topology.devices_size());
    DeviceProto* device_proto = local_topology.add_devices();
    device_proto->set_local_device_ordinal(local_device->device_ordinal());
    device_proto->set_name(desc->name());
    device_proto->set_vendor(desc->device_vendor());
  }

  GlobalTopologyProto global_topology;
  TF_RETURN_IF_ERROR(
      distributed_client->Connect(local_topology, &global_topology));

  std::vector<GlobalDeviceId> gpu_device_ids(local_device_states.size());
  for (const LocalTopologyProto& node : global_topology.nodes()) {
    for (const DeviceProto& device_proto : node.devices()) {
      std::unique_ptr<LocalDeviceState> local_device;
      if (node.node_id() == node_id) {
        TF_RET_CHECK(device_proto.local_device_ordinal() >= 0 &&
                     device_proto.local_device_ordinal() <
                         local_device_states.size());
        TF_RET_CHECK(local_device_states[device_proto.local_device_ordinal()] !=
                     nullptr);
        local_device =
            std::move(local_device_states[device_proto.local_device_ordinal()]);
        gpu_device_ids[device_proto.local_device_ordinal()] =
            GlobalDeviceId(device_proto.global_device_id());
      }
      auto device = absl::make_unique<GpuDevice>(
          device_proto.global_device_id(), std::move(local_device),
          device_proto.name(), node.node_id());
      devices->push_back(std::move(device));
    }
  }
  for (const auto& device : local_device_states) {
    TF_RET_CHECK(device == nullptr);
  }
  gpu_executable_run_options->set_gpu_global_device_ids(
      std::move(gpu_device_ids));
  auto nccl_id_store =
      std::make_shared<NcclIdStore>(node_id, distributed_client);
  gpu_executable_run_options->set_nccl_unique_id_callback(
      [nccl_id_store](const NcclCliqueKey& key) {
        return nccl_id_store->GetNcclUniqueId(key);
      });
  return Status::OK();
}

}  // namespace

GpuDevice::GpuDevice(int id,
                     std::unique_ptr<LocalDeviceState> local_device_state,
                     std::string device_kind, int node_id)
    : Device(id, std::move(local_device_state), kGpuPlatformName,
             std::move(device_kind), node_id) {}

StatusOr<std::shared_ptr<PjRtClient>> GetNvidiaGpuClient(
    bool asynchronous, const GpuAllocatorConfig& allocator_config,
    std::shared_ptr<DistributedRuntimeClient> distributed_client, int node_id) {
  TF_ASSIGN_OR_RETURN(LocalClient * xla_client, GetGpuXlaClient());
  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<LocalDeviceState>> local_device_states,
      BuildLocalDeviceStates(xla_client, asynchronous));
  TF_ASSIGN_OR_RETURN(
      auto allocator,
      GetGpuDeviceAllocator(allocator_config, local_device_states));
  auto host_memory_allocator =
      GetGpuHostAllocator(local_device_states.front()->executor());

  std::vector<std::unique_ptr<Device>> devices;
  auto gpu_run_options = absl::make_unique<GpuExecutableRunOptions>();
  if (distributed_client) {
    TF_RETURN_IF_ERROR(BuildDistributedDevices(
        std::move(local_device_states), std::move(distributed_client), node_id,
        &devices, gpu_run_options.get()));
  } else {
    devices = BuildLocalDevices(std::move(local_device_states));
  }

  std::shared_ptr<PjRtClient> pyclient = std::make_shared<GpuClient>(
      "gpu", xla_client, std::move(devices),
      /*node_id=*/node_id, std::move(allocator),
      std::move(host_memory_allocator),
      /*gpu_run_options=*/std::move(gpu_run_options));
  return pyclient;
}

}  // namespace xla
