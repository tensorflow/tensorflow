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

#include "tensorflow/compiler/xla/pjrt/gpu_device.h"

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h"

#ifdef NCCL_ENABLED
#include "third_party/nccl/nccl.h"
#endif  // NCCL_ENABLED
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/common_runtime/device/device_host_allocator.h"
#include "tensorflow/core/common_runtime/device/device_id.h"
#include "tensorflow/core/common_runtime/device/device_mem_allocator.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/stream_executor/tf_allocator_adapter.h"

namespace xla {
namespace {

// A custom PjRtClient that overrides the device assignment method.
class GpuClient : public xla::PjRtStreamExecutorClient {
 public:
  using xla::PjRtStreamExecutorClient::PjRtStreamExecutorClient;

  xla::StatusOr<xla::DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;
};

xla::StatusOr<xla::DeviceAssignment> GpuClient::GetDefaultDeviceAssignment(
    int num_replicas, int num_partitions) const {
  if (num_partitions == 1 && num_replicas <= addressable_devices().size()) {
    xla::DeviceAssignment assignment(num_replicas, 1);
    for (int i = 0; i < num_replicas; ++i) {
      assignment(i, 0) = addressable_devices().at(i)->id();
    }
    return assignment;
  }
  // Fallback to default global device assignment if we can't run locally.
  return PjRtStreamExecutorClient::GetDefaultDeviceAssignment(num_replicas,
                                                              num_partitions);
}

// Builds an xla::LocalClient for the GPU platform.
StatusOr<LocalClient*> GetGpuXlaClient() {
  // "gpu" will be substitued by the default defined in platform_util.cc
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      PlatformUtil::GetPlatform("gpu"));
  if (platform->VisibleDeviceCount() <= 0) {
    return FailedPrecondition("No visible GPU devices.");
  }
  LocalClientOptions options;
  options.set_platform(platform);
  return ClientLibrary::GetOrCreateLocalClient(options);
}

// Builds a LocalDeviceState for each GPU present.
StatusOr<std::vector<std::unique_ptr<LocalDeviceState>>> BuildLocalDeviceStates(
    LocalClient* xla_client, bool asynchronous) {
  std::vector<std::unique_ptr<LocalDeviceState>> addressable_devices;
  for (int i = 0; i < xla_client->device_count(); ++i) {
    se::StreamExecutor* executor =
        xla_client->backend().stream_executor(i).ValueOrDie();
    addressable_devices.push_back(absl::make_unique<LocalDeviceState>(
        executor, xla_client, LocalDeviceState::kComputeSynchronized,
        asynchronous,
        /*allow_event_reuse=*/true));
  }
  return std::move(addressable_devices);
}

// Builds a BFCAllocator for all local GPUs.
StatusOr<std::unique_ptr<se::MultiDeviceAdapter>> CreateBFCAllocator(
    absl::Span<std::unique_ptr<LocalDeviceState> const> addressable_devices,
    double memory_fraction, bool preallocate) {
  CHECK_GT(addressable_devices.size(), 0);
  const se::Platform* platform =
      addressable_devices.front()->executor()->platform();
  std::vector<se::MultiDeviceAdapter::AllocatorWithStream> allocators;
  bool enable_unified_memory;
  Status status = tensorflow::ReadBoolFromEnvVar("TF_FORCE_UNIFIED_MEMORY",
                                                 false, &enable_unified_memory);
  if (!status.ok()) {
    LOG(ERROR) << "Unable to read TF_FORCE_UNIFIED_MEMORY: "
               << status.error_message();
  }

  for (auto& local_device : addressable_devices) {
    se::StreamExecutor* executor = local_device->executor();
    int device_ordinal = executor->device_ordinal();
    auto sub_allocator = absl::make_unique<tensorflow::DeviceMemAllocator>(
        executor, tensorflow::PlatformDeviceId(device_ordinal),
        /*use_unified_memory=*/enable_unified_memory,
        /*alloc_visitors=*/std::vector<tensorflow::SubAllocator::Visitor>(),
        /*free_visitors=*/std::vector<tensorflow::SubAllocator::Visitor>());

    int64 free_memory;
    int64 total_memory;
    if (!executor->DeviceMemoryUsage(&free_memory, &total_memory)) {
      return Unavailable("Failed to query available memory from device %i",
                         device_ordinal);
    }
    // To allow full GPU memory to be visible to the BFC allocator if using
    // unified memory.
    size_t allocator_memory =
        enable_unified_memory ? total_memory : free_memory * memory_fraction;
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
    absl::Span<std::unique_ptr<LocalDeviceState> const> addressable_devices) {
  std::unique_ptr<se::DeviceMemoryAllocator> allocator;
  if (allocator_config.kind != GpuAllocatorConfig::Kind::kPlatform) {
    TF_ASSIGN_OR_RETURN(allocator,
                        CreateBFCAllocator(addressable_devices,
                                           allocator_config.memory_fraction,
                                           allocator_config.preallocate));
  }
  return std::move(allocator);
}

// Returns a GPU pinned host memory allocator to use when staging host->GPU
// transfers. We use a fixed 64MB pool of pinned memory.
std::unique_ptr<tensorflow::BFCAllocator> GetGpuHostAllocator(
    se::StreamExecutor* executor) {
  tensorflow::SubAllocator* sub_allocator = new tensorflow::DeviceHostAllocator(
      executor, /*numa_node=*/0, /*alloc_visitors=*/{}, /*free_visitors=*/{});
  // TODO(phawkins): allow the user to tune this.
  const int64 kGpuHostMemoryLimitBytes = 64 * (1LL << 30);
  return absl::make_unique<tensorflow::BFCAllocator>(
      sub_allocator, kGpuHostMemoryLimitBytes, /*allow_growth=*/true,
      /*name=*/"xla_gpu_host_bfc");
}

// A table mapping NcclCliqueKeys to ncclUniqueId values encoded as strings.
// In a distributed setup the table of NCCL IDs is kept on the master node
// (node 0). The node of the first participating device will create the unique
// id.
class NcclIdStore {
 public:
  NcclIdStore(int node_id, std::shared_ptr<DistributedRuntimeClient> client,
              absl::flat_hash_map<GlobalDeviceId, int> device_to_node)
      : node_id_(node_id),
        client_(std::move(client)),
        device_to_node_(std::move(device_to_node)) {}

  StatusOr<std::string> GetNcclUniqueId(const gpu::NcclCliqueKey& key);

 private:
  const int node_id_;
  const std::shared_ptr<DistributedRuntimeClient> client_;
  const absl::flat_hash_map<GlobalDeviceId, int> device_to_node_;

  absl::Mutex mu_;
  absl::flat_hash_map<gpu::NcclCliqueKey, std::string> cache_
      ABSL_GUARDED_BY(mu_);
};

StatusOr<std::string> NcclIdStore::GetNcclUniqueId(
    const gpu::NcclCliqueKey& key) {
  // The caller must ensure that threads calling this method concurrently have
  // unique keys, otherwise the global key-value store may hold the wrong value.
  {
    absl::MutexLock lock(&mu_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      return it->second;
    }
  }
  std::string id_string;
  int primary_node_id = device_to_node_.at(key.devices()[0]);
  if (node_id_ == primary_node_id) {
#ifdef NCCL_ENABLED
    ncclUniqueId id;
    ncclResult_t r = ncclGetUniqueId(&id);
    TF_RET_CHECK(r == ncclSuccess);
    id_string = std::string(id.internal, NCCL_UNIQUE_ID_BYTES);
    TF_RETURN_IF_ERROR(client_->KeyValueSet(key.ToString(), id_string));
#else
    return FailedPrecondition("NCCL support was not built into XLA binary.");
#endif
  } else {
    TF_ASSIGN_OR_RETURN(id_string, client_->BlockingKeyValueGet(
                                       key.ToString(), absl::Minutes(5)));
  }
  absl::MutexLock lock(&mu_);
  auto result = cache_.emplace(key, std::move(id_string));
  TF_RET_CHECK(result.second) << "Unique ID already in cache.";
  return result.first->second;
}

std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> BuildLocalDevices(
    std::vector<std::unique_ptr<LocalDeviceState>> local_device_states) {
  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
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
    std::vector<std::unique_ptr<PjRtStreamExecutorDevice>>* devices,
    gpu::GpuExecutableRunOptions* gpu_executable_run_options) {
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
      distributed_client->EnumerateDevices(local_topology, &global_topology));

  std::vector<GlobalDeviceId> gpu_device_ids(local_device_states.size());
  absl::flat_hash_map<GlobalDeviceId, int> device_to_node;
  for (const LocalTopologyProto& node : global_topology.nodes()) {
    for (const DeviceProto& device_proto : node.devices()) {
      GlobalDeviceId global_device_id(device_proto.global_device_id());
      device_to_node[global_device_id] = node.node_id();
      std::unique_ptr<LocalDeviceState> local_device;
      if (node.node_id() == node_id) {
        TF_RET_CHECK(device_proto.local_device_ordinal() >= 0 &&
                     device_proto.local_device_ordinal() <
                         local_device_states.size());
        TF_RET_CHECK(local_device_states[device_proto.local_device_ordinal()] !=
                     nullptr);
        local_device =
            std::move(local_device_states[device_proto.local_device_ordinal()]);
        gpu_device_ids[device_proto.local_device_ordinal()] = global_device_id;
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
  auto nccl_id_store = std::make_shared<NcclIdStore>(
      node_id, distributed_client, device_to_node);
  gpu_executable_run_options->set_nccl_unique_id_callback(
      [nccl_id_store](const gpu::NcclCliqueKey& key) {
        return nccl_id_store->GetNcclUniqueId(key);
      });
  return Status::OK();
}

}  // namespace

GpuDevice::GpuDevice(int id,
                     std::unique_ptr<LocalDeviceState> local_device_state,
                     std::string device_kind, int node_id)
    : PjRtStreamExecutorDevice(id, std::move(local_device_state),
                               std::move(device_kind), node_id) {}

StatusOr<std::unique_ptr<PjRtClient>> GetGpuClient(
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

  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
  auto gpu_run_options = absl::make_unique<gpu::GpuExecutableRunOptions>();
  if (distributed_client) {
    TF_RETURN_IF_ERROR(BuildDistributedDevices(
        std::move(local_device_states), std::move(distributed_client), node_id,
        &devices, gpu_run_options.get()));
  } else {
    devices = BuildLocalDevices(std::move(local_device_states));
  }

  return std::unique_ptr<PjRtClient>(std::make_unique<GpuClient>(
      kGpuName, xla_client, std::move(devices),
      /*node_id=*/node_id, std::move(allocator),
      std::move(host_memory_allocator),
      /*should_stage_host_to_device_transfers=*/true,
      /*gpu_run_options=*/std::move(gpu_run_options)));
}

}  // namespace xla
