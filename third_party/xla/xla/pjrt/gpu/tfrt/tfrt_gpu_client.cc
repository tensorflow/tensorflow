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

#include "xla/pjrt/gpu/tfrt/tfrt_gpu_client.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "xla/client/local_client.h"
#include "xla/pjrt/gpu/gpu_helpers.h"
#include "xla/pjrt/gpu/gpu_topology.h"
#include "xla/pjrt/gpu/gpu_topology.pb.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {

static absl::StatusOr<std::vector<std::unique_ptr<TfrtGpuDevice>>>
GetTfrtGpuDevices(LocalClient* xla_client) {
  std::vector<std::unique_ptr<TfrtGpuDevice>> devices;
  int i = 0;
  for (se::StreamExecutor* executor :
       xla_client->backend().stream_executors()) {
    // TODO(b/382117736): allow GPU allocator parameters to be configurable.
    TF_ASSIGN_OR_RETURN(auto allocator,
                        CreateBFCAllocator(executor, /*memory_fraction=*/0.9,
                                           /*preallocate=*/true, std::nullopt));

    TfrtGpuDevice::Options options;
    options.id = i;
    options.local_device_id = PjRtLocalDeviceId(i);
    options.local_hardware_id = PjRtLocalHardwareId(i);
    options.executor = executor;
    options.allocator = std::move(allocator);
    options.stream_capacity = 4;
    options.max_inflight_computations = 1;
    const se::Platform* platform = executor->GetPlatform();
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<xla::se::DeviceDescription> desc,
        platform->DescriptionForDevice(options.local_hardware_id.value()));
    options.platform_version = desc->name();

    auto device = std::make_unique<TfrtGpuDevice>(std::move(options));
    devices.push_back(std::move(device));
    ++i;
  }
  return std::move(devices);
}

absl::StatusOr<std::unique_ptr<PjRtClient>> GetTfrtGpuClient(
    TfrtGpuClient::Options options) {
  TF_ASSIGN_OR_RETURN(
      LocalClient * xla_client,
      GetGpuXlaClient(options.platform_name, options.allowed_devices));
  EnablePeerAccess(xla_client->backend().stream_executors());
  std::unique_ptr<tsl::Allocator> host_memory_allocator;
  if (!xla_client->backend().stream_executors().empty()) {
    TF_ASSIGN_OR_RETURN(
        host_memory_allocator,
        GetGpuHostAllocator(xla_client->backend().stream_executors().front()));
  }
  TF_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<TfrtGpuDevice>> devices,
                      GetTfrtGpuDevices(xla_client));

  GpuTopologyProto gpu_topology_proto;
  for (const auto& device : devices) {
    if (gpu_topology_proto.platform_version().empty()) {
      gpu_topology_proto.set_platform_version(
          std::string(device->device_kind()));
    }
    gpu_topology_proto.add_device_ids(device->id());
  }

  // TODO(b/382117736): Support multi-host
  gpu_topology_proto.set_num_slices(1);
  gpu_topology_proto.set_num_hosts_per_slice(1);
  gpu_topology_proto.set_num_devices_per_host(devices.size());

  auto gpu_topology = std::shared_ptr<const GpuTopology>(
      GpuTopology::FromProto(gpu_topology_proto));

  return std::unique_ptr<PjRtClient>(std::make_unique<TfrtGpuClient>(
      /*process_index=*/0, xla_client, std::move(devices),
      std::move(host_memory_allocator), gpu_topology));
}

}  // namespace xla
