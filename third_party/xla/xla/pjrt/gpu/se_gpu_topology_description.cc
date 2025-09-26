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
#include "xla/pjrt/gpu/se_gpu_topology_description.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_stream_executor_device_description.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/strings/proto_serialization.h"
#include "xla/xla_data.pb.h"

namespace xla {

/*static*/ void StreamExecutorGpuTopologyDescription::SetupDeviceDescription(
    PjRtStreamExecutorDeviceDescription& description,
    const std::string& device_vendor, const std::string& compute_capability,
    int core_count, int64_t shared_memory_per_block_optin,
    int partition_index) {
  std::vector<int64_t> v_coords(description.coords().begin(),
                                description.coords().end());

  description.SetAttributes(
      {{"coords", xla::PjRtDeviceAttribute(v_coords)},
       {"device_vendor", device_vendor},
       // TODO - b/435521225: `slice_index` is deprecated. Use
       // `partition_index`, which better aligns with NVIDIA's terminology.
       {"slice_index", static_cast<int64_t>(partition_index)},
       {"partition_index", static_cast<int64_t>(partition_index)},
       {"compute_capability", xla::PjRtDeviceAttribute(compute_capability)},
       {"shared_memory_per_block_optin", shared_memory_per_block_optin},
       {"core_count", static_cast<int64_t>(core_count)}});
  description.SetToString(absl::StrFormat(
      "StreamExecutorGpuDevice(device_kind=%s, id=%i, process_index=%i, "
      "partition_index=%i))",
      description.device_kind(), description.id(), description.process_index(),
      partition_index));
  description.SetDebugString(absl::StrFormat(
      "%s_%i(process=%i,(%i))", description.device_kind(), description.id(),
      description.process_index(), v_coords[0]));
}

std::vector<std::unique_ptr<const PjRtDeviceDescription>>
StreamExecutorGpuTopologyDescription::DeviceDescriptions() const {
  std::vector<std::unique_ptr<const PjRtDeviceDescription>> devices;
  if (gpu_topology_->number_of_devices() <= 0) {
    return devices;
  }
  devices.reserve(gpu_topology_->number_of_devices());
  // Instead of "host", we use "process", as it's more accurate and consistent
  // with PjRt terminology. In a multi-process setting, a host can have multiple
  // processes, e.g., one process per GPU.
  const int32_t num_devices_per_process = gpu_topology_->num_devices_per_host();
  const int32_t num_processes_per_partition =
      gpu_topology_->num_hosts_per_partition();
  for (int device_id = 0; device_id < gpu_topology_->number_of_devices();
       ++device_id) {
    // The local_device_id, process_index and partition_index are inferred from
    // the global device id. It requires the global topology is symmetric:
    //  - all partitions have the same number of processes.
    //  - all processes have the same number of devices.
    //  - processes of the same partition are adjacent to each other.
    //
    // And it also requires the ids assignments follows the PjRt topology
    // exchange protocol in xla/pjrt/distributed/topology_util.cc:
    //  - ids are densely assigned and start from 0
    //  - from lower process index to higher process index
    //  - within the process, from lower device ordinal to higher device ordinal
    //
    // If the above requirements are not met, users should get the device
    // description by looking up individual device from PjRt client.
    const int local_device_id = num_devices_per_process == -1
                                    ? 0
                                    : (device_id % num_devices_per_process);
    const int process_index = num_devices_per_process == -1
                                  ? 0
                                  : (device_id / num_devices_per_process);
    const int process_index_in_partition =
        process_index == -1 ? 0 : (process_index % num_processes_per_partition);
    const int partition_index =
        num_processes_per_partition == -1
            ? 0
            : (process_index / num_processes_per_partition);
    auto description = std::make_unique<PjRtStreamExecutorDeviceDescription>(
        device_id, local_device_id, process_index, process_index_in_partition,
        partition_index, std::string(platform_version()));
    if (target_config_.has_value()) {
      std::string compute_capability = "<unknown compute-capability>";
      std::string gpu_vendor = "<unknown gpu vendor>";
      if (target_config_->gpu_device_info().has_cuda_compute_capability()) {
        const auto& cap =
            target_config_->gpu_device_info().cuda_compute_capability();
        compute_capability = absl::StrCat(cap.major(), ".", cap.minor());
        gpu_vendor = "NVIDIA Corporation";
      }

      StreamExecutorGpuTopologyDescription::SetupDeviceDescription(
          *description, gpu_vendor, compute_capability,
          target_config_->gpu_device_info().core_count(),
          target_config_->gpu_device_info().shared_memory_per_block_optin(), 0);
    }
    devices.push_back(std::move(description));
  }
  return devices;
}

absl::StatusOr<std::string> StreamExecutorGpuTopologyDescription::Serialize()
    const {
  std::string result;
  if (!tsl::SerializeToStringDeterministic(gpu_topology_->ToProto(), &result)) {
    return absl::InternalError("Failed to serialize gpu_topology");
  }
  return result;
}

absl::StatusOr<Layout> StreamExecutorGpuTopologyDescription::GetDefaultLayout(
    PrimitiveType element_type, absl::Span<const int64_t> dims) const {
  Shape shape = ShapeUtil::MakeShape(element_type, dims);
  Layout layout = LayoutUtil::GetWithDefaultLayout(shape).layout();
  // `GetWithDefaultLayout` returns a padded layout for sub-byte types since the
  // notion of "default" is context dependent and in this case means the default
  // for literals for historical reasons. Because of this, we need to manually
  // populate the `element_size_in_bits` for sub-byte types here.
  if (primitive_util::IsSubByteNonPredType(element_type)) {
    layout.set_element_size_in_bits(primitive_util::BitWidth(element_type));
  }
  return layout;
}

}  // namespace xla
