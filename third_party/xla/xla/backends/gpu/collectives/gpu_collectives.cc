/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/gpu/collectives/gpu_collectives.h"

#include <cstddef>
#include <optional>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/runtime/device_id.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_interconnect_resource.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"

namespace xla::gpu {

GpuCollectives* GpuCollectives::Default(absl::string_view platform_name) {
  absl::StatusOr<Collectives*> collectives =
      CollectivesRegistry::Default(platform_name);
  CHECK_OK(collectives) << "Failed to get GPU collectives";  // Crash OK

  if (auto* gpu_collectives = absl::down_cast<GpuCollectives*>(*collectives)) {
    return gpu_collectives;
  }

  LOG(FATAL) << "Unsupported collectives implementation for GPU";
}

GpuCollectives::Device::Device(se::StreamExecutor* stream_executor)
    : stream_executor_(stream_executor) {}

se::StreamExecutor* GpuCollectives::Device::stream_executor() const {
  return stream_executor_;
}

GpuCollectives::Executor::Executor(stream_executor::Stream* stream)
    : stream_(stream) {}

stream_executor::Stream* GpuCollectives::Executor::stream() const {
  return stream_;
}

se::DeviceAddressBase GpuCollectives::Slice(se::DeviceAddressBase buff,
                                            PrimitiveType dtype, size_t offset,
                                            size_t count) {
  size_t multiplier = ShapeUtil::ByteSizeOfPrimitiveType(dtype);
  return buff.GetByteSlice(offset * multiplier, count * multiplier);
}

FabricHomogeneity CheckFabricHomogeneity(se::StreamExecutor* executor,
                                         const CliqueKey& clique_key) {
  se::DeviceInterconnectResource* res =
      executor->GetOrNullResource<se::DeviceInterconnectResource>();
  if (!res) {
    return FabricHomogeneity::kUnknown;
  }

  const se::DeviceInterconnectResource::InfoMap& interconnect_map =
      res->interconnect_map();

  std::optional<absl::string_view> cluster_uuid;
  std::optional<absl::string_view> clique_id;

  for (const GlobalDeviceId device : clique_key.devices()) {
    se::DeviceInterconnectResource::InfoMap::const_iterator it =
        interconnect_map.find(device.value());

    if (it == interconnect_map.end()) {
      return FabricHomogeneity::kUnknown;
    }
    const se::DeviceInterconnectInfo& info = it->second;

    // Empty strings indicate uninitialized/unhealthy fabric state.
    if (info.cluster_uuid.empty() || info.clique_id.empty()) {
      return FabricHomogeneity::kUnknown;
    }

    if (!cluster_uuid.has_value()) {
      // Initialize the baseline comparison values.
      cluster_uuid = info.cluster_uuid;
      clique_id = info.clique_id;
    } else if (info.cluster_uuid != *cluster_uuid ||
               info.clique_id != *clique_id) {
      XLA_VLOG_DEVICE(1, executor->device_ordinal())
          << "Fabric mismatch detected in clique for device " << device.value();
      VLOG(1) << "Fabric mismatch detected in clique for device "
              << device.value();
      return FabricHomogeneity::kHeterogeneous;
    }
  }
  return FabricHomogeneity::kHomogeneous;
}

}  // namespace xla::gpu
