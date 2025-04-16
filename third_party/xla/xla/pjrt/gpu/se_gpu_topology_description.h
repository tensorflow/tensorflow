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
#ifndef XLA_PJRT_GPU_SE_GPU_TOPOLOGY_DESCRIPTION_H_
#define XLA_PJRT_GPU_SE_GPU_TOPOLOGY_DESCRIPTION_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/pjrt/gpu/gpu_topology.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_stream_executor_device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {

class StreamExecutorGpuTopologyDescription : public PjRtTopologyDescription {
 public:
  StreamExecutorGpuTopologyDescription(
      const PjRtPlatformId platform_id, const absl::string_view platform_name,
      std::shared_ptr<const GpuTopology> gpu_topology,
      const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& attributes =
          {},
      std::optional<stream_executor::GpuTargetConfigProto> target_config =
          std::nullopt)
      : platform_id_(platform_id),
        platform_name_(platform_name),
        gpu_topology_(std::move(gpu_topology)),
        attributes_(attributes),
        target_config_(std::move(target_config)) {}

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

  const std::optional<stream_executor::GpuTargetConfigProto>& target_config()
      const {
    return target_config_;
  }

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
  std::optional<stream_executor::GpuTargetConfigProto> target_config_;
};
}  // namespace xla

#endif  // XLA_PJRT_GPU_SE_GPU_TOPOLOGY_DESCRIPTION_H_
