/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_PJRT_INTERPRETER_INTERPRETER_TOPOLOGY_DESCRIPTION_H_
#define XLA_PJRT_INTERPRETER_INTERPRETER_TOPOLOGY_DESCRIPTION_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/proto/topology_description.pb.h"
#include "xla/runtime/chip_id.h"
#include "xla/runtime/device_id.h"
#include "xla/service/device_assignment.h"
#include "xla/xla_data.pb.h"

namespace xla {

class InterpreterDescription final : public PjRtDeviceDescription {
 public:
  InterpreterDescription() = default;
  static const InterpreterDescription& Singleton();

  int id() const override { return 0; }

  int process_index() const override { return 0; }

  absl::string_view device_kind() const override { return "interpreter"; }

  absl::string_view DebugString() const override { return "interpreter:0"; }

  absl::string_view ToString() const override {
    return "InterpreterDevice(id=0)";
  }

  const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& Attributes()
      const override {
    return attributes_;
  }

 private:
  absl::flat_hash_map<std::string, PjRtDeviceAttribute> attributes_;
};

class InterpreterTopologyDescription : public PjRtTopologyDescription {
 public:
  InterpreterTopologyDescription() = default;

  bool operator==(const InterpreterTopologyDescription& other) const {
    return this->platform_id() == other.platform_id() &&
           this->platform_name() == other.platform_name() &&
           this->platform_version() == other.platform_version();
  }

  PjRtPlatformId platform_id() const override { return xla::InterpreterId(); }

  absl::string_view platform_name() const override {
    return xla::InterpreterName();
  }

  absl::string_view platform_version() const override { return "<unknown>"; }

  std::vector<std::unique_ptr<const PjRtDeviceDescription>> DeviceDescriptions()
      const override;

  bool is_subslice_topology() const override { return false; }

  absl::StatusOr<int> ProcessCount() const override { return 1; }

  absl::StatusOr<int> ChipsPerProcess() const override { return 1; }

  absl::StatusOr<int> LogicalDeviceCountOfDefaultTypePerChip() const override {
    return 1;
  }

  absl::StatusOr<int> CoreCountOfDefaultTypePerChip() const override {
    return 1;
  }

  absl::StatusOr<PjRtIdContainer<ProcessId>> ProcessIds() const override {
    return PjRtIdContainer<ProcessId>({ProcessId(0)});
  }

  absl::StatusOr<PjRtIdContainer<GlobalDeviceId>>
  LogicalDeviceOfDefaultTypeIdsOnProcess(ProcessId process_id) const override {
    if (process_id.value() != 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid process id: ", process_id.value()));
    }
    return PjRtIdContainer<GlobalDeviceId>({GlobalDeviceId(0)});
  }

  absl::StatusOr<std::pair<ProcessId, int>>
  ProcessIdAndIndexOnProcessForLogicalDeviceOfDefaultType(
      GlobalDeviceId device_id) const override {
    if (device_id.value() != 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid device id: ", device_id.value()));
    }
    return std::make_pair(ProcessId(0), 0);
  }

  absl::StatusOr<std::pair<PjRtDeviceDimensions, int32_t>>
  ChipCoordAndCoreIndexForLogicalDeviceOfDefaultType(
      GlobalDeviceId device_id) const override {
    if (device_id.value() != 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid device id: ", device_id.value()));
    }
    return std::make_pair(PjRtDeviceDimensions{0, 0, 0}, 0);
  }

  absl::StatusOr<uint64_t> Fingerprint() const override;

  const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& Attributes()
      const override {
    return attributes_;
  }

  absl::Span<const int> GetMemorySpaceKindIds() const override;

  absl::StatusOr<Layout> GetDefaultLayout(
      PrimitiveType element_type,
      absl::Span<const int64_t> dims) const override;

  absl::StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int process_index, int num_replicas,
      std::optional<int> num_replicas_per_slice, int num_partitions,
      const MultiSliceConfig* multi_slice_config) const override;

  absl::StatusOr<xla::PjRtTopologyDescriptionProto> ToProto() const override;

  static absl::StatusOr<std::unique_ptr<InterpreterTopologyDescription>>
  FromProto(const xla::PjRtTopologyDescriptionProto& proto);

 private:
  absl::flat_hash_map<std::string, PjRtDeviceAttribute> attributes_;
};

}  // namespace xla

#endif  // XLA_PJRT_INTERPRETER_INTERPRETER_TOPOLOGY_DESCRIPTION_H_
