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

#ifndef XLA_PYTHON_PJRT_IFRT_PJRT_TOPOLOGY_H_
#define XLA_PYTHON_PJRT_IFRT_PJRT_TOPOLOGY_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/topology.h"
#include "xla/xla_data.pb.h"

namespace xla::ifrt {

class PjRtTopology final : public llvm::RTTIExtends<PjRtTopology, Topology> {
 public:
  explicit PjRtTopology(
      std::shared_ptr<const xla::PjRtTopologyDescription> description);

  const std::shared_ptr<const xla::PjRtTopologyDescription>& description()
      const {
    return description_;
  }

  absl::string_view platform_name() const override;
  absl::string_view platform_version() const override;
  PjRtPlatformId platform_id() const override;

  std::vector<std::unique_ptr<const PjRtDeviceDescription>> DeviceDescriptions()
      const override;

  absl::StatusOr<xla::Layout> GetDefaultLayout(
      PrimitiveType element_type,
      absl::Span<const int64_t> dims) const override;

  absl::StatusOr<std::string> Serialize() const override;

  const AttributeMap& Attributes() const override;

  static char ID;  // NOLINT

 private:
  std::shared_ptr<const xla::PjRtTopologyDescription> description_;
  AttributeMap attributes_;
};

}  // namespace xla::ifrt

#endif  // XLA_PYTHON_PJRT_IFRT_PJRT_TOPOLOGY_H_
