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

#ifndef XLA_PYTHON_IFRT_TOPOLOGY_H_
#define XLA_PYTHON_IFRT_TOPOLOGY_H_

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
#include "xla/xla_data.pb.h"

namespace xla::ifrt {

class Topology : public llvm::RTTIExtends<Topology, llvm::RTTIRoot> {
 public:
  // Returns a string that identifies the platform (CPU/GPU/TPU).
  virtual absl::string_view platform_name() const = 0;

  // Returns a string containing human-readable, platform-specific version info
  // (e.g. the CUDA version on GPU or libtpu version on Cloud TPU).
  virtual absl::string_view platform_version() const = 0;

  virtual PjRtPlatformId platform_id() const = 0;

  // Returns an unordered list of descriptions for all devices in this topology.
  // TODO(phawkins): consider introducing an IFRT-specific API here instead of
  // delegating to PJRT.
  virtual std::vector<std::unique_ptr<const PjRtDeviceDescription>>
  DeviceDescriptions() const = 0;

  // Returns the default device layout for a buffer with `element_type` and
  // `dims`. The default layout is a platform-specific layout used when no other
  // layout is specified, e.g. for host-to-device transfers. When compiling, the
  // default layout is used for program arguments and outputs unless
  // user-specified or compiler-chosen layouts are requested via the
  // "mhlo.layout_mode" attribute.
  virtual absl::StatusOr<xla::Layout> GetDefaultLayout(
      PrimitiveType element_type, absl::Span<const int64_t> dims) const = 0;

  // Serializes the topology for use in cache keys. (No guarantees on
  // stability).
  virtual absl::StatusOr<std::string> Serialize() const = 0;

  // Returns vendor specific attributes about the topology.
  virtual const AttributeMap& Attributes() const = 0;

  static char ID;  // NOLINT
};

}  // namespace xla::ifrt

#endif  // XLA_PYTHON_IFRT_TOPOLOGY_H_
