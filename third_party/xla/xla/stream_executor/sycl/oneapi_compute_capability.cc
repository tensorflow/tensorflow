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

#include "xla/stream_executor/sycl/oneapi_compute_capability.h"

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/sycl/oneapi_compute_capability.pb.h"

namespace stream_executor {

OneAPIComputeCapability::OneAPIComputeCapability(XeGeneration generation,
                                                 uint8_t version)
    : generation_(generation) {
  // The first range that matches selects the arch. If none does, the arch
  // stays kUnknown but the generation is kept.
  for (const ArchRange& range : kXeArchRanges) {
    if (GetArchInfo(range.arch).generation == generation &&
        version >= range.min_version && version <= range.max_version) {
      arch_ = range.arch;
      return;
    }
  }
}

OneAPIComputeCapability::OneAPIComputeCapability(absl::string_view arch) {
  for (size_t i = 0; i < static_cast<size_t>(XeArch::kUnknown); ++i) {
    if (absl::EqualsIgnoreCase(kXeArchInfos[i].name, arch)) {
      *this = OneAPIComputeCapability(static_cast<XeArch>(i));
      return;
    }
  }
}

std::string OneAPIComputeCapability::architecture() const {
  return std::string(GetArchInfo(arch_).name);
}

OneAPIComputeCapability OneAPIComputeCapability::FromProto(
    const OneAPIComputeCapabilityProto& proto) {
  OneAPIComputeCapability cc{proto.architecture()};
  // A known architecture name takes precedence; the generation is only needed
  // to identify devices whose architecture is unknown.
  if (cc.arch_ == XeArch::kUnknown && proto.has_generation()) {
    cc.generation_ = static_cast<XeGeneration>(proto.generation());
  }
  return cc;
}

OneAPIComputeCapabilityProto OneAPIComputeCapability::ToProto() const {
  OneAPIComputeCapabilityProto proto;
  proto.set_architecture(architecture());
  if (IsSpecified()) {
    proto.set_generation(static_cast<uint32_t>(generation_));
  }
  return proto;
}

std::string OneAPIComputeCapability::ToString() const {
  if (arch_ == XeArch::kUnknown && IsSpecified()) {
    return absl::StrCat(architecture(),
                        "(gen=", static_cast<uint32_t>(generation_), ")");
  }
  return architecture();
}

}  // namespace stream_executor
