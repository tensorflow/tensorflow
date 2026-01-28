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

#include <cassert>
#include <cstdint>
#include <string>
#include <utility>

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/sycl/oneapi_compute_capability.pb.h"

namespace stream_executor {

OneAPIComputeCapabilityProto OneAPIComputeCapability::ToProto() const {
  OneAPIComputeCapabilityProto proto;
  if (generation_ == 0xc && version_ == 0x3c) {
    proto.set_architecture("PVC");
  } else if (generation_ == 0x14 && version_ == 0x1) {
    proto.set_architecture("BMG");
  } else if (generation_ == 0xc && version_ == 0x37) {
    proto.set_architecture("DG2");
  }
  // This stub implementation currently disregards the "variant" proto field.
  // TODO(intel-tf): Add proper implementation for the proto message.
  proto.set_variant("");
  return proto;
}

std::string OneAPIComputeCapability::ToString() const {
  return absl::StrCat(generation_, ".", version_);
}

std::pair<uint32_t, uint32_t> OneAPIComputeCapability::BaseVersionTupleFor(
    absl::string_view name) {
  std::string architecture = absl::AsciiStrToLower(name);
  if (architecture == "pvc") {
    return {0xc, 0x3c};
  }
  if (architecture == "bmg") {
    return {0x14, 0x1};
  }
  if (architecture == "dg2") {
    return {0xc, 0x37};
  }
  return {0, 0};
}

uint32_t OneAPIComputeCapability::GenericIPVersionFor(absl::string_view name) {
  auto [gen, ver] = BaseVersionTupleFor(name);
  return (gen << 22) | (ver << 14);
}

// This stub implementation currently disregards the "variant" proto field.
// TODO(intel-tf): Add proper implementation for the proto message.
OneAPIComputeCapability OneAPIComputeCapability::FromProto(
    const OneAPIComputeCapabilityProto& proto) {
  return OneAPIComputeCapability{proto.architecture()};
}

}  // namespace stream_executor
