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

#include "xla/pjrt/pjrt_topology_description_registry.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/proto/topology_description.pb.h"

namespace xla {

PjRtTopologyDescriptionRegistry& PjRtTopologyDescriptionRegistry::Global() {
  static auto* registry = new PjRtTopologyDescriptionRegistry();
  return *registry;
}

absl::Status PjRtTopologyDescriptionRegistry::RegisterDeserializer(
    PjRtPlatformId platform_id, absl::string_view platform_name,
    PjRtTopologyDescriptionDeserializer deserializer) {
  absl::MutexLock lock(mu_);
  id_deserializers_[platform_id] = deserializer;
  name_deserializers_[platform_name] = deserializer;
  return absl::OkStatus();
}

void PjRtTopologyDescriptionRegistry::RegisterDynamicCompilerLookup(
    DynamicCompilerLookup lookup) {
  absl::MutexLock lock(mu_);
  dynamic_compiler_lookup_ = std::move(lookup);
}

absl::StatusOr<std::unique_ptr<PjRtCompiler>>
PjRtTopologyDescriptionRegistry::GetDynamicCompiler(
    absl::string_view platform_name) const {
  absl::MutexLock lock(mu_);
  if (!dynamic_compiler_lookup_) {
    return absl::NotFoundError(
        absl::StrCat("No dynamic compiler lookup registered for platform '",
                     platform_name, "'."));
  }
  return dynamic_compiler_lookup_(platform_name);
}

absl::StatusOr<std::unique_ptr<PjRtTopologyDescription>>
PjRtTopologyDescriptionRegistry::Deserialize(
    const PjRtTopologyDescriptionProto& proto) const {
  PjRtTopologyDescriptionDeserializer deserializer;
  {
    absl::MutexLock lock(mu_);
    if (!proto.platform_name().empty()) {
      auto it = name_deserializers_.find(proto.platform_name());
      if (it != name_deserializers_.end()) {
        deserializer = it->second;
      }
    }
    if (!deserializer && proto.platform_id() != 0) {
      auto it = id_deserializers_.find(proto.platform_id());
      if (it != id_deserializers_.end()) {
        deserializer = it->second;
      }
    }
  }
  if (deserializer) {
    return deserializer(proto);
  }
  return absl::NotFoundError(absl::StrCat(
      "No PjRtTopologyDescriptionDeserializer registered for platform_name '",
      proto.platform_name(), "' or platform_id '", proto.platform_id(), "'."));
}

absl::StatusOr<PjRtTopologyDescriptionProto> PjRtTopologyDescriptionToProto(
    const PjRtTopologyDescription* topology_description) {
  if (topology_description == nullptr) {
    return absl::InvalidArgumentError("The topology description is null.");
  }
  return topology_description->ToProto();
}

absl::StatusOr<std::unique_ptr<PjRtTopologyDescription>>
PjRtTopologyDescriptionFromProto(const PjRtTopologyDescriptionProto& proto) {
  // Static C++ Registry (In-tree backends: CPU, CUDA, ROCm, linked TPU).
  auto static_result =
      PjRtTopologyDescriptionRegistry::Global().Deserialize(proto);
  if (static_result.ok() || !absl::IsNotFound(static_result.status())) {
    return static_result;
  }

  // Dynamic C-API Compiler Plugin (Out-of-tree plugins: dynamic lib tpu).
  absl::string_view platform_name = proto.platform_name();
  if (platform_name.empty()) {
    if (proto.platform_id() == CpuId()) {
      platform_name = CpuName();
    } else if (proto.platform_id() == CudaId()) {
      platform_name = CudaName();
    } else if (proto.platform_id() == RocmId()) {
      platform_name = RocmName();
    } else if (proto.platform_id() == TpuId()) {
      platform_name = TpuName();
    }
  }
  if (platform_name.empty()) {
    return absl::InvalidArgumentError(
        "PjRtTopologyDescriptionProto does not specify a valid "
        "platform_name or recognized platform_id.");
  }

  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<PjRtCompiler> compiler,
                   PjRtTopologyDescriptionRegistry::Global().GetDynamicCompiler(
                       platform_name));
  return compiler->DeserializePjRtTopologyDescription(
      proto.SerializeAsString());
}

}  // namespace xla
