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

#ifndef XLA_PJRT_PJRT_TOPOLOGY_DESCRIPTION_REGISTRY_H_
#define XLA_PJRT_PJRT_TOPOLOGY_DESCRIPTION_REGISTRY_H_

#include <functional>
#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/proto/topology_description.pb.h"

namespace xla {

using PjRtTopologyDescriptionDeserializer =
    std::function<absl::StatusOr<std::unique_ptr<PjRtTopologyDescription>>(
        const PjRtTopologyDescriptionProto&)>;

class PjRtTopologyDescriptionRegistry {
 public:
  static PjRtTopologyDescriptionRegistry& Global();

  absl::Status RegisterDeserializer(
      PjRtPlatformId platform_id, absl::string_view platform_name,
      PjRtTopologyDescriptionDeserializer deserializer);

  absl::StatusOr<std::unique_ptr<PjRtTopologyDescription>> Deserialize(
      const PjRtTopologyDescriptionProto& proto) const;

 private:
  PjRtTopologyDescriptionRegistry() = default;

  mutable absl::Mutex mu_;
  absl::flat_hash_map<PjRtPlatformId, PjRtTopologyDescriptionDeserializer>
      id_deserializers_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, PjRtTopologyDescriptionDeserializer>
      name_deserializers_ ABSL_GUARDED_BY(mu_);
};

absl::StatusOr<PjRtTopologyDescriptionProto> PjRtTopologyDescriptionToProto(
    const PjRtTopologyDescription* topology_description);

absl::StatusOr<std::unique_ptr<PjRtTopologyDescription>>
PjRtTopologyDescriptionFromProto(const PjRtTopologyDescriptionProto& proto);

#define REGISTER_PJRT_TOPOLOGY_DESERIALIZER(token_name, platform_id,           \
                                            platform_name, fn)                 \
  [[maybe_unused]] static bool pjrt_topo_deserializer_##token_name = []() {    \
    QCHECK_OK(                                                                 \
        ::xla::PjRtTopologyDescriptionRegistry::Global().RegisterDeserializer( \
            platform_id, platform_name, fn));                                  \
    return true;                                                               \
  }()

}  // namespace xla

#endif  // XLA_PJRT_PJRT_TOPOLOGY_DESCRIPTION_REGISTRY_H_
