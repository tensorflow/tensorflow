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

#include "xla/core/collectives/collectives_registry.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/core/collectives/collectives.h"
#include "xla/service/platform_util.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

struct Registration {
  std::string platform_name;
  std::string name;
  int32_t priority;
  std::unique_ptr<Collectives> collectives;
};

struct Registry {
  absl::Mutex mu;

  // Container for registered collectives implementations.
  std::vector<Registration> collectives ABSL_GUARDED_BY(mu);

  // A map from a canonical platform name to the collectives implementations
  // for that platform ordered by their priority.
  absl::flat_hash_map<std::string,
                      absl::btree_map<int32_t, Collectives*, std::greater<>>>
      platform_collectives ABSL_GUARDED_BY(mu);
};

}  // namespace

static Registry& GetCollectivesRegistry() {
  static auto* const registry = new Registry();
  return *registry;
}

absl::Status CollectivesRegistry::Register(
    absl::string_view platform_name, absl::string_view name, int32_t priority,
    std::unique_ptr<Collectives> collectives) {
  TF_ASSIGN_OR_RETURN(std::string canonical_platform_name,
                      PlatformUtil::CanonicalPlatformName(platform_name));

  auto& registry = GetCollectivesRegistry();
  absl::MutexLock lock(&registry.mu);

  registry.platform_collectives[canonical_platform_name][priority] =
      collectives.get();
  registry.collectives.push_back(Registration{canonical_platform_name,
                                              std::string(name), priority,
                                              std::move(collectives)});

  return absl::OkStatus();
}

absl::StatusOr<Collectives*> CollectivesRegistry::Default(
    absl::string_view platform_name) {
  TF_ASSIGN_OR_RETURN(std::string canonical_platform_name,
                      PlatformUtil::CanonicalPlatformName(platform_name));

  auto& registry = GetCollectivesRegistry();
  absl::MutexLock lock(&registry.mu);

  if (!registry.platform_collectives.contains(canonical_platform_name)) {
    return Internal(
        "No collectives registered for platform: %s (canonical name: %s)",
        platform_name, canonical_platform_name);
  }

  return registry.platform_collectives[canonical_platform_name].begin()->second;
}

}  // namespace xla
