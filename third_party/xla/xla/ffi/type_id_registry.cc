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

#include "xla/ffi/type_id_registry.h"

#include <atomic>
#include <cstdint>
#include <string>
#include <string_view>

#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/util.h"

namespace xla::ffi {

ABSL_CONST_INIT absl::Mutex type_registry_mutex(absl::kConstInit);

using ExternalTypeIdRegistry =
    absl::flat_hash_map<std::string, TypeIdRegistry::TypeId>;

static ExternalTypeIdRegistry& StaticExternalTypeIdRegistry() {
  static auto* registry = new ExternalTypeIdRegistry();
  return *registry;
}

TypeIdRegistry::TypeId TypeIdRegistry::GetNextInternalTypeId() {
  static auto* counter = new std::atomic<int64_t>(1);
  return TypeId(counter->fetch_add(1));
}

TypeIdRegistry::TypeId TypeIdRegistry::GetNextExternalTypeId() {
  static auto* counter = new std::atomic<int64_t>(1);
  return TypeId(counter->fetch_add(1));
}

absl::StatusOr<TypeIdRegistry::TypeId> TypeIdRegistry::AssignExternalTypeId(
    std::string_view name) {
  absl::MutexLock lock(&type_registry_mutex);
  auto& registry = StaticExternalTypeIdRegistry();

  // Try to emplace with unknow type id and fill it with real type id only if we
  // successfully acquired an entry for a given name.
  auto emplaced = registry.emplace(name, kUnknownTypeId);
  if (!emplaced.second) {
    return Internal("Type name %s already registered with type id %d", name,
                    emplaced.first->second.value());
  }

  // Returns true if the registry contains an entry with a given type id.
  auto type_id_is_in_use = [&registry](TypeId type_id) {
    return absl::c_any_of(registry,
                          [&](const auto& e) { return e.second == type_id; });
  };

  // Create a new type id that is not already in use.
  TypeId type_id = GetNextExternalTypeId();
  while (type_id_is_in_use(type_id)) {
    type_id = GetNextExternalTypeId();
  }

  return emplaced.first->second = type_id;
}

absl::Status TypeIdRegistry::RegisterExternalTypeId(absl::string_view name,
                                                    TypeId type_id) {
  absl::MutexLock lock(&type_registry_mutex);
  auto& registry = StaticExternalTypeIdRegistry();

  auto emplaced = registry.emplace(name, type_id);
  if (!emplaced.second && emplaced.first->second != type_id) {
    return Internal("Type name %s already registered with type id %d vs %d)",
                    name, emplaced.first->second.value(), type_id.value());
  }

  return absl::OkStatus();
}

}  // namespace xla::ffi
