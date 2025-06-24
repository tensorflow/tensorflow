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

#include "xla/stream_executor/platform/platform_object_registry.h"

#include <functional>
#include <string>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"

namespace stream_executor {

namespace {
std::string GetPlatformName(Platform::Id platform_id) {
  absl::StatusOr<Platform*> platform =
      PlatformManager::PlatformWithId(platform_id);
  return platform.ok() ? platform.value()->Name() : "<unknown>";
}
}  // namespace

PlatformObjectRegistry& PlatformObjectRegistry::GetGlobalRegistry() {
  static auto registry = new PlatformObjectRegistry();
  return *registry;
}

absl::StatusOr<std::reference_wrapper<const PlatformObjectRegistry::Container>>
PlatformObjectRegistry::FindObject(const std::type_info& type,
                                   Platform::Id platform_id) const {
  absl::MutexLock lock(&mutex_);
  auto it = objects_.find({std::type_index(type), platform_id});
  if (it != objects_.end()) {
    return it->second;
  }

  absl::StatusOr<Platform*> platform =
      PlatformManager::PlatformWithId(platform_id);
  std::string platform_name =
      platform.ok() ? platform.value()->Name() : "<unknown>";

  return absl::NotFoundError(
      absl::StrFormat("Object %s not found for platform %s and trait %s",
                      type.name(), GetPlatformName(platform_id), type.name()));
}

absl::Status PlatformObjectRegistry::RegisterObject(const std::type_info& type,
                                                    Platform::Id platform_id,
                                                    Container object) {
  absl::MutexLock lock(&mutex_);
  const auto [it, inserted] = objects_.insert(std::make_pair(
      std::make_tuple(std::type_index(type), platform_id), std::move(object)));
  if (!inserted) {
    return absl::AlreadyExistsError(absl::StrFormat(
        "Object for trait %s and platform %s is already registered.",
        type.name(), GetPlatformName(platform_id)));
  }
  return absl::OkStatus();
}

}  // namespace stream_executor
