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

#include "xla/stream_executor/abi/runtime_abi_version_manager.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/abi/runtime_abi_version.h"

namespace stream_executor {

RuntimeAbiVersionManager& RuntimeAbiVersionManager::GetInstance() {
  static absl::NoDestructor<RuntimeAbiVersionManager> instance;
  return *instance;
}
absl::Status RuntimeAbiVersionManager::RegisterRuntimeAbiVersionFactory(
    std::string platform_name,
    RuntimeAbiVersionFactory runtime_abi_version_factory) {
  absl::MutexLock lock(mutex_);
  if (runtime_abi_version_factories_.contains(platform_name)) {
    return absl::AlreadyExistsError(
        absl::StrCat("RuntimeAbiVersionFactory for platform ", platform_name,
                     " already exists."));
  }
  runtime_abi_version_factories_[std::move(platform_name)] =
      std::move(runtime_abi_version_factory);
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<RuntimeAbiVersion>>
RuntimeAbiVersionManager::GetRuntimeAbiVersion(
    const RuntimeAbiVersionProto& proto) const {
  absl::string_view platform_name = proto.platform_name();
  absl::MutexLock lock(mutex_);
  auto it = runtime_abi_version_factories_.find(platform_name);
  if (it == runtime_abi_version_factories_.end()) {
    return absl::NotFoundError(
        absl::StrCat("RuntimeAbiVersionFactory for platform ", platform_name,
                     " not found."));
  }
  return it->second(proto.platform_specific_version());
}

}  // namespace stream_executor
