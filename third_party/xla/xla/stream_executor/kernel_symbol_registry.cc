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

#include "xla/stream_executor/kernel_symbol_registry.h"

#include <string>
#include <tuple>

#include "absl/base/no_destructor.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
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

KernelSymbolRegistry& KernelSymbolRegistry::GetGlobalInstance() {
  static absl::NoDestructor<KernelSymbolRegistry> registry;
  return *registry;
}

absl::Status KernelSymbolRegistry::RegisterSymbol(absl::string_view name,
                                                  Platform::Id platform_id,
                                                  void* symbol) {
  absl::MutexLock lock(mutex_);
  bool inserted;
  std::tie(std::ignore, inserted) =
      symbols_.insert({{std::string(name), platform_id}, symbol});
  if (!inserted) {
    return absl::AlreadyExistsError(
        absl::StrCat("Symbol ", name, " is already registered."));
  }
  return absl::OkStatus();
}

absl::StatusOr<void*> KernelSymbolRegistry::FindSymbol(
    absl::string_view name, Platform::Id platform_id) const {
  absl::MutexLock lock(mutex_);
  auto it = symbols_.find({std::string(name), platform_id});
  if (it == symbols_.end()) {
    return absl::NotFoundError(absl::StrCat("Symbol ", name,
                                            " not found for platform ",
                                            GetPlatformName(platform_id)));
  }
  return it->second;
}

}  // namespace stream_executor
