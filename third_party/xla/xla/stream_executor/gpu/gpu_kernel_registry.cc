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

#include "xla/stream_executor/gpu/gpu_kernel_registry.h"

#include <functional>
#include <string>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"

namespace stream_executor::gpu {

namespace {
std::string GetPlatformName(Platform::Id platform_id) {
  absl::StatusOr<Platform*> platform =
      PlatformManager::PlatformWithId(platform_id);
  return platform.ok() ? platform.value()->Name() : "<unknown>";
}
}  // namespace

absl::StatusOr<std::reference_wrapper<const MultiKernelLoaderSpec>>
GpuKernelRegistry::GetKernelSpec(const std::type_info& type,
                                 Platform::Id platform_id) {
  absl::MutexLock lock(&mutex_);
  auto it = kernel_specs_.find({std::type_index(type), platform_id});
  if (it != kernel_specs_.end()) {
    return it->second;
  }

  absl::StatusOr<Platform*> platform =
      PlatformManager::PlatformWithId(platform_id);
  std::string platform_name =
      platform.ok() ? platform.value()->Name() : "<unknown>";

  return absl::NotFoundError(
      absl::StrFormat("Kernel %s not found for platform %s and type %s",
                      type.name(), GetPlatformName(platform_id), type.name()));
}

absl::Status GpuKernelRegistry::RegisterKernel(
    const std::type_info& type, Platform::Id platform_id,
    const MultiKernelLoaderSpec& kernel) {
  absl::MutexLock lock(&mutex_);
  const auto [it, inserted] = kernel_specs_.insert(std::make_pair(
      std::make_tuple(std::type_index(type), platform_id), kernel));
  if (!inserted) {
    return absl::AlreadyExistsError(
        absl::StrFormat("Kernel %s for platform %s is already registered.",
                        type.name(), GetPlatformName(platform_id)));
  }
  return absl::OkStatus();
}

}  // namespace stream_executor::gpu
