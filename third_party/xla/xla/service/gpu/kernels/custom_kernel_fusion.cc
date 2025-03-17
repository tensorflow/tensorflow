/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/kernels/custom_kernel_fusion.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// CustomKernelFusionRegistry
//===----------------------------------------------------------------------===//

CustomKernelFusionRegistry* CustomKernelFusionRegistry::Default() {
  static auto* registry = new CustomKernelFusionRegistry();
  return registry;
}

absl::Status CustomKernelFusionRegistry::Register(
    std::string name, std::unique_ptr<CustomKernelFusion> fusion) {
  absl::MutexLock lock(&mutex_);
  if (auto it = registry_.try_emplace(name, std::move(fusion)); it.second)
    return absl::OkStatus();
  return absl::InternalError(
      absl::StrCat("Custom kernel fusion ", name, " already registered."));
}

CustomKernelFusion* CustomKernelFusionRegistry::Lookup(
    absl::string_view name) const {
  absl::MutexLock lock(&mutex_);
  if (auto it = registry_.find(name); it != registry_.end())
    return it->second.get();
  return nullptr;
}

}  // namespace xla::gpu
