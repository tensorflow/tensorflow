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

#include "xla/pjrt/pjrt_compiler_variant.h"

#include <string>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/no_destructor.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/pjrt/pjrt_common.h"
#include "tsl/platform/fingerprint.h"

namespace xla {

ABSL_ATTRIBUTE_WEAK std::string CompilerVariantToString(
    PjRtCompilerVariant variant) {
  if (variant == LinkedCompilerVariantId()) {
    return std::string(kLinkedVariant);
  }
  return std::string(kUnknownVariant);
}

PjRtCompilerVariantPickerRegistry& PjRtCompilerVariantPickerRegistry::Global() {
  static absl::NoDestructor<PjRtCompilerVariantPickerRegistry> registry;
  return *registry;
}

void PjRtCompilerVariantPickerRegistry::RegisterVariantPicker(
    PjRtPlatformId platform_id, PjRtCompilerVariantPicker picker,
    bool is_weak) {
  absl::MutexLock lock(mu_);
  auto it = variant_pickers_.find(platform_id);
  if (it != variant_pickers_.end()) {
    if (!is_weak || it->second.is_weak) {
      it->second = {std::move(picker), is_weak};
    }
    return;
  }
  variant_pickers_.emplace(platform_id, Entry{std::move(picker), is_weak});
}

void PjRtCompilerVariantPickerRegistry::RegisterVariantPicker(
    absl::string_view platform_name, PjRtCompilerVariantPicker picker,
    bool is_weak) {
  RegisterVariantPicker(tsl::Fingerprint64(platform_name), std::move(picker),
                        is_weak);
}

absl::StatusOr<std::string> PjRtCompilerVariantPickerRegistry::PickVariantName(
    PjRtPlatformId platform_id) const {
  PjRtCompilerVariantPicker picker;
  {
    absl::MutexLock lock(mu_);
    auto it = variant_pickers_.find(platform_id);
    if (it == variant_pickers_.end()) {
      return std::string(kLinkedVariant);
    }
    picker = it->second.picker;
  }
  ABSL_ASSIGN_OR_RETURN(std::string variant_name, picker());
  return variant_name.empty() ? std::string(kLinkedVariant) : variant_name;
}

absl::StatusOr<std::string> PjRtCompilerVariantPickerRegistry::PickVariantName(
    absl::string_view platform_name) const {
  return PickVariantName(tsl::Fingerprint64(platform_name));
}

absl::StatusOr<PjRtCompilerVariant>
PjRtCompilerVariantPickerRegistry::PickVariant(
    PjRtPlatformId platform_id) const {
  ABSL_ASSIGN_OR_RETURN(std::string variant_name, PickVariantName(platform_id));
  return CompilerVariantFromName(variant_name);
}

absl::StatusOr<PjRtCompilerVariant>
PjRtCompilerVariantPickerRegistry::PickVariant(
    absl::string_view platform_name) const {
  return PickVariant(tsl::Fingerprint64(platform_name));
}

void PjRtRegisterCompilerVariantPicker(
    PjRtPlatformId platform_id, PjRtCompilerVariantPicker picker, bool is_weak,
    PjRtCompilerVariantPickerRegistry& registry) {
  registry.RegisterVariantPicker(platform_id, std::move(picker), is_weak);
}

void PjRtRegisterCompilerVariantPicker(
    absl::string_view platform_name, PjRtCompilerVariantPicker picker,
    bool is_weak, PjRtCompilerVariantPickerRegistry& registry) {
  registry.RegisterVariantPicker(platform_name, std::move(picker), is_weak);
}

}  // namespace xla
