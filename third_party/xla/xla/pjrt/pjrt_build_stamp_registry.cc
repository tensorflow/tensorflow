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

#include "xla/pjrt/pjrt_build_stamp_registry.h"

#include <memory>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/call_once.h"
#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/pjrt/pjrt_build_stamp.h"
#include "xla/pjrt/pjrt_compiler_variant.h"

namespace xla {

PjRtBuildStampRegistry& PjRtBuildStampRegistry::Global() {
  static absl::NoDestructor<PjRtBuildStampRegistry> registry;
  return *registry;
}

absl::Status PjRtBuildStampRegistry::RegisterBuildStampFactory(
    absl::string_view platform_name, Factory factory) {
  return RegisterBuildStampFactory(platform_name, kLinkedVariant,
                                   std::move(factory));
}

absl::Status PjRtBuildStampRegistry::RegisterBuildStampFactory(
    absl::string_view platform_name, absl::string_view variant_name,
    Factory factory) {
  const PjRtBuildStampKey key(platform_name, variant_name);
  absl::MutexLock lock(mu_);
  auto [it, inserted] = entries_.try_emplace(key, nullptr);
  if (!inserted) {
    return absl::AlreadyExistsError(
        absl::StrCat("Build stamp factory already registered for platform '",
                     platform_name, "' and variant '", variant_name, "'"));
  }
  it->second = std::make_unique<Entry>(std::move(factory));
  return absl::OkStatus();
}

absl::StatusOr<const PjRtBuildStamp* absl_nonnull>
PjRtBuildStampRegistry::GetBuildStamp(absl::string_view platform_name,
                                      absl::string_view variant_name)
    ABSL_ATTRIBUTE_LIFETIME_BOUND {
  const PjRtBuildStampKey key(platform_name, variant_name);
  Entry* entry = nullptr;
  {
    absl::MutexLock lock(mu_);
    auto it = entries_.find(key);
    if (it == entries_.end()) {
      return absl::NotFoundError(absl::StrCat(
          "No registered build stamp factory for platform '", platform_name,
          "' and compiler variant '", variant_name, "'"));
    }
    entry = it->second.get();
  }

  absl::call_once(entry->once, [&]() { entry->memoized = entry->factory(); });

  if (!entry->memoized.ok()) {
    return entry->memoized.status();
  }
  return &*entry->memoized;
}

void PjRtRegisterBuildStampFactory(absl::string_view platform_name,
                                   PjRtBuildStampRegistry::Factory factory) {
  CHECK_OK(PjRtBuildStampRegistry::Global().RegisterBuildStampFactory(
      platform_name, std::move(factory)));
}

void PjRtRegisterBuildStampFactory(absl::string_view platform_name,
                                   absl::string_view variant_name,
                                   PjRtBuildStampRegistry::Factory factory) {
  CHECK_OK(PjRtBuildStampRegistry::Global().RegisterBuildStampFactory(
      platform_name, variant_name, std::move(factory)));
}

absl::StatusOr<const PjRtBuildStamp* absl_nonnull> PjRtGetBuildStamp(
    absl::string_view platform_name) {
  return PjRtBuildStampRegistry::Global().GetBuildStamp(platform_name);
}

absl::StatusOr<const PjRtBuildStamp* absl_nonnull> PjRtGetBuildStamp(
    absl::string_view platform_name, absl::string_view variant_name) {
  return PjRtBuildStampRegistry::Global().GetBuildStamp(platform_name,
                                                        variant_name);
}

absl::StatusOr<const PjRtBuildStamp* absl_nonnull> PjRtGetBuildStamp(
    absl::string_view platform_name, PjRtCompilerVariant variant) {
  return PjRtGetBuildStamp(platform_name, CompilerVariantToString(variant));
}

}  // namespace xla
