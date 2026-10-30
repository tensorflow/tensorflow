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

#include "xla/hlo/pass/hlo_pass_registry.h"

#include <memory>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/hlo/pass/hlo_pass_context.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/pass/hlo_pass_options.h"

namespace xla {

HloPassRegistry& HloPassRegistry::Global() {
  static absl::NoDestructor<HloPassRegistry> registry;
  return *registry;
}

absl::StatusOr<const HloPassFactory* absl_nonnull> HloPassRegistry::Register(
    std::unique_ptr<HloPassFactory> factory) {
  // Stays valid after the move: the factory object itself doesn't move.
  absl::string_view name = factory->name();
  absl::MutexLock lock(mu_);
  auto [it, inserted] = factories_.try_emplace(name, std::move(factory));
  if (!inserted) {
    return absl::AlreadyExistsError(
        absl::StrCat("HloPassRegistry already has pass '", name, "'"));
  }
  return it->second.get();
}

absl::StatusOr<const HloPassFactory* absl_nonnull> HloPassRegistry::Get(
    absl::string_view name) const {
  absl::MutexLock lock(mu_);
  if (auto it = factories_.find(name); it != factories_.end()) {
    return it->second.get();
  }
  return absl::NotFoundError(
      absl::StrCat("HloPassRegistry has no pass '", name, "'"));
}

absl::StatusOr<std::unique_ptr<HloPassInterface>> HloPassRegistry::Create(
    absl::string_view name, const HloPassContext& context,
    const HloPassOptions& options) const {
  ABSL_ASSIGN_OR_RETURN(const HloPassFactory* factory, Get(name));
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloPassInterface> pass,
                   factory->Create(context, options));
  ABSL_RETURN_IF_ERROR(options.CheckAllConsumed());
  if (pass->name() != name) {
    return absl::InternalError(absl::StrCat("HloPassFactory '", name,
                                            "' created pass with name '",
                                            pass->name(), "'"));
  }
  return pass;
}

}  // namespace xla
