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

#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_handler_registry.h"

#include <optional>
#include <string>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace xla::gpu {

NativeCustomCallHandlerRegistry& NativeCustomCallHandlerRegistry::GetGlobal() {
  static absl::NoDestructor<NativeCustomCallHandlerRegistry> registry;
  return *registry;
}

std::optional<NativeCustomCallHandlerRef>
NativeCustomCallHandlerRegistry::Lookup(absl::string_view target) const {
  auto it = handlers_.find(target);
  if (it == handlers_.end()) {
    return std::nullopt;
  }
  return it->second;
}

absl::Status NativeCustomCallHandlerRegistry::Register(
    absl::string_view target, NativeCustomCallHandler handler) {
  if (handler == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("Null native custom-call handler for target: ", target));
  }
  auto [_, inserted] =
      handlers_.emplace(std::string(target), std::move(handler));
  if (!inserted) {
    return absl::AlreadyExistsError(absl::StrCat(
        "Native custom-call handler already registered for target: ", target));
  }
  return absl::OkStatus();
}

}  // namespace xla::gpu
