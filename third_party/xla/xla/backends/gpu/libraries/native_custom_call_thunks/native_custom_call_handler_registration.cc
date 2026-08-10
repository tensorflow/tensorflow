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

#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_handler_registration.h"

#include <utility>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_handler_registry.h"

namespace xla::gpu {
namespace native_custom_call_internal {

Registrar::Registrar(absl::string_view target,
                     NativeCustomCallHandler handler) {
  CHECK_OK(NativeCustomCallHandlerRegistry::GetGlobal().Register(
      target, std::move(handler)));
}

}  // namespace native_custom_call_internal
}  // namespace xla::gpu
