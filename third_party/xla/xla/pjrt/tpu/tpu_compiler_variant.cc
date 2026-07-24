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

#include "xla/pjrt/tpu/tpu_compiler_variant.h"

#include <string>

#include "absl/base/attributes.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_compiler_variant.h"

namespace xla {
ABSL_ATTRIBUTE_WEAK absl::StatusOr<PjRtCompilerVariant>
PickTpuCompilerVariant() {
  return LinkedCompilerVariantId();
}

namespace {
bool RegisterTpuVariantPicker() {
  PjRtRegisterCompilerVariantPicker(
      "tpu",
      []() -> absl::StatusOr<std::string> {
        ASSIGN_OR_RETURN(PjRtCompilerVariant variant, PickTpuCompilerVariant(),
                         _);
        std::string compiler_variant = CompilerVariantToString(variant);
        return compiler_variant == kLinkedVariant ? "" : compiler_variant;
      },
      /*is_weak=*/true);
  return true;
}
bool tpu_variant_picker_registered = RegisterTpuVariantPicker();
}  // namespace

}  // namespace xla
