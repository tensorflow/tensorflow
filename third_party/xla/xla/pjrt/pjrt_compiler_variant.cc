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

#include "absl/status/statusor.h"

namespace xla {
namespace {

PickCompilerVariantFunc g_pick_compiler_variant_func = nullptr;
CompilerVariantToStringFunc g_compiler_variant_to_string_func = nullptr;

}  // namespace

void RegisterPickCompilerVariantFunc(PickCompilerVariantFunc func) {
  g_pick_compiler_variant_func = func;
}

void RegisterCompilerVariantToStringFunc(CompilerVariantToStringFunc func) {
  g_compiler_variant_to_string_func = func;
}

absl::StatusOr<CompilerVariant> PickCompilerVariant(bool fallback_allowed) {
  if (g_pick_compiler_variant_func != nullptr) {
    return g_pick_compiler_variant_func(fallback_allowed);
  }
  return CompilerVariant::kLinked;
}

std::string CompilerVariantToString(CompilerVariant v) {
  if (g_compiler_variant_to_string_func != nullptr) {
    return g_compiler_variant_to_string_func(v);
  }
  return "";
}

}  // namespace xla
