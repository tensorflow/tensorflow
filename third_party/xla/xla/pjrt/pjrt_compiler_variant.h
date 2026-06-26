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

#ifndef XLA_PJRT_PJRT_COMPILER_VARIANT_H_
#define XLA_PJRT_PJRT_COMPILER_VARIANT_H_

#include <functional>
#include <string>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace xla {

// Represents the PJRT compiler variant to use.
// In OSS, only the default linked variant is defined. Internal implementations
// may define additional variants and cast them to this type.
enum class CompilerVariant : int {
  kLinked = 0,
};

// Function pointer types for overriding default behavior.
using PickCompilerVariantFunc = absl::StatusOr<CompilerVariant> (*)(bool);
using CompilerVariantToStringFunc = std::string (*)(CompilerVariant);

// Registers a custom implementation for PickCompilerVariant.
void RegisterPickCompilerVariantFunc(PickCompilerVariantFunc func);

// Registers a custom implementation for CompilerVariantToString.
void RegisterCompilerVariantToStringFunc(CompilerVariantToStringFunc func);

// Returns the PJRT compiler variant that should be used.
absl::StatusOr<CompilerVariant> PickCompilerVariant(
    bool fallback_allowed = true);

// Helper function to convert CompilerVariant to a string (e.g. for
// registration).
std::string CompilerVariantToString(CompilerVariant v);

}  // namespace xla

#endif  // XLA_PJRT_PJRT_COMPILER_VARIANT_H_
