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

#ifndef XLA_BACKENDS_CPU_CODEGEN_BUILTIN_SYMBOLS_H_
#define XLA_BACKENDS_CPU_CODEGEN_BUILTIN_SYMBOLS_H_

#include "absl/base/nullability.h"
#include "absl/strings/string_view.h"

namespace xla::cpu {

// Returns the address in the current process of the builtin runtime symbol
// `name` that XLA:CPU executables may reference (e.g. memcpy, libm functions
// or compiler-rt conversions), or nullptr if `name` isn't a builtin. `name`
// must not have a global prefix (e.g. the leading underscore on macOS).
void* absl_nullable FindBuiltinSymbol(absl::string_view name);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_BUILTIN_SYMBOLS_H_
