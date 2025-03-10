/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_CODEGEN_SYMBOL_NAME_UTIL_H_
#define XLA_BACKENDS_CPU_CODEGEN_SYMBOL_NAME_UTIL_H_

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace xla::cpu {

// A simple utility function to convert a name to a C name. Does a very simple
// replacement of dots and dashes with underscores.`
// This is relevant for CPU AOT compilation where we need to generate C names
// for symbols.
absl::StatusOr<std::string> ConvertToCName(absl::string_view name);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_SYMBOL_NAME_UTIL_H_
