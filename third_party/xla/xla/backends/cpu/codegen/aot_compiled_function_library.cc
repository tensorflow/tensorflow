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

#include "xla/backends/cpu/codegen/aot_compiled_function_library.h"

#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/runtime/function_library.h"

namespace xla::cpu {

AotCompiledFunctionLibrary::AotCompiledFunctionLibrary(
    absl::flat_hash_map<std::string, FunctionPtr> symbols_map)
    : symbols_map_(std::move(symbols_map)) {}

absl::StatusOr<void*> AotCompiledFunctionLibrary::ResolveFunction(
    TypeId type_id, absl::string_view name) {
  if (auto it = symbols_map_.find(name); it != symbols_map_.end()) {
    // NOTE(basioli) there is no type checking here.
    return it->second;
  }
  return absl::Status(absl::StatusCode::kNotFound,
                      absl::StrFormat("Function %s not found (type id: %d)",
                                      name, type_id.value()));
}

}  // namespace xla::cpu
