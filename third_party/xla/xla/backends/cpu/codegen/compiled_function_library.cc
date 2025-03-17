/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/cpu/codegen/compiled_function_library.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/codegen/execution_engine.h"
#include "xla/backends/cpu/runtime/function_library.h"

namespace xla::cpu {

CompiledFunctionLibrary::CompiledFunctionLibrary(
    std::unique_ptr<ExecutionEngine> execution_engine,
    absl::flat_hash_map<std::string, ResolvedSymbol> symbols_map)
    : execution_engine_(std::move(execution_engine)),
      symbols_map_(std::move(symbols_map)) {
  DCHECK(execution_engine_) << "Execution engine must not be null";
}

absl::StatusOr<void*> CompiledFunctionLibrary::ResolveFunction(
    TypeId type_id, absl::string_view name) {
  if (auto it = symbols_map_.find(name); it != symbols_map_.end()) {
    if (it->second.type_id != type_id) {
      return absl::Status(
          absl::StatusCode::kInternal,
          absl::StrFormat("Symbol %s has type id %d, expected %d", name,
                          it->second.type_id.value(), type_id.value()));
    }
    return it->second.ptr;
  }
  return absl::Status(absl::StatusCode::kNotFound,
                      absl::StrFormat("Function %s not found (type id: %d)",
                                      name, type_id.value()));
}

}  // namespace xla::cpu
