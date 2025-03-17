/* Copyright 2024 The OpenXLA Authors. All Rights Reserved.

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

#include <windows.h>

#include <string>

#include "absl/status/statusor.h"
#include "xla/mlir/tools/mlir_interpreter/dialects/symbol_finder.h"

namespace mlir {
namespace interpreter {

absl::StatusOr<void*> FindSymbolInProcess(const std::string& symbol_name) {
  HMODULE handle = GetModuleHandle(NULL);
  if (handle) {
    void* sym = GetProcAddress(handle, symbol_name.c_str());
    if (!sym) {
      return absl::NotFoundError("Callee not found");
    }
    return sym;
  } else {
    return absl::InternalError("Failed to get module handle");
  }
}
}  // namespace interpreter
}  // namespace mlir
