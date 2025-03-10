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

#ifndef XLA_BACKENDS_CPU_CODEGEN_AOT_COMPILED_FUNCTION_LIBRARY_H_
#define XLA_BACKENDS_CPU_CODEGEN_AOT_COMPILED_FUNCTION_LIBRARY_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/runtime/function_library.h"

namespace xla::cpu {

// A AotCompiledFunctionLibrary is a FunctionLibrary that utilizes a symbol
// table to resolve function names to functions compiled into the object file
// linked with the current library.
class AotCompiledFunctionLibrary : public FunctionLibrary {
 public:
  using FunctionPtr = void*;

  // Constructs a new AotCompiledFunctionLibrary.
  //
  // `symbols_map` is a map from symbol names to resolved symbols.
  explicit AotCompiledFunctionLibrary(
      absl::flat_hash_map<std::string, FunctionPtr> symbols_map);

  // Resolves the function with the given name and type ID.
  absl::StatusOr<void*> ResolveFunction(TypeId type_id,
                                        absl::string_view name) final;

 private:
  // Caches the resolved symbols so we don't have to look them up every time a
  // function is resolved.
  absl::flat_hash_map<std::string, FunctionPtr> symbols_map_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_AOT_COMPILED_FUNCTION_LIBRARY_H_
