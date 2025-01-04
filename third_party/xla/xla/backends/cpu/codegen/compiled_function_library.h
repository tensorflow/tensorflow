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

#ifndef XLA_BACKENDS_CPU_CODEGEN_COMPILED_FUNCTION_LIBRARY_H_
#define XLA_BACKENDS_CPU_CODEGEN_COMPILED_FUNCTION_LIBRARY_H_

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "xla/backends/cpu/runtime/function_library.h"

namespace xla::cpu {

class CompiledFunctionLibrary : public FunctionLibrary {
 public:
  struct ResolvedSymbol {
    TypeId type_id;
    void* ptr;
  };

  CompiledFunctionLibrary(
      std::unique_ptr<llvm::orc::ExecutionSession> execution_session,
      std::unique_ptr<llvm::orc::RTDyldObjectLinkingLayer> object_layer,
      absl::flat_hash_map<std::string, ResolvedSymbol> symbols_map);

  ~CompiledFunctionLibrary() final;

  absl::StatusOr<void*> ResolveFunction(TypeId type_id,
                                        absl::string_view name) final;

 private:
  std::unique_ptr<llvm::orc::ExecutionSession> execution_session_;
  std::unique_ptr<llvm::orc::RTDyldObjectLinkingLayer> object_layer_;
  absl::flat_hash_map<std::string, ResolvedSymbol> symbols_map_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_COMPILED_FUNCTION_LIBRARY_H_
