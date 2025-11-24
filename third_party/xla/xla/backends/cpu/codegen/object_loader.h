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

#ifndef XLA_BACKENDS_CPU_CODEGEN_OBJECT_LOADER_H_
#define XLA_BACKENDS_CPU_CODEGEN_OBJECT_LOADER_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/CoreContainers.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/MemoryBuffer.h"
#include "xla/backends/cpu/codegen/compiled_function_library.h"
#include "xla/backends/cpu/codegen/execution_engine.h"
#include "xla/backends/cpu/runtime/function_library.h"

namespace xla::cpu {

class ObjectLoader {
 public:
  using Symbol = FunctionLibrary::Symbol;
  using ResolvedSymbol = CompiledFunctionLibrary::ResolvedSymbol;

  explicit ObjectLoader(
      size_t num_dylibs, const llvm::DataLayout& data_layout,
      ExecutionEngine::DefinitionGenerator definition_generator);
  explicit ObjectLoader(std::unique_ptr<ExecutionEngine> execution_engine);

  ObjectLoader(ObjectLoader&& other) = default;
  ObjectLoader& operator=(ObjectLoader&& other) = default;

  absl::Status AddObjFile(const std::string& obj_file,
                          const std::string& memory_buffer_name,
                          size_t dylib_index = 0);

  absl::Status AddObjFile(std::unique_ptr<llvm::MemoryBuffer> obj_file,
                          size_t dylib_index = 0);

  absl::StatusOr<llvm::orc::SymbolMap> LookupSymbols(
      absl::Span<const Symbol> symbols);

  absl::StatusOr<std::unique_ptr<FunctionLibrary>> CreateFunctionLibrary(
      absl::Span<const Symbol> symbols, llvm::orc::SymbolMap& symbol_map) &&;

  absl::StatusOr<std::unique_ptr<FunctionLibrary>> Load(
      absl::Span<const Symbol> symbols) &&;

  size_t num_dylibs() const { return execution_engine_->num_dylibs(); }

  llvm::orc::RTDyldObjectLinkingLayer* object_layer() {
    return execution_engine_->object_layer();
  }

  llvm::orc::ExecutionSession* execution_session() {
    return execution_engine_->execution_session();
  }

  absl::StatusOr<llvm::orc::JITDylib*> dylib(size_t dylib_index) {
    return execution_engine_->dylib(dylib_index);
  }

 private:
  std::function<std::string(absl::string_view)> GetMangler();

  std::unique_ptr<ExecutionEngine> execution_engine_;

  // Non-owning pointers to dynamic libraries created for the execution session.
  std::vector<llvm::orc::JITDylib*> dylibs_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_OBJECT_LOADER_H_
