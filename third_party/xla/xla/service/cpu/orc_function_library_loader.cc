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

#include "xla/service/cpu/orc_function_library_loader.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/any_span.h"
#include "absl/types/span.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/MemoryBuffer.h"
#include "xla/backends/cpu/codegen/builtin_definition_generator.h"
#include "xla/backends/cpu/codegen/execution_engine.h"
#include "xla/backends/cpu/codegen/object_loader.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/service/cpu/function_library_loader.h"
#include "xla/util.h"

namespace xla::cpu {

absl::StatusOr<std::unique_ptr<FunctionLibrary>> LoadObjectFilesWithOrc(
    absl::Span<const FunctionLibrary::Symbol> compiled_symbols,
    absl::AnySpan<const ObjFileProto> obj_files,
    absl::string_view data_layout_str) {
  llvm::DataLayout data_layout(
      llvm::StringRef(data_layout_str.data(), data_layout_str.size()));

  // Definition generator to link with XLA:CPU host runtime symbols.
  ExecutionEngine::DefinitionGenerator definition_generator =
      [](const llvm::DataLayout& data_layout) {
        return std::make_unique<BuiltinDefinitionGenerator>(data_layout);
      };

  ObjectLoader object_loader(/*num_dylibs=*/1, data_layout,
                             definition_generator);

  for (size_t i = 0; i < object_loader.num_dylibs(); ++i) {
    ABSL_ASSIGN_OR_RETURN(llvm::orc::JITDylib * dylib, object_loader.dylib(i));
    dylib->addGenerator(
        std::make_unique<BuiltinDefinitionGenerator>(data_layout));
  }

  for (const ObjFileProto& obj_file : obj_files) {
    llvm::StringRef data(obj_file.contents().data(),
                         obj_file.contents().size());
    ABSL_RETURN_IF_ERROR(object_loader.AddObjFile(
        llvm::MemoryBuffer::getMemBuffer(data, obj_file.name())));
  }

  return std::move(object_loader).Load(compiled_symbols);
}

absl::StatusOr<std::unique_ptr<FunctionLibrary>> LoadFunctionLibraryWithOrc(
    const CompilationResultProto& proto) {
  ABSL_ASSIGN_OR_RETURN(std::vector<FunctionLibrary::Symbol> compiled_symbols,
                   GetCompiledSymbolsFromProto(proto.compiled_symbols()));
  if (proto.object_files().empty() && !compiled_symbols.empty()) {
    return FailedPrecondition(
        "CompilationResultProto has no object_files to load with ORC.");
  }
  return LoadObjectFilesWithOrc(compiled_symbols, proto.object_files(),
                                proto.data_layout());
}

}  // namespace xla::cpu
