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

#include "xla/service/cpu/cpu_aot_loader.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "xla/backends/cpu/codegen/cpu_features.h"
#include "xla/backends/cpu/codegen/execution_engine.h"
#include "xla/backends/cpu/codegen/ir_compiler.h"
#include "xla/backends/cpu/codegen/object_loader.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/cpu_aot_compilation_result.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/service/cpu/runtime_symbol_generator.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::cpu {

llvm::TargetOptions CompilerTargetOptions(
    const HloModuleConfig& module_config) {
  llvm::TargetOptions target_options;
  // Always allow FMA fusion. This increases precision instead of decreasing it.
  target_options.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  return target_options;
}

absl::StatusOr<std::vector<FunctionLibrary::Symbol>>
GetCompiledSymbolsFromProto(
    absl::Span<const SymbolProto> compiled_symbols_proto) {
  std::vector<FunctionLibrary::Symbol> compiled_symbols;
  for (const auto& symbol_proto : compiled_symbols_proto) {
    switch (symbol_proto.function_type_id()) {
      case SymbolProto::KERNEL:
        compiled_symbols.push_back(
            FunctionLibrary::Sym<FunctionLibrary::Kernel>(symbol_proto.name()));
        break;
      case SymbolProto::COMPARATOR:
        compiled_symbols.push_back(
            FunctionLibrary::Sym<FunctionLibrary::Comparator>(
                symbol_proto.name()));
        break;
      default:
        return Internal(
            "Unknown function type id %s",
            SymbolProto_FunctionTypeId_Name(symbol_proto.function_type_id()));
    }
  }
  VLOG(3) << "Collected " << compiled_symbols.size() << " compiled symbols";
  for (const auto& symbol : compiled_symbols) {
    VLOG(3) << " Symbol: " << symbol.name;
  }

  return compiled_symbols;
}

absl::StatusOr<std::unique_ptr<FunctionLibrary>> LoadFunctionLibrary(
    const std::vector<FunctionLibrary::Symbol>& compiled_symbols,
    absl::Span<const ObjFileProto> obj_files, const HloModule* hlo_module) {
  const HloModuleConfig& config = hlo_module->config();
  const DebugOptions& debug_options = config.debug_options();
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<llvm::TargetMachine> target_machine,
      IrCompiler::InferTargetMachine(
          std::move(CompilerTargetOptions(hlo_module->config())),
          IrCompiler::GetCodeGenOptLevel(config),
          CpuFeatureFromString(debug_options.xla_cpu_max_isa())));

  // Definition generator to link with XLA:CPU host runtime symbols.
  ExecutionEngine::DefinitionGenerator definition_generator =
      [](const llvm::DataLayout& data_layout) {
        return std::make_unique<RuntimeSymbolGenerator>(data_layout);
      };

  ObjectLoader object_loader(/*num_dylibs=*/1,
                             target_machine->createDataLayout(),
                             definition_generator);

  for (size_t i = 0; i < object_loader.num_dylibs(); ++i) {
    object_loader.dylib(i).value()->addGenerator(
        std::make_unique<RuntimeSymbolGenerator>(
            target_machine->createDataLayout()));
  }

  for (auto& obj_file : obj_files) {
    llvm::StringRef data(obj_file.contents().data(),
                         obj_file.contents().size());
    TF_RETURN_IF_ERROR(object_loader.AddObjFile(
        llvm::MemoryBuffer::getMemBuffer(data, obj_file.name())));
  }

  return std::move(object_loader).Load(compiled_symbols);
}

absl::StatusOr<std::unique_ptr<Executable>> CpuAotLoader::LoadExecutable(
    const std::string& serialized_aot_result) {
  xla::cpu::CompilationResultProto proto;
  if (!proto.ParseFromString(serialized_aot_result)) {
    return Internal("Failed to parse serialized CpuAotCompilationResult.");
  }
  return LoadExecutable(proto);
}

absl::StatusOr<std::unique_ptr<Executable>> CpuAotLoader::LoadExecutable(
    const xla::cpu::CompilationResultProto& aot_result_proto) {
  TF_ASSIGN_OR_RETURN(auto aot_result,
                      LoadAotCompilationResult(aot_result_proto));
  return LoadExecutable(std::move(*aot_result));
}

absl::StatusOr<std::unique_ptr<Executable>> CpuAotLoader::LoadExecutable(
    xla::AotCompilationResult&& compilation_result) {
  return std::move(compilation_result).LoadExecutable(nullptr, nullptr);
}

absl::StatusOr<std::unique_ptr<AotCompilationResult>>
CpuAotLoader::LoadAotCompilationResult(
    const std::string& serialized_aot_result) {
  xla::cpu::CompilationResultProto proto;
  if (!proto.ParseFromString(serialized_aot_result)) {
    return Internal("Failed to parse serialized CpuAotCompilationResult.");
  }
  return LoadAotCompilationResult(proto);
}

absl::StatusOr<std::unique_ptr<AotCompilationResult>>
CpuAotLoader::LoadAotCompilationResult(
    const xla::cpu::CompilationResultProto& aot_result_proto) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> hlo_module,
      HloModule::CreateFromProtoWithConfig(aot_result_proto.hlo_module()));
  std::vector<SymbolProto> compiled_symbols_proto;
  for (const auto& symbol_proto : aot_result_proto.compiled_symbols()) {
    compiled_symbols_proto.push_back(symbol_proto);
  }

  TF_ASSIGN_OR_RETURN(auto compiled_symbols,
                      GetCompiledSymbolsFromProto(compiled_symbols_proto));

  std::vector<ObjFileProto> obj_files;
  for (const auto& obj_file : aot_result_proto.object_files()) {
    obj_files.push_back(obj_file);
  }

  TF_ASSIGN_OR_RETURN(
      auto function_library,
      LoadFunctionLibrary(compiled_symbols, obj_files, hlo_module.get()));

  return CpuAotCompilationResult::FromProto(aot_result_proto,
                                            std::move(function_library));
}

}  // namespace xla::cpu
