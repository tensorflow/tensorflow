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
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "xla/backends/cpu/codegen/builtin_definition_generator.h"
#include "xla/backends/cpu/codegen/cpu_features.h"
#include "xla/backends/cpu/codegen/execution_engine.h"
#include "xla/backends/cpu/codegen/ir_compiler.h"
#include "xla/backends/cpu/codegen/object_loader.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/cpu_aot_compilation_result.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/llvm_ir/llvm_command_line_options.h"
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
    absl::Span<const ObjFileProto> obj_files, const HloModule* hlo_module,
    const TargetMachineOptions& target_machine_options) {
  const HloModuleConfig& config = hlo_module->config();

  auto llvm_options = llvm_ir::ExtractXlaBackendExtraOptions(
      config.debug_options().xla_backend_extra_options());
  llvm_ir::LLVMCommandLineOptionsLock llvm_lock(llvm_options);

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<llvm::TargetMachine> target_machine,
      IrCompiler::InferTargetMachine(
          std::move(CompilerTargetOptions(hlo_module->config())),
          IrCompiler::GetCodeGenOptLevel(config), target_machine_options));

  // Definition generator to link with XLA:CPU host runtime symbols.
  ExecutionEngine::DefinitionGenerator definition_generator =
      [](const llvm::DataLayout& data_layout) {
        return std::make_unique<BuiltinDefinitionGenerator>(data_layout);
      };

  ObjectLoader object_loader(/*num_dylibs=*/1,
                             target_machine->createDataLayout(),
                             definition_generator);

  for (size_t i = 0; i < object_loader.num_dylibs(); ++i) {
    object_loader.dylib(i).value()->addGenerator(
        std::make_unique<BuiltinDefinitionGenerator>(
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
    CompiledModule&& compilation_result) {
  return std::move(compilation_result).LoadExecutable();
}

absl::StatusOr<std::unique_ptr<CompiledModule>>
CpuAotLoader::LoadAotCompilationResult(
    const std::string& serialized_aot_result) {
  xla::cpu::CompilationResultProto proto;
  if (!proto.ParseFromString(serialized_aot_result)) {
    return Internal("Failed to parse serialized CpuAotCompilationResult.");
  }
  return LoadAotCompilationResult(proto);
}

absl::StatusOr<std::unique_ptr<CompiledModule>>
CpuAotLoader::LoadAotCompilationResult(
    const xla::cpu::CompilationResultProto& aot_result_proto) {
  VLOG(3) << "AOT result target machine options: "
          << aot_result_proto.target_machine_options().DebugString();

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> hlo_module,
      HloModule::CreateFromProtoWithConfig(aot_result_proto.hlo_module()));

  auto llvm_options = llvm_ir::ExtractXlaBackendExtraOptions(
      hlo_module->config().debug_options().xla_backend_extra_options());
  llvm_ir::LLVMCommandLineOptionsLock llvm_lock(llvm_options);

  TF_ASSIGN_OR_RETURN(TargetMachineOptions target_machine_options,
                      TargetMachineOptions::FromProto(
                          aot_result_proto.target_machine_options()));

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<llvm::TargetMachine> target_machine,
      IrCompiler::InferTargetMachine(
          std::move(CompilerTargetOptions(hlo_module->config())),
          IrCompiler::GetCodeGenOptLevel(hlo_module->config()),
          target_machine_options));

  llvm::Triple triple(target_machine_options.triple());
  llvm::Triple expected_triple(target_machine->getTargetTriple());
  if (triple.getArchName() != expected_triple.getArchName()) {
    return Internal("Target arch mismatch expected %s got %s.",
                    expected_triple.getArchName(), triple.getArchName());
  }

  llvm::StringMap<bool> host_machine_features = llvm::sys::getHostCPUFeatures();
  std::vector<std::string> compile_machine_features =
      target_machine_options.GetTargetMachineFeaturesVector();
  // Convert the supported features to a vector of strings.
  std::vector<std::string> host_machine_features_vector;
  for (const auto& [feature, supported] : host_machine_features) {
    if (supported) {
      host_machine_features_vector.push_back(feature.str());
    }
  }

  VLOG(3) << "Host machine options:"
          << "\nHost CPU: " << llvm::sys::getHostCPUName().str()
          << "\nHost triple: " << llvm::sys::getDefaultTargetTriple()
          << "\nHost features: "
          << absl::StrJoin(host_machine_features_vector, ",");

  for (const absl::string_view feature : compile_machine_features) {
    if (feature[0] == '+' &&
        (!host_machine_features.contains(feature.substr(1)) ||
         !host_machine_features[feature.substr(1)])) {
      // TODO: b/477590953 - Turn this warning into an absl::Status Internal
      // error once a mechanism for passing CPU topology to host offloaded
      // programs is implemented.
      LOG(ERROR)
          << "Loading XLA:CPU AOT result. Target machine feature " << feature
          << " is not  supported on the host machine. Machine type used for "
             "XLA:CPU compilation doesn't match the machine type for "
             "execution. Compile machine features: ["
          << absl::StrJoin(compile_machine_features, ",")
          << "] vs host machine features: ["
          << absl::StrJoin(host_machine_features_vector, ",") << "]"
          << ". This could lead to execution errors such as SIGILL.";
    }
  }

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
      LoadFunctionLibrary(compiled_symbols, obj_files, hlo_module.get(),
                          target_machine_options));

  return CpuAotCompilationResult::FromProto(aot_result_proto,
                                            std::move(function_library));
}

}  // namespace xla::cpu
