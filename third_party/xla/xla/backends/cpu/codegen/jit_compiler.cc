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

#include "xla/backends/cpu/codegen/jit_compiler.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "xla/backends/cpu/codegen/contiguous_section_memory_manager.h"
#include "xla/backends/cpu/codegen/cpu_features.h"
#include "xla/backends/cpu/codegen/function_library.h"
#include "xla/backends/cpu/codegen/ir_compiler.h"
#include "xla/service/cpu/orc_jit_memory_mapper.h"
#include "xla/util.h"
#include "tsl/platform/cpu_info.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla::cpu {

absl::StatusOr<std::unique_ptr<llvm::TargetMachine>>
JitCompiler::InferTargetMachine(
    const llvm::TargetOptions& target_options, llvm::CodeGenOptLevel opt_level,
    std::optional<tsl::port::CPUFeature> max_cpu_feature) {
  // Detect machine attributes for the target CPU.
  auto result = DetectMachineAttributes(max_cpu_feature);
  llvm::SmallVector<std::string> attrs(result.features.begin(),
                                       result.features.end());

  // If `max_cpu_feature` is newer than the host CPU, we should keep the host
  // CPU name, e.g., we don't want to set the target CPU to Skylake when we are
  // on a Broadwell host.
  std::string_view cpu = result.num_filtered_features
                             ? CpuTargetFromMaxFeature(*max_cpu_feature)
                             : std::string_view(llvm::sys::getHostCPUName());

  std::unique_ptr<llvm::TargetMachine> target_machine(
      llvm::EngineBuilder()
          .setTargetOptions(target_options)
          .setOptLevel(opt_level)
          .selectTarget(
              /*TargetTriple=*/llvm::Triple(), /*MArch=*/"",
              /*MCPU=*/cpu,
              /*MAttrs=*/attrs));

  if (target_machine == nullptr) {
    return Internal("Failed to create target machine for CPU %s", cpu);
  }

  return std::move(target_machine);
}

IrCompiler::TargetMachineBuilder JitCompiler::InferTargetMachineBuilder(
    const llvm::TargetOptions& target_options, llvm::CodeGenOptLevel opt_level,
    std::optional<tsl::port::CPUFeature> max_cpu_feature) {
  return [target_options, opt_level, max_cpu_feature] {
    return InferTargetMachine(target_options, opt_level, max_cpu_feature);
  };
}

absl::StatusOr<JitCompiler> JitCompiler::Create(
    llvm::TargetOptions target_options, llvm::CodeGenOptLevel opt_level,
    Options options) {
  // Initialize LLVM the first time `JitCompiler` is created.
  static bool llvm_initialized = [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    return true;
  }();
  CHECK(llvm_initialized) << "LLVM must be initialized";

  // Infer target machine from the current host CPU.
  IrCompiler::TargetMachineBuilder target_machine_builder =
      InferTargetMachineBuilder(std::move(target_options), opt_level,
                                options.max_cpu_feature);
  TF_ASSIGN_OR_RETURN(auto target_machine, target_machine_builder());

  // LLVM execution session that holds jit-compiled functions.
  auto execution_session = std::make_unique<llvm::orc::ExecutionSession>(
      std::make_unique<llvm::orc::UnsupportedExecutorProcessControl>());

  // Create an instance of IrCompiler for lowering LLVM modules to machine code.
  auto ir_compiler = std::make_unique<IrCompiler>(
      target_machine_builder, std::move(options.ir_compiler_options),
      std::move(options.ir_compiler_hooks));

  return JitCompiler(std::move(target_machine_builder),
                     std::move(target_machine), std::move(execution_session),
                     std::move(ir_compiler), options.num_dylibs);
}

static std::unique_ptr<llvm::orc::RTDyldObjectLinkingLayer>
CreateObjectLinkingLayer(llvm::orc::ExecutionSession& execution_session) {
  return std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(
      execution_session, [] {
        return std::make_unique<ContiguousSectionMemoryManager>(
            orc_jit_memory_mapper::GetInstance());
      });
}

static std::unique_ptr<llvm::orc::IRCompileLayer> CreateCompileLayer(
    llvm::orc::ExecutionSession& execution_session,
    llvm::orc::RTDyldObjectLinkingLayer& object_linking_layer,
    std::unique_ptr<IrCompiler> ir_compiler) {
  return std::make_unique<llvm::orc::IRCompileLayer>(
      execution_session, object_linking_layer, std::move(ir_compiler));
}

JitCompiler::JitCompiler(
    IrCompiler::TargetMachineBuilder target_machine_builder,
    std::shared_ptr<llvm::TargetMachine> target_machine,
    std::unique_ptr<llvm::orc::ExecutionSession> execution_session,
    std::unique_ptr<IrCompiler> ir_compiler, size_t num_dylibs)
    : target_machine_builder_(std::move(target_machine_builder)),
      target_machine_(std::move(target_machine)),
      execution_session_(std::move(execution_session)),
      object_linking_layer_(CreateObjectLinkingLayer(*execution_session_)),
      compile_layer_(CreateCompileLayer(
          *execution_session_, *object_linking_layer_, std::move(ir_compiler))),
      gdb_(llvm::JITEventListener::createGDBRegistrationListener()),
      perf_(llvm::JITEventListener::createPerfJITEventListener()) {
  // Create at least one dynamic library for the given jit compiler.
  dylibs_.resize(std::max<size_t>(1, num_dylibs));
  for (size_t i = 0; i < dylibs_.size(); ++i) {
    dylibs_[i] = &execution_session_->createBareJITDylib(
        absl::StrCat("<xla_jit_dylib_", i, ">"));
  }

  // Register GDB and perf event listeners with the object linking layer.
  if (gdb_) object_linking_layer_->registerJITEventListener(*gdb_);
  if (perf_) object_linking_layer_->registerJITEventListener(*perf_);
}

JitCompiler::~JitCompiler() {
  if (execution_session_) {
    if (auto err = execution_session_->endSession()) {
      execution_session_->reportError(std::move(err));
    }
  }
}

absl::Status JitCompiler::AddModule(llvm::orc::ThreadSafeModule module,
                                    size_t dylib_index) {
  if (dylib_index >= dylibs_.size()) {
    return Internal("Invalid dylib index %d (num dylibs: %d))", dylib_index,
                    dylibs_.size());
  }

  // Set up module for codegen for the target machine at hand.
  module.withModuleDo([&](llvm::Module& m) {
    m.setDataLayout(target_machine_->createDataLayout());
    m.setTargetTriple(target_machine_->getTargetTriple().getTriple());
  });

  // Add module to the selected dynamic library.
  llvm::orc::JITDylib* dylib = dylibs_[dylib_index];
  if (auto err = compile_layer_->add(*dylib, std::move(module))) {
    return Internal("Failed to add module to dylib %d: %s", dylib_index,
                    llvm::toString(std::move(err)));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<FunctionLibrary>> JitCompiler::Compile(
    absl::Span<const Symbol> symbols) && {
  // Mangle symbol names for the target machine data layout.
  llvm::DataLayout data_layout = target_machine_->createDataLayout();
  auto mangle = [&](std::string_view name) {
    llvm::SmallVector<char, 40> mangled;
    llvm::Mangler::getNameWithPrefix(mangled, name, data_layout);
    return std::string(mangled.begin(), mangled.end());
  };

  // Resolve type-erased symbol pointers.
  using ResolvedSymbol = CompiledFunctionLibrary::ResolvedSymbol;
  absl::flat_hash_map<std::string, ResolvedSymbol> symbols_map;

  // TODO(ezhulenev): Use task runner to parallelize symbol lookups.
  for (const auto& symbol : symbols) {
    std::string mangled = mangle(symbol.name);
    VLOG(3) << absl::StreamFormat("Look up symbol: %s (mangled: %s)",
                                  symbol.name, mangled);

    auto symbol_def = execution_session_->lookup(dylibs_, mangled);
    if (auto err = symbol_def.takeError()) {
      return Internal("Failed to lookup symbol %s: %s", symbol.name,
                      llvm::toString(std::move(err)));
    }

    llvm::orc::ExecutorAddr addr = symbol_def->getAddress();
    void* ptr = reinterpret_cast<void*>(addr.getValue());
    symbols_map[symbol.name] = ResolvedSymbol{symbol.type_id, ptr};
  }

  return std::make_unique<CompiledFunctionLibrary>(
      std::move(execution_session_), std::move(symbols_map));
}

JitCompiler::CompiledFunctionLibrary::CompiledFunctionLibrary(
    std::unique_ptr<llvm::orc::ExecutionSession> execution_session,
    absl::flat_hash_map<std::string, ResolvedSymbol> symbols_map)
    : execution_session_(std::move(execution_session)),
      symbols_map_(std::move(symbols_map)) {
  DCHECK(execution_session_) << "Execution session must not be null";
}

JitCompiler::CompiledFunctionLibrary::~CompiledFunctionLibrary() {
  if (auto err = execution_session_->endSession()) {
    execution_session_->reportError(std::move(err));
  }
}

absl::StatusOr<void*> JitCompiler::CompiledFunctionLibrary::ResolveFunction(
    TypeId type_id, std::string_view name) {
  if (auto it = symbols_map_.find(name); it != symbols_map_.end()) {
    if (it->second.type_id != type_id) {
      return Internal("Symbol %s has type id %d, expected %d", name,
                      it->second.type_id.value(), type_id.value());
    }
    return it->second.ptr;
  }
  return NotFound("Function %s not found (type id: %d)", name, type_id.value());
}

}  // namespace xla::cpu
