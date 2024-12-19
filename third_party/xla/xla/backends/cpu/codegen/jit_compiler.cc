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
#include <utility>

#include "absl/base/call_once.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h"
#include "llvm/ExecutionEngine/Orc/TaskDispatch.h"
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
#include "xla/backends/cpu/codegen/ir_compiler.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/service/cpu/orc_jit_memory_mapper.h"
#include "xla/util.h"
#include "tsl/platform/cpu_info.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"

namespace xla::cpu {

using tsl::profiler::TraceMe;
using tsl::profiler::TraceMeEncode;

// Initialize LLVM the first time `JitCompiler` is created.
static void InitializeLLVMTarget() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
}

absl::once_flag initialize_llvm_flag;

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
  absl::string_view cpu = result.num_filtered_features
                              ? CpuTargetFromMaxFeature(*max_cpu_feature)
                              : absl::string_view(llvm::sys::getHostCPUName());

  absl::call_once(initialize_llvm_flag, InitializeLLVMTarget);
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
    llvm::TargetOptions target_options, Options options,
    TaskRunner task_runner) {
  absl::call_once(initialize_llvm_flag, InitializeLLVMTarget);

  // Infer target machine from the current host CPU.
  IrCompiler::TargetMachineBuilder target_machine_builder =
      InferTargetMachineBuilder(std::move(target_options),
                                options.ir_compiler_options.opt_level,
                                options.max_cpu_feature);
  TF_ASSIGN_OR_RETURN(auto target_machine, target_machine_builder());

  // Dispatch compilation tasks using the provided task runner.
  auto task_dispatcher =
      std::make_unique<TaskDispatcher>(std::move(task_runner));
  TaskDispatcher* task_dispatcher_ptr = task_dispatcher.get();

  // LLVM execution session that holds jit-compiled functions.
  auto execution_session = std::make_unique<llvm::orc::ExecutionSession>(
      std::make_unique<llvm::orc::UnsupportedExecutorProcessControl>(
          /*SSP=*/nullptr, std::move(task_dispatcher)));

  execution_session->setErrorReporter([](llvm::Error err) {
    LOG(ERROR) << "LLVM compilation error: " << llvm::toString(std::move(err));
  });

  // Create an instance of IrCompiler for lowering LLVM modules to machine code.
  auto ir_compiler = std::make_unique<IrCompiler>(
      target_machine_builder, std::move(options.ir_compiler_options),
      std::move(options.ir_compiler_hooks));

  return JitCompiler(
      std::move(target_machine_builder), std::move(target_machine),
      task_dispatcher_ptr, std::move(execution_session), std::move(ir_compiler),
      options.num_dylibs, std::move(options.definition_generator));
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
    llvm::orc::RTDyldObjectLinkingLayer& object_layer,
    std::unique_ptr<IrCompiler> ir_compiler) {
  return std::make_unique<llvm::orc::IRCompileLayer>(
      execution_session, object_layer, std::move(ir_compiler));
}

JitCompiler::JitCompiler(
    IrCompiler::TargetMachineBuilder target_machine_builder,
    std::shared_ptr<llvm::TargetMachine> target_machine,
    TaskDispatcher* task_dispatcher,
    std::unique_ptr<llvm::orc::ExecutionSession> execution_session,
    std::unique_ptr<IrCompiler> ir_compiler, size_t num_dylibs,
    DefinitionGenerator definition_generator)
    : target_machine_builder_(std::move(target_machine_builder)),
      target_machine_(std::move(target_machine)),
      task_dispatcher_(task_dispatcher),
      execution_session_(std::move(execution_session)),
      object_layer_(CreateObjectLinkingLayer(*execution_session_)),
      compile_layer_(CreateCompileLayer(*execution_session_, *object_layer_,
                                        std::move(ir_compiler))),
      gdb_(llvm::JITEventListener::createGDBRegistrationListener()),
      perf_(llvm::JITEventListener::createPerfJITEventListener()) {
  // Create at least one dynamic library for the given jit compiler.
  dylibs_.resize(std::max<size_t>(1, num_dylibs));
  for (size_t i = 0; i < dylibs_.size(); ++i) {
    dylibs_[i] = &execution_session_->createBareJITDylib(
        absl::StrCat("<xla_jit_dylib_", i, ">"));
    if (definition_generator) {
      dylibs_[i]->addGenerator(definition_generator(target_machine_.get()));
    }
  }

  // Register GDB and perf event listeners with the object linking layer.
  if (gdb_) object_layer_->registerJITEventListener(*gdb_);
  if (perf_) object_layer_->registerJITEventListener(*perf_);

  // Copied from LLJIT, required to find symbols on Windows.
  if (target_machine_->getTargetTriple().isOSBinFormatCOFF()) {
    object_layer_->setOverrideObjectFlagsWithResponsibilityFlags(true);
    object_layer_->setAutoClaimResponsibilityForObjectSymbols(true);
  }
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

absl::Status JitCompiler::AddObjFile(
    std::unique_ptr<llvm::MemoryBuffer> obj_file, size_t dylib_index) {
  if (dylib_index >= dylibs_.size()) {
    return Internal("Invalid dylib index %d (num dylibs: %d))", dylib_index,
                    dylibs_.size());
  }

  llvm::orc::JITDylib* dylib = dylibs_[dylib_index];
  if (auto err = object_layer_->add(*dylib, std::move(obj_file))) {
    return Internal("Failed to add object file to dylib %d: %s", dylib_index,
                    llvm::toString(std::move(err)));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<FunctionLibrary>> JitCompiler::Compile(
    absl::Span<const Symbol> symbols) && {
  TraceMe trace([&] {
    return TraceMeEncode("JitCompiler::Compile",
                         {{"num_symbols", symbols.size()}});
  });

  // Mangle symbol names for the target machine data layout.
  llvm::DataLayout data_layout = target_machine_->createDataLayout();
  auto mangle = [&](absl::string_view name) {
    llvm::SmallVector<char, 40> mangled;
    llvm::Mangler::getNameWithPrefix(mangled, name, data_layout);
    return std::string(mangled.begin(), mangled.end());
  };

  // Build a symbol lookup set.
  llvm::orc::SymbolLookupSet lookup_set;
  for (const auto& symbol : symbols) {
    VLOG(5) << absl::StreamFormat(" - look up symbol: %s", symbol.name);
    lookup_set.add(execution_session_->intern(mangle(symbol.name)));
  }

  // Build a search order for the dynamic libraries.
  llvm::orc::JITDylibSearchOrder search_order(dylibs_.size());
  for (size_t i = 0; i < dylibs_.size(); ++i) {
    search_order[i] = std::make_pair(
        dylibs_[i], llvm::orc::JITDylibLookupFlags::MatchExportedSymbolsOnly);
  }

  // Look up all requested symbols in the execution session.
  auto symbol_map = execution_session_->lookup(std::move(search_order),
                                               std::move(lookup_set));

  // Wait for all compilation tasks to finish.
  task_dispatcher_->shutdown();

  if (auto err = symbol_map.takeError()) {
    return Internal("%s", llvm::toString(std::move(err)));
  }

  // Resolve type-erased symbol pointers from the symbol map.
  using ResolvedSymbol = CompiledFunctionLibrary::ResolvedSymbol;
  absl::flat_hash_map<std::string, ResolvedSymbol> resolved_map;

  for (const auto& symbol : symbols) {
    auto symbol_name = execution_session_->intern(mangle(symbol.name));
    llvm::orc::ExecutorSymbolDef symbol_def = symbol_map->at(symbol_name);
    llvm::orc::ExecutorAddr symbol_addr = symbol_def.getAddress();
    void* ptr = reinterpret_cast<void*>(symbol_addr.getValue());
    resolved_map[symbol.name] = ResolvedSymbol{symbol.type_id, ptr};
  }

  return std::make_unique<CompiledFunctionLibrary>(
      std::move(execution_session_), std::move(object_layer_),
      std::move(resolved_map));
}

JitCompiler::TaskDispatcher::TaskDispatcher(TaskRunner task_runner)
    : task_runner_(std::move(task_runner)) {}

JitCompiler::TaskDispatcher::~TaskDispatcher() { shutdown(); }

void JitCompiler::TaskDispatcher::dispatch(
    std::unique_ptr<llvm::orc::Task> task) {
  // Dispatch task in the current thread if no task runner is provided.
  if (task_runner_ == nullptr) {
    task->run();
    return;
  }

  // Dispatch task using user-provided task runner.
  absl::MutexLock lock(&mu_);
  ++num_dispatched_tasks_;

  task_runner_([this, task = std::shared_ptr<llvm::orc::Task>(
                          std::move(task))]() mutable {
    TraceMe trace("TaskDispatcher::dispatch");

    // We run and explicitly destroy the task before decrementing the counter
    // and notifying the condition variable, to ensure that the task is fully
    // executed and cleaned up before task dispatcher shut down.
    task->run();
    task.reset();

    absl::MutexLock lock(&mu_);
    --num_dispatched_tasks_;
  });
}

void JitCompiler::TaskDispatcher::shutdown() {
  auto all_tasks_finished = [this]() ABSL_SHARED_LOCKS_REQUIRED(mu_) {
    return num_dispatched_tasks_ == 0;
  };
  absl::MutexLock lock(&mu_, absl::Condition(&all_tasks_finished));
}

JitCompiler::CompiledFunctionLibrary::CompiledFunctionLibrary(
    std::unique_ptr<llvm::orc::ExecutionSession> execution_session,
    std::unique_ptr<llvm::orc::RTDyldObjectLinkingLayer> object_layer,
    absl::flat_hash_map<std::string, ResolvedSymbol> symbols_map)
    : execution_session_(std::move(execution_session)),
      object_layer_(std::move(object_layer)),
      symbols_map_(std::move(symbols_map)) {
  DCHECK(execution_session_) << "Execution session must not be null";
}

JitCompiler::CompiledFunctionLibrary::~CompiledFunctionLibrary() {
  if (auto err = execution_session_->endSession()) {
    execution_session_->reportError(std::move(err));
  }
}

absl::StatusOr<void*> JitCompiler::CompiledFunctionLibrary::ResolveFunction(
    TypeId type_id, absl::string_view name) {
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
