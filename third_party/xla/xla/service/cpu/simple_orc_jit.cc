/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/cpu/simple_orc_jit.h"

#include <stdint.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h"
#include "llvm/ExecutionEngine/Orc/SymbolStringPool.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/IR/FMF.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "xla/backends/cpu/codegen/contiguous_section_memory_manager.h"
#include "xla/backends/cpu/codegen/cpu_features.h"
#include "xla/backends/cpu/codegen/ir_compiler.h"
#include "xla/backends/cpu/codegen/jit_compiler.h"
#include "xla/service/cpu/orc_jit_memory_mapper.h"
#include "xla/service/cpu/runtime_symbol_generator.h"
#include "xla/service/llvm_compiler.h"
#include "tsl/platform/logging.h"

namespace xla::cpu {

SimpleOrcJIT::SimpleOrcJIT(
    std::unique_ptr<llvm::orc::ExecutorProcessControl> target_process_control,
    std::unique_ptr<llvm::orc::ExecutionSession> execution_session,
    const llvm::TargetOptions& target_options, llvm::CodeGenOptLevel opt_level,
    bool optimize_for_size, bool disable_expensive_passes,
    bool disable_slp_vectorizer, llvm::FastMathFlags fast_math_flags,
    LLVMCompiler::ModuleHook pre_optimization_hook,
    LLVMCompiler::ModuleHook post_optimization_hook,
    std::function<void(const llvm::object::ObjectFile&)> post_codegen_hook,
    size_t num_jit_dylibs, absl::string_view max_cpu_isa)
    : target_machine_builder_(JitCompiler::InferTargetMachineBuilder(
          target_options, opt_level, CpuFeatureFromString(max_cpu_isa))),
      target_machine_(target_machine_builder_().value()),
      target_triple_(target_machine_->getTargetTriple()),
      data_layout_(target_machine_->createDataLayout()),
      target_process_control_(std::move(target_process_control)),
      execution_session_(std::move(execution_session)),
      object_layer_(*execution_session_,
                    []() {
                      return std::make_unique<ContiguousSectionMemoryManager>(
                          orc_jit_memory_mapper::GetInstance());
                    }),
      compile_layer_(
          *execution_session_, object_layer_,
          std::make_unique<IrCompiler>(
              target_machine_builder_,
              IrCompiler::Options{
                  /*optimization_level=*/static_cast<int32_t>(opt_level),
                  /*optimize_for_size=*/optimize_for_size,
                  /*fast_math_flags=*/fast_math_flags,
                  /*disable_expensive_passes=*/disable_expensive_passes,
                  /*disable_slp_vectorizer=*/disable_slp_vectorizer,
              },
              IrCompiler::CompilationHooks{
                  std::move(pre_optimization_hook),
                  std::move(post_optimization_hook),
                  std::move(post_codegen_hook),
              })),
      gdb_jit_event_listener_(
          llvm::JITEventListener::createGDBRegistrationListener()),
      perf_jit_event_listener_(
          llvm::JITEventListener::createPerfJITEventListener()) {
  VLOG(1) << "CPU target: " << target_machine_->getTargetCPU().str()
          << " features: " << target_machine_->getTargetFeatureString().str();

  // Always create at least one dylib.
  num_jit_dylibs = std::max(size_t{1}, num_jit_dylibs);
  jit_dylibs_.resize(num_jit_dylibs);
  for (size_t i = 0; i < num_jit_dylibs; ++i) {
    jit_dylibs_[i] = &execution_session_->createBareJITDylib(
        absl::StrCat("<xla_jit_dylib_", i, ">"));
    jit_dylibs_[i]->addGenerator(
        std::make_unique<RuntimeSymbolGenerator>(data_layout_));
  }

  object_layer_.registerJITEventListener(*this);
  if (perf_jit_event_listener_) {
    object_layer_.registerJITEventListener(*perf_jit_event_listener_);
  }

  // Copied from LLJIT, required to find symbols on Windows.
  if (target_triple_.isOSBinFormatCOFF()) {
    object_layer_.setOverrideObjectFlagsWithResponsibilityFlags(true);
    object_layer_.setAutoClaimResponsibilityForObjectSymbols(true);
  }
}

SimpleOrcJIT::~SimpleOrcJIT() {
  if (auto err = execution_session_->endSession()) {
    execution_session_->reportError(std::move(err));
  }
}

llvm::Expected<std::unique_ptr<SimpleOrcJIT>> SimpleOrcJIT::Create(
    const llvm::TargetOptions& target_options, llvm::CodeGenOptLevel opt_level,
    bool optimize_for_size, bool disable_expensive_passes,
    bool disable_slp_vectorizer, llvm::FastMathFlags fast_math_flags,
    LLVMCompiler::ModuleHook pre_optimization_hook,
    LLVMCompiler::ModuleHook post_optimization_hook,
    std::function<void(const llvm::object::ObjectFile&)> post_codegen_hook,
    size_t num_jit_dylibs, absl::string_view max_cpu_isa) {
  auto SSP = std::make_shared<llvm::orc::SymbolStringPool>();
  auto target_process_control =
      llvm::orc::SelfExecutorProcessControl::Create(std::move(SSP));
  if (!target_process_control) {
    return target_process_control.takeError();
  }

  auto execution_session = std::make_unique<llvm::orc::ExecutionSession>(
      std::make_unique<llvm::orc::UnsupportedExecutorProcessControl>());
  return std::make_unique<SimpleOrcJIT>(
      std::move(*target_process_control), std::move(execution_session),
      target_options, opt_level, optimize_for_size, disable_expensive_passes,
      disable_slp_vectorizer, fast_math_flags, std::move(pre_optimization_hook),
      std::move(post_optimization_hook), std::move(post_codegen_hook),
      num_jit_dylibs, std::move(max_cpu_isa));
}

void SimpleOrcJIT::notifyObjectLoaded(
    llvm::JITEventListener::ObjectKey key,
    const llvm::object::ObjectFile& object,
    const llvm::RuntimeDyld::LoadedObjectInfo& object_info) {
  gdb_jit_event_listener_->notifyObjectLoaded(key, object, object_info);
  size_of_generated_code_in_bytes_ += object.getData().size();
}

void SimpleOrcJIT::notifyFreeingObject(llvm::JITEventListener::ObjectKey key) {
  gdb_jit_event_listener_->notifyFreeingObject(key);
}

llvm::Error SimpleOrcJIT::AddObjFile(
    std::unique_ptr<llvm::MemoryBuffer> obj_file, size_t dylib_index) {
  return object_layer_.add(*jit_dylibs_[dylib_index], std::move(obj_file));
}

llvm::Error SimpleOrcJIT::AddModule(llvm::orc::ThreadSafeModule module,
                                    size_t dylib_index) {
  return compile_layer_.add(*jit_dylibs_[dylib_index], std::move(module));
}

void SimpleOrcJIT::DoneCompiling() {
  // The target machine takes a non-trivial amount of memory, so once we are
  // done compiling throw it away.
  target_machine_.reset();
}

llvm::Expected<llvm::orc::ExecutorSymbolDef> SimpleOrcJIT::FindCompiledSymbol(
    const std::string& name) {
  return execution_session_->lookup(jit_dylibs_, name);
}

}  // namespace xla::cpu
