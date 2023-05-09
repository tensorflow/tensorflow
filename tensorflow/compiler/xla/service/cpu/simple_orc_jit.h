/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_SIMPLE_ORC_JIT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_SIMPLE_ORC_JIT_H_

#include <memory>
#include <string>
#include <vector>

#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/SymbolStringPool.h"
#include "llvm/IR/Module.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include "tensorflow/compiler/xla/service/cpu/compiler_functor.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace cpu {

// Simplified LLVM JIT based on the new Orc API.
//
// This class wraps Orc's functionality into a single interface that only
// exposes what we need for XLA.
//
// Supports JIT-ing multiple modules but without cross-module linking.
// Implements eager compilation - the module is lowered to binary as soon as
// it's added to the JIT.
class SimpleOrcJIT : public llvm::JITEventListener {
 public:
  using ObjLayerT = llvm::orc::RTDyldObjectLinkingLayer;
  using CompileLayerT = llvm::orc::IRCompileLayer;

  // Create a new JIT, targeting the host architecture.
  //
  // {pre,post}_optimization_hook is invoked on the module before/after all
  // LLVM IR-level optimizations.  post_codegen_hook is invoked after
  // compiling to machine code.
  SimpleOrcJIT(
      std::unique_ptr<llvm::orc::ExecutorProcessControl> target_process_control,
      std::unique_ptr<llvm::orc::ExecutionSession> execution_session,
      const llvm::TargetOptions& target_options,
      llvm::CodeGenOpt::Level opt_level, bool optimize_for_size,
      bool disable_expensive_passes, bool disable_slp_vectorizer,
      llvm::FastMathFlags fast_math_flags,
      LLVMCompiler::ModuleHook pre_optimization_hook,
      LLVMCompiler::ModuleHook post_optimization_hook,
      std::function<void(const llvm::object::ObjectFile&)> post_codegen_hook);

  static llvm::Expected<std::unique_ptr<SimpleOrcJIT>> Create(
      const llvm::TargetOptions& target_options,
      llvm::CodeGenOpt::Level opt_level, bool optimize_for_size,
      bool disable_expensive_passes, bool disable_slp_vectorizer,
      llvm::FastMathFlags fast_math_flags,
      LLVMCompiler::ModuleHook pre_optimization_hook,
      LLVMCompiler::ModuleHook post_optimization_hook,
      std::function<void(const llvm::object::ObjectFile&)> post_codegen_hook);

  ~SimpleOrcJIT() override;

  const llvm::DataLayout& data_layout() const { return data_layout_; }

  const llvm::Triple& target_triple() const { return target_triple_; }

  llvm::Error AddModule(llvm::orc::ThreadSafeModule module);

  // Discards objects we no longer need once we are done compiling.
  void DoneCompiling();

  // Get the runtime address of the compiled symbol whose name is given. Returns
  // nullptr if the symbol cannot be found.
  llvm::Expected<llvm::orc::ExecutorSymbolDef> FindCompiledSymbol(
      const std::string& name);

  llvm::TargetMachine* target_machine() const { return target_machine_.get(); }

  // Creates an llvm::TargetMachine suitable for JITting code that will run on
  // the current machine.
  static std::unique_ptr<llvm::TargetMachine> InferTargetMachineForJIT(
      const llvm::TargetOptions& target_options,
      llvm::CodeGenOpt::Level opt_level);

  int64_t SizeOfGeneratedCodeInBytes() const {
    return size_of_generated_code_in_bytes_;
  }

 private:
  llvm::orc::ExecutorSymbolDef ResolveRuntimeSymbol(llvm::StringRef name);

  void notifyObjectLoaded(
      llvm::JITEventListener::ObjectKey key,
      const llvm::object::ObjectFile& object,
      const llvm::RuntimeDyld::LoadedObjectInfo& object_info) override;
  void notifyFreeingObject(llvm::JITEventListener::ObjectKey key) override;

  std::unique_ptr<llvm::TargetMachine> target_machine_;
  llvm::Triple target_triple_;
  const llvm::DataLayout data_layout_;
  std::unique_ptr<llvm::orc::ExecutorProcessControl> target_process_control_;
  std::unique_ptr<llvm::orc::ExecutionSession> execution_session_;
  ObjLayerT object_layer_;
  CompileLayerT compile_layer_;
  llvm::orc::JITDylib* main_jit_dylib_;
  int64_t size_of_generated_code_in_bytes_ = 0;

  // Non owning pointer to a JIT event listener that registers the JIT events
  // with an attached GDB.
  //
  // Note: we get a pointer to this event listener using
  // `createGDBRegistrationListener` which makes it look like we're supposed to
  // free this, but the function is poorly named and really just returns a
  // pointer to a static object.
  llvm::JITEventListener* gdb_jit_event_listener_;

  llvm::JITEventListener* perf_jit_event_listener_;
};

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_SIMPLE_ORC_JIT_H_
