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

#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/SymbolStringPool.h"
#include "llvm/IR/Module.h"
#include "llvm/Target/TargetMachine.h"
#include "tensorflow/compiler/xla/service/cpu/compiler_functor.h"
#include "tensorflow/compiler/xla/service/cpu/disassembler.h"
#include "tensorflow/compiler/xla/service/cpu/external_constant_pool.h"
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
class SimpleOrcJIT {
 public:
  using ObjLayerT = llvm::orc::RTDyldObjectLinkingLayer;
  using CompileFtor = std::function<ObjLayerT::ObjectPtr(llvm::Module&)>;
  using CompileLayerT = llvm::orc::IRCompileLayer<ObjLayerT, CompileFtor>;
  using VModuleKeyT = llvm::orc::VModuleKey;

  // Create a new JIT, targeting the host architecture.
  // The |target_options| parameter allows customization of certain code
  // generation properties of the TargetMachine (whether or not float point math
  // can be reassociated, etc.).
  // The |opt_level| parameter controls the optimization level of the code
  // generator.
  // The |optimize_for_size| parameter specifies that the code generator should
  // optimize to reduce code size, potentially at the cost of performance.
  // The |disable_expensive_passes| parameter will disable certain optimization
  // passes
  // The |pre_optimization_hook| is invoked on the module before any IR
  // level optimizations are applied.
  // The |post_optimization_hook| is invoked on the module after all IR
  // level optimizations are applied.
  SimpleOrcJIT(const llvm::TargetOptions& target_options,
               llvm::CodeGenOpt::Level opt_level, bool optimize_for_size,
               bool enable_fast_math, bool disable_expensive_passes,
               LLVMCompiler::ModuleHook pre_optimization_hook,
               LLVMCompiler::ModuleHook post_optimization_hook);

  // Data layout this JIT was created with.
  const llvm::DataLayout& data_layout() const { return data_layout_; }

  // Target triple (host) this JIT was created with.
  const llvm::Triple& target_triple() const {
    return target_machine_->getTargetTriple();
  }

  // Add a module to the JIT. Returns an opaque key that can be used to later
  // remove this module.
  VModuleKeyT AddModule(std::unique_ptr<llvm::Module> module);

  // Remove a module from the JIT and free the memory associated with it.
  void RemoveModule(VModuleKeyT key);

  // Get the runtime address of the compiled symbol whose name is given. Returns
  // nullptr if the symbol cannot be found.
  llvm::JITSymbol FindCompiledSymbol(const std::string& name);

  llvm::TargetMachine* target_machine() const { return target_machine_.get(); }

  ExternalConstantPool* external_constant_pool() {
    return &external_constant_pool_;
  }

 private:
  llvm::JITSymbol ResolveRuntimeSymbol(const std::string& name);

  std::vector<VModuleKeyT> module_keys_;
  std::unique_ptr<llvm::TargetMachine> target_machine_;
  const Disassembler disassembler_;
  const llvm::DataLayout data_layout_;
  llvm::orc::SymbolStringPool string_pool_;
  llvm::orc::ExecutionSession execution_session_;
  std::shared_ptr<llvm::orc::SymbolResolver> symbol_resolver_;
  ObjLayerT object_layer_;
  CompileLayerT compile_layer_;
  ExternalConstantPool external_constant_pool_;
};

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_SIMPLE_ORC_JIT_H_
