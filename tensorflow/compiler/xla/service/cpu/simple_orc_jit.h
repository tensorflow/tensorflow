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

#include "external/llvm/include/llvm/ADT/Triple.h"
#include "external/llvm/include/llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "external/llvm/include/llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "external/llvm/include/llvm/IR/Module.h"
#include "external/llvm/include/llvm/Target/TargetMachine.h"
#include "tensorflow/compiler/xla/service/cpu/disassembler.h"
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
  using ObjLayerT = llvm::orc::ObjectLinkingLayer<>;
  using CompileLayerT = llvm::orc::IRCompileLayer<ObjLayerT>;
  using ModuleHandleT = CompileLayerT::ModuleSetHandleT;

  // Create a new JIT, targeting the host architecture.
  // The |target_options| parameter allows customization of certain code
  // generation properties of the TargetMachine (whether or not float point math
  // can be reassociated, etc.).
  // The |opt_level| parameter controls the optimization level of the code
  // generator.
  SimpleOrcJIT(const llvm::TargetOptions& target_options,
               llvm::CodeGenOpt::Level opt_level);

  // Data layout this JIT was created with.
  const llvm::DataLayout& data_layout() const { return data_layout_; }

  // Target triple (host) this JIT was created with.
  const llvm::Triple& target_triple() const {
    return target_machine_->getTargetTriple();
  }

  // Add a module to the JIT. Returns an opaque handle that can be used to later
  // remove this module.
  ModuleHandleT AddModule(std::unique_ptr<llvm::Module> module);

  // Remove a module from the JIT and free the memory associated with it.
  void RemoveModule(ModuleHandleT handle);

  // Get the runtime address of the compiled symbol whose name is given. Returns
  // nullptr if the symbol cannot be found.
  llvm::JITSymbol FindSymbol(const std::string& name);

 private:
  std::vector<ModuleHandleT> module_handles_;
  std::unique_ptr<llvm::TargetMachine> target_machine_;
  const Disassembler disassembler_;
  const llvm::DataLayout data_layout_;
  ObjLayerT object_layer_;
  CompileLayerT compile_layer_;
};

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_SIMPLE_ORC_JIT_H_
