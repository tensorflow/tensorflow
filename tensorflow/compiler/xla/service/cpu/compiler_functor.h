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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_COMPILER_FUNCTOR_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_COMPILER_FUNCTOR_H_

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "tensorflow/compiler/xla/service/llvm_compiler.h"
#include "tensorflow/tsl/platform/logging.h"

namespace xla {
namespace cpu {

// Functor class for compiling an LLVM module down to an object file. For use by
// Orc JIT compile layer.
class CompilerFunctor : public llvm::orc::IRCompileLayer::IRCompiler {
 public:
  explicit CompilerFunctor(
      llvm::TargetMachine* target_machine, int opt_level,
      bool optimize_for_size, bool disable_expensive_passes,
      bool disable_slp_vectorizer, llvm::FastMathFlags fast_math_flags,
      LLVMCompiler::ModuleHook pre_optimization_hook = nullptr,
      LLVMCompiler::ModuleHook post_optimization_hook = nullptr,
      std::function<void(const llvm::object::ObjectFile&)> post_codegen_hook =
          nullptr,
      bool dfsan_enabled = false,
      const std::vector<std::string>& dfsan_abi_list_files = {},
      const std::vector<std::string>& convert_to_xla_runtime_abi = {})
      : IRCompiler(llvm::orc::IRSymbolMapper::ManglingOptions()),
        target_machine_(target_machine),
        opt_level_(opt_level),
        optimize_for_size_(optimize_for_size),
        disable_expensive_passes_(disable_expensive_passes),
        disable_slp_vectorizer_(disable_slp_vectorizer),
        fast_math_flags_(fast_math_flags),
        pre_optimization_hook_(std::move(pre_optimization_hook)),
        post_optimization_hook_(std::move(post_optimization_hook)),
        post_codegen_hook_(std::move(post_codegen_hook)),
        dfsan_enabled_(dfsan_enabled),
        dfsan_abi_list_files_(dfsan_abi_list_files),
        convert_to_xla_runtime_abi_(convert_to_xla_runtime_abi) {}

  // Compile a Module to an ObjectFile.
  llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> operator()(
      llvm::Module& module) override;

 private:
  llvm::TargetMachine* target_machine_;
  const unsigned opt_level_;
  const bool optimize_for_size_;
  const bool disable_expensive_passes_;
  const bool disable_slp_vectorizer_;
  const llvm::FastMathFlags fast_math_flags_;
  LLVMCompiler::ModuleHook pre_optimization_hook_;
  LLVMCompiler::ModuleHook post_optimization_hook_;
  std::function<void(const llvm::object::ObjectFile&)> post_codegen_hook_;
  const bool dfsan_enabled_ = false;
  const std::vector<std::string> dfsan_abi_list_files_;
  const std::vector<std::string> convert_to_xla_runtime_abi_;
};

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_COMPILER_FUNCTOR_H_
