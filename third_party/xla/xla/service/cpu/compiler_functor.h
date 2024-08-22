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

#ifndef XLA_SERVICE_CPU_COMPILER_FUNCTOR_H_
#define XLA_SERVICE_CPU_COMPILER_FUNCTOR_H_

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/synchronization/mutex.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/IR/FMF.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Target/TargetMachine.h"
#include "xla/service/llvm_compiler.h"

namespace xla::cpu {

// Functor class for compiling an LLVM module down to an object file. For use by
// Orc JIT compile layer.
class CompilerFunctor : public llvm::orc::IRCompileLayer::IRCompiler {
 public:
  // Returns an instance of llvm::TargetMachine for a compilation. It can be
  // a shared TargetMachine if compilation is single threaded, or must be a
  // unique TargetMachine if compilation is multi threaded (because
  // TargetMachine is not thread safe).
  //
  // See `llvm::orc::ConcurrentIRCompiler` to see corresponding API in ORC.
  using TargetMachineBuilder =
      std::function<std::shared_ptr<llvm::TargetMachine>()>;

  explicit CompilerFunctor(
      TargetMachineBuilder target_machine_builder, int opt_level,
      bool optimize_for_size, bool disable_expensive_passes,
      bool disable_slp_vectorizer, llvm::FastMathFlags fast_math_flags,
      LLVMCompiler::ModuleHook pre_optimization_hook = nullptr,
      LLVMCompiler::ModuleHook post_optimization_hook = nullptr,
      absl::AnyInvocable<void(const llvm::object::ObjectFile&)>
          post_codegen_hook = nullptr,
      bool dfsan_enabled = false,
      const std::vector<std::string>& dfsan_abi_list_files = {})
      : IRCompiler(llvm::orc::IRSymbolMapper::ManglingOptions()),
        target_machine_builder_(std::move(target_machine_builder)),
        opt_level_(opt_level),
        optimize_for_size_(optimize_for_size),
        disable_expensive_passes_(disable_expensive_passes),
        disable_slp_vectorizer_(disable_slp_vectorizer),
        fast_math_flags_(fast_math_flags),
        dfsan_enabled_(dfsan_enabled),
        dfsan_abi_list_files_(dfsan_abi_list_files),
        pre_optimization_hook_(std::move(pre_optimization_hook)),
        post_optimization_hook_(std::move(post_optimization_hook)),
        post_codegen_hook_(std::move(post_codegen_hook)) {}

  // Compile a Module to an ObjectFile.
  llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> operator()(
      llvm::Module& module) override;

 private:
  TargetMachineBuilder target_machine_builder_;
  const unsigned opt_level_;
  const bool optimize_for_size_;
  const bool disable_expensive_passes_;
  const bool disable_slp_vectorizer_;
  const llvm::FastMathFlags fast_math_flags_;
  const bool dfsan_enabled_ = false;
  const std::vector<std::string> dfsan_abi_list_files_;

  LLVMCompiler::ModuleHook pre_optimization_hook_ ABSL_GUARDED_BY(mutex_);
  LLVMCompiler::ModuleHook post_optimization_hook_ ABSL_GUARDED_BY(mutex_);
  absl::AnyInvocable<void(const llvm::object::ObjectFile&)> post_codegen_hook_
      ABSL_GUARDED_BY(mutex_);

  // Synchronizes access to user-defined compilation hooks.
  absl::Mutex mutex_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_COMPILER_FUNCTOR_H_
