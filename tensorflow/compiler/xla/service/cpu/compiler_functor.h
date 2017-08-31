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

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "tensorflow/compiler/xla/service/cpu/disassembler.h"
#include "tensorflow/compiler/xla/service/llvm_compiler.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace cpu {

// Functor class for compiling an LLVM module down to an object file. For use by
// Orc JIT compile layer.
class CompilerFunctor {
 public:
  // Describes the set of vector intrinsics available to the generated code.
  struct VectorIntrinsics {
    bool sse_intrinsics;
    bool avx_intrinsics;
    bool neon_intrinsics;
  };

  // Returns a VectorIntrinsics where all intrinsics are available.
  static VectorIntrinsics AllIntrinsics();

  explicit CompilerFunctor(
      llvm::TargetMachine* target_machine, const Disassembler* disassembler,
      int opt_level, bool optimize_for_size, bool enable_fast_math,
      const VectorIntrinsics& available_intrinsics,
      LLVMCompiler::ModuleHook pre_optimization_hook = nullptr,
      LLVMCompiler::ModuleHook post_optimization_hook = nullptr)
      : target_machine_(target_machine),
        disassembler_(CHECK_NOTNULL(disassembler)),
        opt_level_(opt_level),
        optimize_for_size_(optimize_for_size),
        enable_fast_math_(enable_fast_math),
        available_intrinsics_(available_intrinsics),
        pre_optimization_hook_(pre_optimization_hook),
        post_optimization_hook_(post_optimization_hook) {}

  // Compile a Module to an ObjectFile.
  llvm::object::OwningBinary<llvm::object::ObjectFile> operator()(
      llvm::Module& module) const;  // NOLINT

 private:
  // Populates the given pass manager with TargetLibraryInfo and
  // TargetTransformInfo passes.
  void AddTargetInfoPasses(llvm::legacy::PassManagerBase* passes) const;

  // Populates the given pass managers based on the optimization level.
  void AddOptimizationPasses(llvm::legacy::PassManagerBase* module_passes,
                             llvm::legacy::FunctionPassManager* function_passes,
                             unsigned opt_level, unsigned size_level) const;

  llvm::TargetMachine* target_machine_;
  const Disassembler* disassembler_;
  const unsigned opt_level_;
  const bool optimize_for_size_;
  const bool enable_fast_math_;
  const VectorIntrinsics available_intrinsics_;
  LLVMCompiler::ModuleHook pre_optimization_hook_;
  LLVMCompiler::ModuleHook post_optimization_hook_;
};

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_COMPILER_FUNCTOR_H_
