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

#ifndef XLA_BACKENDS_CPU_CODEGEN_FUSION_COMPILER_H_
#define XLA_BACKENDS_CPU_CODEGEN_FUSION_COMPILER_H_

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/llvm_ir_kernel_source.h"
#include "xla/codegen/mlir_kernel_source.h"

namespace xla::cpu {

// FusionCompiler compiles MLIR modules to LLVM IR using the XLA:CPU compilation
// pipeline.
class FusionCompiler {
 public:
  struct CompilationHooks {
    absl::AnyInvocable<void(mlir::ModuleOp)> pre_optimization;
    absl::AnyInvocable<void(mlir::ModuleOp)> post_optimization;
    absl::AnyInvocable<void(mlir::ModuleOp)> post_lowering;
  };

  struct Options {
    int32_t vector_width;
    int32_t verification_level;
    bool fast_min_max;
  };

  explicit FusionCompiler(Options options, CompilationHooks hooks = {})
      : options_(std::move(options)), hooks_(std::move(hooks)) {}

  // Compile a given MLIR module to LLVM, using the provided LLVM context.
  absl::StatusOr<std::unique_ptr<llvm::Module>> Compile(
      llvm::LLVMContext& llvm_context, mlir::ModuleOp mlir_module);
  // Compile a MLIR kernel source to a LLVM kernel source.
  absl::StatusOr<LlvmIrKernelSource> Compile(
      MlirKernelSource mlir_kernel_source);

  // Create a new MLIR context for the compiler with the required dialects for
  // compiling an XLA:CPU fusion.
  static std::unique_ptr<mlir::MLIRContext> CreateContext();

 private:
  Options options_;
  CompilationHooks hooks_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_FUSION_COMPILER_H_
