#include "xla/hlo/ir/hlo_module.h"
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
#include <string>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/FMF.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/codegen/mlir_kernel_source.h"

namespace xla::cpu {

// FusionCompiler compiles MLIR modules to LLVM IR using the XLA:CPU compilation
// pipeline.
class FusionCompiler {
 public:
  struct Options {
    int32_t vector_width;
    int32_t verification_level;
    bool fast_min_max;
    llvm::FastMathFlags fast_math_flags;
    bool use_new_xtile_lowering = false;
    bool msan_enabled = false;
  };

  FusionCompiler(mlir::MLIRContext* context, Options options,
                 const HloModule* hlo_module = nullptr);

  // Compile a given MLIR module to LLVM, using the provided LLVM context.
  absl::StatusOr<std::unique_ptr<llvm::Module>> Compile(
      llvm::LLVMContext& llvm_context, mlir::ModuleOp mlir_module);
  // Compile a MLIR kernel source to a LLVM kernel source.
  absl::StatusOr<LlvmKernelSource> Compile(MlirKernelSource mlir_kernel_source);

  // Create a new MLIR context for the compiler with the required dialects for
  // compiling an XLA:CPU fusion.
  static std::unique_ptr<mlir::MLIRContext> CreateContext();

  // Create a dialect registry for the compiler with the required dialects for
  // compiling an XLA:CPU fusion. If `register_pass_pipelines` is true, this
  // will also register the pass pipelines for the compiler, typically to be
  // used in tests.
  static mlir::DialectRegistry CreateDialectRegistry();

 private:
  Options options_;
  const HloModule* hlo_module_;
  // We have 2 distinct pipelines for scalar and tiled kernels, this is
  // because they differ slightly in their semantics, ideally these would be
  // unified but this is a larger change.
  mlir::PassManager scalar_pass_manager_;
  mlir::PassManager tiled_pass_manager_;
};

// Xtile CPU pipeline contains two stages, the first is the conversion from
// Xtile to the vector dialect, the second is the conversion from the vector
// dialect to LLVM.
void AddXtileToVectorPasses(mlir::OpPassManager& pm, bool msan_enabled);
void AddNewXtileToVectorPasses(mlir::OpPassManager& pm);
void AddVectorToLLVMPasses(mlir::OpPassManager& pm, bool fast_min_max);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_FUSION_COMPILER_H_
