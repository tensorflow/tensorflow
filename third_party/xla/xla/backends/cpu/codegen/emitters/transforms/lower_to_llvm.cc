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

#include <memory>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // IWYU pragma: keep
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/cpu/codegen/emitters/ir/xla_cpu_dialect.h"  // IWYU pragma: keep
#include "xla/backends/cpu/codegen/emitters/transforms/xla_cpu_rewrite_patterns.h"
#include "xla/codegen/emitters/ir/xla_dialect.h"  // IWYU pragma: keep

namespace xla::cpu {

#define GEN_PASS_DECL_LOWERTOLLVMPASS
#define GEN_PASS_DEF_LOWERTOLLVMPASS
#include "xla/backends/cpu/codegen/emitters/transforms/passes.h.inc"

namespace {
class LowerToLLVMPass : public impl::LowerToLLVMPassBase<LowerToLLVMPass> {
  void runOnOperation() override {
    mlir::MLIRContext* mlir_context = &getContext();
    mlir::RewritePatternSet patterns(mlir_context);
    PopulateXlaCpuConversionPatterns(patterns);
    mlir::GreedyRewriteConfig config;
    config.enableFolding(true);
    if (mlir::failed(mlir::applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
      return;
    }
  }
};
}  // namespace

std::unique_ptr<mlir::Pass> CreateLowerToLLVMPass() {
  return std::make_unique<LowerToLLVMPass>();
}

}  // namespace xla::cpu
