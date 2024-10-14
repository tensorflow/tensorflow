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
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "xla/backends/cpu/codegen/ir/xla_cpu_dialect.h"  // IWYU pragma: keep
#include "xla/backends/cpu/codegen/transforms/xla_cpu_rewrite_patterns.h"

namespace xla::cpu {

#define GEN_PASS_DECL_LOWERTRIVIALPASS
#define GEN_PASS_DEF_LOWERTRIVIALPASS
#include "xla/backends/cpu/codegen/transforms/passes.h.inc"

namespace {
class LowerTrivialPass : public impl::LowerTrivialPassBase<LowerTrivialPass> {
  void runOnOperation() override {
    mlir::TypeConverter converter;
    mlir::ConversionTarget target(getContext());

    converter.addConversion([](mlir::Type type) { return type; });
    PopulateXlaCpuTypeConversionAndLegality(converter, target);

    mlir::RewritePatternSet patterns(&getContext());
    PopulateXlaCpuConversionPatterns(patterns);

    // Add conversion patterns for function signatures.
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
        patterns, converter);

    // Set up basic legality constraints.
    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();

    // Add dynamic legality constraints to apply conversions defined above.
    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
        [&](mlir::func::FuncOp op) {
          return converter.isSignatureLegal(op.getFunctionType());
        });

    if (mlir::failed(mlir::applyFullConversion(getOperation(), target,
                                               std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<mlir::Pass> CreateLowerTrivialPass() {
  return std::make_unique<LowerTrivialPass>();
}

}  // namespace xla::cpu
