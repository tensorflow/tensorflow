/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <iterator>
#include <memory>
#include <utility>

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/transforms/gpu/custom_calls.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"

namespace xla {
namespace gpu {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/xla/mlir/transforms/gpu/passes.h.inc"

using namespace mlir;  // NOLINT

using mlir::lmhlo::InfeedOp;
using mlir::lmhlo::OutfeedOp;

class ConvertLmhloToGpuRuntimePass
    : public ConvertLmhloToGpuRuntimePassBase<ConvertLmhloToGpuRuntimePass> {
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mlir::func::FuncDialect, mlir::arith::ArithmeticDialect>();
  }
};

//===----------------------------------------------------------------------===//

template <typename IoFeedOp>
class IoFeedOpLowering : public OpRewritePattern<IoFeedOp> {
  static StringRef Target(InfeedOp) { return "xla.gpu.infeed"; }
  static StringRef Target(OutfeedOp) { return "xla.gpu.outfeed"; }

 public:
  IoFeedOpLowering(MLIRContext* ctx, CustomCalls& custom_calls)
      : OpRewritePattern<IoFeedOp>(ctx), custom_calls_(custom_calls) {}

  LogicalResult matchAndRewrite(IoFeedOp op,
                                PatternRewriter& rewriter) const override {
    // Get or create a custom call function declaration.
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    func::FuncOp callee = custom_calls_.GetOrCreate(b, Target(op), op);

    llvm::SmallVector<NamedAttribute> attrs = {
        {b.getStringAttr("config"), op.getConfigAttr()}};

    // Call the runtime intrinsic with the original operands.
    auto call = rewriter.replaceOpWithNewOp<func::CallOp>(
        op, callee.getName(), TypeRange(), op.getOperands());
    AppendCustomCallAttrs(call, attrs);

    return success();
  }

 private:
  CustomCalls& custom_calls_;
};

class InfeedOpLowering : public IoFeedOpLowering<InfeedOp> {
 public:
  using IoFeedOpLowering::IoFeedOpLowering;
};

class OutfeedOpLowering : public IoFeedOpLowering<OutfeedOp> {
 public:
  using IoFeedOpLowering::IoFeedOpLowering;
};

//===----------------------------------------------------------------------===//

void ConvertLmhloToGpuRuntimePass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext* ctx = module.getContext();

  // Keep track of the custom calls created from the lowered operations.
  SymbolTable sym_table(module);
  CustomCalls custom_calls(std::move(sym_table));

  // Convert lmhlo operations to XLA gpu runtime custom calls.
  RewritePatternSet patterns(ctx);
  patterns.insert<InfeedOpLowering, OutfeedOpLowering>(ctx, custom_calls);

  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertLmhloToGpuRuntimePass() {
  return std::make_unique<ConvertLmhloToGpuRuntimePass>();
}

}  // namespace gpu
}  // namespace xla
