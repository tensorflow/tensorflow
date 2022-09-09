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
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/utils/runtime/custom_calls.h"

namespace xla {
namespace gpu {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/xla/mlir/transforms/gpu/passes.h.inc"

using namespace mlir;  // NOLINT

using mlir::gpu::GPUModuleOp;
using mlir::gpu::LaunchFuncOp;
using mlir::gpu::MemcpyOp;
using mlir::gpu::MemsetOp;

using xla::runtime::CustomCallDeclarations;

class ConvertGpuToGpuRuntimePass
    : public ConvertGpuToGpuRuntimePassBase<ConvertGpuToGpuRuntimePass> {
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mlir::func::FuncDialect, mlir::arith::ArithmeticDialect>();
  }
};

//===----------------------------------------------------------------------===//

class GpuModuleOpLowering : public OpRewritePattern<GPUModuleOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(GPUModuleOp op,
                                PatternRewriter& rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//

class MemcpyOpLowering : public OpRewritePattern<MemcpyOp> {
 public:
  MemcpyOpLowering(MLIRContext* ctx, CustomCallDeclarations& custom_calls)
      : OpRewritePattern(ctx), custom_calls_(custom_calls) {}

  // We use a heuristic to identify the direction of the memcpy operation, if
  // the operand was allocated by alloca op or is a global memref, then it must
  // be a memref on the host.
  static bool IsHostMemRef(Value value) {
    auto* op = value.getDefiningOp();
    return llvm::isa_and_nonnull<memref::AllocaOp, memref::GetGlobalOp>(op);
  }

  // Identify the direction of the memcpy operation.
  static StringRef Target(MemcpyOp op) {
    if (IsHostMemRef(op.dst())) return "xla.gpu.memcpy.d2h";
    if (IsHostMemRef(op.src())) return "xla.gpu.memcpy.h2d";
    return "xla.gpu.memcpy.d2d";
  }

  LogicalResult matchAndRewrite(MemcpyOp op,
                                PatternRewriter& rewriter) const override {
    // Get or create a custom call function declaration.
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    func::FuncOp callee = custom_calls_.GetOrCreate(b, Target(op), op);

    // Create a function launch call operation.
    rewriter.replaceOpWithNewOp<func::CallOp>(op, callee.getName(), TypeRange(),
                                              op.getOperands());

    return success();
  }

 private:
  CustomCallDeclarations& custom_calls_;
};

//===----------------------------------------------------------------------===//

class MemsetOpLowering : public OpRewritePattern<MemsetOp> {
 private:
  static constexpr const char kCustomCallTarget[] = "xla.gpu.memset";

 public:
  MemsetOpLowering(MLIRContext* ctx, CustomCallDeclarations& custom_calls)
      : OpRewritePattern(ctx), custom_calls_(custom_calls) {}

  LogicalResult matchAndRewrite(MemsetOp op,
                                PatternRewriter& rewriter) const override {
    // Get or create a custom call function declaration.
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    func::FuncOp callee = custom_calls_.GetOrCreate(b, kCustomCallTarget, op);

    // Create a function launch call operation.
    rewriter.replaceOpWithNewOp<func::CallOp>(op, callee.getName(), TypeRange(),
                                              op.getOperands());

    return success();
  }

 private:
  CustomCallDeclarations& custom_calls_;
};

//===----------------------------------------------------------------------===//

class LaunchFuncOpLowering : public OpRewritePattern<LaunchFuncOp> {
 private:
  static constexpr const char kCustomCallTarget[] = "xla.gpu.func.launch";

 public:
  LaunchFuncOpLowering(MLIRContext* ctx, CustomCallDeclarations& custom_calls)
      : OpRewritePattern(ctx), custom_calls_(custom_calls) {}

  LogicalResult matchAndRewrite(LaunchFuncOp op,
                                PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Cast grid and block dimensions to i32 before passing to the custom call.
    auto cast = [&](mlir::Value value) {
      return b.create<arith::IndexCastOp>(b.getI32Type(), value);
    };

    // Prepare arguments for the custom call.
    llvm::SmallVector<Value> args = {
        cast(op.gridSizeX()),  cast(op.gridSizeY()),  cast(op.gridSizeZ()),
        cast(op.blockSizeX()), cast(op.blockSizeY()), cast(op.blockSizeZ())};

    // Add kernel arguments.
    llvm::copy(op.operands(), std::back_inserter(args));

    // Get or create a custom call function declaration.
    func::FuncOp callee = custom_calls_.GetOrCreate(
        b, "xla.gpu.func.launch", TypeRange(ValueRange(args)), TypeRange());

    // Create a function launch call operation.
    auto call = b.create<func::CallOp>(callee.getName(), TypeRange(), args);
    call->setAttr(b.getStringAttr("kernel"), op.getKernelName());

    // Erase the original gpu launch operation.
    rewriter.eraseOp(op);

    return success();
  }

 private:
  CustomCallDeclarations& custom_calls_;
};

//===----------------------------------------------------------------------===//

void ConvertGpuToGpuRuntimePass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext* ctx = module.getContext();

  // Keep track of the custom calls created from the lowered operations.
  SymbolTable sym_table(module);
  CustomCallDeclarations custom_calls(std::move(sym_table));

  // Convert gpu operations to XLA gpu runtime custom calls.
  RewritePatternSet patterns(ctx);
  patterns.insert<GpuModuleOpLowering>(ctx);
  patterns.insert<LaunchFuncOpLowering, MemcpyOpLowering, MemsetOpLowering>(
      ctx, custom_calls);

  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertGpuToGpuRuntimePass() {
  return std::make_unique<ConvertGpuToGpuRuntimePass>();
}

}  // namespace gpu
}  // namespace xla
