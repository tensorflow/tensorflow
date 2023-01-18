/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include <string>
#include <utility>

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"

constexpr llvm::StringRef
    mlir::kernel_gen::tf_framework ::JITCompileFromStrOp::kJITEntryFunctionName;

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {

constexpr int64_t i32Limit = 4294967296;
using shape::ShapeOfOp;

bool IsSingleResultTFOperation(Operation *op) {
  assert(op != nullptr && "expect op");
  if (op->getDialect() !=
      op->getContext()->getLoadedDialect<TF::TensorFlowDialect>())
    return false;
  if (op->getNumResults() != 1) return false;
  return true;
}

bool IsUnaryTFOperation(Operation *op) {
  return IsSingleResultTFOperation(op) && op->getNumOperands() == 1;
}

bool IsBinaryTFOperation(Operation *op) {
  return IsSingleResultTFOperation(op) && op->getNumOperands() == 2;
}

struct TFToJITInvocationsPattern : public RewritePattern {
  explicit TFToJITInvocationsPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Apply to all single result TF ops except those that are already in a
    // JIT-compiled region.
    if (!IsSingleResultTFOperation(op) ||
        op->getParentOfType<tf_framework::JITCompileOp>())
      return failure();

    Location loc = op->getLoc();
    Value op_result = op->getResults().front();

    // Create the JIT compile op.
    auto jit_compile_op = rewriter.create<tf_framework::JITCompileOp>(
        loc, rewriter.getType<tf_framework::JITCallableType>(),
        /*ctx=*/llvm::None);

    // Move the TF operation into the body.
    {
      OpBuilder::InsertionGuard guard(rewriter);
      llvm::SmallVector<Location> locs(op->getNumOperands(), loc);
      Block *block = rewriter.createBlock(&jit_compile_op.getBody(), {},
                                          op->getOperandTypes(), locs);

      // Map operands.
      IRMapping bvm;
      for (auto it : llvm::zip(op->getOperands(), block->getArguments()))
        bvm.map(std::get<0>(it), std::get<1>(it));

      rewriter.setInsertionPointToStart(block);
      rewriter.clone(*op, bvm);
      rewriter.create<tf_framework::JITCompileYieldOp>(loc,
                                                       bvm.lookup(op_result));
    }

    // Create JIT execute op.
    rewriter.replaceOpWithNewOp<tf_framework::JITExecuteOp>(
        op, op_result.getType(), /*ctx=*/Value(), jit_compile_op.getResult(),
        op->getOperands());
    return success();
  }
};

struct TFToI64JITInvocationForLargeTensorsPattern : public RewritePattern {
  explicit TFToI64JITInvocationForLargeTensorsPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if ((!IsUnaryTFOperation(op) && !IsBinaryTFOperation(op)) ||
        !llvm::isa<func::FuncOp>(op->getParentOp())) {
      return failure();
    }

    // Create large argument condition.
    auto loc = op->getLoc();
    auto arg_1 = op->getOperands().front();
    auto shape_1 = rewriter.create<shape::ShapeOfOp>(loc, arg_1);
    auto num_elems_1 = rewriter.create<shape::NumElementsOp>(loc, shape_1);
    Value cst_i32_limit =
        rewriter.create<arith::ConstantIndexOp>(loc, i32Limit);
    Value large_tensor_predicate = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, num_elems_1, cst_i32_limit);
    if (IsBinaryTFOperation(op)) {
      auto arg_2 = op->getOperands().back();
      auto shape_2 = rewriter.create<shape::ShapeOfOp>(loc, arg_2);
      auto num_elems_2 = rewriter.create<shape::NumElementsOp>(loc, shape_2);
      large_tensor_predicate = rewriter.create<arith::OrIOp>(
          loc, large_tensor_predicate,
          // Compare op to check size of the second op
          rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                         num_elems_2, cst_i32_limit));
    }

    // Create dispatch code.
    auto jit_body_builder_fn = [&](OpBuilder &b, Location loc) {
      // Create JIT compile op.
      auto callable_ty = b.getType<tf_framework::JITCallableType>();
      auto jit_compile_op = b.create<tf_framework::JITCompileOp>(
          loc, callable_ty, /*ctx=*/Value());
      IRMapping bvm;
      {
        OpBuilder::InsertionGuard g(b);
        Block *block =
            b.createBlock(&jit_compile_op.getBody(), {}, op->getOperandTypes(),
                          SmallVector<Location>(op->getNumOperands(), loc));
        for (auto it : llvm::zip(op->getOperands(), block->getArguments()))
          bvm.map(std::get<0>(it), std::get<1>(it));
        b.setInsertionPointToStart(block);
        Operation *cloned_op = b.clone(*op, bvm);
        b.create<tf_framework::JITCompileYieldOp>(
            loc, cloned_op->getResults().front());
      }

      // Create JIT execute op.
      assert(op->getOperands().size() == 1 || op->getOperands().size() == 2);
      auto jit_execute_op = b.create<tf_framework::JITExecuteOp>(
          loc, op->getResultTypes().front(), /*ctx=*/Value(),
          jit_compile_op.getResult(), op->getOperands());
      b.create<scf::YieldOp>(loc, jit_execute_op.getResult());
    };
    auto aot_body_builder_fn = [&](OpBuilder &b, Location loc) {
      Operation *cloned_op = b.clone(*op);
      b.create<scf::YieldOp>(loc, cloned_op->getResults().front());
    };

    // Create and replace in two steps to clone the original op.
    auto ifOp = rewriter.create<scf::IfOp>(
        loc, large_tensor_predicate, jit_body_builder_fn, aot_body_builder_fn);
    rewriter.replaceOp(op, ifOp.getResults());
    return success();
  }
};

struct PackJITCompileOpPattern
    : public OpRewritePattern<tf_framework::JITCompileOp> {
  using OpRewritePattern<tf_framework::JITCompileOp>::OpRewritePattern;

  explicit PackJITCompileOpPattern(MLIRContext *ctx,
                                   llvm::ArrayRef<int64_t> tile_sizes,
                                   llvm::ArrayRef<int64_t> unroll_factors,
                                   int64_t max_supported_rank, bool enable_ftz,
                                   bool index_64bit, bool cpu_codegen)
      : OpRewritePattern<tf_framework::JITCompileOp>(ctx),
        tile_sizes(tile_sizes),
        unroll_factors(unroll_factors),
        max_supported_rank(max_supported_rank),
        enable_ftz(enable_ftz),
        index_64bit(index_64bit),
        cpu_codegen(cpu_codegen) {}

  LogicalResult matchAndRewrite(tf_framework::JITCompileOp op,
                                PatternRewriter &rewriter) const override {
    Block *body = op.SingleBlock::getBody();
    auto yield_op =
        llvm::cast<tf_framework::JITCompileYieldOp>(body->getTerminator());

    // Temporarily, build the module that would be JIT-compiled. This is only to
    // obtain the serialized code attribute.
    auto loc = op->getLoc();
    auto jit_module = rewriter.create<ModuleOp>(loc);
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(jit_module.SingleBlock::getBody());
      auto jit_function = rewriter.create<func::FuncOp>(
          loc, tf_framework::JITCompileFromStrOp::kJITEntryFunctionName,
          rewriter.getFunctionType(body->getArgumentTypes(),
                                   yield_op->getOperandTypes()));
      jit_function->setAttr(tf_framework::TFFrameworkDialect::kTFEntryAttrName,
                            rewriter.getUnitAttr());
      jit_function.getBody().takeBody(op.getBodyRegion());
      rewriter.setInsertionPointToEnd(&jit_function.getBody().front());
      rewriter.create<func::ReturnOp>(loc, yield_op.getResult());
      rewriter.eraseOp(yield_op);
    }

    // Serialize JIT module.
    std::string code;
    llvm::raw_string_ostream ss(code);
    assert(succeeded(jit_module.verify()));
    mlir::OpPrintingFlags flags;
    jit_module.print(ss, flags.assumeVerified());

    // Remove temporary module.
    rewriter.eraseOp(jit_module);

    // Finally, create the new JIT compile op.
    rewriter.replaceOpWithNewOp<tf_framework::JITCompileFromStrOp>(
        op, op->getResultTypes(), op.getCtx(), rewriter.getStringAttr(code),
        rewriter.getI64ArrayAttr(tile_sizes),
        rewriter.getI64ArrayAttr(unroll_factors),
        rewriter.getI64IntegerAttr(max_supported_rank),
        rewriter.getBoolAttr(enable_ftz), rewriter.getBoolAttr(index_64bit),
        rewriter.getBoolAttr(cpu_codegen));

    return success();
  }

 private:
  llvm::ArrayRef<int64_t> tile_sizes;
  llvm::ArrayRef<int64_t> unroll_factors;
  int64_t max_supported_rank;
  bool enable_ftz;
  bool index_64bit;
  bool cpu_codegen;
};

#define GEN_PASS_DEF_TFTOJITINVOCATIONPASS
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

struct TFToJITInvocationPass
    : public impl::TFToJITInvocationPassBase<TFToJITInvocationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::kernel_gen::tf_framework::TFFrameworkDialect,
                    scf::SCFDialect, shape::ShapeDialect>();
  }
  explicit TFToJITInvocationPass(llvm::ArrayRef<int64_t> tile_sizes,
                                 llvm::ArrayRef<int64_t> unroll_factors,
                                 int64_t max_supported_rank, bool enable_ftz,
                                 bool index_64bit, bool cpu_codegen,
                                 bool jit_i64_indexed_for_large_tensors) {
    tile_sizes_ = tile_sizes;
    unroll_factors_ = unroll_factors;
    max_supported_rank_ = max_supported_rank;
    enable_ftz_ = enable_ftz;
    index_64bit_ = index_64bit;
    cpu_codegen_ = cpu_codegen;
    jit_i64_indexed_for_large_tensors_ = jit_i64_indexed_for_large_tensors;
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    PopulateTFToJITInvocationPatterns(ctx, &patterns, tile_sizes_,
                                      unroll_factors_, max_supported_rank_,
                                      enable_ftz_, index_64bit_, cpu_codegen_,
                                      jit_i64_indexed_for_large_tensors_);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

void PopulateTFToJITInvocationPatterns(
    MLIRContext *ctx, RewritePatternSet *patterns,
    llvm::ArrayRef<int64_t> tile_sizes, llvm::ArrayRef<int64_t> unroll_factors,
    int64_t max_supported_rank, bool enable_ftz, bool index_64bit,
    bool cpu_codegen, bool jit_i64_indexed_for_large_tensors) {
  if (jit_i64_indexed_for_large_tensors) {
    patterns->add<TFToI64JITInvocationForLargeTensorsPattern>(ctx);
  } else {
    patterns->add<TFToJITInvocationsPattern>(ctx);
  }

  bool index_64bit_if_jit_compiling =
      jit_i64_indexed_for_large_tensors ? true : index_64bit;
  patterns->add<PackJITCompileOpPattern>(
      ctx, tile_sizes, unroll_factors, max_supported_rank, enable_ftz,
      index_64bit_if_jit_compiling, cpu_codegen);
}

std::unique_ptr<OperationPass<func::FuncOp>> CreateTFToJITInvocationPass(
    llvm::ArrayRef<int64_t> tile_sizes, llvm::ArrayRef<int64_t> unroll_factors,
    int64_t max_supported_rank, bool enable_ftz, bool index_64bit,
    bool cpu_codegen, bool jit_i64_indexed_for_large_tensors) {
  return std::make_unique<TFToJITInvocationPass>(
      tile_sizes, unroll_factors, max_supported_rank, enable_ftz, index_64bit,
      cpu_codegen, jit_i64_indexed_for_large_tensors);
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
