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

#include <string>
#include <utility>

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
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

constexpr int64_t i32BitLimit = 4294967296;
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
      Block *block = rewriter.createBlock(&jit_compile_op.body(), {},
                                          op->getOperandTypes(), locs);

      // Map operands.
      BlockAndValueMapping bvm;
      for (auto it : llvm::zip(op->getOperands(), block->getArguments()))
        bvm.map(std::get<0>(it), std::get<1>(it));

      rewriter.setInsertionPointToStart(block);
      rewriter.clone(*op, bvm);
      rewriter.create<tf_framework::JITCompileYieldOp>(loc,
                                                       bvm.lookup(op_result));
    }

    // Create JIT execute op.
    rewriter.replaceOpWithNewOp<tf_framework::JITExecuteOp>(
        op, op_result.getType(), /*ctx=*/Value(), jit_compile_op.result(),
        op->getOperands());
    return success();
  }
};

struct TFToI64JITInvocationForLargeTensorsPattern : public RewritePattern {
  explicit TFToI64JITInvocationForLargeTensorsPattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!IsUnaryTFOperation(op) ||
        !llvm::isa<func::FuncOp>(op->getParentOp())) {
      return failure();
    }

    auto results = llvm::to_vector<16>(op->getResults());
    auto operand_types = llvm::to_vector<16>(llvm::map_range(
        op->getOperands(), [](Value v) { return v.getType(); }));
    auto result_types = llvm::to_vector<16>(
        llvm::map_range(results, [](Value v) { return v.getType(); }));

    // Create the JIT compile op.
    auto loc = op->getLoc();
    Value shape_size_limit =
        rewriter.create<arith::ConstantIndexOp>(loc, i32BitLimit);
    auto arg = op->getOperands().front();
    auto shape = rewriter.create<shape::ShapeOfOp>(loc, arg);
    auto num_elems = rewriter.create<shape::NumElementsOp>(loc, shape);
    Value coniditon_check_main = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, num_elems, shape_size_limit);

    Value conditional_path =
        rewriter
            .create<scf::IfOp>(
                loc, op->getResultTypes(), coniditon_check_main,
                [&](OpBuilder &b, Location l) {
                  auto jit_compile_op =
                      rewriter.create<tf_framework::JITCompileOp>(
                          loc,
                          rewriter.getType<tf_framework::JITCallableType>(),
                          llvm::None);
                  BlockAndValueMapping bvm;
                  {
                    OpBuilder::InsertionGuard guard(rewriter);
                    Block *block = rewriter.createBlock(
                        &jit_compile_op.body(), {}, operand_types,
                        SmallVector<Location>(operand_types.size(), loc));
                    for (auto it :
                         llvm::zip(op->getOperands(), block->getArguments()))
                      bvm.map(std::get<0>(it), std::get<1>(it));
                    rewriter.setInsertionPointToStart(block);
                    rewriter.clone(*op, bvm);
                    auto new_op = rewriter.clone(*op, bvm);
                    rewriter.create<tf_framework::JITCompileYieldOp>(
                        loc, TypeRange{}, new_op->getResults());
                  }
                  auto jit_execute_op =
                      rewriter.create<tf_framework::JITExecuteOp>(
                          loc, result_types, Value(), jit_compile_op.result(),
                          op->getOperands());
                  b.create<scf::YieldOp>(l, jit_execute_op.result());
                },
                [&](OpBuilder &b, Location l) {
                  auto new_op = rewriter.clone(*op);
                  b.create<scf::YieldOp>(l, new_op->getResult(0));
                })
            .getResult(0);

    rewriter.replaceOp(op, conditional_path);
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
                                   bool index_64bit_if_jit_compiling,
                                   bool cpu_codegen)
      : OpRewritePattern<tf_framework::JITCompileOp>(ctx),
        tile_sizes(tile_sizes),
        unroll_factors(unroll_factors),
        max_supported_rank(max_supported_rank),
        enable_ftz(enable_ftz),
        index_64bit_if_jit_compiling(index_64bit_if_jit_compiling),
        cpu_codegen(cpu_codegen) {}

  LogicalResult matchAndRewrite(tf_framework::JITCompileOp op,
                                PatternRewriter &rewriter) const override {
    Block *body = op.SingleBlock::getBody();
    auto yield_op =
        llvm::cast<tf_framework::JITCompileYieldOp>(body->getTerminator());

    // Temporarily, build the module that would be JIT-compiled. This is only to
    // obtain the serialized code attribute.
    auto loc = op->getLoc();
    OpBuilder tmp_module_builder(getContext(), rewriter.getListener());
    auto jit_module = tmp_module_builder.create<ModuleOp>(loc);
    tmp_module_builder.setInsertionPointToStart(
        jit_module.SingleBlock::getBody());
    auto jit_function = tmp_module_builder.create<func::FuncOp>(
        loc, tf_framework::JITCompileFromStrOp::kJITEntryFunctionName,
        tmp_module_builder.getFunctionType(body->getArgumentTypes(),
                                           yield_op->getOperandTypes()));
    jit_function->setAttr(tf_framework::TFFrameworkDialect::kTFEntryAttrName,
                          tmp_module_builder.getUnitAttr());
    jit_function.getBody().takeBody(op.getBodyRegion());
    tmp_module_builder.setInsertionPointToEnd(&jit_function.getBody().front());
    tmp_module_builder.create<func::ReturnOp>(loc, yield_op.result());
    rewriter.eraseOp(yield_op);

    // Serialize JIT module.
    std::string code;
    llvm::raw_string_ostream ss(code);
    jit_module.print(ss);

    // Finally, create the new JIT compile op.
    rewriter.replaceOpWithNewOp<tf_framework::JITCompileFromStrOp>(
        op, op->getResultTypes(), op.ctx(), rewriter.getStringAttr(code),
        rewriter.getI64ArrayAttr(tile_sizes),
        rewriter.getI64ArrayAttr(unroll_factors),
        rewriter.getI64IntegerAttr(max_supported_rank),
        rewriter.getBoolAttr(enable_ftz),
        rewriter.getBoolAttr(index_64bit_if_jit_compiling),
        rewriter.getBoolAttr(cpu_codegen));

    return success();
  }

 private:
  llvm::ArrayRef<int64_t> tile_sizes;
  llvm::ArrayRef<int64_t> unroll_factors;
  int64_t max_supported_rank;
  bool enable_ftz;
  bool index_64bit_if_jit_compiling;
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
