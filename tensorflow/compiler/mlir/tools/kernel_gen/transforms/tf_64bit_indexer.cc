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
#include <iostream>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SCF/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
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
#include "mlir/Transforms/Bufferize.h"  // from @llvm-project


namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {


using shape::ShapeOfOp;
static constexpr StringRef kEmitCInterfaceAttrName = "llvm.emit_c_interface";

bool IsTFOperation(Operation *op) {
  return op != nullptr &&
         op->getDialect() ==
             op->getContext()->getLoadedDialect<TF::TensorFlowDialect>();
}

struct TFT64BitIndexerPattern : public RewritePattern {
  explicit TFT64BitIndexerPattern(MLIRContext *ctx, bool index_64bit)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {index_64bit_ = index_64bit;}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!IsTFOperation(op) || op->getParentOfType<tf_framework::JITCompileOp>() ||
          !llvm::isa<FuncOp>(op->getParentOp()))
    {
        return failure();
    }
    SmallVector<Operation *, 16> cluster;
    llvm::SmallPtrSet<Value, 16> operand_set, result_set;
    Operation *it = op;
    while (IsTFOperation(it)) {
      // Find results that escape the JIT compile region.
      for (auto &use : it->getUses()) {
        if (!llvm::is_contained(cluster, use.getOwner()))
          result_set.insert(use.get());
      }

      // Update JIT region operands and results.
      for (Value v : it->getResults()) operand_set.erase(v);
      for (Value v : it->getOperands()) operand_set.insert(v);
      cluster.push_back(it);
      it = it->getPrevNode();
    }

    // Introduce order to the operands and results.
    auto operands = llvm::to_vector<16>(operand_set);
    auto results = llvm::to_vector<16>(result_set);
    auto operand_types = llvm::to_vector<16>(
        llvm::map_range(operands, [](Value v) { return v.getType(); }));
    auto result_types = llvm::to_vector<16>(
        llvm::map_range(results, [](Value v) { return v.getType(); }));

    // Create the JIT compile op.
    auto loc = op->getLoc();
    Value shape_size_limit = rewriter.create<arith::ConstantIndexOp>(loc, 4294967296);
    auto arg = op->getOperands().front(); 
    auto shape = rewriter.create<shape::ShapeOfOp>(loc, arg);
    auto num_elems = rewriter.create<shape::NumElementsOp>(loc, shape);
    Value coniditon_check_main = rewriter.create<arith::CmpIOp>(
                    loc, arith::CmpIPredicate::sgt, num_elems, shape_size_limit);

    Value conditional_path =
                rewriter.create<scf::IfOp>(
                  loc, op->getResultTypes(), coniditon_check_main,
                  [&](OpBuilder &b, Location l) {
                    auto jit_compile_op = rewriter.create<tf_framework::JITCompileOp>(
                      loc, rewriter.getType<tf_framework::JITCallableType>(), llvm::None);
                      BlockAndValueMapping bvm;
                      {
                        OpBuilder::InsertionGuard guard(rewriter);
                        Block *block =
                            rewriter.createBlock(&jit_compile_op.body(), {}, operand_types);
                        for (auto it : llvm::zip(operands, block->getArguments()))
                          bvm.map(std::get<0>(it), std::get<1>(it));
                        rewriter.setInsertionPointToStart(block);
                        for (Operation *it : llvm::reverse(cluster)) rewriter.clone(*it, bvm);
                        auto mapped_results = llvm::to_vector<16>(
                            llvm::map_range(results, [&](Value v) { return bvm.lookup(v); }));
                        rewriter.create<tf_framework::JITCompileYieldOp>(loc, TypeRange{},
                                                                        mapped_results);
                      }
                    auto jit_execute_op = rewriter.create<tf_framework::JITExecuteOp>(
                        loc, result_types, Value(), jit_compile_op.result(), operands);
                          b.create<scf::YieldOp>(l, jit_execute_op.results());
                     },
                     [&](OpBuilder &b, Location l) {
                       Operation *op_backup = rewriter.clone(*op);
                       b.create<scf::YieldOp>(l, op_backup->getResult(0));
                     })
                    .getResult(0);

  rewriter.replaceOp(op, conditional_path);
  return success();
  }
  private:
  bool index_64bit_;
};

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

struct IndexSizerPass
    : public IndexSizerPassBase<IndexSizerPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::kernel_gen::tf_framework::TFFrameworkDialect, 
    scf::SCFDialect, shape::ShapeDialect>();
  }
  explicit IndexSizerPass(llvm::ArrayRef<std::string> architectures,
                                 llvm::ArrayRef<int64_t> tile_sizes,
                                 llvm::ArrayRef<int64_t> unroll_factors,
                                 int64_t max_supported_rank, bool enable_ftz,
                                 bool index_64bit, bool cpu_codegen) {
    architectures_ = architectures;
    tile_sizes_ = tile_sizes;
    unroll_factors_ = unroll_factors;
    max_supported_rank_ = max_supported_rank;
    enable_ftz_ = enable_ftz;
    index_64bit_ = index_64bit;
    cpu_codegen_ = cpu_codegen;
  }

  void runOnFunction() override {
    MLIRContext *ctx = &getContext();
    BufferizeTypeConverter converter;
    RewritePatternSet patterns(ctx);
    auto architecture_refs = llvm::to_vector<16>(llvm::map_range(
        architectures_, [](std::string &arch) { return StringRef(arch); }));
    Populate64BitIndexerPatterns(
        ctx, &patterns, architecture_refs, tile_sizes_, unroll_factors_,
        max_supported_rank_, enable_ftz_, index_64bit_, cpu_codegen_);
    if (failed(
            applyPatternsAndFoldGreedily(getFunction(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

void Populate64BitIndexerPatterns(MLIRContext *ctx,
                                       RewritePatternSet *patterns,
                                       llvm::ArrayRef<StringRef> architectures,
                                       llvm::ArrayRef<int64_t> tile_sizes,
                                       llvm::ArrayRef<int64_t> unroll_factors,
                                       int64_t max_supported_rank,
                                       bool enable_ftz, bool index_64bit,
                                       bool cpu_codegen) {
  patterns->insert<TFT64BitIndexerPattern>(ctx, index_64bit);
}

std::unique_ptr<FunctionPass> Create64BitIndexerInvocationPass(
    llvm::ArrayRef<std::string> architectures,
    llvm::ArrayRef<int64_t> tile_sizes, llvm::ArrayRef<int64_t> unroll_factors,
    int64_t max_supported_rank, bool enable_ftz, bool index_64bit, bool cpu_codegen) {
  return std::make_unique<IndexSizerPass>(
      architectures, tile_sizes, unroll_factors, max_supported_rank, enable_ftz,
      index_64bit, cpu_codegen);
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
