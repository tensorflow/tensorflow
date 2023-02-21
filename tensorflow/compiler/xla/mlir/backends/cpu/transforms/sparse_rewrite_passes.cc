/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <cassert>
#include <memory>
#include <utility>

#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/backends/cpu/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace xla {
namespace cpu {
namespace {

#define GEN_PASS_DEF_SPARSECUSTOMCALLTOPACKPASS
#include "tensorflow/compiler/xla/mlir/backends/cpu/transforms/passes.h.inc"

using namespace mlir;  // NOLINT

class SparseCustomCallToPackPass
    : public impl::SparseCustomCallToPackPassBase<SparseCustomCallToPackPass> {
  void runOnOperation() override;
};

class SparseCustomCallToPackRewriter
    : public OpRewritePattern<mhlo::CustomCallOp> {
  using OpRewritePattern<mhlo::CustomCallOp>::OpRewritePattern;
  // Rewrites a CustomCallOp to target 'sparse_tensor_pack/unpack' to
  // the corresponding sparse_tensor::PackOp and sparse_tensor::UnpackOp.
  LogicalResult matchAndRewrite(mhlo::CustomCallOp op,
                                PatternRewriter& rewriter) const override {
    const StringRef sparse_pack_call_name = "sparse_tensor_pack";
    const StringRef sparse_unpack_call_name = "sparse_tensor_unpack";

    if (op.getCallTargetName().equals(sparse_pack_call_name)) {
      assert(op.getInputs().size() == 2 && "Need two arrays (data/indices)");
      assert(op.getResults().size() == 1 && "Must be packing into one tensor");
      Value ret_sp_tensor = op.getResults()[0];
      rewriter.replaceOpWithNewOp<sparse_tensor::PackOp>(
          op, ret_sp_tensor.getType(), op.getInputs()[0], op.getInputs()[1]);
      return success();
    } else if (op.getCallTargetName().equals(sparse_unpack_call_name)) {
      assert(op.getResults().size() == 3 &&
             "Must be unpacking into data/indices/nnz");
      assert(op.getInputs().size() == 1 &&
             "Must be unpacking from one sparse tensor");

      SmallVector<Type, 3> unpack_ret_tp(op.getResults().getTypes());
      // A scalar is treated as a zero-ranked tensor type from frontend.
      auto nnz_type = unpack_ret_tp.back().cast<RankedTensorType>();
      assert(nnz_type.getRank() == 0 && "nnz tensor must be zero ranked");
      unpack_ret_tp.back() = nnz_type.getElementType();

      // Constructs the UnpackOp.
      auto unpack_op = rewriter.create<sparse_tensor::UnpackOp>(
          op.getLoc(), unpack_ret_tp, op.getInputs());

      // Converts the scalar nnz returned from UnpackOp back to tensor type.
      SmallVector<Value, 3> unpack_ret_v(unpack_op.getResults());
      auto scalar_nnz = unpack_op.getNnz();
      Value tensor_nnz = rewriter.create<tensor::EmptyOp>(
          op.getLoc(), ArrayRef<int64_t>{}, scalar_nnz.getType());
      tensor_nnz = rewriter.create<tensor::InsertOp>(op.getLoc(), scalar_nnz,
                                                     tensor_nnz, ValueRange{});
      unpack_ret_v.back() = tensor_nnz;
      rewriter.replaceOp(op, unpack_ret_v);
      return success();
    }
    // Returns failure on unmatched call target.
    return failure();
  }
};

class ReallocToAllocRewriter : public OpRewritePattern<memref::ReallocOp> {
  using OpRewritePattern::OpRewritePattern;
  // Rewrites a Realloc to alloc + copy
  LogicalResult matchAndRewrite(memref::ReallocOp op,
                                PatternRewriter& rewriter) const override {
    Value alloc = rewriter.create<memref::AllocOp>(
        op.getLoc(), op.getType(), op.getOperands().drop_front(1),
        op.getAlignmentAttr());
    rewriter.create<memref::CopyOp>(op.getLoc(), op.getSource(), alloc);
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void SparseCustomCallToPackPass::runOnOperation() {
  func::FuncOp func = getOperation();
  MLIRContext* ctx = func.getContext();

  RewritePatternSet patterns(ctx);
  patterns.insert<SparseCustomCallToPackRewriter>(ctx);

  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createSparseCustomCallToPackUnpackOpPass() {
  return std::make_unique<SparseCustomCallToPackPass>();
}

}  // namespace cpu
}  // namespace xla
