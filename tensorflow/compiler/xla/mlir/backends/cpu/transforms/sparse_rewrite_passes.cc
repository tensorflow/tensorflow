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
#include <cstdint>
#include <functional>
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

#define GEN_PASS_DEF_SPARSECUSTOMCALLREWRITINGPASS
#include "tensorflow/compiler/xla/mlir/backends/cpu/transforms/passes.h.inc"

using namespace mlir;  // NOLINT

class SparseCustomCallRewritingPass
    : public impl::SparseCustomCallRewritingPassBase<
          SparseCustomCallRewritingPass> {
  void runOnOperation() override;
};

struct SparsePackCallRewriter {
  LogicalResult operator()(mhlo::CustomCallOp op, PatternRewriter& rewriter) {
    assert(op.getInputs().size() == 2 && "Need two arrays (data/indices)");
    assert(op.getResults().size() == 1 && "Must be packing into one tensor");
    Value ret_sp_tensor = op.getResults()[0];
    rewriter.replaceOpWithNewOp<sparse_tensor::PackOp>(
        op, ret_sp_tensor.getType(), op.getInputs()[0], op.getInputs()[1]);
    return success();
  }
};

struct SparseUnpackCallRewriter {
  LogicalResult operator()(mhlo::CustomCallOp op, PatternRewriter& rewriter) {
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
    auto scalar_nnz = unpack_op.getNse();
    Value tensor_nnz = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), ArrayRef<int64_t>{}, scalar_nnz.getType());
    tensor_nnz = rewriter.create<tensor::InsertOp>(op.getLoc(), scalar_nnz,
                                                   tensor_nnz, ValueRange{});
    unpack_ret_v.back() = tensor_nnz;
    rewriter.replaceOp(op, unpack_ret_v);
    return success();
  }
};

struct SparseTransposeCallRewriter {
  LogicalResult operator()(mhlo::CustomCallOp op, PatternRewriter& rewriter) {
    assert(op.getInputs().size() == 2 && "Need argument and permutation");
    assert(op.getResults().size() == 1 && "Need one output tensor");

    // The permutation is passed in as a constant of dense int elements.
    auto permutation_constant =
        op.getInputs()[1].getDefiningOp<mhlo::ConstantOp>();
    auto permutation =
        permutation_constant.getValue().cast<DenseIntElementsAttr>();

    // Reconstruct the transpose operation.
    Value ret_sp_tensor = op.getResults()[0];
    rewriter.replaceOpWithNewOp<mhlo::TransposeOp>(
        op, ret_sp_tensor.getType(), op.getInputs()[0], permutation);
    return success();
  }
};

struct SparseConcatenateCallRewriter {
  LogicalResult operator()(mhlo::CustomCallOp op, PatternRewriter& rewriter) {
    assert(op.getResults().size() == 1 && "Need one output tensor");

    // The concatenation dimension.
    auto concat_dim = op.getInputs().back().getDefiningOp<mhlo::ConstantOp>();
    auto concat_dim_attr = concat_dim.getValue().cast<DenseIntElementsAttr>();
    // Reconstruct the concatenate operation.
    Value ret_sp_tensor = op.getResults()[0];
    // Depending on test setup, we can get either a 32-bit integer or a 64-bit
    // integer.
    if (concat_dim_attr.getElementType().isInteger(32)) {
      rewriter.replaceOpWithNewOp<sparse_tensor::ConcatenateOp>(
          op, ret_sp_tensor.getType(), op.getInputs().drop_back(),
          rewriter.getIndexAttr(concat_dim_attr.getValues<uint32_t>()[0]));
    } else {
      assert(concat_dim_attr.getElementType().isInteger(64));
      rewriter.replaceOpWithNewOp<sparse_tensor::ConcatenateOp>(
          op, ret_sp_tensor.getType(), op.getInputs().drop_back(),
          rewriter.getIndexAttr(concat_dim_attr.getValues<uint64_t>()[0]));
    }

    return success();
  }
};

struct SparseBroadcastInDimCallRewriter {
  LogicalResult operator()(mhlo::CustomCallOp op, PatternRewriter& rewriter) {
    assert(op.getInputs().size() == 2 &&
           "Need argument and broadcast dimensions");
    assert(op.getResults().size() == 1 && "Need one output tensor");

    // Broadcast dimensions are passed in as a constant of dense int elements.
    auto dims_constant = op.getInputs()[1].getDefiningOp<mhlo::ConstantOp>();
    auto broadcast_dimensions =
        dims_constant.getValue().cast<DenseIntElementsAttr>();

    // Reconstruct the broadcast_in_dim operation.
    Value ret_sp_tensor = op.getResults()[0];
    rewriter.replaceOpWithNewOp<mhlo::BroadcastInDimOp>(
        op, ret_sp_tensor.getType(), op.getInputs()[0], broadcast_dimensions);
    return success();
  }
};

class SparseCustomCallRewriter : public OpRewritePattern<mhlo::CustomCallOp> {
  using OpRewritePattern<mhlo::CustomCallOp>::OpRewritePattern;
  using SparseCustomTargetRewriter = std::function<LogicalResult(
      mhlo::CustomCallOp op, PatternRewriter& rewriter)>;

  const llvm::StringMap<SparseCustomTargetRewriter> rewriter_map_{
      std::make_pair("sparse_tensor_sparse_pack", SparsePackCallRewriter()),
      std::make_pair("sparse_tensor_sparse_unpack", SparseUnpackCallRewriter()),
      std::make_pair("sparse_tensor_transpose", SparseTransposeCallRewriter()),
      std::make_pair("sparse_tensor_broadcast_in_dim",
                     SparseBroadcastInDimCallRewriter()),
      std::make_pair("sparse_tensor_concatenate",
                     SparseConcatenateCallRewriter()),
  };

  // Rewrites a CustomCallOp to target 'sparse_tensor_pack/unpack' to
  // the corresponding sparse_tensor::PackOp and sparse_tensor::UnpackOp.
  LogicalResult matchAndRewrite(mhlo::CustomCallOp op,
                                PatternRewriter& rewriter) const override {
    if (auto it = rewriter_map_.find(op.getCallTargetName());
        it != rewriter_map_.end()) {
      return it->second(op, rewriter);
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

void SparseCustomCallRewritingPass::runOnOperation() {
  func::FuncOp func = getOperation();
  MLIRContext* ctx = func.getContext();

  RewritePatternSet patterns(ctx);
  patterns.insert<SparseCustomCallRewriter>(ctx);

  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createSparseCustomCallRewritingPass() {
  return std::make_unique<SparseCustomCallRewritingPass>();
}

}  // namespace cpu
}  // namespace xla
