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

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_cpurt_passes.h"

namespace tensorflow {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_cpurt_passes.h.inc"

// Convert 2D memref into a 0D memref (scalar).
mlir::Value MemrefToScalar(mlir::OpBuilder& builder, mlir::Location loc,
                           mlir::Value memref) {
  auto memref_type = memref.getType().cast<mlir::MemRefType>();
  auto scalar_type = mlir::MemRefType::get({}, memref_type.getElementType());

  std::array<int64_t, 0> empty;
  return builder.create<mlir::memref::ReinterpretCastOp>(
      loc, scalar_type, memref, /*offset=*/0,
      /*sizes=*/empty, /*strides=*/empty);
}

// Convert 2D memref into a 1D memref (vector).
mlir::Value MemrefToVector(mlir::OpBuilder& builder, mlir::Location loc,
                           mlir::Value memref, mlir::Value size,
                           int64_t static_size) {
  assert(static_size >= 0 || static_size == mlir::ShapedType::kDynamicSize);
  auto memref_type = memref.getType().cast<mlir::MemRefType>();
  auto vec_type =
      mlir::MemRefType::get({static_size}, memref_type.getElementType());

  auto static_offsets = builder.getI64ArrayAttr({0});
  auto static_sizes = builder.getI64ArrayAttr({static_size});
  auto static_strided = builder.getI64ArrayAttr({1});

  auto empty = mlir::ValueRange();
  auto sizes = static_size == mlir::ShapedType::kDynamicSize
                   ? mlir::ValueRange(size)
                   : mlir::ValueRange();

  return builder.create<mlir::memref::ReinterpretCastOp>(
      loc, vec_type, memref, /*offsets=*/empty,
      /*sizes=*/sizes, /*strides=*/empty, static_offsets, static_sizes,
      static_strided);
}

struct LinalgMatmulSpecializationPattern
    : public mlir::OpRewritePattern<mlir::linalg::MatmulOp> {
  using OpRewritePattern<mlir::linalg::MatmulOp>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(
      mlir::linalg::MatmulOp matmul,
      mlir::PatternRewriter& rewriter) const override;
};
mlir::LogicalResult LinalgMatmulSpecializationPattern::matchAndRewrite(
    mlir::linalg::MatmulOp matmul, mlir::PatternRewriter& rewriter) const {
  if (matmul->hasAttr("__tf_cpurt_specialized")) {
    return rewriter.notifyMatchFailure(matmul,
                                       "operation was already specialized");
  }

  auto rhs = matmul.getInputOperand(1)->get();
  auto lhs = matmul.getInputOperand(0)->get();
  auto out = matmul.getOutputOperand(0)->get();

  // We do not support inputs or outputs that are not contiguous in memory.
  if (!IsContiguousMemref(lhs) || !IsContiguousMemref(rhs) ||
      !IsContiguousMemref(out)) {
    return rewriter.notifyMatchFailure(
        matmul, "inputs and output must be contiguous memrefs");
  }

  auto loc = matmul.getLoc();

  // Matmul dimensions: [m, k] x [k, n]
  mlir::Value m = rewriter.create<mlir::memref::DimOp>(loc, lhs, 0);
  mlir::Value k = rewriter.create<mlir::memref::DimOp>(loc, lhs, 1);
  mlir::Value n = rewriter.create<mlir::memref::DimOp>(loc, rhs, 1);

  // Matmul static dimensions if they are known (can be ShapedType::kDynamicSize
  // if not known statically).
  int64_t m_static = lhs.getType().cast<mlir::MemRefType>().getDimSize(0);
  int64_t k_static = lhs.getType().cast<mlir::MemRefType>().getDimSize(1);
  int64_t n_static = rhs.getType().cast<mlir::MemRefType>().getDimSize(1);

  auto one = rewriter.create<mlir::arith::ConstantOp>(
      loc, rewriter.getIndexType(), rewriter.getIndexAttr(1));
  auto m_is_one = rewriter.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, m, one);
  auto n_is_one = rewriter.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, n, one);

  auto m_not_one = rewriter.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::ne, m, one);
  auto n_not_one = rewriter.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::ne, n, one);

  // linalg.dot: n == 1 && m == 1
  auto is_dot_product =
      rewriter.create<mlir::arith::AndIOp>(loc, m_is_one, n_is_one);
  // linalg.vecmat m == 1 && n != 1
  auto is_vecmat =
      rewriter.create<mlir::arith::AndIOp>(loc, m_is_one, n_not_one);
  // linalg.matvec n == 1 && m != 1
  auto is_matvec =
      rewriter.create<mlir::arith::AndIOp>(loc, n_is_one, m_not_one);

  // Build a linalg.dot operation casting inputs to vectors.
  auto dot = [&](mlir::OpBuilder& builder, mlir::Location nestedLoc) {
    auto lhs_vec = MemrefToVector(builder, nestedLoc, lhs, k, k_static);
    auto rhs_vec = MemrefToVector(builder, nestedLoc, rhs, k, k_static);
    auto out_scalar = MemrefToScalar(builder, nestedLoc, out);

    builder.create<mlir::linalg::DotOp>(nestedLoc,
                                        mlir::ValueRange({lhs_vec, rhs_vec}),
                                        mlir::ValueRange({out_scalar}));
    builder.create<mlir::scf::YieldOp>(nestedLoc);
  };

  // Build a linalg.vecmat operation casting lhs to vector.
  auto vecmat = [&](mlir::OpBuilder& builder, mlir::Location nestedLoc) {
    auto lhs_vec = MemrefToVector(builder, nestedLoc, lhs, k, k_static);
    auto out_vec = MemrefToVector(builder, nestedLoc, out, n, n_static);

    builder.create<mlir::linalg::VecmatOp>(nestedLoc,
                                           mlir::ValueRange({lhs_vec, rhs}),
                                           mlir::ValueRange({out_vec}));
    builder.create<mlir::scf::YieldOp>(nestedLoc);
  };

  // Build a linalg.matvec operation casting rhs to vector.
  auto matvec = [&](mlir::OpBuilder& builder, mlir::Location nestedLoc) {
    auto rhs_vec = MemrefToVector(builder, nestedLoc, rhs, k, k_static);
    auto out_vec = MemrefToVector(builder, nestedLoc, out, m, m_static);

    builder.create<mlir::linalg::MatvecOp>(nestedLoc,
                                           mlir::ValueRange({lhs, rhs_vec}),
                                           mlir::ValueRange({out_vec}));
    builder.create<mlir::scf::YieldOp>(nestedLoc);
  };

  // Build a generic linalg.matmul operation when it can't be matched to any of
  // the specializations.
  auto generic = [&](mlir::OpBuilder& builder, mlir::Location nestedLoc) {
    llvm::SmallVector<mlir::Value> inputs = matmul.getInputOperands();
    llvm::SmallVector<mlir::Value> outputs = matmul.getOutputOperands();
    auto specialized =
        builder.create<mlir::linalg::MatmulOp>(nestedLoc, inputs, outputs);
    specialized->setAttr("__tf_cpurt_specialized", rewriter.getUnitAttr());
    builder.create<mlir::scf::YieldOp>(nestedLoc);
  };

  // TODO(ezhulenev): Simplify to scf.switch operation.
  // if (is_dot_product) ===>>> linalg.dot    ------------------------------- //
  auto dispatch = rewriter.create<mlir::scf::IfOp>(
      loc, is_dot_product, dot,
      [&](mlir::OpBuilder& builder, mlir::Location nestedLoc) {
        // else if (is_vecmat)  ===>>> linalg.vecmat    --------------------- //
        rewriter.create<mlir::scf::IfOp>(
            nestedLoc, is_vecmat, vecmat,
            [&](mlir::OpBuilder& builder, mlir::Location nestedLoc) {
              // else if (is_matvec)  ===>>> linalg.matvec    --------------- //
              // else                 ===>>> linalg.matmul    --------------- //
              rewriter.create<mlir::scf::IfOp>(nestedLoc, is_matvec, matvec,
                                               generic);
              builder.create<mlir::scf::YieldOp>(nestedLoc);
            });
        builder.create<mlir::scf::YieldOp>(nestedLoc);
      });

  rewriter.replaceOp(matmul, dispatch.results());
  return mlir::success();
}

// -------------------------------------------------------------------------- //
// Dispatch linalg.matmul to one of the more specialized operations at runtime.
// -------------------------------------------------------------------------- //
struct LinalgMatmulSpecializationPass
    : public LinalgMatmulSpecializationBase<LinalgMatmulSpecializationPass> {
  void runOnFunction() override {
    mlir::FuncOp function = getFunction();
    mlir::MLIRContext* ctx = function.getContext();

    mlir::RewritePatternSet patterns(ctx);
    patterns.insert<LinalgMatmulSpecializationPattern>(ctx);

    (void)mlir::applyPatternsAndFoldGreedily(function, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<mlir::FunctionPass> CreateLinalgMatmulSpecializationPass() {
  return std::make_unique<LinalgMatmulSpecializationPass>();
}

}  // namespace tensorflow
