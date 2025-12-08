/* Copyright 2025 The OpenXLA Authors.

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
#include <memory>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h"

namespace xla::cpu {

#define GEN_PASS_DEF_UNPACKSUBBYTEVECTORWRITEPASS
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h.inc"

namespace {

mlir::Value Get1DVectorElement(mlir::OpBuilder& builder, mlir::Location loc,
                               mlir::TypedValue<mlir::VectorType> vector,
                               int64_t idx) {
  auto vector_1d_type =
      mlir::VectorType::get({1}, vector.getType().getElementType());
  mlir::Value element =
      mlir::vector::ExtractOp::create(builder, loc, vector, idx);
  return mlir::vector::FromElementsOp::create(builder, loc, vector_1d_type,
                                              element);
}

mlir::ArrayAttr GetBoundsAttr(mlir::OpBuilder& builder) {
  // I'm not confident what the semantics of inbounds are with subbyte types so
  // lets be conservative and set them all to false, the canonicalization pass
  // will set it to true where possible.
  llvm::SmallVector<bool> in_bounds{false};
  return builder.getBoolArrayAttr(in_bounds);
}

template <typename TransferOp>
mlir::LogicalResult CheckCanUnroll(TransferOp op,
                                   mlir::PatternRewriter& rewriter) {
  if (!mlir::isa<mlir::MemRefType>(op.getBase().getType())) {
    return rewriter.notifyMatchFailure(op, "base is not a memref.");
  }

  auto vector_type = op.getVectorType();

  if (!vector_type.getElementType().isIntOrIndexOrFloat() ||
      vector_type.getElementType().getIntOrFloatBitWidth() >= 8) {
    return rewriter.notifyMatchFailure(op,
                                       "element type is not a sub-byte type.");
  }

  auto num_elements = vector_type.getNumElements();
  if (vector_type.getRank() != 1 || num_elements == 1) {
    return rewriter.notifyMatchFailure(op, "vector is already trivial.");
  }

  return mlir::success();
}

mlir::LogicalResult UnrollTransferRead(mlir::vector::TransferReadOp op,
                                       mlir::PatternRewriter& rewriter) {
  if (auto status = CheckCanUnroll(op, rewriter); mlir::failed(status)) {
    return status;
  }

  auto vector_type = op.getVectorType();
  auto num_elements = vector_type.getNumElements();

  auto in_bounds_attr = GetBoundsAttr(rewriter);

  llvm::SmallVector<mlir::Value> elements;
  for (int64_t idx = 0; idx != num_elements; ++idx) {
    auto vector_1d_type =
        mlir::VectorType::get({1}, vector_type.getElementType());

    mlir::Value mask = op.getMask() ? Get1DVectorElement(rewriter, op->getLoc(),
                                                         op.getMask(), idx)
                                    : nullptr;

    llvm::SmallVector<mlir::Value> offsets = op.getIndices();
    offsets.back() = mlir::arith::AddIOp::create(
        rewriter, op->getLoc(), offsets.back(),
        mlir::arith::ConstantIndexOp::create(rewriter, op->getLoc(), idx));

    mlir::Value element_vector = mlir::vector::TransferReadOp::create(
        rewriter, op->getLoc(), vector_1d_type, op.getBase(), offsets,
        op.getPermutationMapAttr(), op.getPadding(), mask, in_bounds_attr);
    mlir::Value element = mlir::vector::ExtractOp::create(
        rewriter, op->getLoc(), element_vector, 0);
    elements.push_back(element);
  }

  rewriter.replaceOpWithNewOp<mlir::vector::FromElementsOp>(op, vector_type,
                                                            elements);
  return mlir::success();
}

mlir::LogicalResult UnrollTransferWrite(mlir::vector::TransferWriteOp op,
                                        mlir::PatternRewriter& rewriter) {
  if (auto status = CheckCanUnroll(op, rewriter); mlir::failed(status)) {
    return status;
  }

  auto num_elements = op.getVectorType().getNumElements();

  auto in_bounds_attr = GetBoundsAttr(rewriter);
  for (int64_t idx = 0; idx != num_elements; ++idx) {
    mlir::Value element =
        Get1DVectorElement(rewriter, op->getLoc(), op.getValueToStore(), idx);
    mlir::Value mask = op.getMask() ? Get1DVectorElement(rewriter, op->getLoc(),
                                                         op.getMask(), idx)
                                    : nullptr;

    llvm::SmallVector<mlir::Value> offsets = op.getIndices();
    offsets.back() = mlir::arith::AddIOp::create(
        rewriter, op->getLoc(), offsets.back(),
        mlir::arith::ConstantIndexOp::create(rewriter, op->getLoc(), idx));

    mlir::vector::TransferWriteOp::create(
        rewriter, op->getLoc(), mlir::TypeRange{}, element, op.getBase(),
        offsets, op.getPermutationMapAttr(), mask, in_bounds_attr);
  }

  rewriter.eraseOp(op);
  return mlir::success();
}

class UnpackSubByteVectorWritePass
    : public impl::UnpackSubByteVectorWritePassBase<
          UnpackSubByteVectorWritePass> {
 public:
  using UnpackSubByteVectorWritePassBase::UnpackSubByteVectorWritePassBase;

  void runOnOperation() override {
    mlir::MLIRContext* context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add(UnrollTransferRead);
    patterns.add(UnrollTransferWrite);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateUnpackSubByteVectorWritePass() {
  return std::make_unique<UnpackSubByteVectorWritePass>();
}

}  // namespace xla::cpu
