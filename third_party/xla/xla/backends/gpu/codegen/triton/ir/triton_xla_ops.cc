/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"

#include <cassert>
#include <cstdint>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"  // IWYU pragma: keep
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"  // IWYU pragma: keep
#include "mlir/IR/MLIRContext.h"  // IWYU pragma: keep
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"  // IWYU pragma: keep
#include "mlir/IR/TypeUtilities.h"  // IWYU pragma: keep
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_dialect.cc.inc"

using mlir::LogicalResult;
using mlir::Type;

namespace mlir::triton::xla {
//===----------------------------------------------------------------------===//
// ExtractOp
//===----------------------------------------------------------------------===//

void ExtractOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "extracted_tile");
}

LogicalResult ExtractOp::verify() {
  int64_t rank = getResultType().getRank();
  if (rank == 0) {
    return emitError("cannot extract a 0-d tensor");
  }
  if (rank != getLayout().size()) {
    return emitError("layout attribute has a wrong size");
  }
  return success();
}

void ExtractOp::build(OpBuilder &b, OperationState &result,
                      RankedTensorType result_type, Value src,
                      ArrayRef<OpFoldResult> offsets,
                      ArrayRef<OpFoldResult> strides, ArrayRef<int64_t> layout,
                      ArrayRef<NamedAttribute> attrs) {
  SmallVector<int64_t> static_offsets, static_sizes, static_strides;
  SmallVector<Value> dynamic_offsets, dynamic_sizes, dynamic_strides;
  dispatchIndexOpFoldResults(offsets, dynamic_offsets, static_offsets);
  dispatchIndexOpFoldResults(strides, dynamic_strides, static_strides);
  result.addAttribute(InsertOp::getLayoutAttrName(OperationName(
                          InsertOp::getOperationName(), b.getContext())),
                      b.getDenseI64ArrayAttr(layout));
  result.addAttributes(attrs);
  build(b, result, result_type, src, dynamic_offsets, {}, dynamic_strides,
        b.getDenseI64ArrayAttr(static_offsets),
        b.getDenseI64ArrayAttr(result_type.getShape()),
        b.getDenseI64ArrayAttr(static_strides), {});
}

void ExtractOp::build(OpBuilder &b, OperationState &result,
                      RankedTensorType result_type, Value src,
                      ValueRange offsets, ValueRange strides,
                      ArrayRef<int64_t> layout,
                      ArrayRef<NamedAttribute> attrs) {
  build(b, result, result_type, src, getAsOpFoldResult(offsets),
        getAsOpFoldResult(strides), layout, attrs);
}

class ExtractOpOffsetsSizesStridesFolder final
    : public OpRewritePattern<ExtractOp> {
 public:
  using OpRewritePattern<ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<OpFoldResult> mixed_offsets(op.getMixedOffsets());
    SmallVector<OpFoldResult> mixed_strides(op.getMixedStrides());

    // No constant operands were folded, just return;
    if (failed(foldDynamicIndexList(mixed_offsets, /*onlyNonNegative=*/true)) &&
        failed(foldDynamicIndexList(mixed_strides))) {
      return failure();
    }
    // Create the new op in canonical form.
    rewriter.replaceOpWithNewOp<ExtractOp>(
        op, op.getResultType(), op.getSrc(), mixed_offsets, mixed_strides,
        op.getLayout(), llvm::to_vector(op->getDiscardableAttrs()));
    return success();
  }
};

void ExtractOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<ExtractOpOffsetsSizesStridesFolder>(context);
}

//===----------------------------------------------------------------------===//
// InsertOp
//===----------------------------------------------------------------------===//

void InsertOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "inserted_tile");
}

LogicalResult InsertOp::verify() {
  int64_t rank = getSrcType().getRank();
  if (rank == 0) {
    return emitError("cannot insert a 0-d tensor");
  }
  if (rank != getLayout().size()) {
    return emitError("layout attribute has a wrong size");
  }
  return success();
}

void InsertOp::build(OpBuilder &b, OperationState &result, Value src, Value dst,
                     ArrayRef<OpFoldResult> offsets,
                     ArrayRef<OpFoldResult> strides, ArrayRef<int64_t> layout,
                     ArrayRef<NamedAttribute> attrs) {
  RankedTensorType src_type = mlir::cast<RankedTensorType>(src.getType());
  RankedTensorType dst_type = mlir::cast<RankedTensorType>(dst.getType());
  SmallVector<int64_t> static_offsets, static_sizes, static_strides;
  SmallVector<Value> dynamic_offsets, dynamic_sizes, dynamic_strides;
  dispatchIndexOpFoldResults(offsets, dynamic_offsets, static_offsets);
  dispatchIndexOpFoldResults(strides, dynamic_strides, static_strides);
  result.addAttribute(InsertOp::getLayoutAttrName(OperationName(
                          InsertOp::getOperationName(), b.getContext())),
                      b.getDenseI64ArrayAttr(layout));
  result.addAttributes(attrs);
  build(b, result, dst_type, src, dst, dynamic_offsets, {}, dynamic_strides,
        b.getDenseI64ArrayAttr(static_offsets),
        b.getDenseI64ArrayAttr(src_type.getShape()),
        b.getDenseI64ArrayAttr(static_strides), {});
}

void InsertOp::build(OpBuilder &b, OperationState &result, Value src, Value dst,
                     ValueRange offsets, ValueRange strides,
                     ArrayRef<int64_t> layout, ArrayRef<NamedAttribute> attrs) {
  build(b, result, src, dst, getAsOpFoldResult(offsets),
        getAsOpFoldResult(strides), layout, attrs);
}

class InsertOpOffsetsSizesStridesFolder final
    : public OpRewritePattern<InsertOp> {
 public:
  using OpRewritePattern<InsertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<OpFoldResult> mixed_offsets(op.getMixedOffsets());
    SmallVector<OpFoldResult> mixed_strides(op.getMixedStrides());
    // No constant operands were folded, just return;
    if (failed(foldDynamicIndexList(mixed_offsets, /*onlyNonNegative=*/true)) &&
        failed(foldDynamicIndexList(mixed_strides))) {
      return failure();
    }
    // Create the new op in canonical form.
    rewriter.replaceOpWithNewOp<InsertOp>(
        op, op.getSrc(), op.getDst(), mixed_offsets, mixed_strides,
        op.getLayout(), llvm::to_vector(op->getDiscardableAttrs()));
    return success();
  }
};

void InsertOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<InsertOpOffsetsSizesStridesFolder>(context);
}

}  // namespace mlir::triton::xla

#define GET_OP_CLASSES
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.cc.inc"
