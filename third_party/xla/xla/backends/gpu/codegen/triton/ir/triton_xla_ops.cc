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

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"  // IWYU pragma: keep
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"  // IWYU pragma: keep
#include "mlir/IR/MLIRContext.h"  // IWYU pragma: keep
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"  // IWYU pragma: keep
#include "mlir/IR/TypeUtilities.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_dialect.cc.inc"
#include "xla/codegen/xtile/ir/xtile_attrs.h"  // IWYU pragma: keep

using mlir::LogicalResult;
using mlir::Type;

namespace mlir::triton::xla {

using ::xla::xtile::LayoutAttr;

// Parser hook for triton_xla.extract/insert ops assembly format.
ParseResult parseAsMemRefType(OpAsmParser& parser, Type& type,
                              DenseI64ArrayAttr& shape,
                              DenseI64ArrayAttr& order) {
  MemRefType memref_type;
  if (parser.parseCustomTypeWithFallback(memref_type)) {
    return failure();
  };

  int address_space = 1;
  if (auto attr = dyn_cast_or_null<IntegerAttr>(memref_type.getMemorySpace())) {
    address_space = attr.getInt();
  }
  type = PointerType::get(memref_type.getElementType(), address_space);
  shape = DenseI64ArrayAttr::get(parser.getContext(), memref_type.getShape());

  LayoutAttr layout = dyn_cast<LayoutAttr>(memref_type.getLayout());
  if (!layout) {
    parser.emitError(parser.getCurrentLocation())
        << "expected layout attribute";
    return failure();
  }
  order = layout.getMinorToMajor();

  return success();
}

// Printer hook for triton_xla.extract/insert ops assembly format.
void printAsMemRefType(OpAsmPrinter& printer, Operation* op, PointerType type,
                       DenseI64ArrayAttr shape, DenseI64ArrayAttr order) {
  auto layout = LayoutAttr::get(
      op->getContext(), DenseI64ArrayAttr::get(op->getContext(), order));
  Attribute memory_space;
  if (int addr_space = type.getAddressSpace(); addr_space != 1) {
    memory_space = Builder(op).getI32IntegerAttr(addr_space);
  }
  printer << MemRefType::get(shape, type.getPointeeType(), layout,
                             memory_space);
}

static LogicalResult produceSliceErrorMsg(SliceVerificationResult result,
                                          Operation* op,
                                          RankedTensorType expected_type) {
  switch (result) {
    case SliceVerificationResult::Success:
      return success();
    case SliceVerificationResult::RankTooLarge:
      return op->emitError("expected rank to be smaller or equal to ")
             << "the other rank. ";
    case SliceVerificationResult::SizeMismatch:
      return op->emitError("expected type to be ")
             << expected_type << " or a rank-reduced version. (size mismatch) ";
    case SliceVerificationResult::ElemTypeMismatch:
      return op->emitError("expected element type to be ")
             << expected_type.getElementType();
    default:
      llvm_unreachable("unexpected extract_slice op verification result");
  }
}

static LogicalResult verifyExtractInsert(
    Operation* op, RankedTensorType tensor_type, PointerType pointer_type,
    DenseI64ArrayAttr layout, ArrayRef<int64_t> shape, ArrayRef<int64_t> sizes,
    ArrayRef<int64_t> strides) {
  if (tensor_type.getRank() == 0) {
    return op->emitError("unsupported 0-d tensor");
  }
  if (ShapedType::isDynamicShape(sizes)) {
    return op->emitError("dynamic sizes are not supported");
  }
  if (ShapedType::isDynamicShape(strides)) {
    return op->emitError("dynamic strides are not supported");
  }
  if (failed(LayoutAttr::get(op->getContext(), layout).verifyLayout(shape, [&] {
        return op->emitError();
      }))) {
    return failure();
  }
  auto expected_type =
      RankedTensorType::get(sizes, pointer_type.getPointeeType());
  SliceVerificationResult result =
      isRankReducedType(expected_type, tensor_type);
  if (result != SliceVerificationResult::Success) {
    return produceSliceErrorMsg(result, op, expected_type);
  }
  // Note: other than tensor.extract/insert, offsets, sizes, strides may run
  // out-of-bounds with respect to the source/destination.
  return success();
}

//===----------------------------------------------------------------------===//
// ExtractOp
//===----------------------------------------------------------------------===//

void ExtractOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "extracted_tile");
}

LogicalResult ExtractOp::verify() {
  return verifyExtractInsert(getOperation(), getType(), getSrc().getType(),
                             getSrcLayoutAttr(), getSrcShape(),
                             getStaticSizes(), getStaticStrides());
}

void ExtractOp::build(OpBuilder& b, OperationState& result,
                      RankedTensorType result_type, Value src,
                      ArrayRef<OpFoldResult> offsets, ArrayRef<int64_t> sizes,
                      ArrayRef<int64_t> strides, ArrayRef<int64_t> src_shape,
                      ArrayRef<int64_t> src_layout) {
  SmallVector<int64_t> static_offsets;
  SmallVector<Value> dynamic_offsets;
  dispatchIndexOpFoldResults(offsets, dynamic_offsets, static_offsets);
  build(b, result, result_type, src, dynamic_offsets, /*sizes=*/{},
        /*strides=*/{}, b.getDenseI64ArrayAttr(static_offsets),
        b.getDenseI64ArrayAttr(sizes), b.getDenseI64ArrayAttr(strides),
        b.getDenseI64ArrayAttr(src_shape), b.getDenseI64ArrayAttr(src_layout));
}

void ExtractOp::build(OpBuilder& b, OperationState& result,
                      RankedTensorType result_type, Value src,
                      ValueRange offsets, ArrayRef<int64_t> sizes,
                      ArrayRef<int64_t> strides, ArrayRef<int64_t> shape,
                      ArrayRef<int64_t> layout) {
  build(b, result, result_type, src, getAsOpFoldResult(offsets), sizes, strides,
        shape, layout);
}

class ExtractOpOffsetsSizesStridesFolder final
    : public OpRewritePattern<ExtractOp> {
 public:
  using OpRewritePattern<ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<OpFoldResult> mixed_offsets(op.getMixedOffsets());
    if (failed(foldDynamicIndexList(mixed_offsets, /*onlyNonNegative=*/true))) {
      // No constant operands were folded, just return;
      return failure();
    }
    // Create the new op in canonical form.
    auto disable_attrs = to_vector(op->getDiscardableAttrs());
    auto new_op = rewriter.replaceOpWithNewOp<ExtractOp>(
        op, op.getType(), op.getSrc(), mixed_offsets, op.getStaticSizes(),
        op.getStaticStrides(), op.getSrcShape(), op.getSrcLayout());
    new_op->setDiscardableAttrs(disable_attrs);
    return success();
  }
};

void ExtractOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                            MLIRContext* context) {
  results.add<ExtractOpOffsetsSizesStridesFolder>(context);
}

//===----------------------------------------------------------------------===//
// InsertOp
//===----------------------------------------------------------------------===//

LogicalResult InsertOp::verify() {
  return verifyExtractInsert(
      getOperation(), getSrc().getType(), getDst().getType(),
      getDstLayoutAttr(), getDstShape(), getStaticSizes(), getStaticStrides());
}

void InsertOp::build(OpBuilder& b, OperationState& result, Value src, Value dst,
                     ArrayRef<OpFoldResult> offsets, ArrayRef<int64_t> sizes,
                     ArrayRef<int64_t> strides, ArrayRef<int64_t> dst_shape,
                     ArrayRef<int64_t> dst_layout) {
  SmallVector<int64_t> static_offsets;
  SmallVector<Value> dynamic_offsets;
  dispatchIndexOpFoldResults(offsets, dynamic_offsets, static_offsets);
  build(b, result, /*resultTypes=*/{}, src, dst, dynamic_offsets, /*sizes=*/{},
        /*strides=*/{}, b.getDenseI64ArrayAttr(static_offsets),
        b.getDenseI64ArrayAttr(sizes), b.getDenseI64ArrayAttr(strides),
        b.getDenseI64ArrayAttr(dst_shape), b.getDenseI64ArrayAttr(dst_layout));
}

void InsertOp::build(OpBuilder& b, OperationState& result, Value src, Value dst,
                     ValueRange offsets, ArrayRef<int64_t> sizes,
                     ArrayRef<int64_t> strides, ArrayRef<int64_t> shape,
                     ArrayRef<int64_t> layout) {
  build(b, result, src, dst, getAsOpFoldResult(offsets), sizes, strides, shape,
        layout);
}

class InsertOpOffsetsSizesStridesFolder final
    : public OpRewritePattern<InsertOp> {
 public:
  using OpRewritePattern<InsertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<OpFoldResult> mixed_offsets(op.getMixedOffsets());
    // No constant operands were folded, just return;
    if (failed(foldDynamicIndexList(mixed_offsets, /*onlyNonNegative=*/true))) {
      return failure();
    }
    // Create the new op in canonical form.
    auto disable_attrs = to_vector(op->getDiscardableAttrs());
    auto new_op = rewriter.replaceOpWithNewOp<InsertOp>(
        op, op.getSrc(), op.getDst(), mixed_offsets, op.getStaticSizes(),
        op.getStaticStrides(), op.getDstShape(), op.getDstLayout());
    new_op->setDiscardableAttrs(disable_attrs);
    return success();
  }
};

void InsertOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                           MLIRContext* context) {
  results.add<InsertOpOffsetsSizesStridesFolder>(context);
}

OpFoldResult MemrefToPtrOp::fold(FoldAdaptor adaptor) {
  if (auto ptr_to_memref = getOperand().getDefiningOp<PtrToMemrefOp>()) {
    // memref_to_ptr(ptr_to_memref(x)) -> x
    return ptr_to_memref.getOperand();
  }

  return {};
}

LogicalResult MemrefToPtrOp::verify() {
  mlir::MemRefType src_type = getSrc().getType();
  if (src_type.getElementType() != getType().getPointeeType()) {
    getOperation()->emitError(
        "source element type does not match result pointee type");
    return failure();
  }

  // It is only safe to directly convert a pointer to a memref if the memref
  // has no offset.
  llvm::SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (src_type.getStridesAndOffset(strides, offset).failed()) {
    getOperation()->emitError("failed to get strides and offset") << src_type;
    return failure();
  }
  if (offset != 0) {
    getOperation()->emitError("memref has non-zero offset");
    return failure();
  }

  return success();
}

LogicalResult PtrToMemrefOp::verify() {
  mlir::MemRefType result_type = getType();
  if (getSrc().getType().getPointeeType() != result_type.getElementType()) {
    getOperation()->emitError(
        "source pointee type does not match result element type");
    return failure();
  }

  // It is only safe to directly convert a pointer to a memref if the memref
  // has no offset.
  llvm::SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (result_type.getStridesAndOffset(strides, offset).failed()) {
    getOperation()->emitError("failed to get strides and offset")
        << result_type;
    return failure();
  }
  if (offset != 0) {
    getOperation()->emitError("memref has non-zero offset");
    return failure();
  }

  return success();
}

}  // namespace mlir::triton::xla

#define GET_OP_CLASSES
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.cc.inc"
