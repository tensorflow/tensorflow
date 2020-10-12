/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This file defines the operations used in the LMHLO dialect.

#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h.inc"
#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops_structs.cc.inc"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace lmhlo {

LmhloDialect::LmhloDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<LmhloDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.cc.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ConstOp.
//===----------------------------------------------------------------------===//

/// An lho.constant on an memref that is locally allocated and with no other
/// users (other than dealloc's) can be erased.
// TODO: This can be generalized to an arbitrary op by making use of memory
// effects (write memory effect).
struct EraseConstOp : public OpRewritePattern<ConstOp> {
  using OpRewritePattern<ConstOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConstOp op,
                                PatternRewriter& rewriter) const override {
    Value memref = op.output();
    if (!memref.getDefiningOp<AllocOp>()) {
      return failure();
    }

    // Check that all uses of the memref are either DeallocOps or this op.
    for (Operation* user : memref.getUsers())
      if (user != op && !isa<DeallocOp>(user)) return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

void ConstOp::getCanonicalizationPatterns(OwningRewritePatternList& results,
                                          MLIRContext* context) {
  results.insert<EraseConstOp>(context);
}

//===----------------------------------------------------------------------===//
// StaticMemRefCastOp
//===----------------------------------------------------------------------===//

Value StaticMemRefCastOp::getViewSource() { return *getODSOperands(0).begin(); }

static LogicalResult Verify(StaticMemRefCastOp op) {
  if (!op.operand().getType().cast<ShapedType>().hasStaticShape())
    return op.emitOpError("operand must have static shape");
  if (!op.getType().hasStaticShape())
    return op.emitOpError("result must have static shape");
  return success();
}

//===----------------------------------------------------------------------===//
// DynamicMemRefCastOp
//===----------------------------------------------------------------------===//

Value DynamicMemRefCastOp::getViewSource() {
  return *getODSOperands(0).begin();
}

static LogicalResult Verify(DynamicMemRefCastOp op) {
  // Check if `sizes` and `strides` args are compatible with the result type.
  if (op.sizes().size() != op.getType().getRank())
    return op.emitOpError(
        "`sizes` args count must be equal to the rank of the output memref");
  return success();
}

//===----------------------------------------------------------------------===//
// ReshapeMemrefCastOp
//===----------------------------------------------------------------------===//

Value ReshapeMemRefCastOp::getViewSource() { return operand(); }

static LogicalResult Verify(ReshapeMemRefCastOp op) {
  Type operandType = op.operand().getType();
  Type resultType = op.result().getType();

  Type operandElementType = operandType.cast<ShapedType>().getElementType();
  Type resultElementType = resultType.cast<ShapedType>().getElementType();
  if (operandElementType != resultElementType)
    return op.emitOpError(
        "element types of source and destination memref "
        "types should be the same");

  if (auto operandMemRefType = operandType.dyn_cast<MemRefType>())
    if (!operandMemRefType.getAffineMaps().empty())
      return op.emitOpError(
          "operand memref type should have identity affine map");

  int64_t shapeSize = op.shape().getType().cast<MemRefType>().getDimSize(0);
  auto resultMemRefType = resultType.dyn_cast<MemRefType>();
  if (resultMemRefType) {
    if (shapeSize == ShapedType::kDynamicSize)
      return op.emitOpError(
          "cannot use shape operand with dynamic length to "
          "cast statically-ranked memref type");
    if (shapeSize != resultMemRefType.getRank())
      return op.emitOpError(
          "length of shape operand differs from the result's memref rank");

    if (!resultMemRefType.getAffineMaps().empty())
      return op.emitOpError(
          "result memref type should have identity affine map");
  }
  return success();
}

}  // namespace lmhlo
}  // namespace mlir

#define GET_OP_CLASSES
#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.cc.inc"

namespace mlir {
namespace lmhlo {

// TODO(cheshire): Support folding, reuse code from hlo_ops.cc.

void FusionOp::build(OpBuilder &builder, OperationState &result,
                     ArrayRef<NamedAttribute> attributes) {
  result.addAttributes(attributes);
  Region *bodyRegion = result.addRegion();
  FusionOp::ensureTerminator(*bodyRegion, builder, result.location);
}

}  // namespace lmhlo
}  // namespace mlir
