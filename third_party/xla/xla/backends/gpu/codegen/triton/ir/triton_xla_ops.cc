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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/Builders.h"  // IWYU pragma: keep
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"  // IWYU pragma: keep
#include "mlir/IR/MLIRContext.h"  // IWYU pragma: keep
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"  // IWYU pragma: keep
#include "mlir/IR/TypeUtilities.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_dialect.cc.inc"
#include "triton/Dialect/TritonGPU/IR/Types.h"

using mlir::LogicalResult;
using mlir::RankedTensorType;
using mlir::Type;

namespace mlir::triton::xla {

//===----------------------------------------------------------------------===//
// TileOp
//===----------------------------------------------------------------------===//

void TileOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "tiled_tensor");
}

template <typename DenseIntArrayAttrType>
mlir::ParseResult parseDenseIntArrayAttr(mlir::AsmParser& parser,
                                         DenseIntArrayAttrType& array) {
  array = mlir::dyn_cast_or_null<DenseIntArrayAttrType>(
      DenseIntArrayAttrType::parse(parser, mlir::Type{}));
  if (!array) return mlir::failure();
  return mlir::success();
}

ParseResult TileOp::parse(OpAsmParser& parser, OperationState& result) {
  OpAsmParser::UnresolvedOperand src;
  TiledTensorType tiled_tensor_type;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> offsets, sizes, strides;
  if (parser.parseOperand(src) ||
      parser.parseOperandList(offsets, OpAsmParser::Delimiter::Square) ||
      parser.parseOperandList(sizes, OpAsmParser::Delimiter::Square) ||
      parser.parseOperandList(strides, OpAsmParser::Delimiter::Square) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(tiled_tensor_type)) {
    return failure();
  }
  auto offset_type = parser.getBuilder().getI32Type();
  auto size_and_stride_type = parser.getBuilder().getI64Type();
  if (parser.resolveOperand(src, tiled_tensor_type.getOriginalType(),
                            result.operands) ||
      parser.resolveOperands(offsets, offset_type, result.operands) ||
      parser.resolveOperands(sizes, size_and_stride_type, result.operands) ||
      parser.resolveOperands(strides, size_and_stride_type, result.operands)) {
    return failure();
  }
  result.addTypes(tiled_tensor_type);
  return success();
}

void TileOp::print(OpAsmPrinter& p) {
  p << ' ' << getTensor();
  p << '[';
  llvm::interleaveComma(getOffsets(), p);
  p << "][";
  llvm::interleaveComma(getSizes(), p);
  p << "][";
  llvm::interleaveComma(getStrides(), p);
  p << "] : " << getType();
}

LogicalResult TileOp::verify() {
  if (getTensor().getType().getRank() == 0) {
    return emitError("cannot tile a 0-d tensor");
  }
  auto tensor_rank = getTensor().getType().getRank();
  if (tensor_rank != getOffsets().size() || tensor_rank != getSizes().size() ||
      tensor_rank != getStrides().size())
    return emitError(
        "mismatch between tensor rank and one or more of "
        "offsets/sizes/strides");
  return success();
}

//===----------------------------------------------------------------------===//
// ExtractOp
//===----------------------------------------------------------------------===//

void ExtractOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "extracted_tile");
}

ParseResult ExtractOp::parse(OpAsmParser& parser, OperationState& result) {
  Builder& builder = parser.getBuilder();

  OpAsmParser::UnresolvedOperand tiled_tensor;
  Type tile_type, original_type;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> offsets;
  if (parser.parseOperand(tiled_tensor) ||
      parser.parseOperandList(offsets, OpAsmParser::Delimiter::Square) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(original_type) || parser.parseKeyword("to") ||
      parser.parseType(tile_type)) {
    return failure();
  }
  auto tiled_tensor_type = TiledTensorType::get(
      parser.getContext(), mlir::cast<RankedTensorType>(tile_type),
      mlir::cast<RankedTensorType>(original_type));
  auto offset_type = builder.getI32Type();
  if (parser.resolveOperand(tiled_tensor, tiled_tensor_type, result.operands) ||
      parser.resolveOperands(offsets, offset_type, result.operands)) {
    return failure();
  }
  result.addTypes(tile_type);
  return success();
}

void ExtractOp::print(OpAsmPrinter& p) {
  TiledTensorType tiled_type = getSrc().getType();
  p << ' ' << getSrc() << '[';
  llvm::interleaveComma(getOffsets(), p);
  p << ']';
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << tiled_type.getOriginalType() << " to "
    << tiled_type.getTileType();
}

LogicalResult ExtractOp::verify() {
  if (getResult().getType().getRank() == 0) {
    return emitError("cannot extract a 0-d tensor");
  }
  if (getSrc().getType().getRank() != getOffsets().size())
    return emitError("source tensor rank does not match number of offsets");
  return success();
}

//===----------------------------------------------------------------------===//
// InsertOp
//===----------------------------------------------------------------------===//

void InsertOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "inserted_tile");
}

ParseResult InsertOp::parse(OpAsmParser& parser, OperationState& result) {
  Builder& builder = parser.getBuilder();

  OpAsmParser::UnresolvedOperand tile, tiled_tensor;
  Type tile_type, original_type;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> offsets;
  if (parser.parseOperand(tile) || parser.parseKeyword("into") ||
      parser.parseOperand(tiled_tensor) ||
      parser.parseOperandList(offsets, OpAsmParser::Delimiter::Square) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(tile_type) || parser.parseKeyword("into") ||
      parser.parseType(original_type) ||
      parser.resolveOperand(tile, tile_type, result.operands)) {
    return failure();
  }
  auto tiled_tensor_type = TiledTensorType::get(
      parser.getContext(), mlir::cast<RankedTensorType>(tile_type),
      mlir::cast<RankedTensorType>(original_type));

  auto offset_type = builder.getI32Type();
  if (parser.resolveOperand(tiled_tensor, tiled_tensor_type, result.operands) ||
      parser.resolveOperands(offsets, offset_type, result.operands)) {
    return failure();
  }
  result.addTypes(original_type);
  return success();
}

void InsertOp::print(OpAsmPrinter& p) {
  TiledTensorType tiled_type = getDst().getType();
  p << ' ' << getSrc() << " into " << getDst() << "[";
  llvm::interleaveComma(getOffsets(), p);
  p << ']';
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << tiled_type.getTileType() << " into "
    << tiled_type.getOriginalType();
}

LogicalResult InsertOp::verify() {
  if (getSrc().getType().getRank() == 0) {
    return emitError("cannot insert a 0-d tensor");
  }
  if (getDst().getType().getRank() != getOffsets().size())
    return emitError(
        "destination tensor rank does not match number of offsets");
  return success();
}

}  // namespace mlir::triton::xla

#define GET_OP_CLASSES
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.cc.inc"
