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

ParseResult ExtractOp::parse(OpAsmParser& parser, OperationState& result) {
  OpAsmParser::UnresolvedOperand src;
  Type tile_type, original_type;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> offsets, tile_sizes, strides;
  if (parser.parseOperand(src) ||
      parser.parseOperandList(offsets, OpAsmParser::Delimiter::Square) ||
      parser.parseOperandList(strides, OpAsmParser::Delimiter::Square) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(original_type) || parser.parseKeyword("to") ||
      parser.parseType(tile_type)) {
    return failure();
  }

  auto index_type = parser.getBuilder().getIndexType();
  if (parser.resolveOperand(src, original_type, result.operands) ||
      parser.resolveOperands(offsets, index_type, result.operands) ||
      parser.resolveOperands(strides, index_type, result.operands)) {
    return failure();
  }

  result.addTypes(tile_type);
  return success();
}

void ExtractOp::print(OpAsmPrinter& p) {
  p << ' ' << getSrc() << '[';
  llvm::interleaveComma(getOffsets(), p);
  p << "][";
  llvm::interleaveComma(getStrides(), p);
  p << "] {layout = array<i64:";
  llvm::interleaveComma(getLayout(), p);
  p << ">} : " << getSrc().getType() << " to " << getResult().getType();
}

LogicalResult ExtractOp::verify() {
  if (getResult().getType().getRank() == 0) {
    return emitError("cannot extract a 0-d tensor");
  }

  auto tensor_rank = getSrc().getType().getRank();
  auto dst_rank = getResult().getType().getRank();
  if (tensor_rank != dst_rank || tensor_rank != getOffsets().size() ||
      tensor_rank != getStrides().size()) {
    return emitError(
        "ranks of source/destination tensor and offsets/strides do not match");
  }

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
  OpAsmParser::UnresolvedOperand src, dst;
  Type tile_type, original_type;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> offsets, strides;
  if (parser.parseOperand(src) || parser.parseKeyword("into") ||
      parser.parseOperand(dst) ||
      parser.parseOperandList(offsets, OpAsmParser::Delimiter::Square) ||
      parser.parseOperandList(strides, OpAsmParser::Delimiter::Square) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(tile_type) || parser.parseKeyword("into") ||
      parser.parseType(original_type)) {
    return failure();
  }

  auto index_type = parser.getBuilder().getIndexType();
  if (parser.resolveOperand(src, tile_type, result.operands) ||
      parser.resolveOperand(dst, original_type, result.operands) ||
      parser.resolveOperands(offsets, index_type, result.operands) ||
      parser.resolveOperands(strides, index_type, result.operands)) {
    return failure();
  }

  result.addTypes(original_type);
  return success();
}

void InsertOp::print(OpAsmPrinter& p) {
  p << ' ' << getSrc() << " into " << getDst() << "[";
  llvm::interleaveComma(getOffsets(), p);
  p << "][";
  llvm::interleaveComma(getStrides(), p);
  p << "] {layout = array<i64:";
  llvm::interleaveComma(getLayout(), p);
  p << ">} : " << getSrc().getType() << " into " << getDst().getType();
}

LogicalResult InsertOp::verify() {
  if (getSrc().getType().getRank() == 0) {
    return emitError("cannot insert a 0-d tensor");
  }

  auto tensor_rank = getSrc().getType().getRank();
  auto dst_rank = getDst().getType().getRank();
  if (tensor_rank != dst_rank || tensor_rank != getOffsets().size() ||
      tensor_rank != getStrides().size()) {
    return emitError(
        "ranks of source/destination tensor and offsets/strides do not match");
  }

  return success();
}

}  // namespace mlir::triton::xla

#define GET_OP_CLASSES
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.cc.inc"
