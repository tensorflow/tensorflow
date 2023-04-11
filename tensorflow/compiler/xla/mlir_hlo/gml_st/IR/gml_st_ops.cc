/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "gml_st/IR/gml_st_ops.h"

#include <utility>

#include "llvm/ADT/SetVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/RegionUtils.h"

// Generated dialect definitions.
#include "gml_st/IR/gml_st_dialect.cc.inc"

namespace mlir {
namespace gml_st {
namespace {

//===----------------------------------------------------------------------===//
// GmlSt Dialect Interfaces
//===----------------------------------------------------------------------===//

struct GmlStInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  // Operations in GmlSt dialect are always legal to inline since they are
  // pure.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
  // Handle the given inlined terminator by replacing it with a new operation
  // as necessary. Required when the region has only one block.
  void handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToRepl) const final {
    auto yieldOp = dyn_cast<gml_st::YieldOp>(op);
    if (!yieldOp) return;

    for (auto [valueToRepl, operand] :
         llvm::zip(valuesToRepl, yieldOp.getOperands())) {
      valueToRepl.replaceAllUsesWith(operand);
    }
  }
};
}  // namespace

//===----------------------------------------------------------------------===//
// GmlStDialect
//===----------------------------------------------------------------------===//

void GmlStDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "gml_st/IR/gml_st_ops.cc.inc"
      >();
  addInterfaces<GmlStInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// FusionOp
//===----------------------------------------------------------------------===//

YieldOp FusionOp::getTerminator() {
  return cast<YieldOp>(getBody()->getTerminator());
}

void FusionOp::print(OpAsmPrinter &p) {
  p << " (";
  llvm::interleaveComma(
      llvm::zip(getBody()->getArguments(), getInputs()), p, [&](auto it) {
        Value inputRegionArg, input;
        std::tie(inputRegionArg, input) = it;
        p << inputRegionArg << " = " << input << ": " << input.getType();
      });
  p << ") ";

  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false);

  p.printOptionalAttrDict(getOperation()->getAttrs());

  if (!getResultTypes().empty()) {
    p << " : ";
    llvm::interleave(getResultTypes(), p, ", ");
  }
}

ParseResult FusionOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands, regionOperands;
  SmallVector<Type, 4> operandTypes;

  auto parseElt = [&]() -> ParseResult {
    if (parser.parseOperand(regionOperands.emplace_back(),
                            /*allowResultNumber=*/false) ||
        parser.parseEqual()) {
      return failure();
    }
    if (parser.parseOperand(operands.emplace_back()) || parser.parseColon() ||
        parser.parseType(operandTypes.emplace_back())) {
      return failure();
    }
    return success();
  };

  // Parse argument list.
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Paren, parseElt))
    return failure();

  SMLoc loc = parser.getCurrentLocation();
  if (parser.resolveOperands(operands, operandTypes, loc, result.operands))
    return failure();

  // Parse region.
  SmallVector<OpAsmParser::Argument, 4> regionArgs;
  for (auto argAndType : llvm::zip(regionOperands, operandTypes)) {
    auto &arg = regionArgs.emplace_back();
    std::tie(arg.ssaName, arg.type) = argAndType;
  }
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs)) return failure();

  // Parse attributes.
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();

  // Parser result types.
  if (parser.parseOptionalColonTypeList(result.types)) return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

LogicalResult YieldOp::verify() { return success(); }

}  // namespace gml_st
}  // namespace mlir

// Generated op classes.
#define GET_OP_CLASSES
#include "gml_st/IR/gml_st_ops.cc.inc"
