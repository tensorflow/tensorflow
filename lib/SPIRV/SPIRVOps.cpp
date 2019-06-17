//===- SPIRVOps.cpp - MLIR SPIR-V operations ------------------------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file defines the operations in the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/SPIRV/SPIRVOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/SPIRV/SPIRVTypes.h"

using namespace mlir;

static constexpr const char kValueAttrName[] = "value";

//===----------------------------------------------------------------------===//
// Common utility functions
//===----------------------------------------------------------------------===//

// Parses an op that has no inputs and no outputs.
static ParseResult parseNoIOOp(OpAsmParser *parser, OperationState *state) {
  if (parser->parseOptionalAttributeDict(state->attributes))
    return failure();
  return success();
}

// Prints an op that has no inputs and no outputs.
static void printNoIOOp(Operation *op, OpAsmPrinter *printer) {
  *printer << op->getName();
  printer->printOptionalAttrDict(op->getAttrs());
}

// Verifies that the given op can only be placed in a `spv.module`.
static LogicalResult verifyModuleOnly(Operation *op) {
  if (!llvm::isa_and_nonnull<spirv::ModuleOp>(op->getParentOp()))
    return op->emitOpError("can only be used in a 'spv.module' block");

  return success();
}

//===----------------------------------------------------------------------===//
// spv.constant
//===----------------------------------------------------------------------===//

static ParseResult parseConstant(OpAsmParser *parser, OperationState *state) {
  Attribute value;
  if (parser->parseAttribute(value, kValueAttrName, state->attributes))
    return failure();

  Type type;
  if (value.getType().isa<NoneType>()) {
    if (parser->parseColonType(type))
      return failure();
  } else {
    type = value.getType();
  }

  return parser->addTypeToList(type, state->types);
}

static void printConstant(spirv::ConstantOp constOp, OpAsmPrinter *printer) {
  *printer << spirv::ConstantOp::getOperationName() << " " << constOp.value()
           << " : " << constOp.getType();
}

static LogicalResult verifyConstant(spirv::ConstantOp constOp) {
  auto opType = constOp.getType();
  auto value = constOp.value();
  auto valueType = value.getType();

  // ODS already generates checks to make sure the result type is valid. We just
  // need to additionally check that the value's attribute type is consistent
  // with the result type.
  switch (value.getKind()) {
  case StandardAttributes::Bool:
  case StandardAttributes::Integer:
  case StandardAttributes::Float:
  case StandardAttributes::DenseElements:
  case StandardAttributes::SparseElements: {
    if (valueType != opType)
      return constOp.emitOpError("result type (")
             << opType << ") does not match value type (" << valueType << ")";
    return success();
  } break;
  case StandardAttributes::Array: {
    auto arrayType = opType.dyn_cast<spirv::ArrayType>();
    if (!arrayType)
      return constOp.emitOpError(
          "must have spv.array result type for array value");
    auto elemType = arrayType.getElementType();
    for (auto element : value.cast<ArrayAttr>().getValue()) {
      if (element.getType() != elemType)
        return constOp.emitOpError(
            "has array element that are not of result array element type");
    }
  } break;
  default:
    return constOp.emitOpError("cannot have value of type ") << valueType;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spv.module
//===----------------------------------------------------------------------===//

static void ensureModuleEnd(Region *region, Builder builder, Location loc) {
  if (region->empty())
    region->push_back(new Block);

  Block &block = region->back();
  if (!block.empty() && llvm::isa<spirv::ModuleEndOp>(block.back()))
    return;

  OperationState state(builder.getContext(), loc,
                       spirv::ModuleEndOp::getOperationName());
  spirv::ModuleEndOp::build(&builder, &state);
  block.push_back(Operation::create(state));
}

static ParseResult parseModule(OpAsmParser *parser, OperationState *state) {
  Region *body = state->addRegion();

  if (parser->parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}) ||
      parser->parseKeyword("attributes") ||
      parser->parseOptionalAttributeDict(state->attributes))
    return failure();

  ensureModuleEnd(body, parser->getBuilder(), state->location);

  return success();
}

static void printModule(spirv::ModuleOp moduleOp, OpAsmPrinter *printer) {
  auto *op = moduleOp.getOperation();
  *printer << spirv::ModuleOp::getOperationName();
  printer->printRegion(op->getRegion(0), /*printEntryBlockArgs=*/false,
                       /*printBlockTerminators=*/false);
  *printer << " attributes";
  printer->printOptionalAttrDict(op->getAttrs());
}

static LogicalResult verifyModule(spirv::ModuleOp moduleOp) {
  auto &op = *moduleOp.getOperation();
  auto *dialect = op.getDialect();
  auto &body = op.getRegion(0).front();

  for (auto &op : body) {
    if (op.getDialect() == dialect)
      continue;

    auto funcOp = llvm::dyn_cast<FuncOp>(op);
    if (!funcOp)
      return op.emitError("'spv.module' can only contain func and spv.* ops");

    if (funcOp.isExternal())
      return op.emitError("'spv.module' cannot contain external functions");

    for (auto &block : funcOp)
      for (auto &op : block) {
        if (op.getDialect() == dialect)
          continue;

        if (llvm::isa<FuncOp>(op))
          return op.emitError("'spv.module' cannot contain nested functions");

        return op.emitError(
            "functions in 'spv.module' can only contain spv.* ops");
      }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// spv.Return
//===----------------------------------------------------------------------===//

static LogicalResult verifyReturn(spirv::ReturnOp returnOp) {
  auto funcOp =
      llvm::dyn_cast_or_null<FuncOp>(returnOp.getOperation()->getParentOp());
  if (!funcOp)
    return returnOp.emitOpError("must appear in a 'func' op");

  auto numOutputs = funcOp.getType().getNumResults();
  if (numOutputs != 0)
    return returnOp.emitOpError("cannot be used in functions returning value")
           << (numOutputs > 1 ? "s" : "");

  return success();
}

namespace mlir {
namespace spirv {

#define GET_OP_CLASSES
#include "mlir/SPIRV/SPIRVOps.cpp.inc"

} // namespace spirv
} // namespace mlir
