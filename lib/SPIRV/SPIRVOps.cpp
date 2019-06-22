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

static constexpr const char kBindingAttrName[] = "binding";
static constexpr const char kDescriptorSetAttrName[] = "descriptor_set";
static constexpr const char kStorageClassAttrName[] = "storage_class";
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

static ParseResult parseConstantOp(OpAsmParser *parser, OperationState *state) {
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

static void print(spirv::ConstantOp constOp, OpAsmPrinter *printer) {
  *printer << spirv::ConstantOp::getOperationName() << " " << constOp.value()
           << " : " << constOp.getType();
}

static LogicalResult verify(spirv::ConstantOp constOp) {
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
  impl::ensureRegionTerminator<spirv::ModuleEndOp>(*region, builder, loc);
}

void spirv::ModuleOp::build(Builder *builder, OperationState *state) {
  ensureModuleEnd(state->addRegion(), *builder, state->location);
}

static ParseResult parseModuleOp(OpAsmParser *parser, OperationState *state) {
  Region *body = state->addRegion();

  if (parser->parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}) ||
      parser->parseKeyword("attributes") ||
      parser->parseOptionalAttributeDict(state->attributes))
    return failure();

  ensureModuleEnd(body, parser->getBuilder(), state->location);

  return success();
}

static void print(spirv::ModuleOp moduleOp, OpAsmPrinter *printer) {
  auto *op = moduleOp.getOperation();
  *printer << spirv::ModuleOp::getOperationName();
  printer->printRegion(op->getRegion(0), /*printEntryBlockArgs=*/false,
                       /*printBlockTerminators=*/false);
  *printer << " attributes";
  printer->printOptionalAttrDict(op->getAttrs());
}

static LogicalResult verify(spirv::ModuleOp moduleOp) {
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

//===----------------------------------------------------------------------===//
// spv.Variable
//===----------------------------------------------------------------------===//

static ParseResult parseVariableOp(OpAsmParser *parser, OperationState *state) {
  // Parse optional initializer
  Optional<OpAsmParser::OperandType> initInfo;
  if (succeeded(parser->parseOptionalKeyword("init"))) {
    initInfo = OpAsmParser::OperandType();
    if (parser->parseLParen() || parser->parseOperand(*initInfo) ||
        parser->parseRParen())
      return failure();
  }

  // Parse optional descriptor binding
  Attribute set, binding;
  if (succeeded(parser->parseOptionalKeyword("bind"))) {
    Type i32Type = parser->getBuilder().getIntegerType(32);
    if (parser->parseLParen() ||
        parser->parseAttribute(set, i32Type, kDescriptorSetAttrName,
                               state->attributes) ||
        parser->parseComma() ||
        parser->parseAttribute(binding, i32Type, kBindingAttrName,
                               state->attributes) ||
        parser->parseRParen())
      return failure();
  }

  // Parse other attributes
  if (parser->parseOptionalAttributeDict(state->attributes))
    return failure();

  // Parse result pointer type
  Type type;
  if (parser->parseColon())
    return failure();
  auto loc = parser->getCurrentLocation();
  if (parser->parseType(type))
    return failure();

  auto ptrType = type.dyn_cast<spirv::PointerType>();
  if (!ptrType)
    return parser->emitError(loc, "expected spv.ptr type");
  state->addTypes(ptrType);

  // Resolve the initializer operand
  SmallVector<Value *, 1> init;
  if (initInfo) {
    if (parser->resolveOperand(*initInfo, ptrType.getPointeeType(), init))
      return failure();
    state->addOperands(init);
  }

  // TODO(antiagainst): The enum attribute should be integer backed so we don't
  // have these excessive string conversions.
  auto attr = parser->getBuilder().getStringAttr(ptrType.getStorageClassStr());
  state->addAttribute(kStorageClassAttrName, attr);

  return success();
}

static void print(spirv::VariableOp varOp, OpAsmPrinter *printer) {
  auto *op = varOp.getOperation();
  SmallVector<StringRef, 4> elidedAttrs{kStorageClassAttrName};
  *printer << spirv::VariableOp::getOperationName();

  // Print optional initializer
  if (op->getNumOperands() > 0) {
    *printer << " init(";
    printer->printOperands(varOp.initializer());
    *printer << ")";
  }

  // Print optional descriptor binding
  auto set = varOp.getAttr(kDescriptorSetAttrName);
  auto binding = varOp.getAttr(kBindingAttrName);
  if (set && binding) {
    elidedAttrs.push_back(kDescriptorSetAttrName);
    elidedAttrs.push_back(kBindingAttrName);
    *printer << " bind(" << set << ", " << binding << ")";
  }

  printer->printOptionalAttrDict(op->getAttrs(), elidedAttrs);
  *printer << " : " << varOp.getType();
}

static LogicalResult verify(spirv::VariableOp varOp) {
  // SPIR-V spec: "Storage Class is the Storage Class of the memory holding the
  // object. It cannot be Generic. It must be the same as the Storage Class
  // operand of the Result Type."
  if (varOp.storage_class() == "Generic")
    return varOp.emitOpError("storage class cannot be 'Generic'");

  auto pointerType = varOp.pointer()->getType().cast<spirv::PointerType>();
  if (varOp.storage_class() != pointerType.getStorageClassStr())
    return varOp.emitOpError(
        "storage class must match result pointer's storage class");

  if (varOp.getNumOperands() != 0) {
    // SPIR-V spec: "Initializer must be an <id> from a constant instruction or
    // a global (module scope) OpVariable instruction".
    bool valid = false;
    if (auto *initOp = varOp.getOperand(0)->getDefiningOp()) {
      if (llvm::isa<spirv::ConstantOp>(initOp)) {
        valid = true;
      } else if (llvm::isa<spirv::VariableOp>(initOp)) {
        valid = llvm::isa_and_nonnull<spirv::ModuleOp>(initOp->getParentOp());
      }
    }
    if (!valid)
      return varOp.emitOpError("initializer must be the result of a "
                               "spv.Constant or module-level spv.Variable op");
  }

  return success();
}

namespace mlir {
namespace spirv {

#define GET_OP_CLASSES
#include "mlir/SPIRV/SPIRVOps.cpp.inc"

} // namespace spirv
} // namespace mlir
