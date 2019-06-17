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

using namespace mlir;

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
static ParseResult printNoIOOp(Operation *op, OpAsmPrinter *printer) {
  *printer << op->getName();
  printer->printOptionalAttrDict(op->getAttrs(),
                                 /*elidedAttrs=*/{});
  return success();
}

// Verifies that the given op can only be placed in a `spv.module`.
static LogicalResult verifyModuleOnly(Operation *op) {
  if (!llvm::isa_and_nonnull<spirv::ModuleOp>(op->getParentOp()))
    return op->emitOpError("can only be used in a 'spv.module' block");

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

static ParseResult printModule(spirv::ModuleOp moduleOp,
                               OpAsmPrinter *printer) {
  auto *op = moduleOp.getOperation();
  *printer << spirv::ModuleOp::getOperationName();
  printer->printRegion(op->getRegion(0), /*printEntryBlockArgs=*/false,
                       /*printBlockTerminators=*/false);
  *printer << " attributes";
  printer->printOptionalAttrDict(op->getAttrs(),
                                 /*elidedAttrs=*/{});
  return success();
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
