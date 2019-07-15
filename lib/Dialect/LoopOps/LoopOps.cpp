//===- Ops.cpp - Loop MLIR Operations -------------------------------------===//
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

#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/StandardOps/Ops.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Support/STLExtras.h"

using namespace mlir;
using namespace mlir::loop;

//===----------------------------------------------------------------------===//
// LoopOpsDialect
//===----------------------------------------------------------------------===//

LoopOpsDialect::LoopOpsDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/LoopOps/LoopOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

// Check that if a "block" is not empty, it has a `TerminatorOp` terminator.
static LogicalResult checkHasTerminator(OpState &op, Block &block) {
  if (block.empty() || isa<TerminatorOp>(block.back()))
    return success();

  return op.emitOpError("expects regions to end with '" +
                        TerminatorOp::getOperationName() + "'")
             .attachNote()
         << "in custom textual format, the absence of terminator implies '"
         << TerminatorOp::getOperationName() << "'";
}

void mlir::loop::ensureLoopTerminator(Region &region, Builder &builder,
                                      Location loc) {
  impl::ensureRegionTerminator<TerminatorOp>(region, builder, loc);
}

void ForOp::build(Builder *builder, OperationState *result, Value *lb,
                  Value *ub, Value *step) {
  result->addOperands({lb, ub, step});
  Region *bodyRegion = result->addRegion();
  ensureLoopTerminator(*bodyRegion, *builder, result->location);
  bodyRegion->front().addArgument(builder->getIndexType());
}

LogicalResult verify(ForOp op) {
  if (auto cst = dyn_cast_or_null<ConstantIndexOp>(op.step()->getDefiningOp()))
    if (cst.getValue() <= 0)
      return op.emitOpError("constant step operand must be nonnegative");

  // Check that the body defines as single block argument for the induction
  // variable.
  auto *body = op.body();
  if (body->getNumArguments() != 1 ||
      !body->getArgument(0)->getType().isIndex())
    return op.emitOpError("expected body to have a single index argument for "
                          "the induction variable");
  if (failed(checkHasTerminator(op, *body)))
    return failure();
  return success();
}

static void print(OpAsmPrinter *p, ForOp op) {
  *p << op.getOperationName() << " " << *op.getInductionVar() << " = "
     << *op.lowerBound() << " to " << *op.upperBound() << " step "
     << *op.step();
  p->printRegion(op.region(),
                 /*printEntryBlockArgs=*/false,
                 /*printBlockTerminators=*/false);
  p->printOptionalAttrDict(op.getAttrs());
}

static ParseResult parseForOp(OpAsmParser *parser, OperationState *result) {
  auto &builder = parser->getBuilder();
  OpAsmParser::OperandType inductionVariable, lb, ub, step;
  // Parse the induction variable followed by '='.
  if (parser->parseRegionArgument(inductionVariable) || parser->parseEqual())
    return failure();

  // Parse loop bounds.
  Type indexType = builder.getIndexType();
  if (parser->parseOperand(lb) ||
      parser->resolveOperand(lb, indexType, result->operands) ||
      parser->parseKeyword("to") || parser->parseOperand(ub) ||
      parser->resolveOperand(ub, indexType, result->operands) ||
      parser->parseKeyword("step") || parser->parseOperand(step) ||
      parser->resolveOperand(step, indexType, result->operands))
    return failure();

  // Parse the body region.
  Region *body = result->addRegion();
  if (parser->parseRegion(*body, inductionVariable, indexType))
    return failure();

  ensureLoopTerminator(*body, builder, result->location);

  // Parse the optional attribute list.
  if (parser->parseOptionalAttributeDict(result->attributes))
    return failure();

  return success();
}

ForOp mlir::loop::getForInductionVarOwner(Value *val) {
  auto *ivArg = dyn_cast<BlockArgument>(val);
  if (!ivArg)
    return ForOp();
  assert(ivArg->getOwner() && "unlinked block argument");
  auto *containingInst = ivArg->getOwner()->getContainingOp();
  return dyn_cast_or_null<ForOp>(containingInst);
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

void IfOp::build(Builder *builder, OperationState *result, Value *cond,
                 bool withElseRegion) {
  result->addOperands(cond);
  Region *thenRegion = result->addRegion();
  Region *elseRegion = result->addRegion();
  ensureLoopTerminator(*thenRegion, *builder, result->location);
  if (withElseRegion)
    ensureLoopTerminator(*elseRegion, *builder, result->location);
}

static LogicalResult verify(IfOp op) {
  // Verify that the entry of each child region does not have arguments.
  for (auto &region : op.getOperation()->getRegions()) {
    if (region.empty())
      continue;

    // TODO(riverriddle) We currently do not allow multiple blocks in child
    // regions.
    if (std::next(region.begin()) != region.end())
      return op.emitOpError("expected one block per 'then' or 'else' regions");
    if (failed(checkHasTerminator(op, region.front())))
      return failure();

    for (auto &b : region)
      if (b.getNumArguments() != 0)
        return op.emitOpError(
            "requires that child entry blocks have no arguments");
  }
  return success();
}

static ParseResult parseIfOp(OpAsmParser *parser, OperationState *result) {
  // Create the regions for 'then'.
  result->regions.reserve(2);
  Region *thenRegion = result->addRegion();
  Region *elseRegion = result->addRegion();

  auto &builder = parser->getBuilder();
  OpAsmParser::OperandType cond;
  Type i1Type = builder.getIntegerType(1);
  if (parser->parseOperand(cond) ||
      parser->resolveOperand(cond, i1Type, result->operands))
    return failure();

  // Parse the 'then' region.
  if (parser->parseRegion(*thenRegion, {}, {}))
    return failure();
  ensureLoopTerminator(*thenRegion, parser->getBuilder(), result->location);

  // If we find an 'else' keyword then parse the 'else' region.
  if (!parser->parseOptionalKeyword("else")) {
    if (parser->parseRegion(*elseRegion, {}, {}))
      return failure();
    ensureLoopTerminator(*elseRegion, parser->getBuilder(), result->location);
  }

  // Parse the optional attribute list.
  if (parser->parseOptionalAttributeDict(result->attributes))
    return failure();

  return success();
}

static void print(OpAsmPrinter *p, IfOp op) {
  *p << IfOp::getOperationName() << " " << *op.condition();
  p->printRegion(op.thenRegion(),
                 /*printEntryBlockArgs=*/false,
                 /*printBlockTerminators=*/false);

  // Print the 'else' regions if it exists and has a block.
  auto &elseRegion = op.elseRegion();
  if (!elseRegion.empty()) {
    *p << " else";
    p->printRegion(elseRegion,
                   /*printEntryBlockArgs=*/false,
                   /*printBlockTerminators=*/false);
  }

  p->printOptionalAttrDict(op.getAttrs());
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/LoopOps/LoopOps.cpp.inc"
