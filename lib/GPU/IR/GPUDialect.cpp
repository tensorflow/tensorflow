//===- GPUDialect.cpp - MLIR Dialect for GPU Kernels implementation -------===//
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
// This file implements the GPU kernel-related dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/GPU/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/StandardOps/Ops.h"

using namespace mlir;
using namespace mlir::gpu;

StringRef GPUDialect::getDialectName() { return "gpu"; }

bool GPUDialect::isKernel(FuncOp function) {
  UnitAttr isKernelAttr =
      function.getAttrOfType<UnitAttr>(getKernelFuncAttrName());
  return static_cast<bool>(isKernelAttr);
}

GPUDialect::GPUDialect(MLIRContext *context)
    : Dialect(getDialectName(), context) {
  addOperations<LaunchOp, LaunchFuncOp,
#define GET_OP_LIST
#include "mlir/GPU/GPUOps.cpp.inc"
                >();
}

#define GET_OP_CLASSES
#include "mlir/GPU/GPUOps.cpp.inc"

//===----------------------------------------------------------------------===//
// LaunchOp
//===----------------------------------------------------------------------===//

static SmallVector<Type, 4> getValueTypes(ArrayRef<Value *> values) {
  SmallVector<Type, 4> types;
  types.reserve(values.size());
  for (Value *v : values)
    types.push_back(v->getType());
  return types;
}

void LaunchOp::build(Builder *builder, OperationState *result, Value *gridSizeX,
                     Value *gridSizeY, Value *gridSizeZ, Value *blockSizeX,
                     Value *blockSizeY, Value *blockSizeZ,
                     ArrayRef<Value *> operands) {
  // Add grid and block sizes as op operands, followed by the data operands.
  result->addOperands(
      {gridSizeX, gridSizeY, gridSizeZ, blockSizeX, blockSizeY, blockSizeZ});
  result->addOperands(operands);

  // Create a kernel body region with kNumConfigRegionAttributes + N arguments,
  // where the first kNumConfigRegionAttributes arguments have `index` type and
  // the rest have the same types as the data operands.
  Region *kernelRegion = result->addRegion();
  Block *body = new Block();
  body->addArguments(
      std::vector<Type>(kNumConfigRegionAttributes, builder->getIndexType()));
  body->addArguments(getValueTypes(operands));
  kernelRegion->push_back(body);
}

Region &LaunchOp::getBody() { return getOperation()->getRegion(0); }

KernelDim3 LaunchOp::getBlockIds() {
  assert(!getBody().getBlocks().empty() && "FuncOp body must not be empty.");
  auto args = getBody().getBlocks().front().getArguments();
  return KernelDim3{args[0], args[1], args[2]};
}

KernelDim3 LaunchOp::getThreadIds() {
  assert(!getBody().getBlocks().empty() && "FuncOp body must not be empty.");
  auto args = getBody().getBlocks().front().getArguments();
  return KernelDim3{args[3], args[4], args[5]};
}

KernelDim3 LaunchOp::getGridSize() {
  assert(!getBody().getBlocks().empty() && "FuncOp body must not be empty.");
  auto args = getBody().getBlocks().front().getArguments();
  return KernelDim3{args[6], args[7], args[8]};
}

KernelDim3 LaunchOp::getBlockSize() {
  assert(!getBody().getBlocks().empty() && "FuncOp body must not be empty.");
  auto args = getBody().getBlocks().front().getArguments();
  return KernelDim3{args[9], args[10], args[11]};
}

LaunchOp::operand_range LaunchOp::getKernelOperandValues() {
  return llvm::drop_begin(getOperands(), kNumConfigOperands);
}

LaunchOp::operand_type_range LaunchOp::getKernelOperandTypes() {
  return llvm::drop_begin(getOperandTypes(), kNumConfigOperands);
}

KernelDim3 LaunchOp::getGridSizeOperandValues() {
  return KernelDim3{getOperand(0), getOperand(1), getOperand(2)};
}

KernelDim3 LaunchOp::getBlockSizeOperandValues() {
  return KernelDim3{getOperand(3), getOperand(4), getOperand(5)};
}

llvm::iterator_range<Block::args_iterator> LaunchOp::getKernelArguments() {
  auto args = getBody().getBlocks().front().getArguments();
  return llvm::drop_begin(args, LaunchOp::kNumConfigRegionAttributes);
}

LogicalResult LaunchOp::verify() {
  // Kernel launch takes kNumConfigOperands leading operands for grid/block
  // sizes and transforms them into kNumConfigRegionAttributes region arguments
  // for block/thread identifiers and grid/block sizes.
  if (!getBody().empty()) {
    Block &entryBlock = getBody().front();
    if (entryBlock.getNumArguments() != kNumConfigOperands + getNumOperands())
      return emitError("unexpected number of region arguments");
  }

  // Block terminators without successors are expected to exit the kernel region
  // and must be `gpu.launch`.
  for (Block &block : getBody()) {
    if (block.empty())
      continue;
    if (block.back().getNumSuccessors() != 0)
      continue;
    if (!isa<gpu::Return>(&block.back())) {
      return block.back()
                 .emitError("expected 'gpu.terminator' or a terminator with "
                            "successors")
                 .attachNote(getLoc())
             << "in '" << getOperationName() << "' body region";
    }
  }

  return success();
}

// Pretty-print the kernel grid/block size assignment as
//   (%iter-x, %iter-y, %iter-z) in
//   (%size-x = %ssa-use, %size-y = %ssa-use, %size-z = %ssa-use)
// where %size-* and %iter-* will correspond to the body region arguments.
static void printSizeAssignment(OpAsmPrinter *p, KernelDim3 size,
                                ArrayRef<Value *> operands, KernelDim3 ids) {
  *p << '(' << *ids.x << ", " << *ids.y << ", " << *ids.z << ") in (";
  *p << *size.x << " = " << *operands[0] << ", ";
  *p << *size.y << " = " << *operands[1] << ", ";
  *p << *size.z << " = " << *operands[2] << ')';
}

void LaunchOp::print(OpAsmPrinter *p) {
  SmallVector<Value *, 12> operandContainer(operand_begin(), operand_end());
  ArrayRef<Value *> operands(operandContainer);

  // Print the launch configuration.
  *p << getOperationName() << ' ' << getBlocksKeyword();
  printSizeAssignment(p, getGridSize(), operands.take_front(3), getBlockIds());
  *p << ' ' << getThreadsKeyword();
  printSizeAssignment(p, getBlockSize(), operands.slice(3, 3), getThreadIds());

  // From now on, the first kNumConfigOperands operands corresponding to grid
  // and block sizes are irrelevant, so we can drop them.
  operands = operands.drop_front(kNumConfigOperands);

  // Print the data argument remapping.
  if (!getBody().empty() && !operands.empty()) {
    *p << ' ' << getArgsKeyword() << '(';
    for (unsigned i = 0, e = operands.size(); i < e; ++i) {
      if (i != 0)
        *p << ", ";
      *p << *getBody().front().getArgument(kNumConfigRegionAttributes + i)
         << " = " << *operands[i];
    }
    *p << ") ";
  }

  // Print the types of data arguments.
  if (!operands.empty()) {
    *p << ": ";
    for (unsigned i = 0, e = operands.size(); i < e; ++i) {
      if (i != 0)
        *p << ", ";
      *p << operands[i]->getType();
    }
  }

  p->printRegion(getBody(), /*printEntryBlockArgs=*/false);
  p->printOptionalAttrDict(getAttrs());
}

// Parse the size assignment blocks for blocks and threads.  These have the form
//   (%region_arg, %region_arg, %region_arg) in
//   (%region_arg = %operand, %region_arg = %operand, %region_arg = %operand)
// where %region_arg are percent-identifiers for the region arguments to be
// introduced futher (SSA defs), and %operand are percent-identifiers for the
// SSA value uses.
static ParseResult
parseSizeAssignment(OpAsmParser *parser,
                    MutableArrayRef<OpAsmParser::OperandType> sizes,
                    MutableArrayRef<OpAsmParser::OperandType> regionSizes,
                    MutableArrayRef<OpAsmParser::OperandType> indices) {
  if (parser->parseLParen() || parser->parseRegionArgument(indices[0]) ||
      parser->parseComma() || parser->parseRegionArgument(indices[1]) ||
      parser->parseComma() || parser->parseRegionArgument(indices[2]) ||
      parser->parseRParen() || parser->parseKeyword("in") ||
      parser->parseLParen())
    return failure();

  for (int i = 0; i < 3; ++i) {
    if (i != 0 && parser->parseComma())
      return failure();
    if (parser->parseRegionArgument(regionSizes[i]) || parser->parseEqual() ||
        parser->parseOperand(sizes[i]))
      return failure();
  }

  return parser->parseRParen();
}

// Parses a Launch operation.
// operation ::= `gpu.launch` `blocks` `(` ssa-id-list `)` `in` ssa-reassignment
//                           `threads` `(` ssa-id-list `)` `in` ssa-reassignment
//                             (`args` ssa-reassignment `:` type-list)?
//                             region attr-dict?
// ssa-reassignment ::= `(` ssa-id `=` ssa-use (`,` ssa-id `=` ssa-use)* `)`
ParseResult LaunchOp::parse(OpAsmParser *parser, OperationState *result) {
  // Sizes of the grid and block.
  SmallVector<OpAsmParser::OperandType, kNumConfigOperands> sizes(
      kNumConfigOperands);
  MutableArrayRef<OpAsmParser::OperandType> sizesRef(sizes);

  // Actual (data) operands passed to the kernel.
  SmallVector<OpAsmParser::OperandType, 4> dataOperands;

  // Region arguments to be created.
  SmallVector<OpAsmParser::OperandType, 16> regionArgs(
      kNumConfigRegionAttributes);
  MutableArrayRef<OpAsmParser::OperandType> regionArgsRef(regionArgs);

  // Parse the size assignment segments: the first segment assigns grid siezs
  // and defines values for block identifiers; the second segment assigns block
  // sies and defines values for thread identifiers.  In the region argument
  // list, identifiers preceed sizes, and block-related values preceed
  // thread-related values.
  if (parser->parseKeyword(getBlocksKeyword().data()) ||
      parseSizeAssignment(parser, sizesRef.take_front(3),
                          regionArgsRef.slice(6, 3),
                          regionArgsRef.slice(0, 3)) ||
      parser->parseKeyword(getThreadsKeyword().data()) ||
      parseSizeAssignment(parser, sizesRef.drop_front(3),
                          regionArgsRef.slice(9, 3),
                          regionArgsRef.slice(3, 3)) ||
      parser->resolveOperands(sizes, parser->getBuilder().getIndexType(),
                              result->operands))
    return failure();

  // If kernel argument renaming segment is present, parse it.  When present,
  // the segment should have at least one element.  If this segment is present,
  // so is the trailing type list.  Parse it as well and use the parsed types
  // to resolve the operands passed to the kernel arguments.
  SmallVector<Type, 4> dataTypes;
  if (!parser->parseOptionalKeyword(getArgsKeyword().data())) {
    llvm::SMLoc argsLoc = parser->getCurrentLocation();

    regionArgs.push_back({});
    dataOperands.push_back({});
    if (parser->parseLParen() ||
        parser->parseRegionArgument(regionArgs.back()) ||
        parser->parseEqual() || parser->parseOperand(dataOperands.back()))
      return failure();

    while (!parser->parseOptionalComma()) {
      regionArgs.push_back({});
      dataOperands.push_back({});
      if (parser->parseRegionArgument(regionArgs.back()) ||
          parser->parseEqual() || parser->parseOperand(dataOperands.back()))
        return failure();
    }

    if (parser->parseRParen() || parser->parseColonTypeList(dataTypes) ||
        parser->resolveOperands(dataOperands, dataTypes, argsLoc,
                                result->operands))
      return failure();
  }

  // Introduce the body region and parse it.  The region has
  // kNumConfigRegionAttributes leading arguments that correspond to
  // block/thread identifiers and grid/block sizes, all of the `index` type.
  // Follow the actual kernel arguments.
  Type index = parser->getBuilder().getIndexType();
  dataTypes.insert(dataTypes.begin(), kNumConfigRegionAttributes, index);
  Region *body = result->addRegion();
  return failure(parser->parseRegion(*body, regionArgs, dataTypes) ||
                 parser->parseOptionalAttributeDict(result->attributes));
}

void LaunchOp::eraseKernelArgument(unsigned index) {
  Block &entryBlock = getBody().front();
  assert(index < entryBlock.getNumArguments() - kNumConfigRegionAttributes &&
         "kernel argument index overflow");
  entryBlock.eraseArgument(kNumConfigRegionAttributes + index);
  getOperation()->eraseOperand(kNumConfigOperands + index);
}

namespace {
// Clone any known constants passed as operands to the kernel into its body.
class PropagateConstantBounds : public OpRewritePattern<LaunchOp> {
  using OpRewritePattern<LaunchOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(LaunchOp launchOp,
                                     PatternRewriter &rewriter) const override {
    auto oringInsertionPoint = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(&launchOp.getBody().front());

    // Traverse operands passed to kernel and check if some of them are known
    // constants.  If so, clone the constant operation inside the kernel region
    // and use it instead of passing the value from the parent region.  Perform
    // the traversal in the inverse order to simplify index arithmetics when
    // dropping arguments.
    SmallVector<Value *, 8> operands(launchOp.getKernelOperandValues().begin(),
                                     launchOp.getKernelOperandValues().end());
    SmallVector<Value *, 8> kernelArgs(launchOp.getKernelArguments().begin(),
                                       launchOp.getKernelArguments().end());
    bool found = false;
    for (unsigned i = operands.size(); i > 0; --i) {
      unsigned index = i - 1;
      Value *operand = operands[index];
      if (!isa_and_nonnull<ConstantOp>(operand->getDefiningOp())) {
        continue;
      }

      found = true;
      Value *internalConstant =
          rewriter.clone(*operand->getDefiningOp())->getResult(0);
      Value *kernelArg = kernelArgs[index];
      kernelArg->replaceAllUsesWith(internalConstant);
      launchOp.eraseKernelArgument(index);
    }
    rewriter.restoreInsertionPoint(oringInsertionPoint);

    if (!found)
      return matchFailure();

    rewriter.updatedRootInPlace(launchOp);
    return matchSuccess();
  }
};
} // end namespace

void LaunchOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
  RewriteListBuilder<PropagateConstantBounds>::build(results, context);
}

//===----------------------------------------------------------------------===//
// LaunchFuncOp
//===----------------------------------------------------------------------===//

void LaunchFuncOp::build(Builder *builder, OperationState *result,
                         FuncOp kernelFunc, Value *gridSizeX, Value *gridSizeY,
                         Value *gridSizeZ, Value *blockSizeX, Value *blockSizeY,
                         Value *blockSizeZ, ArrayRef<Value *> kernelOperands) {
  // Add grid and block sizes as op operands, followed by the data operands.
  result->addOperands(
      {gridSizeX, gridSizeY, gridSizeZ, blockSizeX, blockSizeY, blockSizeZ});
  result->addOperands(kernelOperands);
  result->addAttribute(getKernelAttrName(),
                       builder->getFunctionAttr(kernelFunc));
}

void LaunchFuncOp::build(Builder *builder, OperationState *result,
                         FuncOp kernelFunc, KernelDim3 gridSize,
                         KernelDim3 blockSize,
                         ArrayRef<Value *> kernelOperands) {
  build(builder, result, kernelFunc, gridSize.x, gridSize.y, gridSize.z,
        blockSize.x, blockSize.y, blockSize.z, kernelOperands);
}

StringRef LaunchFuncOp::kernel() {
  return getAttrOfType<FunctionAttr>(getKernelAttrName()).getValue();
}

unsigned LaunchFuncOp::getNumKernelOperands() {
  return getNumOperands() - kNumConfigOperands;
}

Value *LaunchFuncOp::getKernelOperand(unsigned i) {
  return getOperation()->getOperand(i + kNumConfigOperands);
}

KernelDim3 LaunchFuncOp::getGridSizeOperandValues() {
  return KernelDim3{getOperand(0), getOperand(1), getOperand(2)};
}

KernelDim3 LaunchFuncOp::getBlockSizeOperandValues() {
  return KernelDim3{getOperand(3), getOperand(4), getOperand(5)};
}

LogicalResult LaunchFuncOp::verify() {
  auto kernelAttr = this->getAttr(getKernelAttrName());
  if (!kernelAttr) {
    return emitOpError("attribute 'kernel' must be specified");
  } else if (!kernelAttr.isa<FunctionAttr>()) {
    return emitOpError("attribute 'kernel' must be a function");
  }

  auto module = getParentOfType<ModuleOp>();
  FuncOp kernelFunc = module.lookupSymbol<FuncOp>(kernel());
  if (!kernelFunc)
    return emitError() << "kernel function '" << kernelAttr << "' is undefined";

  if (!kernelFunc.getAttrOfType<mlir::UnitAttr>(
          GPUDialect::getKernelFuncAttrName())) {
    return emitError("kernel function is missing the '")
           << GPUDialect::getKernelFuncAttrName() << "' attribute";
  }
  unsigned numKernelFuncArgs = kernelFunc.getNumArguments();
  if (getNumKernelOperands() != numKernelFuncArgs) {
    return emitOpError("got ")
           << getNumKernelOperands() << " kernel operands but expected "
           << numKernelFuncArgs;
  }
  auto functionType = kernelFunc.getType();
  for (unsigned i = 0; i < numKernelFuncArgs; ++i) {
    if (getKernelOperand(i)->getType() != functionType.getInput(i)) {
      return emitOpError("type of function argument ")
             << i << " does not match";
    }
  }
  return success();
}
