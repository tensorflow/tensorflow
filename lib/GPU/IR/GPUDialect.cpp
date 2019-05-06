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
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace mlir::gpu;

StringRef GPUDialect::getDialectName() { return "gpu"; }

GPUDialect::GPUDialect(MLIRContext *context)
    : Dialect(getDialectName(), context) {
  addOperations<LaunchOp, LaunchFuncOp>();
}

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
  auto args = getBody().getBlocks().front().getArguments();
  return KernelDim3{args[0], args[1], args[2]};
}

KernelDim3 LaunchOp::getThreadIds() {
  auto args = getBody().getBlocks().front().getArguments();
  return KernelDim3{args[3], args[4], args[5]};
}

KernelDim3 LaunchOp::getGridSize() {
  auto args = getBody().getBlocks().front().getArguments();
  return KernelDim3{args[6], args[7], args[8]};
}

KernelDim3 LaunchOp::getBlockSize() {
  auto args = getBody().getBlocks().front().getArguments();
  return KernelDim3{args[9], args[10], args[11]};
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
static bool
parseSizeAssignment(OpAsmParser *parser,
                    MutableArrayRef<OpAsmParser::OperandType> sizes,
                    MutableArrayRef<OpAsmParser::OperandType> regionSizes,
                    MutableArrayRef<OpAsmParser::OperandType> indices) {
  if (parser->parseLParen() || parser->parseRegionArgument(indices[0]) ||
      parser->parseComma() || parser->parseRegionArgument(indices[1]) ||
      parser->parseComma() || parser->parseRegionArgument(indices[2]) ||
      parser->parseRParen() || parser->parseKeyword("in") ||
      parser->parseLParen())
    return true;

  for (int i = 0; i < 3; ++i) {
    if (i != 0 && parser->parseComma())
      return true;
    if (parser->parseRegionArgument(regionSizes[i]) || parser->parseEqual() ||
        parser->parseOperand(sizes[i]))
      return true;
  }

  return parser->parseRParen();
}

// Parses a Launch operation.
// operation ::= `gpu.launch` `blocks` `(` ssa-id-list `)` `in` ssa-reassignment
//                           `threads` `(` ssa-id-list `)` `in` ssa-reassignment
//                             (`args` ssa-reassignment `:` type-list)?
//                             region attr-dict?
// ssa-reassignment ::= `(` ssa-id `=` ssa-use (`,` ssa-id `=` ssa-use)* `)`
bool LaunchOp::parse(OpAsmParser *parser, OperationState *result) {
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
    return true;

  // If kernel argument renaming segment is present, parse it.  When present,
  // the segment should have at least one element.  If this segment is present,
  // so is the trailing type list.  Parse it as well and use the parsed types
  // to resolve the operands passed to the kernel arguments.
  SmallVector<Type, 4> dataTypes;
  if (!parser->parseOptionalKeyword(getArgsKeyword().data())) {
    llvm::SMLoc argsLoc;

    regionArgs.push_back({});
    dataOperands.push_back({});
    if (parser->getCurrentLocation(&argsLoc) || parser->parseLParen() ||
        parser->parseRegionArgument(regionArgs.back()) ||
        parser->parseEqual() || parser->parseOperand(dataOperands.back()))
      return true;

    while (!parser->parseOptionalComma()) {
      regionArgs.push_back({});
      dataOperands.push_back({});
      if (parser->parseRegionArgument(regionArgs.back()) ||
          parser->parseEqual() || parser->parseOperand(dataOperands.back()))
        return true;
    }

    if (parser->parseRParen() || parser->parseColonTypeList(dataTypes) ||
        parser->resolveOperands(dataOperands, dataTypes, argsLoc,
                                result->operands))
      return true;
  }

  // Introduce the body region and parse it.  The region has
  // kNumConfigRegionAttributes leading arguments that correspond to
  // block/thread identifiers and grid/block sizes, all of the `index` type.
  // Follow the actual kernel arguments.
  Type index = parser->getBuilder().getIndexType();
  dataTypes.insert(dataTypes.begin(), kNumConfigRegionAttributes, index);
  Region *body = result->addRegion();
  return parser->parseRegion(*body, regionArgs, dataTypes) ||
         parser->parseOptionalAttributeDict(result->attributes);
}


//===----------------------------------------------------------------------===//
// LaunchFuncOp
//===----------------------------------------------------------------------===//
Function *LaunchFuncOp::kernel() {
  return this->getAttr("kernel").dyn_cast<FunctionAttr>().getValue();
}

unsigned LaunchFuncOp::getNumKernelOperands() {
  return getNumOperands() - kNumConfigOperands;
}

Value *LaunchFuncOp::getKernelOperand(unsigned i) {
  return getOperation()->getOperand(i + kNumConfigOperands);
}

LogicalResult LaunchFuncOp::verify() {
  auto kernelAttr = this->getAttr("kernel");
  if (!kernelAttr) {
    return emitOpError("attribute 'kernel' must be specified");
  } else if (!kernelAttr.isa<FunctionAttr>()) {
    return emitOpError("attribute 'kernel' must be a function");
  }
  Function *kernelFunc = this->kernel();
  unsigned numKernelFuncArgs = kernelFunc->getNumArguments();
  if (getNumKernelOperands() != numKernelFuncArgs) {
    return emitOpError("got " + Twine(getNumKernelOperands()) +
                       " kernel operands but expected " +
                       Twine(numKernelFuncArgs));
  }
  for (unsigned i = 0; i < numKernelFuncArgs; ++i) {
    if (getKernelOperand(i)->getType() !=
        kernelFunc->getArgument(i)->getType()) {
      return emitOpError("type of function argument " + Twine(i) +
                         " does not match");
    }
  }
  return success();
}
