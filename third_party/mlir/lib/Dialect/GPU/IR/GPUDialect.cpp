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

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace mlir::gpu;

//===----------------------------------------------------------------------===//
// GPUDialect
//===----------------------------------------------------------------------===//

StringRef GPUDialect::getDialectName() { return "gpu"; }

bool GPUDialect::isKernel(Operation *op) {
  UnitAttr isKernelAttr = op->getAttrOfType<UnitAttr>(getKernelFuncAttrName());
  return static_cast<bool>(isKernelAttr);
}

GPUDialect::GPUDialect(MLIRContext *context)
    : Dialect(getDialectName(), context) {
  addOperations<LaunchOp, LaunchFuncOp,
#define GET_OP_LIST
#include "mlir/Dialect/GPU/GPUOps.cpp.inc"
                >();
}

LogicalResult GPUDialect::verifyOperationAttribute(Operation *op,
                                                   NamedAttribute attr) {
  if (!attr.second.isa<UnitAttr>() ||
      !attr.first.is(getContainerModuleAttrName()))
    return success();

  auto module = dyn_cast<ModuleOp>(op);
  if (!module)
    return op->emitError("expected '")
           << getContainerModuleAttrName() << "' attribute to be attached to '"
           << ModuleOp::getOperationName() << '\'';

  auto walkResult = module.walk([&module](LaunchFuncOp launchOp) -> WalkResult {
    // Ignore launches that are nested more or less deep than functions in the
    // module we are currently checking.
    if (!launchOp.getParentOp() ||
        launchOp.getParentOp()->getParentOp() != module)
      return success();

    // Ignore launch ops with missing attributes here. The errors will be
    // reported by the verifiers of those ops.
    if (!launchOp.getAttrOfType<StringAttr>(
            LaunchFuncOp::getKernelAttrName()) ||
        !launchOp.getAttrOfType<SymbolRefAttr>(
            LaunchFuncOp::getKernelModuleAttrName()))
      return success();

    // Check that `launch_func` refers to a well-formed GPU kernel module.
    StringRef kernelModuleName = launchOp.getKernelModuleName();
    auto kernelModule = module.lookupSymbol<ModuleOp>(kernelModuleName);
    if (!kernelModule)
      return launchOp.emitOpError()
             << "kernel module '" << kernelModuleName << "' is undefined";
    if (!kernelModule.getAttrOfType<UnitAttr>(
            GPUDialect::getKernelModuleAttrName()))
      return launchOp.emitOpError("module '")
             << kernelModuleName << "' is missing the '"
             << GPUDialect::getKernelModuleAttrName() << "' attribute";

    // Check that `launch_func` refers to a well-formed kernel function.
    StringRef kernelName = launchOp.kernel();
    Operation *kernelFunc = kernelModule.lookupSymbol(kernelName);
    auto kernelStdFunction = dyn_cast_or_null<FuncOp>(kernelFunc);
    auto kernelLLVMFunction = dyn_cast_or_null<LLVM::LLVMFuncOp>(kernelFunc);
    if (!kernelStdFunction && !kernelLLVMFunction)
      return launchOp.emitOpError("kernel function '")
             << kernelName << "' is undefined";
    if (!kernelFunc->getAttrOfType<mlir::UnitAttr>(
            GPUDialect::getKernelFuncAttrName()))
      return launchOp.emitOpError("kernel function is missing the '")
             << GPUDialect::getKernelFuncAttrName() << "' attribute";

    unsigned actualNumArguments = launchOp.getNumKernelOperands();
    unsigned expectedNumArguments = kernelLLVMFunction
                                        ? kernelLLVMFunction.getNumArguments()
                                        : kernelStdFunction.getNumArguments();
    if (expectedNumArguments != actualNumArguments)
      return launchOp.emitOpError("got ")
             << actualNumArguments << " kernel operands but expected "
             << expectedNumArguments;

    // Due to the ordering of the current impl of lowering and LLVMLowering,
    // type checks need to be temporarily disabled.
    // TODO(ntv,zinenko,herhut): reactivate checks once "changing gpu.launchFunc
    // to encode target module" has landed.
    // auto functionType = kernelFunc.getType();
    // for (unsigned i = 0; i < numKernelFuncArgs; ++i) {
    //   if (getKernelOperand(i)->getType() != functionType.getInput(i)) {
    //     return emitOpError("type of function argument ")
    //            << i << " does not match";
    //   }
    // }

    return success();
  });

  return walkResult.wasInterrupted() ? failure() : success();
}

template <typename T> static LogicalResult verifyIndexOp(T op) {
  auto dimension = op.dimension();
  if (dimension != "x" && dimension != "y" && dimension != "z")
    return op.emitError("dimension \"") << dimension << "\" is invalid";
  return success();
}

static LogicalResult verifyAllReduce(gpu::AllReduceOp allReduce) {
  if (allReduce.body().empty() != allReduce.op().hasValue())
    return allReduce.emitError(
        "expected either an op attribute or a non-empty body");
  if (!allReduce.body().empty()) {
    if (allReduce.body().front().getNumArguments() != 2)
      return allReduce.emitError("expected two region arguments");
    for (auto *argument : allReduce.body().front().getArguments()) {
      if (argument->getType() != allReduce.getType())
        return allReduce.emitError("incorrect region argument type");
    }
    unsigned yieldCount = 0;
    for (Block &block : allReduce.body()) {
      if (auto yield = dyn_cast<gpu::YieldOp>(block.getTerminator())) {
        if (yield.getNumOperands() != 1)
          return allReduce.emitError("expected one gpu.yield operand");
        if (yield.getOperand(0)->getType() != allReduce.getType())
          return allReduce.emitError("incorrect gpu.yield type");
        ++yieldCount;
      }
    }
    if (yieldCount == 0)
      return allReduce.emitError("expected gpu.yield op in region");
  }
  return success();
}

// Namespace avoids ambiguous ReturnOpOperandAdaptor.
namespace mlir {
namespace gpu {
#define GET_OP_CLASSES
#include "mlir/Dialect/GPU/GPUOps.cpp.inc"
} // namespace gpu
} // namespace mlir

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

void LaunchOp::build(Builder *builder, OperationState &result, Value *gridSizeX,
                     Value *gridSizeY, Value *gridSizeZ, Value *blockSizeX,
                     Value *blockSizeY, Value *blockSizeZ,
                     ArrayRef<Value *> operands) {
  // Add grid and block sizes as op operands, followed by the data operands.
  result.addOperands(
      {gridSizeX, gridSizeY, gridSizeZ, blockSizeX, blockSizeY, blockSizeZ});
  result.addOperands(operands);

  // Create a kernel body region with kNumConfigRegionAttributes + N arguments,
  // where the first kNumConfigRegionAttributes arguments have `index` type and
  // the rest have the same types as the data operands.
  Region *kernelRegion = result.addRegion();
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
      return emitOpError("unexpected number of region arguments");
  }

  // Block terminators without successors are expected to exit the kernel region
  // and must be `gpu.launch`.
  for (Block &block : getBody()) {
    if (block.empty())
      continue;
    if (block.back().getNumSuccessors() != 0)
      continue;
    if (!isa<gpu::ReturnOp>(&block.back())) {
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
static void printSizeAssignment(OpAsmPrinter &p, KernelDim3 size,
                                ArrayRef<Value *> operands, KernelDim3 ids) {
  p << '(' << *ids.x << ", " << *ids.y << ", " << *ids.z << ") in (";
  p << *size.x << " = " << *operands[0] << ", ";
  p << *size.y << " = " << *operands[1] << ", ";
  p << *size.z << " = " << *operands[2] << ')';
}

void LaunchOp::print(OpAsmPrinter &p) {
  SmallVector<Value *, 12> operandContainer(operand_begin(), operand_end());
  ArrayRef<Value *> operands(operandContainer);

  // Print the launch configuration.
  p << getOperationName() << ' ' << getBlocksKeyword();
  printSizeAssignment(p, getGridSize(), operands.take_front(3), getBlockIds());
  p << ' ' << getThreadsKeyword();
  printSizeAssignment(p, getBlockSize(), operands.slice(3, 3), getThreadIds());

  // From now on, the first kNumConfigOperands operands corresponding to grid
  // and block sizes are irrelevant, so we can drop them.
  operands = operands.drop_front(kNumConfigOperands);

  // Print the data argument remapping.
  if (!getBody().empty() && !operands.empty()) {
    p << ' ' << getArgsKeyword() << '(';
    for (unsigned i = 0, e = operands.size(); i < e; ++i) {
      if (i != 0)
        p << ", ";
      p << *getBody().front().getArgument(kNumConfigRegionAttributes + i)
        << " = " << *operands[i];
    }
    p << ") ";
  }

  // Print the types of data arguments.
  if (!operands.empty()) {
    p << ": ";
    for (unsigned i = 0, e = operands.size(); i < e; ++i) {
      if (i != 0)
        p << ", ";
      p << operands[i]->getType();
    }
  }

  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict(getAttrs());
}

// Parse the size assignment blocks for blocks and threads.  These have the form
//   (%region_arg, %region_arg, %region_arg) in
//   (%region_arg = %operand, %region_arg = %operand, %region_arg = %operand)
// where %region_arg are percent-identifiers for the region arguments to be
// introduced further (SSA defs), and %operand are percent-identifiers for the
// SSA value uses.
static ParseResult
parseSizeAssignment(OpAsmParser &parser,
                    MutableArrayRef<OpAsmParser::OperandType> sizes,
                    MutableArrayRef<OpAsmParser::OperandType> regionSizes,
                    MutableArrayRef<OpAsmParser::OperandType> indices) {
  assert(indices.size() == 3 && "space for three indices expected");
  SmallVector<OpAsmParser::OperandType, 3> args;
  if (parser.parseRegionArgumentList(args, /*requiredOperandCount=*/3,
                                     OpAsmParser::Delimiter::Paren) ||
      parser.parseKeyword("in") || parser.parseLParen())
    return failure();
  std::move(args.begin(), args.end(), indices.begin());

  for (int i = 0; i < 3; ++i) {
    if (i != 0 && parser.parseComma())
      return failure();
    if (parser.parseRegionArgument(regionSizes[i]) || parser.parseEqual() ||
        parser.parseOperand(sizes[i]))
      return failure();
  }

  return parser.parseRParen();
}

// Parses a Launch operation.
// operation ::= `gpu.launch` `blocks` `(` ssa-id-list `)` `in` ssa-reassignment
//                           `threads` `(` ssa-id-list `)` `in` ssa-reassignment
//                             (`args` ssa-reassignment `:` type-list)?
//                             region attr-dict?
// ssa-reassignment ::= `(` ssa-id `=` ssa-use (`,` ssa-id `=` ssa-use)* `)`
ParseResult LaunchOp::parse(OpAsmParser &parser, OperationState &result) {
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

  // Parse the size assignment segments: the first segment assigns grid sizes
  // and defines values for block identifiers; the second segment assigns block
  // sizes and defines values for thread identifiers.  In the region argument
  // list, identifiers precede sizes, and block-related values precede
  // thread-related values.
  if (parser.parseKeyword(getBlocksKeyword().data()) ||
      parseSizeAssignment(parser, sizesRef.take_front(3),
                          regionArgsRef.slice(6, 3),
                          regionArgsRef.slice(0, 3)) ||
      parser.parseKeyword(getThreadsKeyword().data()) ||
      parseSizeAssignment(parser, sizesRef.drop_front(3),
                          regionArgsRef.slice(9, 3),
                          regionArgsRef.slice(3, 3)) ||
      parser.resolveOperands(sizes, parser.getBuilder().getIndexType(),
                             result.operands))
    return failure();

  // If kernel argument renaming segment is present, parse it.  When present,
  // the segment should have at least one element.  If this segment is present,
  // so is the trailing type list.  Parse it as well and use the parsed types
  // to resolve the operands passed to the kernel arguments.
  SmallVector<Type, 4> dataTypes;
  if (!parser.parseOptionalKeyword(getArgsKeyword())) {
    llvm::SMLoc argsLoc = parser.getCurrentLocation();

    regionArgs.push_back({});
    dataOperands.push_back({});
    if (parser.parseLParen() || parser.parseRegionArgument(regionArgs.back()) ||
        parser.parseEqual() || parser.parseOperand(dataOperands.back()))
      return failure();

    while (!parser.parseOptionalComma()) {
      regionArgs.push_back({});
      dataOperands.push_back({});
      if (parser.parseRegionArgument(regionArgs.back()) ||
          parser.parseEqual() || parser.parseOperand(dataOperands.back()))
        return failure();
    }

    if (parser.parseRParen() || parser.parseColonTypeList(dataTypes) ||
        parser.resolveOperands(dataOperands, dataTypes, argsLoc,
                               result.operands))
      return failure();
  }

  // Introduce the body region and parse it.  The region has
  // kNumConfigRegionAttributes leading arguments that correspond to
  // block/thread identifiers and grid/block sizes, all of the `index` type.
  // Follow the actual kernel arguments.
  Type index = parser.getBuilder().getIndexType();
  dataTypes.insert(dataTypes.begin(), kNumConfigRegionAttributes, index);
  Region *body = result.addRegion();
  return failure(parser.parseRegion(*body, regionArgs, dataTypes) ||
                 parser.parseOptionalAttrDict(result.attributes));
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
    auto origInsertionPoint = rewriter.saveInsertionPoint();
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
    rewriter.restoreInsertionPoint(origInsertionPoint);

    if (!found)
      return matchFailure();

    rewriter.updatedRootInPlace(launchOp);
    return matchSuccess();
  }
};
} // end namespace

void LaunchOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
  results.insert<PropagateConstantBounds>(context);
}

//===----------------------------------------------------------------------===//
// LaunchFuncOp
//===----------------------------------------------------------------------===//

void LaunchFuncOp::build(Builder *builder, OperationState &result,
                         FuncOp kernelFunc, Value *gridSizeX, Value *gridSizeY,
                         Value *gridSizeZ, Value *blockSizeX, Value *blockSizeY,
                         Value *blockSizeZ, ArrayRef<Value *> kernelOperands) {
  // Add grid and block sizes as op operands, followed by the data operands.
  result.addOperands(
      {gridSizeX, gridSizeY, gridSizeZ, blockSizeX, blockSizeY, blockSizeZ});
  result.addOperands(kernelOperands);
  result.addAttribute(getKernelAttrName(),
                      builder->getStringAttr(kernelFunc.getName()));
  auto kernelModule = kernelFunc.getParentOfType<ModuleOp>();
  if (Optional<StringRef> kernelModuleName = kernelModule.getName())
    result.addAttribute(getKernelModuleAttrName(),
                        builder->getSymbolRefAttr(*kernelModuleName));
}

void LaunchFuncOp::build(Builder *builder, OperationState &result,
                         FuncOp kernelFunc, KernelDim3 gridSize,
                         KernelDim3 blockSize,
                         ArrayRef<Value *> kernelOperands) {
  build(builder, result, kernelFunc, gridSize.x, gridSize.y, gridSize.z,
        blockSize.x, blockSize.y, blockSize.z, kernelOperands);
}

StringRef LaunchFuncOp::kernel() {
  return getAttrOfType<StringAttr>(getKernelAttrName()).getValue();
}

unsigned LaunchFuncOp::getNumKernelOperands() {
  return getNumOperands() - kNumConfigOperands;
}

StringRef LaunchFuncOp::getKernelModuleName() {
  return getAttrOfType<SymbolRefAttr>(getKernelModuleAttrName())
      .getRootReference();
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
  auto module = getParentOfType<ModuleOp>();
  if (!module)
    return emitOpError("expected to belong to a module");

  if (!module.getAttrOfType<UnitAttr>(GPUDialect::getContainerModuleAttrName()))
    return emitOpError("expected the closest surrounding module to have the '" +
                       GPUDialect::getContainerModuleAttrName() +
                       "' attribute");

  auto kernelAttr = getAttrOfType<StringAttr>(getKernelAttrName());
  if (!kernelAttr)
    return emitOpError("string attribute '" + getKernelAttrName() +
                       "' must be specified");

  auto kernelModuleAttr =
      getAttrOfType<SymbolRefAttr>(getKernelModuleAttrName());
  if (!kernelModuleAttr)
    return emitOpError("symbol reference attribute '" +
                       getKernelModuleAttrName() + "' must be specified");

  return success();
}
