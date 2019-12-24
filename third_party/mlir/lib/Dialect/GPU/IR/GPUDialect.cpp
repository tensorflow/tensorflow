//===- GPUDialect.cpp - MLIR Dialect for GPU Kernels implementation -------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the GPU kernel-related dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/FunctionImplementation.h"
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
  addOperations<
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
    auto kernelGPUFunction = dyn_cast_or_null<gpu::GPUFuncOp>(kernelFunc);
    auto kernelLLVMFunction = dyn_cast_or_null<LLVM::LLVMFuncOp>(kernelFunc);
    if (!kernelGPUFunction && !kernelLLVMFunction)
      return launchOp.emitOpError("kernel function '")
             << kernelName << "' is undefined";
    if (!kernelFunc->getAttrOfType<mlir::UnitAttr>(
            GPUDialect::getKernelFuncAttrName()))
      return launchOp.emitOpError("kernel function is missing the '")
             << GPUDialect::getKernelFuncAttrName() << "' attribute";

    unsigned actualNumArguments = launchOp.getNumKernelOperands();
    unsigned expectedNumArguments = kernelLLVMFunction
                                        ? kernelLLVMFunction.getNumArguments()
                                        : kernelGPUFunction.getNumArguments();
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
    for (auto argument : allReduce.body().front().getArguments()) {
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

static LogicalResult verifyShuffleOp(gpu::ShuffleOp shuffleOp) {
  auto type = shuffleOp.value()->getType();
  if (shuffleOp.result()->getType() != type) {
    return shuffleOp.emitOpError()
           << "requires the same type for value operand and result";
  }
  if (!type.isIntOrFloat() || type.getIntOrFloatBitWidth() != 32) {
    return shuffleOp.emitOpError()
           << "requires value operand type to be f32 or i32";
  }
  return success();
}

static void printShuffleOp(OpAsmPrinter &p, ShuffleOp op) {
  p << ShuffleOp::getOperationName() << ' ';
  p.printOperands(op.getOperands());
  p << ' ' << op.mode() << " : ";
  p.printType(op.value()->getType());
}

static ParseResult parseShuffleOp(OpAsmParser &parser, OperationState &state) {
  SmallVector<OpAsmParser::OperandType, 3> operandInfo;
  if (parser.parseOperandList(operandInfo, 3))
    return failure();

  StringRef mode;
  if (parser.parseKeyword(&mode))
    return failure();
  state.addAttribute("mode", parser.getBuilder().getStringAttr(mode));

  Type valueType;
  Type int32Type = parser.getBuilder().getIntegerType(32);
  Type int1Type = parser.getBuilder().getI1Type();
  if (parser.parseColonType(valueType) ||
      parser.resolveOperands(operandInfo, {valueType, int32Type, int32Type},
                             parser.getCurrentLocation(), state.operands) ||
      parser.addTypesToList({valueType, int1Type}, state.types))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// LaunchOp
//===----------------------------------------------------------------------===//

static SmallVector<Type, 4> getValueTypes(ValueRange values) {
  SmallVector<Type, 4> types;
  types.reserve(values.size());
  for (Value v : values)
    types.push_back(v->getType());
  return types;
}

void LaunchOp::build(Builder *builder, OperationState &result, Value gridSizeX,
                     Value gridSizeY, Value gridSizeZ, Value blockSizeX,
                     Value blockSizeY, Value blockSizeZ, ValueRange operands) {
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

KernelDim3 LaunchOp::getBlockIds() {
  assert(!body().getBlocks().empty() && "FuncOp body must not be empty.");
  auto args = body().getBlocks().front().getArguments();
  return KernelDim3{args[0], args[1], args[2]};
}

KernelDim3 LaunchOp::getThreadIds() {
  assert(!body().getBlocks().empty() && "FuncOp body must not be empty.");
  auto args = body().getBlocks().front().getArguments();
  return KernelDim3{args[3], args[4], args[5]};
}

KernelDim3 LaunchOp::getGridSize() {
  assert(!body().getBlocks().empty() && "FuncOp body must not be empty.");
  auto args = body().getBlocks().front().getArguments();
  return KernelDim3{args[6], args[7], args[8]};
}

KernelDim3 LaunchOp::getBlockSize() {
  assert(!body().getBlocks().empty() && "FuncOp body must not be empty.");
  auto args = body().getBlocks().front().getArguments();
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

iterator_range<Block::args_iterator> LaunchOp::getKernelArguments() {
  auto args = body().getBlocks().front().getArguments();
  return llvm::drop_begin(args, LaunchOp::kNumConfigRegionAttributes);
}

LogicalResult verify(LaunchOp op) {
  // Kernel launch takes kNumConfigOperands leading operands for grid/block
  // sizes and transforms them into kNumConfigRegionAttributes region arguments
  // for block/thread identifiers and grid/block sizes.
  if (!op.body().empty()) {
    Block &entryBlock = op.body().front();
    if (entryBlock.getNumArguments() !=
        LaunchOp::kNumConfigOperands + op.getNumOperands())
      return op.emitOpError("unexpected number of region arguments");
  }

  // Block terminators without successors are expected to exit the kernel region
  // and must be `gpu.launch`.
  for (Block &block : op.body()) {
    if (block.empty())
      continue;
    if (block.back().getNumSuccessors() != 0)
      continue;
    if (!isa<gpu::ReturnOp>(&block.back())) {
      return block.back()
                 .emitError("expected 'gpu.terminator' or a terminator with "
                            "successors")
                 .attachNote(op.getLoc())
             << "in '" << LaunchOp::getOperationName() << "' body region";
    }
  }

  return success();
}

// Pretty-print the kernel grid/block size assignment as
//   (%iter-x, %iter-y, %iter-z) in
//   (%size-x = %ssa-use, %size-y = %ssa-use, %size-z = %ssa-use)
// where %size-* and %iter-* will correspond to the body region arguments.
static void printSizeAssignment(OpAsmPrinter &p, KernelDim3 size,
                                ValueRange operands, KernelDim3 ids) {
  p << '(' << *ids.x << ", " << *ids.y << ", " << *ids.z << ") in (";
  p << *size.x << " = " << *operands[0] << ", ";
  p << *size.y << " = " << *operands[1] << ", ";
  p << *size.z << " = " << *operands[2] << ')';
}

void printLaunchOp(OpAsmPrinter &p, LaunchOp op) {
  ValueRange operands = op.getOperands();

  // Print the launch configuration.
  p << LaunchOp::getOperationName() << ' ' << op.getBlocksKeyword();
  printSizeAssignment(p, op.getGridSize(), operands.take_front(3),
                      op.getBlockIds());
  p << ' ' << op.getThreadsKeyword();
  printSizeAssignment(p, op.getBlockSize(), operands.slice(3, 3),
                      op.getThreadIds());

  // From now on, the first kNumConfigOperands operands corresponding to grid
  // and block sizes are irrelevant, so we can drop them.
  operands = operands.drop_front(LaunchOp::kNumConfigOperands);

  // Print the data argument remapping.
  if (!op.body().empty() && !operands.empty()) {
    p << ' ' << op.getArgsKeyword() << '(';
    Block *entryBlock = &op.body().front();
    interleaveComma(llvm::seq<int>(0, operands.size()), p, [&](int i) {
      p << *entryBlock->getArgument(LaunchOp::kNumConfigRegionAttributes + i)
        << " = " << *operands[i];
    });
    p << ") ";
  }

  // Print the types of data arguments.
  if (!operands.empty())
    p << ": " << operands.getTypes();

  p.printRegion(op.body(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict(op.getAttrs());
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
ParseResult parseLaunchOp(OpAsmParser &parser, OperationState &result) {
  // Sizes of the grid and block.
  SmallVector<OpAsmParser::OperandType, LaunchOp::kNumConfigOperands> sizes(
      LaunchOp::kNumConfigOperands);
  MutableArrayRef<OpAsmParser::OperandType> sizesRef(sizes);

  // Actual (data) operands passed to the kernel.
  SmallVector<OpAsmParser::OperandType, 4> dataOperands;

  // Region arguments to be created.
  SmallVector<OpAsmParser::OperandType, 16> regionArgs(
      LaunchOp::kNumConfigRegionAttributes);
  MutableArrayRef<OpAsmParser::OperandType> regionArgsRef(regionArgs);

  // Parse the size assignment segments: the first segment assigns grid sizes
  // and defines values for block identifiers; the second segment assigns block
  // sizes and defines values for thread identifiers.  In the region argument
  // list, identifiers precede sizes, and block-related values precede
  // thread-related values.
  if (parser.parseKeyword(LaunchOp::getBlocksKeyword().data()) ||
      parseSizeAssignment(parser, sizesRef.take_front(3),
                          regionArgsRef.slice(6, 3),
                          regionArgsRef.slice(0, 3)) ||
      parser.parseKeyword(LaunchOp::getThreadsKeyword().data()) ||
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
  if (!parser.parseOptionalKeyword(LaunchOp::getArgsKeyword())) {
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
  dataTypes.insert(dataTypes.begin(), LaunchOp::kNumConfigRegionAttributes,
                   index);
  Region *body = result.addRegion();
  return failure(parser.parseRegion(*body, regionArgs, dataTypes) ||
                 parser.parseOptionalAttrDict(result.attributes));
}

void LaunchOp::eraseKernelArgument(unsigned index) {
  Block &entryBlock = body().front();
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
    rewriter.startRootUpdate(launchOp);
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&launchOp.body().front());

    // Traverse operands passed to kernel and check if some of them are known
    // constants.  If so, clone the constant operation inside the kernel region
    // and use it instead of passing the value from the parent region.  Perform
    // the traversal in the inverse order to simplify index arithmetics when
    // dropping arguments.
    auto operands = launchOp.getKernelOperandValues();
    auto kernelArgs = launchOp.getKernelArguments();
    bool found = false;
    for (unsigned i = operands.size(); i > 0; --i) {
      unsigned index = i - 1;
      Value operand = operands[index];
      if (!isa_and_nonnull<ConstantOp>(operand->getDefiningOp()))
        continue;

      found = true;
      Value internalConstant =
          rewriter.clone(*operand->getDefiningOp())->getResult(0);
      Value kernelArg = *std::next(kernelArgs.begin(), index);
      kernelArg->replaceAllUsesWith(internalConstant);
      launchOp.eraseKernelArgument(index);
    }

    if (!found) {
      rewriter.cancelRootUpdate(launchOp);
      return matchFailure();
    }

    rewriter.finalizeRootUpdate(launchOp);
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
                         GPUFuncOp kernelFunc, Value gridSizeX, Value gridSizeY,
                         Value gridSizeZ, Value blockSizeX, Value blockSizeY,
                         Value blockSizeZ, ValueRange kernelOperands) {
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
                         GPUFuncOp kernelFunc, KernelDim3 gridSize,
                         KernelDim3 blockSize, ValueRange kernelOperands) {
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

Value LaunchFuncOp::getKernelOperand(unsigned i) {
  return getOperation()->getOperand(i + kNumConfigOperands);
}

KernelDim3 LaunchFuncOp::getGridSizeOperandValues() {
  return KernelDim3{getOperand(0), getOperand(1), getOperand(2)};
}

KernelDim3 LaunchFuncOp::getBlockSizeOperandValues() {
  return KernelDim3{getOperand(3), getOperand(4), getOperand(5)};
}

LogicalResult verify(LaunchFuncOp op) {
  auto module = op.getParentOfType<ModuleOp>();
  if (!module)
    return op.emitOpError("expected to belong to a module");

  if (!module.getAttrOfType<UnitAttr>(GPUDialect::getContainerModuleAttrName()))
    return op.emitOpError(
        "expected the closest surrounding module to have the '" +
        GPUDialect::getContainerModuleAttrName() + "' attribute");

  auto kernelAttr = op.getAttrOfType<StringAttr>(op.getKernelAttrName());
  if (!kernelAttr)
    return op.emitOpError("string attribute '" + op.getKernelAttrName() +
                          "' must be specified");

  auto kernelModuleAttr =
      op.getAttrOfType<SymbolRefAttr>(op.getKernelModuleAttrName());
  if (!kernelModuleAttr)
    return op.emitOpError("symbol reference attribute '" +
                          op.getKernelModuleAttrName() + "' must be specified");

  return success();
}

//===----------------------------------------------------------------------===//
// GPUFuncOp
//===----------------------------------------------------------------------===//

void GPUFuncOp::build(Builder *builder, OperationState &result, StringRef name,
                      FunctionType type, ArrayRef<Type> workgroupAttributions,
                      ArrayRef<Type> privateAttributions,
                      ArrayRef<NamedAttribute> attrs) {
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder->getStringAttr(name));
  result.addAttribute(getTypeAttrName(), TypeAttr::get(type));
  result.addAttribute(getNumWorkgroupAttributionsAttrName(),
                      builder->getI64IntegerAttr(workgroupAttributions.size()));
  result.addAttributes(attrs);
  Region *body = result.addRegion();
  Block *entryBlock = new Block;
  entryBlock->addArguments(type.getInputs());
  entryBlock->addArguments(workgroupAttributions);
  entryBlock->addArguments(privateAttributions);

  body->getBlocks().push_back(entryBlock);
}

/// Parses a GPU function memory attribution.
///
/// memory-attribution ::= (`workgroup` `(` ssa-id-and-type-list `)`)?
///                        (`private` `(` ssa-id-and-type-list `)`)?
///
/// Note that this function parses only one of the two similar parts, with the
/// keyword provided as argument.
static ParseResult
parseAttributions(OpAsmParser &parser, StringRef keyword,
                  SmallVectorImpl<OpAsmParser::OperandType> &args,
                  SmallVectorImpl<Type> &argTypes) {
  // If we could not parse the keyword, just assume empty list and succeed.
  if (failed(parser.parseOptionalKeyword(keyword)))
    return success();

  if (failed(parser.parseLParen()))
    return failure();

  // Early exit for an empty list.
  if (succeeded(parser.parseOptionalRParen()))
    return success();

  do {
    OpAsmParser::OperandType arg;
    Type type;

    if (parser.parseRegionArgument(arg) || parser.parseColonType(type))
      return failure();

    args.push_back(arg);
    argTypes.push_back(type);
  } while (succeeded(parser.parseOptionalComma()));

  return parser.parseRParen();
}

/// Parses a GPU function.
///
/// <operation> ::= `gpu.func` symbol-ref-id `(` argument-list `)`
///                 (`->` function-result-list)? memory-attribution `kernel`?
///                 function-attributes? region
static ParseResult parseGPUFuncOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 8> entryArgs;
  SmallVector<SmallVector<NamedAttribute, 2>, 1> argAttrs;
  SmallVector<SmallVector<NamedAttribute, 2>, 1> resultAttrs;
  SmallVector<Type, 8> argTypes;
  SmallVector<Type, 4> resultTypes;
  bool isVariadic;

  // Parse the function name.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, ::mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  auto signatureLocation = parser.getCurrentLocation();
  if (failed(impl::parseFunctionSignature(
          parser, /*allowVariadic=*/false, entryArgs, argTypes, argAttrs,
          isVariadic, resultTypes, resultAttrs)))
    return failure();

  if (entryArgs.empty() && !argTypes.empty())
    return parser.emitError(signatureLocation)
           << "gpu.func requires named arguments";

  // Construct the function type. More types will be added to the region, but
  // not to the functiont type.
  Builder &builder = parser.getBuilder();
  auto type = builder.getFunctionType(argTypes, resultTypes);
  result.addAttribute(GPUFuncOp::getTypeAttrName(), TypeAttr::get(type));

  // Parse workgroup memory attributions.
  if (failed(parseAttributions(parser, GPUFuncOp::getWorkgroupKeyword(),
                               entryArgs, argTypes)))
    return failure();

  // Store the number of operands we just parsed as the number of workgroup
  // memory attributions.
  unsigned numWorkgroupAttrs = argTypes.size() - type.getNumInputs();
  result.addAttribute(GPUFuncOp::getNumWorkgroupAttributionsAttrName(),
                      builder.getI64IntegerAttr(numWorkgroupAttrs));

  // Parse private memory attributions.
  if (failed(parseAttributions(parser, GPUFuncOp::getPrivateKeyword(),
                               entryArgs, argTypes)))
    return failure();

  // Parse the kernel attribute if present.
  if (succeeded(parser.parseOptionalKeyword(GPUFuncOp::getKernelKeyword())))
    result.addAttribute(GPUDialect::getKernelFuncAttrName(),
                        builder.getUnitAttr());

  // Parse attributes.
  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();
  mlir::impl::addArgAndResultAttrs(builder, result, argAttrs, resultAttrs);

  // Parse the region. If no argument names were provided, take all names
  // (including those of attributions) from the entry block.
  auto *body = result.addRegion();
  return parser.parseRegion(*body, entryArgs, argTypes);
}

static void printAttributions(OpAsmPrinter &p, StringRef keyword,
                              ArrayRef<BlockArgument> values) {
  if (values.empty())
    return;

  p << ' ' << keyword << '(';
  interleaveComma(values, p,
                  [&p](BlockArgument v) { p << *v << " : " << v->getType(); });
  p << ')';
}

/// Prints a GPU Func op.
void printGPUFuncOp(OpAsmPrinter &p, GPUFuncOp op) {
  p << GPUFuncOp::getOperationName() << ' ';
  p.printSymbolName(op.getName());

  FunctionType type = op.getType();
  impl::printFunctionSignature(p, op.getOperation(), type.getInputs(),
                               /*isVariadic=*/false, type.getResults());

  printAttributions(p, op.getWorkgroupKeyword(), op.getWorkgroupAttributions());
  printAttributions(p, op.getPrivateKeyword(), op.getPrivateAttributions());
  if (op.isKernel())
    p << ' ' << op.getKernelKeyword();

  impl::printFunctionAttributes(p, op.getOperation(), type.getNumInputs(),
                                type.getNumResults(),
                                {op.getNumWorkgroupAttributionsAttrName(),
                                 GPUDialect::getKernelFuncAttrName()});
  p.printRegion(op.getBody(), /*printEntryBlockArgs=*/false);
}

void GPUFuncOp::setType(FunctionType newType) {
  auto oldType = getType();
  assert(newType.getNumResults() == oldType.getNumResults() &&
         "unimplemented: changes to the number of results");

  SmallVector<char, 16> nameBuf;
  for (int i = newType.getNumInputs(), e = oldType.getNumInputs(); i < e; i++)
    removeAttr(getArgAttrName(i, nameBuf));

  setAttr(getTypeAttrName(), TypeAttr::get(newType));
}

/// Hook for FunctionLike verifier.
LogicalResult GPUFuncOp::verifyType() {
  Type type = getTypeAttr().getValue();
  if (!type.isa<FunctionType>())
    return emitOpError("requires '" + getTypeAttrName() +
                       "' attribute of function type");
  return success();
}

static LogicalResult verifyAttributions(Operation *op,
                                        ArrayRef<BlockArgument> attributions,
                                        unsigned memorySpace) {
  for (Value v : attributions) {
    auto type = v->getType().dyn_cast<MemRefType>();
    if (!type)
      return op->emitOpError() << "expected memref type in attribution";

    if (type.getMemorySpace() != memorySpace) {
      return op->emitOpError()
             << "expected memory space " << memorySpace << " in attribution";
    }
  }
  return success();
}

/// Verifies the body of the function.
LogicalResult GPUFuncOp::verifyBody() {
  unsigned numFuncArguments = getNumArguments();
  unsigned numWorkgroupAttributions = getNumWorkgroupAttributions();
  unsigned numBlockArguments = front().getNumArguments();
  if (numBlockArguments < numFuncArguments + numWorkgroupAttributions)
    return emitOpError() << "expected at least "
                         << numFuncArguments + numWorkgroupAttributions
                         << " arguments to body region";

  ArrayRef<Type> funcArgTypes = getType().getInputs();
  for (unsigned i = 0; i < numFuncArguments; ++i) {
    Type blockArgType = front().getArgument(i)->getType();
    if (funcArgTypes[i] != blockArgType)
      return emitOpError() << "expected body region argument #" << i
                           << " to be of type " << funcArgTypes[i] << ", got "
                           << blockArgType;
  }

  if (failed(verifyAttributions(getOperation(), getWorkgroupAttributions(),
                                GPUDialect::getWorkgroupAddressSpace())) ||
      failed(verifyAttributions(getOperation(), getPrivateAttributions(),
                                GPUDialect::getPrivateAddressSpace())))
    return failure();

  return success();
}

// Namespace avoids ambiguous ReturnOpOperandAdaptor.
namespace mlir {
namespace gpu {
#define GET_OP_CLASSES
#include "mlir/Dialect/GPU/GPUOps.cpp.inc"
} // namespace gpu
} // namespace mlir
