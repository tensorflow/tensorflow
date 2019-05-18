//===- DialectConversion.cpp - MLIR dialect conversion generic pass -------===//
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
// This file implements a generic pass for converting between MLIR dialects.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Transforms/Utils.h"

using namespace mlir;

namespace {
/// This class implements a pattern rewriter for DialectOpConversion patterns.
/// It automatically performs remapping of replaced operation values.
struct DialectConversionRewriter final : public PatternRewriter {
  DialectConversionRewriter(Function *fn) : PatternRewriter(fn) {}
  ~DialectConversionRewriter() = default;

  // Implement the hook for replacing an operation with new values.
  void replaceOp(Operation *op, ArrayRef<Value *> newValues,
                 ArrayRef<Value *> valuesToRemoveIfDead) override {
    assert(newValues.size() == op->getNumResults());
    for (unsigned i = 0, e = newValues.size(); i < e; ++i)
      mapping.map(op->getResult(i), newValues[i]);
  }

  // Implement the hook for creating operations, and make sure that newly
  // created ops are added to the worklist for processing.
  Operation *createOperation(const OperationState &state) override {
    return FuncBuilder::createOperation(state);
  }

  void lookupValues(Operation::operand_range operands,
                    SmallVectorImpl<Value *> &remapped) {
    remapped.reserve(llvm::size(operands));
    for (Value *operand : operands) {
      Value *value = mapping.lookupOrNull(operand);
      assert(value && "converting op before ops defining its operands");
      remapped.push_back(value);
    }
  }

  // Mapping between values(blocks) in the original function and in the new
  // function.
  BlockAndValueMapping mapping;
};
} // end anonymous namespace

/// Rewrite the IR rooted at the specified operation with the result of
/// this pattern, generating any new operations with the specified
/// builder.  If an unexpected error is encountered (an internal
/// compiler error), it is emitted through the normal MLIR diagnostic
/// hooks and the IR is left in a valid state.
void DialectOpConversion::rewrite(Operation *op,
                                  PatternRewriter &rewriter) const {
  SmallVector<Value *, 4> operands;
  auto &dialectRewriter = static_cast<DialectConversionRewriter &>(rewriter);
  dialectRewriter.lookupValues(op->getOperands(), operands);

  // If this operation has no successors, invoke the rewrite directly.
  if (op->getNumSuccessors() == 0)
    return rewrite(op, operands, rewriter);

  // Otherwise, we need to remap the successors.
  SmallVector<Block *, 2> destinations;
  destinations.reserve(op->getNumSuccessors());

  SmallVector<ArrayRef<Value *>, 2> operandsPerDestination;
  unsigned firstSuccessorOperand = op->getSuccessorOperandIndex(0);
  for (unsigned i = 0, seen = 0, e = op->getNumSuccessors(); i < e; ++i) {
    // Lookup the successor.
    auto *successor = dialectRewriter.mapping.lookupOrNull(op->getSuccessor(i));
    assert(successor && "block was not remapped");
    destinations.push_back(successor);

    // Lookup the successors operands.
    unsigned n = op->getNumSuccessorOperands(i);
    operandsPerDestination.push_back(
        llvm::makeArrayRef(operands.data() + firstSuccessorOperand + seen, n));
    seen += n;
  }

  // Rewrite the operation.
  rewrite(op,
          llvm::makeArrayRef(operands.data(),
                             operands.data() + firstSuccessorOperand),
          destinations, operandsPerDestination, rewriter);
}

namespace mlir {
namespace impl {
// Implementation detail class of the DialectConversion pass.  Performs
// function-by-function conversions by creating new functions, filling them in
// with converted blocks, updating the function attributes, and replacing the
// old functions with the new ones in the module.
class FunctionConversion {
public:
  // Constructs a FunctionConversion by storing the hooks.
  explicit FunctionConversion(DialectConversion *conversion, Function *func,
                              RewritePatternMatcher &matcher)
      : dialectConversion(conversion), rewriter(func), matcher(matcher) {}

  // Converts the current function to the dialect using hooks defined in
  // `dialectConversion`.  Returns the converted function or `nullptr` on error.
  Function *convertFunction();

  // Converts the given region starting from the entry block and following the
  // block successors.  Returns the converted region or `nullptr` on error.
  template <typename RegionParent>
  std::unique_ptr<Region> convertRegion(MLIRContext *context, Region *region,
                                        RegionParent *parent);

  // Converts a block by traversing its operations sequentially, looking for
  // the first pattern match and dispatching the operation conversion to
  // either `convertOp` or `convertOpWithSuccessors` depending on the presence
  // of successors.  If there is no match, clones the operation.
  //
  // After converting operations, traverses the successor blocks unless they
  // have been visited already as indicated in `visitedBlocks`.
  LogicalResult convertBlock(Block *block,
                             llvm::DenseSet<Block *> &visitedBlocks);

  // Pointer to a specific dialect pass.
  DialectConversion *dialectConversion;

  /// The writer used when rewriting operations.
  DialectConversionRewriter rewriter;

  /// The matcher use when converting operations.
  RewritePatternMatcher &matcher;
};
} // end namespace impl
} // end namespace mlir

LogicalResult
impl::FunctionConversion::convertBlock(Block *block,
                                       llvm::DenseSet<Block *> &visitedBlocks) {
  // First, add the current block to the list of visited blocks.
  visitedBlocks.insert(block);
  // Setup the builder to the insert to the converted block.
  rewriter.setInsertionPointToStart(rewriter.mapping.lookupOrNull(block));

  // Iterate over ops and convert them.
  for (Operation &op : *block) {
    if (matcher.matchAndRewrite(&op, rewriter))
      continue;

    // If there is no conversion provided for the op, clone the op and convert
    // its regions, if any.
    auto *newOp = rewriter.cloneWithoutRegions(op, rewriter.mapping);
    for (int i = 0, e = op.getNumRegions(); i < e; ++i) {
      auto newRegion = convertRegion(op.getContext(), &op.getRegion(i), &op);
      newOp->getRegion(i).takeBody(*newRegion);
    }
  }

  // Recurse to children unless they have been already visited.
  for (Block *succ : block->getSuccessors()) {
    if (visitedBlocks.count(succ) != 0)
      continue;
    if (failed(convertBlock(succ, visitedBlocks)))
      return failure();
  }
  return success();
}

template <typename RegionParent>
std::unique_ptr<Region>
impl::FunctionConversion::convertRegion(MLIRContext *context, Region *region,
                                        RegionParent *parent) {
  assert(region && "expected a region");
  auto newRegion = llvm::make_unique<Region>(parent);
  if (region->empty())
    return newRegion;

  auto emitError = [context](llvm::Twine f) -> std::unique_ptr<Region> {
    context->emitError(UnknownLoc::get(context), f.str());
    return nullptr;
  };

  // Create new blocks and convert their arguments.
  for (Block &block : *region) {
    auto *newBlock = new Block;
    newRegion->push_back(newBlock);
    rewriter.mapping.map(&block, newBlock);
    for (auto *arg : block.getArguments()) {
      auto convertedType = dialectConversion->convertType(arg->getType());
      if (!convertedType)
        return emitError("could not convert block argument type");
      newBlock->addArgument(convertedType);
      rewriter.mapping.map(arg, *newBlock->args_rbegin());
    }
  }

  // Start a DFS-order traversal of the CFG to make sure defs are converted
  // before uses in dominated blocks.
  llvm::DenseSet<Block *> visitedBlocks;
  if (failed(convertBlock(&region->front(), visitedBlocks)))
    return nullptr;

  // If some blocks are not reachable through successor chains, they should have
  // been removed by the DCE before this.
  if (visitedBlocks.size() != std::distance(region->begin(), region->end()))
    return emitError("unreachable blocks were not converted");
  return newRegion;
}

Function *impl::FunctionConversion::convertFunction() {
  Function *f = rewriter.getFunction();
  MLIRContext *context = f->getContext();
  auto emitError = [context](llvm::Twine f) -> Function * {
    context->emitError(UnknownLoc::get(context), f.str());
    return nullptr;
  };

  // Create a new function with argument types and result types converted. Wrap
  // it into a unique_ptr to make sure it is cleaned up in case of error.
  SmallVector<NamedAttributeList, 4> newFunctionArgAttrs;
  Type newFunctionType = dialectConversion->convertFunctionSignatureType(
      f->getType(), f->getAllArgAttrs(), newFunctionArgAttrs);
  if (!newFunctionType)
    return emitError("could not convert function type");
  auto newFunction = llvm::make_unique<Function>(
      f->getLoc(), f->getName().strref(), newFunctionType.cast<FunctionType>(),
      f->getAttrs(), newFunctionArgAttrs);

  // Return early if the function is external.
  if (f->isExternal())
    return newFunction.release();

  auto newBody = convertRegion(context, &f->getBody(), f);
  if (!newBody)
    return emitError("could not convert function body");
  newFunction->getBody().takeBody(*newBody);

  return newFunction.release();
}

// Create a function type with arguments and results converted, and argument
// attributes passed through.
FunctionType DialectConversion::convertFunctionSignatureType(
    FunctionType type, ArrayRef<NamedAttributeList> argAttrs,
    SmallVectorImpl<NamedAttributeList> &convertedArgAttrs) {
  SmallVector<Type, 8> arguments;
  SmallVector<Type, 4> results;

  arguments.reserve(type.getNumInputs());
  for (auto t : type.getInputs())
    arguments.push_back(convertType(t));

  results.reserve(type.getNumResults());
  for (auto t : type.getResults())
    results.push_back(convertType(t));

  // Note this will cause an extra allocation only if we need
  // to grow the caller-provided resulting attribute vector.
  convertedArgAttrs.reserve(arguments.size());
  for (auto attr : argAttrs)
    convertedArgAttrs.push_back(attr);

  return FunctionType::get(arguments, results, type.getContext());
}

// Converts the module as follows.
// 1. Call `convertFunction` on each function of the module and collect the
// mapping between old and new functions.
// 2. Remap all function attributes in the new functions to point to the new
// functions instead of the old ones.
// 3. Replace old functions with the new in the module.
LogicalResult DialectConversion::convert(Module *module) {
  if (!module)
    return failure();

  // Grab the conversion patterns from the converter and create the pattern
  // matcher.
  MLIRContext *context = module->getContext();
  OwningRewritePatternList patterns;
  initConverters(patterns, context);
  RewritePatternMatcher matcher(std::move(patterns));

  // Convert the functions but don't add them to the module yet to avoid
  // converted functions to be converted again.
  SmallVector<Function *, 0> originalFuncs, convertedFuncs;
  DenseMap<Attribute, FunctionAttr> functionAttrRemapping;
  originalFuncs.reserve(module->getFunctions().size());
  for (auto &func : *module)
    originalFuncs.push_back(&func);
  convertedFuncs.reserve(module->getFunctions().size());
  for (auto *func : originalFuncs) {
    impl::FunctionConversion converter(this, func, matcher);
    Function *converted = converter.convertFunction();
    if (!converted)
      return failure();

    auto origFuncAttr = FunctionAttr::get(func);
    auto convertedFuncAttr = FunctionAttr::get(converted);
    convertedFuncs.push_back(converted);
    functionAttrRemapping.insert({origFuncAttr, convertedFuncAttr});
  }

  // Remap function attributes in the converted functions (they are not yet in
  // the module).  Original functions will disappear anyway so there is no
  // need to remap attributes in them.
  for (const auto &funcPair : functionAttrRemapping)
    remapFunctionAttrs(*funcPair.getSecond().getValue(), functionAttrRemapping);

  // Remove original functions from the module, then insert converted
  // functions.  The order is important to avoid name collisions.
  for (auto &func : originalFuncs)
    func->erase();
  for (auto *func : convertedFuncs)
    module->getFunctions().push_back(func);

  return success();
}
