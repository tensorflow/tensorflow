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

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Transforms/Utils.h"

using namespace mlir;
using namespace mlir::impl;

//===----------------------------------------------------------------------===//
// ProducerGenerator
//===----------------------------------------------------------------------===//
namespace {
/// This class provides a simple interface for generating fake producers during
/// the conversion process. These fake producers are used when replacing the
/// results of an operation with values of a new, legal, type. The producer
/// provides a definition for the remaining uses of the old value while they
/// await conversion.
struct ProducerGenerator {
  ProducerGenerator(MLIRContext *ctx)
      : producerOpName(kProducerName, ctx), loc(UnknownLoc::get(ctx)) {}

  /// Cleanup any generated conversion values. Returns failure if there are any
  /// dangling references to a producer operation, success otherwise.
  LogicalResult cleanupGeneratedOps() {
    for (auto *op : producerOps) {
      if (!op->use_empty()) {
        auto diag = op->getContext()->emitError(loc)
                    << "Converter did not convert all uses of replaced value "
                       "with illegal type";
        for (auto *user : op->getResult(0)->getUsers())
          diag.attachNote(user->getLoc())
              << "user was not converted : " << *user;
        return diag;
      }
      op->destroy();
    }
    return success();
  }

  /// Generate a producer value for 'oldValue'. These new producers replace all
  /// of the current uses of the original value, and record a mapping between
  /// for replacement with the 'newValue'.
  void generateAndReplace(Value *oldValue, Value *newValue,
                          BlockAndValueMapping &mapping) {
    if (oldValue->use_empty())
      return;

    // Otherwise, generate a new producer operation for the given value type.
    auto *producer = Operation::create(
        loc, producerOpName, llvm::None, oldValue->getType(), llvm::None,
        llvm::None, 0, false, oldValue->getContext());

    // Replace the uses of the old value and record the mapping.
    oldValue->replaceAllUsesWith(producer->getResult(0));
    mapping.map(producer->getResult(0), newValue);
    producerOps.push_back(producer);
  }

  /// This is an operation name for a fake operation that is inserted during the
  /// conversion process. Operations of this type are guaranteed to never escape
  /// the converter.
  static constexpr StringLiteral kProducerName = "__mlir_conversion.producer";
  OperationName producerOpName;

  /// This is a collection of producer values that were generated during the
  /// conversion process.
  std::vector<Operation *> producerOps;

  /// An instance of the unknown location that is used when generating
  /// producers.
  UnknownLoc loc;
};

//===----------------------------------------------------------------------===//
// DialectConversionRewriter
//===----------------------------------------------------------------------===//

/// This class implements a pattern rewriter for DialectConversionPattern
/// patterns. It automatically performs remapping of replaced operation values.
struct DialectConversionRewriter final : public PatternRewriter {
  DialectConversionRewriter(Function *fn)
      : PatternRewriter(fn), tempGenerator(fn->getContext()) {}
  ~DialectConversionRewriter() = default;

  // Implement the hook for replacing an operation with new values.
  void replaceOp(Operation *op, ArrayRef<Value *> newValues,
                 ArrayRef<Value *> valuesToRemoveIfDead) override {
    assert(newValues.size() == op->getNumResults());
    for (unsigned i = 0, e = newValues.size(); i < e; ++i) {
      Value *result = op->getResult(i);
      if (result->getType() != newValues[i]->getType())
        tempGenerator.generateAndReplace(result, newValues[i], mapping);
      else
        result->replaceAllUsesWith(newValues[i]);
    }
    op->erase();
  }

  // Implement the hook for creating operations, and make sure that newly
  // created ops are added to the worklist for processing.
  Operation *createOperation(const OperationState &state) override {
    return FuncBuilder::createOperation(state);
  }

  void lookupValues(Operation::operand_range operands,
                    SmallVectorImpl<Value *> &remapped) {
    remapped.reserve(llvm::size(operands));
    for (Value *operand : operands)
      remapped.push_back(mapping.lookupOrDefault(operand));
  }

  // Mapping between values(blocks) in the original function and in the new
  // function.
  BlockAndValueMapping mapping;

  /// Utility used to create temporary producers operations.
  ProducerGenerator tempGenerator;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// DialectConversionPattern
//===----------------------------------------------------------------------===//

/// Rewrite the IR rooted at the specified operation with the result of
/// this pattern, generating any new operations with the specified
/// builder.  If an unexpected error is encountered (an internal
/// compiler error), it is emitted through the normal MLIR diagnostic
/// hooks and the IR is left in a valid state.
void DialectConversionPattern::rewrite(Operation *op,
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
    destinations.push_back(op->getSuccessor(i));

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

//===----------------------------------------------------------------------===//
// FunctionConverter
//===----------------------------------------------------------------------===//
namespace {
// Implementation detail class of the DialectConversion utility.  Performs
// function-by-function conversions by creating new functions, filling them in
// with converted blocks, updating the function attributes, and replacing the
// old functions with the new ones in the module.
class FunctionConverter {
public:
  // Constructs a FunctionConverter.
  explicit FunctionConverter(MLIRContext *ctx, DialectConversion *conversion,
                             RewritePatternMatcher &matcher)
      : dialectConversion(conversion), matcher(matcher) {}

  // Converts the given function to the dialect using hooks defined in
  // `dialectConversion`.  Returns the converted function or `nullptr` on error.
  Function *convertFunction(Function *f);

  // Converts the given region starting from the entry block and following the
  // block successors. Returns failure on error, success otherwise.
  template <typename RegionParent>
  LogicalResult convertRegion(DialectConversionRewriter &rewriter,
                              Region &region, RegionParent *parent);

  // Converts a block by traversing its operations sequentially, attempting to
  // match a pattern. If there is no match, recurses the operations regions if
  // it has any.
  //
  // After converting operations, traverses the successor blocks unless they
  // have been visited already as indicated in `visitedBlocks`.
  LogicalResult convertBlock(DialectConversionRewriter &rewriter, Block *block,
                             DenseSet<Block *> &visitedBlocks);

  // Converts the type of the given block argument. Returns success if the
  // argument type could be successfully converted, failure otherwise.
  LogicalResult convertArgument(DialectConversionRewriter &rewriter,
                                BlockArgument *arg, Location loc);

  // Pointer to a specific dialect conversion info.
  DialectConversion *dialectConversion;

  /// The matcher to use when converting operations.
  RewritePatternMatcher &matcher;
};
} // end anonymous namespace

LogicalResult
FunctionConverter::convertArgument(DialectConversionRewriter &rewriter,
                                   BlockArgument *arg, Location loc) {
  auto convertedType = dialectConversion->convertType(arg->getType());
  if (!convertedType)
    return arg->getContext()->emitError(loc)
           << "could not convert block argument of type : " << arg->getType();

  // Generate a replacement value, with the new type, for this argument.
  if (convertedType != arg->getType()) {
    rewriter.tempGenerator.generateAndReplace(arg, arg, rewriter.mapping);
    arg->setType(convertedType);
  }
  return success();
}

LogicalResult
FunctionConverter::convertBlock(DialectConversionRewriter &rewriter,
                                Block *block,
                                DenseSet<Block *> &visitedBlocks) {
  // First, add the current block to the list of visited blocks.
  visitedBlocks.insert(block);

  // Preserve the successors before rewriting the operations.
  SmallVector<Block *, 4> successors(block->getSuccessors());

  // Iterate over ops and convert them.
  for (Operation &op : llvm::make_early_inc_range(*block)) {
    rewriter.setInsertionPoint(&op);
    if (matcher.matchAndRewrite(&op, rewriter))
      continue;

    // If a rewrite wasn't matched, update any mapped operands in place.
    for (auto &operand : op.getOpOperands())
      if (auto *newOperand = rewriter.mapping.lookupOrNull(operand.get()))
        operand.set(newOperand);

    // Traverse any held regions.
    for (auto &region : op.getRegions())
      if (!region.empty() && failed(convertRegion(rewriter, region, &op)))
        return failure();
  }

  // Recurse to children that haven't been visited.
  for (Block *succ : successors) {
    if (visitedBlocks.count(succ))
      continue;
    if (failed(convertBlock(rewriter, succ, visitedBlocks)))
      return failure();
  }
  return success();
}

template <typename RegionParent>
LogicalResult
FunctionConverter::convertRegion(DialectConversionRewriter &rewriter,
                                 Region &region, RegionParent *parent) {
  assert(!region.empty() && "expected non-empty region");

  // Create the arguments of each of the blocks in the region.
  for (Block &block : region)
    for (auto *arg : block.getArguments())
      if (failed(convertArgument(rewriter, arg, parent->getLoc())))
        return failure();

  // Start a DFS-order traversal of the CFG to make sure defs are converted
  // before uses in dominated blocks.
  llvm::DenseSet<Block *> visitedBlocks;
  if (failed(convertBlock(rewriter, &region.front(), visitedBlocks)))
    return failure();

  // If some blocks are not reachable through successor chains, they should have
  // been removed by the DCE before this.
  if (visitedBlocks.size() != std::distance(region.begin(), region.end()))
    return parent->emitError("unreachable blocks were not converted");
  return success();
}

Function *FunctionConverter::convertFunction(Function *f) {
  // Convert the function type using the dialect converter.
  SmallVector<NamedAttributeList, 4> newFunctionArgAttrs;
  Type newFunctionType = dialectConversion->convertFunctionSignatureType(
      f->getType(), f->getAllArgAttrs(), newFunctionArgAttrs);
  if (!newFunctionType)
    return f->emitError("could not convert function type"), nullptr;

  // Create a new function using the mapped function type and arg attributes.
  auto *newFunc = new Function(f->getLoc(), f->getName().strref(),
                               newFunctionType.cast<FunctionType>(),
                               f->getAttrs(), newFunctionArgAttrs);
  f->getModule()->getFunctions().push_back(newFunc);

  // If this is not an external function, we need to convert the body.
  if (!f->isExternal()) {
    DialectConversionRewriter rewriter(f);
    f->getBody().cloneInto(&newFunc->getBody(), rewriter.mapping,
                           f->getContext());
    rewriter.mapping.clear();
    if (failed(convertRegion(rewriter, newFunc->getBody(), &*newFunc))) {
      f->getModule()->getFunctions().pop_back();
      return nullptr;
    }

    // Cleanup any temp producer operations that were generated by the rewriter.
    if (failed(rewriter.tempGenerator.cleanupGeneratedOps())) {
      f->getModule()->getFunctions().pop_back();
      return nullptr;
    }
  }
  return newFunc;
}

//===----------------------------------------------------------------------===//
// DialectConversion
//===----------------------------------------------------------------------===//

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

  SmallVector<Function *, 0> originalFuncs, convertedFuncs;
  DenseMap<Attribute, FunctionAttr> functionAttrRemapping;
  originalFuncs.reserve(module->getFunctions().size());
  for (auto &func : *module)
    originalFuncs.push_back(&func);
  convertedFuncs.reserve(originalFuncs.size());

  // Convert each function.
  FunctionConverter converter(context, this, matcher);
  for (auto *func : originalFuncs) {
    Function *converted = converter.convertFunction(func);
    if (!converted) {
      // Make sure to erase any previously converted functions.
      while (!convertedFuncs.empty())
        convertedFuncs.pop_back_val()->erase();
      return failure();
    }

    convertedFuncs.push_back(converted);
    auto origFuncAttr = FunctionAttr::get(func);
    auto convertedFuncAttr = FunctionAttr::get(converted);
    functionAttrRemapping.insert({origFuncAttr, convertedFuncAttr});
  }

  // Remap function attributes in the converted functions. Original functions
  // will disappear anyway so there is no need to remap attributes in them.
  for (const auto &funcPair : functionAttrRemapping)
    remapFunctionAttrs(*funcPair.getSecond().getValue(), functionAttrRemapping);

  // Remove the original functions from the module and update the names of the
  // converted functions.
  for (unsigned i = 0, e = originalFuncs.size(); i != e; ++i) {
    convertedFuncs[i]->takeName(*originalFuncs[i]);
    originalFuncs[i]->erase();
  }

  return success();
}
