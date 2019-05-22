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
// ArgConverter
//===----------------------------------------------------------------------===//
namespace {
/// This class provides a simple interface for converting the types of block
/// arguments. This is done by inserting fake cast operations for the illegal
/// type that allow for updating the real type to return the correct type.
struct ArgConverter {
  ArgConverter(MLIRContext *ctx)
      : castOpName(kCastName, ctx), loc(UnknownLoc::get(ctx)) {}

  /// Cleanup and undo any generated conversion values.
  void discardRewrites() {
    // On failure drop all uses of the cast operation and destroy it.
    for (auto *op : castOps) {
      op->getResult(0)->dropAllUses();
      op->destroy();
    }
    castOps.clear();
  }

  /// Replace usages of the cast operations with the argument directly.
  void applyRewrites() {
    // On success, we update the type of the block argument and replace uses of
    // the cast.
    for (auto *op : castOps) {
      op->getOperand(0)->setType(op->getResult(0)->getType());
      op->getResult(0)->replaceAllUsesWith(op->getOperand(0));
      op->destroy();
    }
  }

  /// Generate a cast operation for 'arg' that produces the new, legal, type.
  void castArgument(BlockArgument *arg, Type newType,
                    BlockAndValueMapping &mapping) {
    // Otherwise, generate a new cast operation for the given value type.
    auto *cast = Operation::create(loc, castOpName, arg, newType, llvm::None,
                                   llvm::None, 0, false, arg->getContext());

    // Replace the uses of the argument and record the mapping.
    mapping.map(arg, cast->getResult(0));
    castOps.push_back(cast);
  }

  /// This is an operation name for a fake operation that is inserted during the
  /// conversion process. Operations of this type are guaranteed to never escape
  /// the converter.
  static constexpr StringLiteral kCastName = "__mlir_conversion.cast";
  OperationName castOpName;

  /// This is a collection of cast values that were generated during the
  /// conversion process.
  std::vector<Operation *> castOps;

  /// An instance of the unknown location that is used when generating
  /// producers.
  UnknownLoc loc;
};

constexpr StringLiteral ArgConverter::kCastName;

//===----------------------------------------------------------------------===//
// DialectConversionRewriter
//===----------------------------------------------------------------------===//

/// This class implements a pattern rewriter for DialectConversionPattern
/// patterns. It automatically performs remapping of replaced operation values.
struct DialectConversionRewriter final : public PatternRewriter {
  /// This class represents one requested operation replacement via 'replaceOp'.
  struct OpReplacement {
    OpReplacement() = default;
    OpReplacement(Operation *op, ArrayRef<Value *> newValues)
        : op(op), newValues(newValues.begin(), newValues.end()) {}

    Operation *op;
    SmallVector<Value *, 2> newValues;
  };

  DialectConversionRewriter(Function *fn)
      : PatternRewriter(fn), argConverter(fn->getContext()) {}
  ~DialectConversionRewriter() = default;

  /// Cleanup and destroy any generated rewrite operations. This method is
  /// invoked when the conversion process fails.
  void discardRewrites() {
    argConverter.discardRewrites();
    for (auto *op : createdOps) {
      op->dropAllDefinedValueUses();
      op->erase();
    }
  }

  /// Apply all requested operation rewrites. This method is invoked when the
  /// conversion process succeeds.
  void applyRewrites() {
    argConverter.applyRewrites();

    // Apply all of the rewrites replacements requested during conversion.
    for (auto &repl : replacements) {
      for (unsigned i = 0, e = repl.newValues.size(); i != e; ++i)
        repl.op->getResult(i)->replaceAllUsesWith(repl.newValues[i]);
      repl.op->erase();
    }
  }

  /// PatternRewriter hook for replacing the results of an operation.
  void replaceOp(Operation *op, ArrayRef<Value *> newValues,
                 ArrayRef<Value *> valuesToRemoveIfDead) override {
    assert(newValues.size() == op->getNumResults());
    // Create mappings for any type changes.
    for (unsigned i = 0, e = newValues.size(); i < e; ++i)
      if (op->getResult(i)->getType() != newValues[i]->getType())
        mapping.map(op->getResult(i), newValues[i]);

    // Record the requested operation replacement.
    replacements.emplace_back(op, newValues);
  }

  /// PatternRewriter hook for creating a new operation.
  Operation *createOperation(const OperationState &state) override {
    auto *result = FuncBuilder::createOperation(state);
    createdOps.push_back(result);
    return result;
  }

  /// PatternRewriter hook for updating the root operation in-place.
  void notifyRootUpdated(Operation *op) override {
    // The rewriter caches changes to the IR to allow for operating in-place and
    // backtracking. The rewrite is currently not capable of backtracking
    // in-place modifications.
    llvm_unreachable("in-place operation updates are not supported");
  }

  /// Remap the given operands to those with potentially different types.
  void remapValues(Operation::operand_range operands,
                   SmallVectorImpl<Value *> &remapped) {
    remapped.reserve(llvm::size(operands));
    for (Value *operand : operands)
      remapped.push_back(mapping.lookupOrDefault(operand));
  }

  // Mapping between replaced values that differ in type. This happens when
  // replacing a value with one of a different type.
  BlockAndValueMapping mapping;

  /// Utility used to convert block arguments.
  ArgConverter argConverter;

  /// Ordered vector of all of the newly created operations during conversion.
  SmallVector<Operation *, 4> createdOps;

  /// Ordered vector of any requested operation replacements.
  SmallVector<OpReplacement, 4> replacements;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// DialectConversionPattern
//===----------------------------------------------------------------------===//

/// Rewrite the IR rooted at the specified operation with the result of this
/// pattern.  If an unexpected error is encountered (an internal compiler
/// error), it is emitted through the normal MLIR diagnostic hooks and the IR is
/// left in a valid state.
void DialectConversionPattern::rewrite(Operation *op,
                                       PatternRewriter &rewriter) const {
  SmallVector<Value *, 4> operands;
  auto &dialectRewriter = static_cast<DialectConversionRewriter &>(rewriter);
  dialectRewriter.remapValues(op->getOperands(), operands);

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
// This class converts a single function using a given DialectConversion
// structure.
class FunctionConverter {
public:
  // Constructs a FunctionConverter.
  explicit FunctionConverter(MLIRContext *ctx, DialectConversion *conversion,
                             RewritePatternMatcher &matcher)
      : dialectConversion(conversion), matcher(matcher) {}

  /// Converts the given function to the dialect using hooks defined in
  /// `dialectConversion`. Returns failure on error, success otherwise.
  LogicalResult convertFunction(Function *f);

  /// Converts the given region starting from the entry block and following the
  /// block successors. Returns failure on error, success otherwise.
  template <typename RegionParent>
  LogicalResult convertRegion(DialectConversionRewriter &rewriter,
                              Region &region, RegionParent *parent);

  /// Converts a block by traversing its operations sequentially, attempting to
  /// match a pattern. If there is no match, recurses the operations regions if
  /// it has any.
  //
  /// After converting operations, traverses the successor blocks unless they
  /// have been visited already as indicated in `visitedBlocks`.
  LogicalResult convertBlock(DialectConversionRewriter &rewriter, Block *block,
                             DenseSet<Block *> &visitedBlocks);

  /// Converts the type of the given block argument. Returns success if the
  /// argument type could be successfully converted, failure otherwise.
  LogicalResult convertArgument(DialectConversionRewriter &rewriter,
                                BlockArgument *arg, Location loc);

  /// Pointer to a specific dialect conversion info.
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
  if (convertedType != arg->getType())
    rewriter.argConverter.castArgument(arg, convertedType, rewriter.mapping);
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

LogicalResult FunctionConverter::convertFunction(Function *f) {
  // If this is an external function, there is nothing else to do.
  if (f->isExternal())
    return success();

  // Rewrite the function body.
  DialectConversionRewriter rewriter(f);
  if (failed(convertRegion(rewriter, f->getBody(), f))) {
    // Reset any of the converted arguments.
    rewriter.argConverter.discardRewrites();
    return failure();
  }

  // Otherwise the conversion succeeded, so apply all rewrites.
  rewriter.applyRewrites();
  return success();
}

//===----------------------------------------------------------------------===//
// DialectConversion
//===----------------------------------------------------------------------===//

namespace {
/// This class represents a function to be converted. It allows for converting
/// the body of functions and the signature in two phases.
struct ConvertedFunction {
  ConvertedFunction(Function *fn, FunctionType newType,
                    ArrayRef<NamedAttributeList> newFunctionArgAttrs)
      : fn(fn), newType(newType),
        newFunctionArgAttrs(newFunctionArgAttrs.begin(),
                            newFunctionArgAttrs.end()) {}

  /// The function to convert.
  Function *fn;
  /// The new type and argument attributes for the function.
  FunctionType newType;
  SmallVector<NamedAttributeList, 4> newFunctionArgAttrs;
};
} // end anonymous namespace

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

  // Try to convert each of the functions within the module. Defer updating the
  // signatures of the functions until after all of the bodies have been
  // converted. This allows for the conversion patterns to still rely on the
  // public signatures of the functions within the module before they are
  // updated.
  std::vector<ConvertedFunction> toConvert;
  toConvert.reserve(module->getFunctions().size());
  for (auto &func : *module) {
    // Convert the function type using the dialect converter.
    SmallVector<NamedAttributeList, 4> newFunctionArgAttrs;
    FunctionType newType = convertFunctionSignatureType(
        func.getType(), func.getAllArgAttrs(), newFunctionArgAttrs);
    if (!newType || !newType.isa<FunctionType>())
      return func.emitError("could not convert function type");

    // Convert the body of this function.
    FunctionConverter converter(context, this, matcher);
    if (failed(converter.convertFunction(&func)))
      return failure();

    // Add function signature to be updated.
    toConvert.emplace_back(&func, newType.cast<FunctionType>(),
                           newFunctionArgAttrs);
  }

  // Finally, update the signatures of all of the converted functions.
  for (auto &it : toConvert) {
    it.fn->setType(it.newType);
    it.fn->setAllArgAttrs(it.newFunctionArgAttrs);
  }

  return success();
}
