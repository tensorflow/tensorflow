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

namespace mlir {
namespace impl {
// Implementation detail class of the DialectConversion pass.  Performs
// function-by-function conversions by creating new functions, filling them in
// with converted blocks, updating the function attributes, and replacing the
// old functions with the new ones in the module.
class FunctionConversion {
public:
  // Entry point.  Uses hooks defined in `conversion` to obtain the list of
  // conversion patterns and to convert function and block argument types.
  // Converts the `module` in-place by replacing all existing functions with the
  // converted ones.
  static bool convert(DialectConversion *conversion, Module *module);

private:
  // Constructs a FunctionConversion by storing the hooks.
  explicit FunctionConversion(DialectConversion *conversion)
      : dialectConversion(conversion) {}

  // Utility that looks up a list of value in the value remapping table. Returns
  // an empty vector if one of the values is not mapped yet.
  SmallVector<Value *, 4>
  lookupValues(const llvm::iterator_range<Instruction::const_operand_iterator>
                   &operands);

  // Converts the given function to the dialect using hooks defined in
  // `dialectConversion`.  Returns the converted function or `nullptr` on error.
  Function *convertFunction(Function *f);

  // Converts an operation with successors.  Extracts the converted operands
  // from `valueRemapping` and the converted blocks from `blockRemapping`, and
  // passes them to `converter->rewriteTerminator` function defined in the
  // pattern, together with `builder`.
  bool convertOpWithSuccessors(DialectOpConversion *converter, Instruction *op,
                               FuncBuilder &builder);

  // Converts an operation without successors.  Extracts the converted operands
  // from `valueRemapping` and passes them to the `converter->rewrite` function
  // defined in the pattern, together with `builder`.
  bool convertOp(DialectOpConversion *converter, Instruction *op,
                 FuncBuilder &builder);

  // Converts a block by traversing its instructions sequentially, looking for
  // the first pattern match and dispatching the instruction conversion to
  // either `convertOp` or `convertOpWithSuccessors` depending on the presence
  // of successors.  If there is no match, clones the operation.
  //
  // After converting operations, traverses the successor blocks unless they
  // have been visited already as indicated in `visitedBlocks`.
  //
  // Return `true` on error.
  bool convertBlock(Block *block, FuncBuilder &builder,
                    llvm::DenseSet<Block *> &visitedBlocks);

  // Converts the module as follows.
  // 1. Call `convertFunction` on each function of the module and collect the
  // mapping between old and new functions.
  // 2. Remap all function attributes in the new functions to point to the new
  // functions instead of the old ones.
  // 3. Replace old functions with the new in the module.
  bool run(Module *m);

  // Pointer to a specific dialect pass.
  DialectConversion *dialectConversion;

  // Set of known conversion patterns.
  llvm::DenseSet<DialectOpConversion *> conversions;

  // Mapping between values(blocks) in the original function and in the new
  // function.
  BlockAndValueMapping mapping;
};
} // end namespace impl
} // end namespace mlir

SmallVector<Value *, 4> impl::FunctionConversion::lookupValues(
    const llvm::iterator_range<Instruction::const_operand_iterator> &operands) {
  SmallVector<Value *, 4> remapped;
  remapped.reserve(llvm::size(operands));
  for (const Value *operand : operands) {
    Value *value = mapping.lookupOrNull(operand);
    if (!value)
      return {};
    remapped.push_back(value);
  }
  return remapped;
}

bool impl::FunctionConversion::convertOpWithSuccessors(
    DialectOpConversion *converter, Instruction *op, FuncBuilder &builder) {
  SmallVector<Block *, 2> destinations;
  destinations.reserve(op->getNumSuccessors());
  SmallVector<Value *, 4> operands = lookupValues(op->getOperands());
  assert((!operands.empty() || op->getNumOperands() == 0) &&
         "converting op before ops defining its operands");

  SmallVector<ArrayRef<Value *>, 2> operandsPerDestination;
  unsigned numSuccessorOperands = 0;
  for (unsigned i = 0, e = op->getNumSuccessors(); i < e; ++i)
    numSuccessorOperands += op->getNumSuccessorOperands(i);
  unsigned seen = 0;
  unsigned firstSuccessorOperand = op->getNumOperands() - numSuccessorOperands;
  for (unsigned i = 0, e = op->getNumSuccessors(); i < e; ++i) {
    Block *successor = mapping.lookupOrNull(op->getSuccessor(i));
    assert(successor && "block was not remapped");
    destinations.push_back(successor);
    unsigned n = op->getNumSuccessorOperands(i);
    operandsPerDestination.push_back(
        llvm::makeArrayRef(operands.data() + firstSuccessorOperand + seen, n));
    seen += n;
  }
  converter->rewriteTerminator(
      op,
      llvm::makeArrayRef(operands.data(),
                         operands.data() + firstSuccessorOperand),
      destinations, operandsPerDestination, builder);
  return false;
}

bool impl::FunctionConversion::convertOp(DialectOpConversion *converter,
                                         Instruction *op,
                                         FuncBuilder &builder) {
  auto operands = lookupValues(op->getOperands());
  assert((!operands.empty() || op->getNumOperands() == 0) &&
         "converting op before ops defining its operands");

  auto results = converter->rewrite(op, operands, builder);
  if (results.size() != op->getNumResults())
    return op->emitError("rewriting produced a different number of results");

  for (unsigned i = 0, e = results.size(); i < e; ++i)
    mapping.map(op->getResult(i), results[i]);
  return false;
}

bool impl::FunctionConversion::convertBlock(
    Block *block, FuncBuilder &builder,
    llvm::DenseSet<Block *> &visitedBlocks) {
  // First, add the current block to the list of visited blocks.
  visitedBlocks.insert(block);
  // Setup the builder to the insert to the converted block.
  builder.setInsertionPointToStart(mapping.lookupOrNull(block));

  // Iterate over ops and convert them.
  for (Instruction &inst : *block) {
    if (inst.getNumBlockLists() != 0) {
      inst.emitError("unsupported region instruction");
      return true;
    }

    // Find the first matching conversion and apply it.
    bool converted = false;
    for (auto *conversion : conversions) {
      if (!conversion->match(&inst))
        continue;

      if (inst.isTerminator() && inst.getNumSuccessors() > 0) {
        if (convertOpWithSuccessors(conversion, &inst, builder))
          return true;
      } else if (convertOp(conversion, &inst, builder)) {
        return true;
      }
      converted = true;
      break;
    }
    // If there is no conversion provided for the op, clone the op as is.
    if (!converted)
      builder.clone(inst, mapping);
  }

  // Recurse to children unless they have been already visited.
  for (Block *succ : block->getSuccessors()) {
    if (visitedBlocks.count(succ) != 0)
      continue;
    if (convertBlock(succ, builder, visitedBlocks))
      return true;
  }
  return false;
}

Function *impl::FunctionConversion::convertFunction(Function *f) {
  assert(f && "expected function");
  MLIRContext *context = f->getContext();
  auto emitError = [context](llvm::Twine f) -> Function * {
    context->emitError(UnknownLoc::get(context), f.str());
    return nullptr;
  };

  // Create a new function with argument types and result types converted.  Wrap
  // it into a unique_ptr to make sure it is cleaned up in case of error.
  Type newFunctionType = dialectConversion->convertType(f->getType());
  if (!newFunctionType)
    return emitError("could not convert function type");
  auto newFunction = llvm::make_unique<Function>(
      f->getLoc(), f->getName().strref(), newFunctionType.cast<FunctionType>(),
      f->getAttrs());

  // Return early if the function has no blocks.
  if (f->getBlocks().empty())
    return newFunction.release();

  // Create blocks in the new function and convert types of their arguments.
  FuncBuilder builder(newFunction.get());
  for (Block &block : *f) {
    auto *newBlock = builder.createBlock();
    mapping.map(&block, newBlock);
    for (auto *arg : block.getArguments()) {
      auto convertedType = dialectConversion->convertType(arg->getType());
      if (!convertedType)
        return emitError("could not convert block argument type");
      newBlock->addArgument(convertedType);
      mapping.map(arg, *newBlock->args_rbegin());
    }
  }

  // Start a DFS-order traversal of the CFG to make sure defs are converted
  // before uses in dominated blocks.
  llvm::DenseSet<Block *> visitedBlocks;
  if (convertBlock(&f->front(), builder, visitedBlocks))
    return nullptr;

  // If some blocks are not reachable through successor chains, they should have
  // been removed by the DCE before this.
  if (visitedBlocks.size() != f->getBlocks().size())
    return emitError("unreachable blocks were not converted");

  return newFunction.release();
}

bool impl::FunctionConversion::convert(DialectConversion *conversion,
                                       Module *module) {
  return impl::FunctionConversion(conversion).run(module);
}

bool impl::FunctionConversion::run(Module *module) {
  if (!module)
    return true;

  MLIRContext *context = module->getContext();
  conversions = dialectConversion->initConverters(context);

  // Convert the functions but don't add them to the module yet to avoid
  // converted functions to be converted again.
  SmallVector<Function *, 0> originalFuncs, convertedFuncs;
  DenseMap<Attribute, FunctionAttr> functionAttrRemapping;
  originalFuncs.reserve(module->getFunctions().size());
  for (auto &func : *module)
    originalFuncs.push_back(&func);
  convertedFuncs.reserve(module->getFunctions().size());
  for (auto *func : originalFuncs) {
    Function *converted = convertFunction(func);
    if (!converted)
      return true;

    auto origFuncAttr = FunctionAttr::get(func, context);
    auto convertedFuncAttr = FunctionAttr::get(converted, context);
    convertedFuncs.push_back(converted);
    functionAttrRemapping.insert({origFuncAttr, convertedFuncAttr});
  }

  // Remap function attributes in the converted functions (they are not yet in
  // the module).  Original functions will disappear anyway so there is no
  // need to remap attributes in them.
  for (const auto &funcPair : functionAttrRemapping) {
    remapFunctionAttrs(*funcPair.getSecond().getValue(), functionAttrRemapping);
  }

  // Remove original functions from the module, then insert converted
  // functions.  The order is important to avoid name collisions.
  for (auto &func : originalFuncs)
    func->erase();
  for (auto *func : convertedFuncs)
    module->getFunctions().push_back(func);

  return false;
}

PassResult DialectConversion::runOnModule(Module *m) {
  return impl::FunctionConversion::convert(this, m) ? failure() : success();
}
