//===- Canonicalizer.cpp - Canonicalize MLIR operations -------------------===//
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
// This transformation pass converts operations into their canonical forms by
// folding constants, applying operation identity transformations etc.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/StandardOps/StandardOps.h"
#include "mlir/Transforms/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
using namespace mlir;

//===----------------------------------------------------------------------===//
// Definition of a few patterns for canonicalizing operations.
//===----------------------------------------------------------------------===//

namespace {
/// This is a common class used for patterns of the form
/// "someop(memrefcast) -> someop".  It folds the source of any memref_cast
/// into the root operation directly.
struct MemRefCastFolder : public Pattern {
  /// The rootOpName is the name of the root operation to match against.
  MemRefCastFolder(StringRef rootOpName, MLIRContext *context)
      : Pattern(rootOpName, context, 1) {}

  std::pair<PatternBenefit, std::unique_ptr<PatternState>>
  match(Operation *op) const override {
    for (auto *operand : op->getOperands())
      if (auto *memref = operand->getDefiningOperation())
        if (memref->isa<MemRefCastOp>())
          return matchSuccess();

    return matchFailure();
  }

  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    for (unsigned i = 0, e = op->getNumOperands(); i != e; ++i)
      if (auto *memref = op->getOperand(i)->getDefiningOperation())
        if (auto cast = memref->dyn_cast<MemRefCastOp>())
          op->setOperand(i, cast->getOperand());
    rewriter.updatedRootInPlace(op);
  }
};
} // end anonymous namespace.

namespace {
/// subi(x,x) -> 0
///
struct SimplifyXMinusX : public Pattern {
  SimplifyXMinusX(MLIRContext *context)
      : Pattern(SubIOp::getOperationName(), context, 1) {}

  std::pair<PatternBenefit, std::unique_ptr<PatternState>>
  match(Operation *op) const override {
    auto subi = op->cast<SubIOp>();
    if (subi->getOperand(0) == subi->getOperand(1))
      return matchSuccess();

    return matchFailure();
  }
  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    auto subi = op->cast<SubIOp>();
    auto result =
        rewriter.create<ConstantIntOp>(op->getLoc(), 0, subi->getType());

    rewriter.replaceSingleResultOp(op, result);
  }
};
} // end anonymous namespace.

namespace {
/// addi(x, 0) -> x
///
struct SimplifyAddX0 : public Pattern {
  SimplifyAddX0(MLIRContext *context)
      : Pattern(AddIOp::getOperationName(), context, 1) {}

  std::pair<PatternBenefit, std::unique_ptr<PatternState>>
  match(Operation *op) const override {
    auto addi = op->cast<AddIOp>();
    if (auto *operandOp = addi->getOperand(1)->getDefiningOperation())
      // TODO: Support splatted zero as well.  We need a general zero pattern.
      if (auto cst = operandOp->dyn_cast<ConstantIntOp>()) {
        if (cst->getValue() == 0)
          return matchSuccess();
      }

    return matchFailure();
  }
  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    rewriter.replaceSingleResultOp(op, op->getOperand(0));
  }
};
} // end anonymous namespace.

namespace {
/// Fold constant dimensions into an alloc instruction.
struct SimplifyAllocConst : public Pattern {
  SimplifyAllocConst(MLIRContext *context)
      : Pattern(AllocOp::getOperationName(), context, 1) {}

  std::pair<PatternBenefit, std::unique_ptr<PatternState>>
  match(Operation *op) const override {
    auto alloc = op->cast<AllocOp>();

    // Check to see if any dimensions operands are constants.  If so, we can
    // substitute and drop them.
    for (auto *operand : alloc->getOperands())
      if (auto *opOperation = operand->getDefiningOperation())
        if (opOperation->isa<ConstantIndexOp>())
          return matchSuccess();
    return matchFailure();
  }

  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    auto allocOp = op->cast<AllocOp>();
    auto memrefType = allocOp->getType();

    // Ok, we have one or more constant operands.  Collect the non-constant ones
    // and keep track of the resultant memref type to build.
    SmallVector<int, 4> newShapeConstants;
    newShapeConstants.reserve(memrefType->getRank());
    SmallVector<SSAValue *, 4> newOperands;
    SmallVector<SSAValue *, 4> droppedOperands;

    unsigned dynamicDimPos = 0;
    for (unsigned dim = 0, e = memrefType->getRank(); dim < e; ++dim) {
      int dimSize = memrefType->getDimSize(dim);
      // If this is already static dimension, keep it.
      if (dimSize != -1) {
        newShapeConstants.push_back(dimSize);
        continue;
      }
      auto *defOp = allocOp->getOperand(dynamicDimPos)->getDefiningOperation();
      OpPointer<ConstantIndexOp> constantIndexOp;
      if (defOp && (constantIndexOp = defOp->dyn_cast<ConstantIndexOp>())) {
        // Dynamic shape dimension will be folded.
        newShapeConstants.push_back(constantIndexOp->getValue());
        // Record to check for zero uses later below.
        droppedOperands.push_back(constantIndexOp);
      } else {
        // Dynamic shape dimension not folded; copy operand from old memref.
        newShapeConstants.push_back(-1);
        newOperands.push_back(allocOp->getOperand(dynamicDimPos));
      }
      dynamicDimPos++;
    }

    // Create new memref type (which will have fewer dynamic dimensions).
    auto *newMemRefType = MemRefType::get(
        newShapeConstants, memrefType->getElementType(),
        memrefType->getAffineMaps(), memrefType->getMemorySpace());
    assert(newOperands.size() == newMemRefType->getNumDynamicDims());

    // Create and insert the alloc op for the new memref.
    auto newAlloc =
        rewriter.create<AllocOp>(allocOp->getLoc(), newMemRefType, newOperands);
    // Insert a cast so we have the same type as the old alloc.
    auto resultCast = rewriter.create<MemRefCastOp>(allocOp->getLoc(), newAlloc,
                                                    allocOp->getType());

    rewriter.replaceSingleResultOp(op, resultCast, droppedOperands);
  }
};
} // end anonymous namespace.

//===----------------------------------------------------------------------===//
// The actual Canonicalizer Pass.
//===----------------------------------------------------------------------===//

namespace {
class CanonicalizerRewriter;

/// Canonicalize operations in functions.
struct Canonicalizer : public FunctionPass {
  PassResult runOnCFGFunction(CFGFunction *f) override;
  PassResult runOnMLFunction(MLFunction *f) override;

  void simplifyFunction(Function *currentFunction,
                        CanonicalizerRewriter &rewriter);

  void addToWorklist(Operation *op) {
    worklistMap[op] = worklist.size();
    worklist.push_back(op);
  }

  Operation *popFromWorklist() {
    auto *op = worklist.back();
    worklist.pop_back();

    // This operation is no longer in the worklist, keep worklistMap up to date.
    if (op)
      worklistMap.erase(op);
    return op;
  }

  /// If the specified operation is in the worklist, remove it.  If not, this is
  /// a no-op.
  void removeFromWorklist(Operation *op) {
    auto it = worklistMap.find(op);
    if (it != worklistMap.end()) {
      assert(worklist[it->second] == op && "malformed worklist data structure");
      worklist[it->second] = nullptr;
    }
  }

private:
  /// The worklist for this transformation keeps track of the operations that
  /// need to be revisited, plus their index in the worklist.  This allows us to
  /// efficiently remove operations from the worklist when they are removed even
  /// if they aren't the root of a pattern.
  std::vector<Operation *> worklist;
  DenseMap<Operation *, unsigned> worklistMap;

  /// As part of canonicalization, we move constants to the top of the entry
  /// block of the current function and de-duplicate them.  This keeps track of
  /// constants we have done this for.
  DenseMap<std::pair<Attribute *, Type *>, Operation *> uniquedConstants;
};
} // end anonymous namespace

namespace {
class CanonicalizerRewriter : public PatternRewriter {
public:
  CanonicalizerRewriter(Canonicalizer &thePass, MLIRContext *context)
      : PatternRewriter(context), thePass(thePass) {}

  virtual void setInsertionPoint(Operation *op) = 0;

  // If an operation is about to be removed, make sure it is not in our
  // worklist anymore because we'd get dangling references to it.
  void notifyOperationRemoved(Operation *op) override {
    thePass.removeFromWorklist(op);
  }

  Canonicalizer &thePass;
};

} // end anonymous namespace

PassResult Canonicalizer::runOnCFGFunction(CFGFunction *fn) {
  worklist.reserve(64);
  for (auto &bb : *fn)
    for (auto &op : bb)
      addToWorklist(&op);

  class CFGFuncRewriter : public CanonicalizerRewriter {
  public:
    CFGFuncRewriter(Canonicalizer &thePass, CFGFuncBuilder &builder)
        : CanonicalizerRewriter(thePass, builder.getContext()),
          builder(builder) {}

    // Implement the hook for creating operations, and make sure that newly
    // created ops are added to the worklist for processing.
    Operation *createOperation(const OperationState &state) override {
      auto *result = builder.createOperation(state);
      thePass.addToWorklist(result);
      return result;
    }

    // When the root of a pattern is about to be replaced, it can trigger
    // simplifications to its users - make sure to add them to the worklist
    // before the root is changed.
    void notifyRootReplaced(Operation *op) override {
      auto *opStmt = cast<OperationInst>(op);
      for (auto *result : opStmt->getResults())
        // TODO: Add a result->getUsers() iterator.
        for (auto &user : result->getUses()) {
          if (auto *op = dyn_cast<OperationInst>(user.getOwner()))
            thePass.addToWorklist(op);
        }

      // TODO: Walk the operand list dropping them as we go.  If any of them
      // drop to zero uses, then add them to the worklist to allow them to be
      // deleted as dead.
    }

    void setInsertionPoint(Operation *op) override {
      // Any new operations should be added before this instruction.
      builder.setInsertionPoint(cast<OperationInst>(op));
    }

  private:
    CFGFuncBuilder &builder;
  };

  CFGFuncBuilder cfgBuilder(fn);
  CFGFuncRewriter rewriter(*this, cfgBuilder);
  simplifyFunction(fn, rewriter);
  return success();
}

PassResult Canonicalizer::runOnMLFunction(MLFunction *fn) {
  worklist.reserve(64);

  fn->walk([&](OperationStmt *stmt) { addToWorklist(stmt); });

  class MLFuncRewriter : public CanonicalizerRewriter {
  public:
    MLFuncRewriter(Canonicalizer &thePass, MLFuncBuilder &builder)
        : CanonicalizerRewriter(thePass, builder.getContext()),
          builder(builder) {}

    // Implement the hook for creating operations, and make sure that newly
    // created ops are added to the worklist for processing.
    Operation *createOperation(const OperationState &state) override {
      auto *result = builder.createOperation(state);
      thePass.addToWorklist(result);
      return result;
    }

    // When the root of a pattern is about to be replaced, it can trigger
    // simplifications to its users - make sure to add them to the worklist
    // before the root is changed.
    void notifyRootReplaced(Operation *op) override {
      auto *opStmt = cast<OperationStmt>(op);
      for (auto *result : opStmt->getResults())
        // TODO: Add a result->getUsers() iterator.
        for (auto &user : result->getUses()) {
          if (auto *op = dyn_cast<OperationStmt>(user.getOwner()))
            thePass.addToWorklist(op);
        }

      // TODO: Walk the operand list dropping them as we go.  If any of them
      // drop to zero uses, then add them to the worklist to allow them to be
      // deleted as dead.
    }

    void setInsertionPoint(Operation *op) override {
      // Any new operations should be added before this statement.
      builder.setInsertionPoint(cast<OperationStmt>(op));
    }

  private:
    MLFuncBuilder &builder;
  };

  MLFuncBuilder mlBuilder(fn);
  MLFuncRewriter rewriter(*this, mlBuilder);
  simplifyFunction(fn, rewriter);
  return success();
}

void Canonicalizer::simplifyFunction(Function *currentFunction,
                                     CanonicalizerRewriter &rewriter) {
  auto *context = rewriter.getContext();

  // TODO: Instead of a hard coded list of patterns, ask the registered dialects
  // for their canonicalization patterns.
  Pattern *patterns[] = {
      new SimplifyXMinusX(context), new SimplifyAddX0(context),
      new SimplifyAllocConst(context),
      /// load(memrefcast) -> load
      new MemRefCastFolder(LoadOp::getOperationName(), context),
      /// store(memrefcast) -> store
      new MemRefCastFolder(StoreOp::getOperationName(), context),
      /// dealloc(memrefcast) -> dealloc
      new MemRefCastFolder(DeallocOp::getOperationName(), context),
      /// dma_start(memrefcast) -> dma_start
      new MemRefCastFolder(DmaStartOp::getOperationName(), context),
      /// dma_wait(memrefcast) -> dma_wait
      new MemRefCastFolder(DmaWaitOp::getOperationName(), context)};
  PatternMatcher matcher(patterns);

  // These are scratch vectors used in the constant folding loop below.
  SmallVector<Attribute *, 8> operandConstants, resultConstants;

  while (!worklist.empty()) {
    auto *op = popFromWorklist();

    // Nulls get added to the worklist when operations are removed, ignore them.
    if (op == nullptr)
      continue;

    // If we have a constant op, unique it into the entry block.
    if (auto constant = op->dyn_cast<ConstantOp>()) {
      // If this constant is dead, remove it, being careful to keep
      // uniquedConstants up to date.
      if (constant->use_empty()) {
        auto it =
            uniquedConstants.find({constant->getValue(), constant->getType()});
        if (it != uniquedConstants.end() && it->second == op)
          uniquedConstants.erase(it);
        constant->erase();
        continue;
      }

      // Check to see if we already have a constant with this type and value:
      auto &entry = uniquedConstants[std::make_pair(constant->getValue(),
                                                    constant->getType())];
      if (entry) {
        // If this constant is already our uniqued one, then leave it alone.
        if (entry == op)
          continue;

        // Otherwise replace this redundant constant with the uniqued one.  We
        // know this is safe because we move constants to the top of the
        // function when they are uniqued, so we know they dominate all uses.
        constant->replaceAllUsesWith(entry->getResult(0));
        constant->erase();
        continue;
      }

      // If we have no entry, then we should unique this constant as the
      // canonical version.  To ensure safe dominance, move the operation to the
      // top of the function.
      entry = op;

      if (auto *cfgFunc = dyn_cast<CFGFunction>(currentFunction)) {
        auto &entryBB = cfgFunc->front();
        cast<OperationInst>(op)->moveBefore(&entryBB, entryBB.begin());
      } else {
        auto *mlFunc = cast<MLFunction>(currentFunction);
        cast<OperationStmt>(op)->moveBefore(mlFunc, mlFunc->begin());
      }

      continue;
    }

    // If the operation has no side effects, and no users, then it is trivially
    // dead - remove it.
    if (op->hasNoSideEffect() && op->use_empty()) {
      op->erase();
      continue;
    }

    // Check to see if any operands to the instruction is constant and whether
    // the operation knows how to constant fold itself.
    operandConstants.clear();
    for (auto *operand : op->getOperands()) {
      Attribute *operandCst = nullptr;
      if (auto *operandOp = operand->getDefiningOperation()) {
        if (auto operandConstantOp = operandOp->dyn_cast<ConstantOp>())
          operandCst = operandConstantOp->getValue();
      }
      operandConstants.push_back(operandCst);
    }

    // If constant folding was successful, create the result constants, RAUW the
    // operation and remove it.
    resultConstants.clear();
    if (!op->constantFold(operandConstants, resultConstants)) {
      rewriter.setInsertionPoint(op);

      for (unsigned i = 0, e = op->getNumResults(); i != e; ++i) {
        auto *res = op->getResult(i);
        if (res->use_empty()) // ignore dead uses.
          continue;

        // If we already have a canonicalized version of this constant, just
        // reuse it.  Otherwise create a new one.
        SSAValue *cstValue;
        auto it = uniquedConstants.find({resultConstants[i], res->getType()});
        if (it != uniquedConstants.end())
          cstValue = it->second->getResult(0);
        else
          cstValue = rewriter.create<ConstantOp>(
              op->getLoc(), resultConstants[i], res->getType());
        res->replaceAllUsesWith(cstValue);
      }

      assert(op->hasNoSideEffect() && "Constant folded op with side effects?");
      op->erase();
      continue;
    }

    // If this is an associative binary operation with a constant on the LHS,
    // move it to the right side.
    if (operandConstants.size() == 2 && operandConstants[0] &&
        !operandConstants[1]) {
      auto *newLHS = op->getOperand(1);
      op->setOperand(1, op->getOperand(0));
      op->setOperand(0, newLHS);
    }

    // Check to see if we have any patterns that match this node.
    auto match = matcher.findMatch(op);
    if (!match.first)
      continue;

    // Make sure that any new operations are inserted at this point.
    rewriter.setInsertionPoint(op);
    match.first->rewrite(op, std::move(match.second), rewriter);
  }

  uniquedConstants.clear();
}

/// Create a Canonicalizer pass.
FunctionPass *mlir::createCanonicalizerPass() { return new Canonicalizer(); }
