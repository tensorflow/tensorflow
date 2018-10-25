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
#include <memory>
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

/// Canonicalize operations in functions.
struct Canonicalizer : public FunctionPass {
  PassResult runOnCFGFunction(CFGFunction *f) override;
  PassResult runOnMLFunction(MLFunction *f) override;
  PassResult runOnFunction(Function *fn);
};
} // end anonymous namespace


PassResult Canonicalizer::runOnCFGFunction(CFGFunction *fn) {
  return runOnFunction(fn);
}

PassResult Canonicalizer::runOnMLFunction(MLFunction *fn) {
  return runOnFunction(fn);
}

PassResult Canonicalizer::runOnFunction(Function *fn) {
  auto *context = fn->getContext();

  // TODO: Instead of a hard coded list of patterns, ask the operations
  // for their canonicalization patterns.
  OwningPatternList patterns;

  patterns.push_back(std::make_unique<SimplifyXMinusX>(context));
  patterns.push_back(std::make_unique<SimplifyAddX0>(context));
  patterns.push_back(std::make_unique<SimplifyAllocConst>(context));
  /// load(memrefcast) -> load
  patterns.push_back(
      std::make_unique<MemRefCastFolder>(LoadOp::getOperationName(), context));
  /// store(memrefcast) -> store
  patterns.push_back(
      std::make_unique<MemRefCastFolder>(StoreOp::getOperationName(), context));
  /// dealloc(memrefcast) -> dealloc
  patterns.push_back(std::make_unique<MemRefCastFolder>(
      DeallocOp::getOperationName(), context));
  /// dma_start(memrefcast) -> dma_start
  patterns.push_back(std::make_unique<MemRefCastFolder>(
      DmaStartOp::getOperationName(), context));
  /// dma_wait(memrefcast) -> dma_wait
  patterns.push_back(std::make_unique<MemRefCastFolder>(
      DmaWaitOp::getOperationName(), context));

  applyPatternsGreedily(fn, std::move(patterns));
  return success();
}

/// Create a Canonicalizer pass.
FunctionPass *mlir::createCanonicalizerPass() { return new Canonicalizer(); }
