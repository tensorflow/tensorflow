//===- CSE.cpp - Common Sub-expression Elimination ------------------------===//
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
// This transformation pass performs a simple common sub-expression elimination
// algorithm on operations within a function.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Dominance.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/StmtVisitor.h"
#include "mlir/Pass.h"
#include "mlir/Support/Functional.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/RecyclingAllocator.h"
#include <deque>

using namespace mlir;

namespace {
/// Simple common sub-expression elimination.
struct CSE : public FunctionPass {
  CSE() : FunctionPass(&CSE::passID) {}

  PassResult runOnCFGFunction(CFGFunction *f) override;
  PassResult runOnMLFunction(MLFunction *f) override;

  static char passID;
};

// TODO(riverriddle) Handle commutative operations.
struct SimpleOperationInfo : public llvm::DenseMapInfo<OperationInst *> {
  static unsigned getHashValue(const OperationInst *op) {
    // Hash the operations based upon their:
    //   - OperationInst Name
    //   - Attributes
    //   - Result Types
    //   - Operands
    return hash_combine(
        op->getName(), op->getAttrs(),
        hash_combine_range(op->result_type_begin(), op->result_type_end()),
        hash_combine_range(op->operand_begin(), op->operand_end()));
  }
  static bool isEqual(const OperationInst *lhs, const OperationInst *rhs) {
    if (lhs == rhs)
      return true;
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;

    // Compare the operation name.
    if (lhs->getName() != rhs->getName())
      return false;
    // Check operand and result type counts.
    if (lhs->getNumOperands() != rhs->getNumOperands() ||
        lhs->getNumResults() != rhs->getNumResults())
      return false;
    // Compare attributes.
    if (lhs->getAttrs() != rhs->getAttrs())
      return false;
    // Compare operands.
    if (!std::equal(lhs->operand_begin(), lhs->operand_end(),
                    rhs->operand_begin()))
      return false;
    // Compare result types.
    return std::equal(lhs->result_type_begin(), lhs->result_type_end(),
                      rhs->result_type_begin());
  }
};

/// Shared implementation of operation elimination and scoped map definitions.
struct CSEImpl {
  using AllocatorTy = llvm::RecyclingAllocator<
      llvm::BumpPtrAllocator,
      llvm::ScopedHashTableVal<OperationInst *, OperationInst *>>;
  using ScopedMapTy = llvm::ScopedHashTable<OperationInst *, OperationInst *,
                                            SimpleOperationInfo, AllocatorTy>;

  /// Erase any operations that were marked as dead during simplification.
  void eraseDeadOperations() {
    for (auto *op : opsToErase)
      op->erase();
  }

  /// Attempt to eliminate a redundant operation.
  void simplifyOperation(OperationInst *op) {
    // TODO(riverriddle) We currently only eliminate non side-effecting
    // operations.
    if (!op->hasNoSideEffect())
      return;

    // If the operation is already trivially dead just add it to the erase list.
    if (op->use_empty()) {
      opsToErase.push_back(op);
      return;
    }

    // Look for an existing definition for the operation.
    if (auto *existing = knownValues.lookup(op)) {
      // If we find one then replace all uses of the current operation with the
      // existing one and mark it for deletion.
      for (unsigned i = 0, e = existing->getNumResults(); i != e; ++i)
        op->getResult(i)->replaceAllUsesWith(existing->getResult(i));
      opsToErase.push_back(op);

      // If the existing operation has an unknown location and the current
      // operation doesn't, then set the existing op's location to that of the
      // current op.
      if (existing->getLoc().isa<UnknownLoc>() &&
          !op->getLoc().isa<UnknownLoc>()) {
        existing->setLoc(op->getLoc());
      }
    } else {
      // Otherwise, we add this operation to the known values map.
      knownValues.insert(op, op);
    }
  }

  /// A scoped hash table of defining operations within a function.
  ScopedMapTy knownValues;

  /// Operations marked as dead and to be erased.
  std::vector<OperationInst *> opsToErase;
};

/// Common sub-expression elimination for CFG functions.
struct CFGCSE : public CSEImpl {
  /// Represents a single entry in the depth first traversal of a CFG.
  struct CFGStackNode {
    CFGStackNode(ScopedMapTy &knownValues, DominanceInfoNode *node)
        : scope(knownValues), node(node), childIterator(node->begin()),
          processed(false) {}

    /// Scope for the known values.
    ScopedMapTy::ScopeTy scope;

    DominanceInfoNode *node;
    DominanceInfoNode::iterator childIterator;

    /// If this node has been fully processed yet or not.
    bool processed;
  };

  void run(CFGFunction *f) {
    // Note, deque is being used here because there was significant performance
    // gains over vector when the container becomes very large due to the
    // specific access patterns. If/when these performance issues are no
    // longer a problem we can change this to vector. For more information see
    // the llvm mailing list discussion on this:
    // http://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20120116/135228.html
    std::deque<std::unique_ptr<CFGStackNode>> stack;

    // Process the nodes of the dom tree.
    DominanceInfo domInfo(f);
    stack.emplace_back(
        std::make_unique<CFGStackNode>(knownValues, domInfo.getRootNode()));

    while (!stack.empty()) {
      auto &currentNode = stack.back();

      // Check to see if we need to process this node.
      if (!currentNode->processed) {
        currentNode->processed = true;
        simplifyBasicBlock(currentNode->node->getBlock());
        // Otherwise, check to see if we need to process a child node.
      } else if (currentNode->childIterator != currentNode->node->end()) {
        auto *childNode = *(currentNode->childIterator++);
        stack.emplace_back(
            std::make_unique<CFGStackNode>(knownValues, childNode));
      } else {
        // Finally, if the node and all of its children have been processed
        // then we delete the node.
        stack.pop_back();
      }
    }

    // Erase any operations marked as redundant.
    eraseDeadOperations();
  }

  void simplifyBasicBlock(BasicBlock *bb) {
    for (auto &i : *bb)
      if (auto *opInst = dyn_cast<OperationInst>(&i))
        simplifyOperation(opInst);
  }
};

/// Common sub-expression elimination for ML functions.
struct MLCSE : public CSEImpl, StmtWalker<MLCSE> {
  using StmtWalker<MLCSE>::walk;

  void run(MLFunction *f) {
    // Walk the function statements.
    walk(f);

    // Finally, erase any redundant operations.
    eraseDeadOperations();
  }

  // Insert a scope for each statement range.
  template <class Iterator> void walk(Iterator Start, Iterator End) {
    ScopedMapTy::ScopeTy scope(knownValues);
    StmtWalker<MLCSE>::walk(Start, End);
  }

  void visitOperationInst(OperationInst *stmt) { simplifyOperation(stmt); }
};

} // end anonymous namespace

char CSE::passID = 0;

PassResult CSE::runOnCFGFunction(CFGFunction *f) {
  CFGCSE().run(f);
  return success();
}

PassResult CSE::runOnMLFunction(MLFunction *f) {
  MLCSE().run(f);
  return success();
}

FunctionPass *mlir::createCSEPass() { return new CSE(); }

static PassRegistration<CSE>
    pass("cse", "Eliminate common sub-expressions in functions");
