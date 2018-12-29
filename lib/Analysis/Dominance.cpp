//===- Dominance.cpp - Dominator analysis for CFG Functions ---------------===//
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
// Implementation of dominance related classes and instantiations of extern
// templates.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Dominance.h"
#include "mlir/IR/Instructions.h"
#include "llvm/Support/GenericDomTreeConstruction.h"
using namespace mlir;

template class llvm::DominatorTreeBase<Block, false>;
template class llvm::DominatorTreeBase<Block, true>;
template class llvm::DomTreeNodeBase<Block>;

/// Compute the immediate-dominators map.
DominanceInfo::DominanceInfo(Function *function) : DominatorTreeBase() {
  // Build the dominator tree for the function.
  recalculate(function->getBlockList());
}

bool DominanceInfo::properlyDominates(const Block *a, const Block *b) {
  // A block dominates itself but does not properly dominate itself.
  if (a == b)
    return false;

  // If both blocks are in the same block list, then standard dominator
  // information can resolve the query.
  auto *blockListA = a->getParent(), *blockListB = b->getParent();
  if (blockListA == blockListB)
    return DominatorTreeBase::properlyDominates(a, b);

  // Otherwise, 'a' dominates 'b' if 'b' is defined in an IfInst/ForInst that
  // (recursively) ends up being dominated by 'a'.  Walk up the list of
  // containers enclosing B.
  while (blockListA != blockListB) {
    // If 'b' is at a the top level function, then 'a' is defined inside some
    // other instruction that doesn't dominate 'b'.
    auto *containerInst = blockListB->getContainingInst();
    if (!containerInst)
      return false;

    blockListB = containerInst->getBlock()->getParent();
  }

  // Block 'A' is an ancestor of 'B', we know that A dominates B.
  return true;
}

/// Return true if instruction A properly dominates instruction B.
bool DominanceInfo::properlyDominates(const Instruction *a,
                                      const Instruction *b) {
  auto *aBlock = a->getBlock();
  auto *bBlock = b->getBlock();

  // If the blocks are the same, then we do a linear scan.
  if (aBlock == bBlock) {
    // If a/b are the same, then they don't properly dominate each other.
    if (a == b)
      return false;

    // If one is a terminator, then the other dominates it.
    if (a->isTerminator())
      return false;

    if (b->isTerminator())
      return true;

    // Otherwise, do a linear scan to determine whether B comes after A.
    // TODO: This is an O(n) scan that can be bad for very large blocks.
    auto aIter = Block::const_iterator(a);
    auto bIter = Block::const_iterator(b);
    auto fIter = aBlock->begin();
    while (bIter != fIter) {
      --bIter;
      if (aIter == bIter)
        return true;
    }
    return false;
  }

  // If the blocks are different, but in the same function-level block list,
  // then a standard block dominance query is sufficient.
  if (aBlock->getParent()->getContainingFunction() &&
      bBlock->getParent()->getContainingFunction())
    return DominatorTreeBase::properlyDominates(aBlock, bBlock);

  // Traverse up b's hierarchy to check if b's block is contained in a's.
  if (auto *bAncestor = aBlock->findAncestorInstInBlock(*b)) {
    // a and bAncestor are in the same block; check if 'a' dominates
    // bAncestor.
    return properlyDominates(a, bAncestor);
  }

  return false;
}

/// Return true if value A properly dominates instruction B.
bool DominanceInfo::properlyDominates(const Value *a, const Instruction *b) {
  if (auto *aInst = a->getDefiningInst())
    return properlyDominates(aInst, b);

  // The induction variable of a ForInst properly dominantes its body, so we
  // can just do a simple block dominance check.
  if (auto *forInst = dyn_cast<ForInst>(a))
    return dominates(forInst->getBody(), b->getBlock());

  // block arguments properly dominate all instructions in their own block, so
  // we use a dominates check here, not a properlyDominates check.
  return dominates(cast<BlockArgument>(a)->getOwner(), b->getBlock());
}
