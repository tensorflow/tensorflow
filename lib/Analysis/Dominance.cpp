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
#include "mlir/IR/Statements.h"
#include "llvm/Support/GenericDomTreeConstruction.h"
using namespace mlir;

template class llvm::DominatorTreeBase<BasicBlock, false>;
template class llvm::DominatorTreeBase<BasicBlock, true>;
template class llvm::DomTreeNodeBase<BasicBlock>;

/// Compute the immediate-dominators map.
DominanceInfo::DominanceInfo(Function *function) : DominatorTreeBase() {
  // Build the dominator tree for the function.
  recalculate(function->getBlockList());
}

/// Return true if instruction A properly dominates instruction B.
bool DominanceInfo::properlyDominates(const Instruction *a,
                                      const Instruction *b) {
  auto *aBlock = a->getBlock(), *bBlock = b->getBlock();

  // If the blocks are different, it's as easy as whether A's block
  // dominates B's block.
  if (aBlock != bBlock)
    return properlyDominates(a->getBlock(), b->getBlock());

  // If a/b are the same, then they don't properly dominate each other.
  if (a == b)
    return false;

  // If one is a terminator, then the other dominates it.
  if (a->isTerminator())
    return false;

  if (b->isTerminator())
    return true;

  // Otherwise, do a linear scan to determine whether B comes after A.
  auto aIter = BasicBlock::const_iterator(a);
  auto bIter = BasicBlock::const_iterator(b);
  auto fIter = aBlock->begin();
  while (bIter != fIter) {
    --bIter;
    if (aIter == bIter)
      return true;
  }

  return false;
}

/// Return true if value A properly dominates instruction B.
bool DominanceInfo::properlyDominates(const Value *a, const Instruction *b) {
  if (auto *aInst = a->getDefiningInst())
    return properlyDominates(aInst, b);

  // bbarguments properly dominate all instructions in their own block, so we
  // use a dominates check here, not a properlyDominates check.
  return dominates(cast<BlockArgument>(a)->getOwner(), b->getBlock());
}
