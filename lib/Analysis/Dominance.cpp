//===- Dominance.cpp - Dominator analysis for functions -------------------===//
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
#include "mlir/IR/Operation.h"
#include "llvm/Support/GenericDomTreeConstruction.h"
using namespace mlir;
using namespace mlir::detail;

template class llvm::DominatorTreeBase<Block, /*IsPostDom=*/false>;
template class llvm::DominatorTreeBase<Block, /*IsPostDom=*/true>;
template class llvm::DomTreeNodeBase<Block>;

//===----------------------------------------------------------------------===//
// DominanceInfoBase
//===----------------------------------------------------------------------===//

/// Recalculate the dominance info for the provided function.
template <bool IsPostDom>
void DominanceInfoBase<IsPostDom>::recalculate(Function *function) {
  dominanceInfos.clear();

  // Build the top level function dominance.
  auto functionDominance = llvm::make_unique<base>();
  functionDominance->recalculate(function->getBody());
  dominanceInfos.try_emplace(&function->getBody(),
                             std::move(functionDominance));

  /// Build the dominance for each of the operation regions.
  function->walk([&](Instruction *inst) {
    for (auto &region : inst->getRegions()) {
      // Don't compute dominance if the region is empty.
      if (region.empty())
        continue;
      auto opDominance = llvm::make_unique<base>();
      opDominance->recalculate(region);
      dominanceInfos.try_emplace(&region, std::move(opDominance));
    }
  });
}

/// Return true if the specified block A properly dominates block B.
template <bool IsPostDom>
bool DominanceInfoBase<IsPostDom>::properlyDominates(Block *a, Block *b) {
  // A block dominates itself but does not properly dominate itself.
  if (a == b)
    return false;

  // If both blocks are not in the same region, 'a' properly dominates 'b' if
  // 'b' is defined in an instruction region that (recursively) ends up being
  // dominated by 'a'. Walk up the list of containers enclosing B.
  auto *regionA = a->getParent(), *regionB = b->getParent();
  if (regionA != regionB) {
    Instruction *bAncestor;
    do {
      bAncestor = regionB->getContainingOp();
      // If 'bAncestor' is the top level function, then 'a' is a block
      // that post dominates 'b'.
      if (!bAncestor)
        return IsPostDom;

      regionB = bAncestor->getBlock()->getParent();
    } while (regionA != regionB);

    // Check to see if the ancestor of 'b' is the same block as 'a'.
    b = bAncestor->getBlock();
    if (a == b)
      return true;
  }

  // Otherwise, use the standard dominance functionality.
  auto baseInfoIt = dominanceInfos.find(regionA);
  assert(baseInfoIt != dominanceInfos.end() && "region info not found");
  return baseInfoIt->second->properlyDominates(a, b);
}

template class mlir::detail::DominanceInfoBase</*IsPostDom=*/true>;
template class mlir::detail::DominanceInfoBase</*IsPostDom=*/false>;

//===----------------------------------------------------------------------===//
// DominanceInfo
//===----------------------------------------------------------------------===//

/// Return true if instruction A properly dominates instruction B.
bool DominanceInfo::properlyDominates(Instruction *a, Instruction *b) {
  auto *aBlock = a->getBlock(), *bBlock = b->getBlock();

  // If the blocks are the same, then check if b is before a in the block.
  if (aBlock == bBlock)
    return a->isBeforeInBlock(b);

  // Traverse up b's hierarchy to check if b's block is contained in a's.
  if (auto *bAncestor = aBlock->findAncestorInstInBlock(*b)) {
    // Since we already know that aBlock != bBlock, here bAncestor != b.
    // a and bAncestor are in the same block; check if 'a' dominates
    // bAncestor.
    return dominates(a, bAncestor);
  }

  // If the blocks are different, check if a's block dominates b's.
  return properlyDominates(aBlock, bBlock);
}

/// Return true if value A properly dominates instruction B.
bool DominanceInfo::properlyDominates(Value *a, Instruction *b) {
  if (auto *aInst = a->getDefiningOp())
    return properlyDominates(aInst, b);

  // block arguments properly dominate all instructions in their own block, so
  // we use a dominates check here, not a properlyDominates check.
  return dominates(cast<BlockArgument>(a)->getOwner(), b->getBlock());
}

//===----------------------------------------------------------------------===//
// PostDominanceInfo
//===----------------------------------------------------------------------===//

/// Returns true if statement 'a' properly postdominates statement b.
bool PostDominanceInfo::properlyPostDominates(Instruction *a, Instruction *b) {
  auto *aBlock = a->getBlock(), *bBlock = b->getBlock();

  // If the blocks are the same, check if b is before a in the block.
  if (aBlock == bBlock)
    return b->isBeforeInBlock(a);

  // Traverse up b's hierarchy to check if b's block is contained in a's.
  if (auto *bAncestor = a->getBlock()->findAncestorInstInBlock(*b))
    // Since we already know that aBlock != bBlock, here bAncestor != b.
    // a and bAncestor are in the same block; check if 'a' postdominates
    // bAncestor.
    return postDominates(a, bAncestor);

  // If the blocks are different, check if a's block post dominates b's.
  return properlyDominates(aBlock, bBlock);
}
