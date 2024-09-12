/* Copyright 2023 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MLIR_HLO_DEALLOCATION_UTILS_UTIL_H_
#define MLIR_HLO_DEALLOCATION_UTILS_UTIL_H_

#include <optional>
#include <set>

#include "llvm/ADT/EquivalenceClasses.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

namespace mlir {
namespace deallocation {

struct RegionEdge {
  // The op in the predecessor that holds the values that are passed. This is
  // either the parent op or a terminator in the predecessor region.
  Operation* predecessorOp;
  // The index in `opWithOperands`' operands where `operands` start.
  int64_t predecessorOperandIndex;
  // The op or region where the values are passed. If the successor is the
  // parent region, this is the parent op.
  llvm::PointerUnion<Operation*, Region*> successorOpOrRegion;
  // The index in the successor's arguments or op results where `operands`
  // start.
  int64_t successorValueIndex;
  RegionBranchPoint predecessorRegionPoint = RegionBranchPoint::parent();
  RegionBranchPoint successorRegionPoint = RegionBranchPoint::parent();

  ValueRange getPredecessorOperands() const {
    return predecessorOp->getOperands().drop_front(predecessorOperandIndex);
  }

  Value getPredecessorOperand(unsigned successorIndex) const {
    return getPredecessorOperands()[successorIndex - successorValueIndex];
  }

  ValueRange getSuccessorValues() const {
    if (successorOpOrRegion.is<Operation*>()) {
      return successorOpOrRegion.get<Operation*>()->getResults().drop_front(
          successorValueIndex);
    }
    return successorOpOrRegion.get<Region*>()->getArguments().drop_front(
        successorValueIndex);
  }

  Value getSuccessorValue(unsigned predecessorIndex) const {
    return getSuccessorValues()[predecessorIndex - predecessorOperandIndex];
  }
};

// Returns predecessors of the given region.
SmallVector<RegionEdge> getPredecessorRegions(RegionBranchOpInterface op,
                                              RegionBranchPoint index);

SmallVector<RegionEdge> getSuccessorRegions(RegionBranchOpInterface op,
                                            RegionBranchPoint index);

// Replaces the op with a new op with proper return types. The old op is not
// removed and it still has uses.
RegionBranchOpInterface moveRegionsToNewOpButKeepOldOp(
    RegionBranchOpInterface op);

namespace detail {
// An arbitrary deterministic Value order.
struct ValueComparator {
  bool operator()(const Value& lhs, const Value& rhs) const {
    if (lhs == rhs) return false;

    // Block arguments are less than results.
    bool lhsIsBBArg = isa<BlockArgument>(lhs);
    if (lhsIsBBArg != isa<BlockArgument>(rhs)) {
      return lhsIsBBArg;
    }

    Region* lhsRegion;
    Region* rhsRegion;
    if (lhsIsBBArg) {
      auto lhsBBArg = llvm::cast<BlockArgument>(lhs);
      auto rhsBBArg = llvm::cast<BlockArgument>(rhs);
      if (lhsBBArg.getArgNumber() != rhsBBArg.getArgNumber()) {
        return lhsBBArg.getArgNumber() < rhsBBArg.getArgNumber();
      }
      lhsRegion = lhsBBArg.getParentRegion();
      rhsRegion = rhsBBArg.getParentRegion();
      assert(lhsRegion != rhsRegion &&
             "lhsRegion == rhsRegion implies lhs == rhs");
    } else if (lhs.getDefiningOp() == rhs.getDefiningOp()) {
      return llvm::cast<OpResult>(lhs).getResultNumber() <
             llvm::cast<OpResult>(rhs).getResultNumber();
    } else {
      lhsRegion = lhs.getDefiningOp()->getParentRegion();
      rhsRegion = rhs.getDefiningOp()->getParentRegion();
      if (lhsRegion == rhsRegion) {
        return lhs.getDefiningOp()->isBeforeInBlock(rhs.getDefiningOp());
      }
    }

    // lhsRegion != rhsRegion, so if we look at their ancestor chain, they
    // - have different heights
    // - or there's a spot where their region numbers differ
    // - or their parent regions are the same and their parent ops are
    //   different.
    while (lhsRegion && rhsRegion) {
      if (lhsRegion->getRegionNumber() != rhsRegion->getRegionNumber()) {
        return lhsRegion->getRegionNumber() < rhsRegion->getRegionNumber();
      }
      if (lhsRegion->getParentRegion() == rhsRegion->getParentRegion()) {
        return lhsRegion->getParentOp()->isBeforeInBlock(
            rhsRegion->getParentOp());
      }
      lhsRegion = lhsRegion->getParentRegion();
      rhsRegion = rhsRegion->getParentRegion();
    }
    if (rhsRegion) return true;
    assert(lhsRegion && "this should only happen if lhs == rhs");
    return false;
  }
};
}  // namespace detail

namespace breaks_if_you_move_ops {

// The comparator depends on the location of ops, so if you insert an op into
// a set and then move it, it may end up in the wrong location.
using ValueEquivalenceClasses =
    llvm::EquivalenceClasses<Value, detail::ValueComparator>;
using ValueSet = std::set<Value, detail::ValueComparator>;
template <typename T>
using ValueMap = std::map<Value, T, detail::ValueComparator>;

}  // namespace breaks_if_you_move_ops

}  // namespace deallocation
}  // namespace mlir

#endif  // MLIR_HLO_DEALLOCATION_UTILS_UTIL_H_
