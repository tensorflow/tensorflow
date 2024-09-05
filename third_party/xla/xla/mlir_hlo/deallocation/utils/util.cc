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

#include "deallocation/utils/util.h"

#include <optional>

#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
namespace deallocation {

SmallVector<RegionEdge> getSuccessorRegions(RegionBranchOpInterface op,
                                            RegionBranchPoint point) {
  SmallVector<RegionEdge> edges;
  if (Region* region = point.getRegionOrNull()) {
    if (region->empty()) {
      return edges;
    }
  }

  SmallVector<RegionSuccessor> successors;
  op.getSuccessorRegions(point, successors);

  for (const auto& successor : successors) {
    auto& edge = edges.emplace_back();
    edge.predecessorRegionPoint = point;
    auto* region = point.getRegionOrNull();
    edge.predecessorOp =
        region ? region->front().getTerminator() : op.getOperation();
    edge.predecessorOperandIndex = edge.predecessorOp->getNumOperands() -
                                   successor.getSuccessorInputs().size();

    if (successor.isParent()) {
      edge.successorRegionPoint = point.parent();
      edge.successorOpOrRegion = op.getOperation();
      edge.successorValueIndex = 0;
    } else {
      edge.successorRegionPoint = successor.getSuccessor();
      edge.successorOpOrRegion = successor.getSuccessor();
      edge.successorValueIndex = llvm::isa<scf::ForOp>(op) ? 1 : 0;
    }
  }

  return edges;
}

SmallVector<RegionEdge> getPredecessorRegions(RegionBranchOpInterface op,
                                              RegionBranchPoint point) {
  SmallVector<RegionEdge> result;
  auto checkPredecessor = [&](RegionBranchPoint possiblePredecessorPoint) {
    for (const auto& successor :
         getSuccessorRegions(op, possiblePredecessorPoint)) {
      if (successor.successorRegionPoint == point) {
        result.push_back(successor);
      }
    }
  };
  checkPredecessor(point.parent());
  for (Region& region : op->getRegions()) {
    checkPredecessor(region);
  }
  return result;
}

RegionBranchOpInterface moveRegionsToNewOpButKeepOldOp(
    RegionBranchOpInterface op) {
  OpBuilder b(op);
  RegionBranchOpInterface newOp;
  if (llvm::isa<scf::ForOp>(op)) {
    newOp = b.create<scf::ForOp>(op.getLoc(), op->getOperands()[0],
                                 op->getOperands()[1], op->getOperands()[2],
                                 op->getOperands().drop_front(3));
  } else if (llvm::isa<scf::WhileOp>(op)) {
    newOp = b.create<scf::WhileOp>(
        op.getLoc(),
        TypeRange{op->getRegion(0).front().getTerminator()->getOperands()}
            .drop_front(),
        op->getOperands());
  } else if (llvm::isa<scf::IfOp>(op)) {
    newOp = b.create<scf::IfOp>(
        op.getLoc(),
        TypeRange{op->getRegion(0).front().getTerminator()->getOperands()},
        op->getOperands()[0], op->getNumRegions() > 1);
  } else if (llvm::isa<scf::ParallelOp>(op)) {
    auto parallel = llvm::cast<scf::ParallelOp>(op);
    newOp = b.create<scf::ParallelOp>(
        op.getLoc(), parallel.getLowerBound(), parallel.getUpperBound(),
        parallel.getStep(), parallel.getInitVals());
  } else {
    llvm_unreachable("unsupported");
  }

  newOp->setAttrs(op->getAttrs());
  for (auto [oldRegion, newRegion] :
       llvm::zip(op->getRegions(), newOp->getRegions())) {
    newRegion.takeBody(oldRegion);
  }

  return newOp;
}

}  // namespace deallocation
}  // namespace mlir
