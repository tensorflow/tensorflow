/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
                                            std::optional<unsigned> index) {
  SmallVector<RegionEdge> edges;
  if (index && op->getRegion(*index).empty()) {
    return edges;
  }

  SmallVector<RegionSuccessor> successors;
  op.getSuccessorRegions(index, successors);

  for (const auto& successor : successors) {
    auto& edge = edges.emplace_back();
    edge.predecessorRegionIndex = index;
    edge.predecessorOp = index ? op->getRegion(*index).front().getTerminator()
                               : op.getOperation();
    edge.predecessorOperandIndex = edge.predecessorOp->getNumOperands() -
                                   successor.getSuccessorInputs().size();

    if (successor.isParent()) {
      edge.successorRegionIndex = std::nullopt;
      edge.successorOpOrRegion = op.getOperation();
      edge.successorValueIndex = 0;
    } else {
      edge.successorRegionIndex = successor.getSuccessor()->getRegionNumber();
      edge.successorOpOrRegion = successor.getSuccessor();
      edge.successorValueIndex = llvm::isa<scf::ForOp>(op) ? 1 : 0;
    }
  }

  return edges;
}

SmallVector<RegionEdge> getPredecessorRegions(RegionBranchOpInterface op,
                                              std::optional<unsigned> index) {
  SmallVector<RegionEdge> result;
  auto checkPredecessor = [&](std::optional<unsigned> possiblePredecessor) {
    for (const auto& successor : getSuccessorRegions(op, possiblePredecessor)) {
      if (successor.successorRegionIndex == index) {
        result.push_back(successor);
      }
    }
  };
  checkPredecessor(std::nullopt);
  for (unsigned i = 0; i < op->getNumRegions(); ++i) {
    checkPredecessor(i);
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

Type getUnrankedMemrefType(Type ty) {
  if (ty.isa<UnrankedMemRefType>()) {
    return ty;
  }
  MemRefType memRefTy = llvm::cast<MemRefType>(ty);
  return UnrankedMemRefType::get(memRefTy.getElementType(),
                                 memRefTy.getMemorySpace());
}

Type getUnrankedMemrefType(Value v) {
  return getUnrankedMemrefType(v.getType());
}

}  // namespace deallocation
}  // namespace mlir
