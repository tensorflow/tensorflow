//===- ConstraintAnalysisGraph.cpp - Graphs type for constraints ----------===//
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

#include "mlir/Quantizer/Support/ConstraintAnalysisGraph.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Quantizer/Support/Configuration.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::quantizer;

void CAGNode::replaceIncoming(CAGNode *otherNode) {
  if (this == otherNode)
    return;
  for (CAGNode *parentNode : incoming) {
    for (CAGNode *&it : parentNode->outgoing) {
      if (it == this) {
        it = otherNode;
        otherNode->incoming.push_back(parentNode);
      }
    }
  }
  incoming.clear();
}

void CAGNode::addOutgoing(CAGNode *toNode) {
  if (!llvm::is_contained(outgoing, toNode)) {
    outgoing.push_back(toNode);
    toNode->incoming.push_back(this);
  }
}

CAGOperandAnchor::CAGOperandAnchor(Operation *op, unsigned operandIdx)
    : CAGAnchorNode(Kind::OperandAnchor, op->getOperand(operandIdx)->getType()),
      op(op), operandIdx(operandIdx) {}

CAGResultAnchor::CAGResultAnchor(Operation *op, unsigned resultIdx)
    : CAGAnchorNode(Kind::ResultAnchor, op->getResult(resultIdx)->getType()),
      resultValue(op->getResult(resultIdx)) {}

CAGSlice::CAGSlice(SolverContext &context) : context(context) {}
CAGSlice::~CAGSlice() { llvm::DeleteContainerPointers(allNodes); }

CAGOperandAnchor *CAGSlice::getOperandAnchor(Operation *op,
                                             unsigned operandIdx) {
  assert(operandIdx < op->getNumOperands() && "illegal operand index");

  // Dedup.
  auto key = std::make_pair(op, operandIdx);
  auto foundIt = operandAnchors.find(key);
  if (foundIt != operandAnchors.end()) {
    return foundIt->second;
  }

  // Create.
  auto anchor = std::make_unique<CAGOperandAnchor>(op, operandIdx);
  auto *unowned = anchor.release();
  unowned->nodeId = allNodes.size();
  allNodes.push_back(unowned);
  operandAnchors.insert(std::make_pair(key, unowned));
  return unowned;
}

CAGResultAnchor *CAGSlice::getResultAnchor(Operation *op, unsigned resultIdx) {
  assert(resultIdx < op->getNumResults() && "illegal result index");

  // Dedup.
  auto key = std::make_pair(op, resultIdx);
  auto foundIt = resultAnchors.find(key);
  if (foundIt != resultAnchors.end()) {
    return foundIt->second;
  }

  // Create.
  auto anchor = std::make_unique<CAGResultAnchor>(op, resultIdx);
  auto *unowned = anchor.release();
  unowned->nodeId = allNodes.size();
  allNodes.push_back(unowned);
  resultAnchors.insert(std::make_pair(key, unowned));
  return unowned;
}

void CAGSlice::enumerateImpliedConnections(
    std::function<void(CAGAnchorNode *from, CAGAnchorNode *to)> callback) {
  // Discover peer identity pairs (i.e. implied edges from Result->Operand and
  // Arg->Call). Use an intermediate vector so that the callback can modify.
  std::vector<std::pair<CAGAnchorNode *, CAGAnchorNode *>> impliedPairs;
  for (auto &resultAnchorPair : resultAnchors) {
    CAGResultAnchor *resultAnchor = resultAnchorPair.second;
    Value *resultValue = resultAnchor->getValue();
    for (auto &use : resultValue->getUses()) {
      Operation *operandOp = use.getOwner();
      unsigned operandIdx = use.getOperandNumber();
      auto foundIt = operandAnchors.find(std::make_pair(operandOp, operandIdx));
      if (foundIt != operandAnchors.end()) {
        impliedPairs.push_back(std::make_pair(resultAnchor, foundIt->second));
      }
    }
  }

  // Callback for each pair.
  for (auto &impliedPair : impliedPairs) {
    callback(impliedPair.first, impliedPair.second);
  }
}

unsigned CAGSlice::propagate(const TargetConfiguration &config) {
  std::vector<CAGNode *> dirtyNodes;
  dirtyNodes.reserve(allNodes.size());
  // Note that because iteration happens in nodeId order, there is no need
  // to sort in order to make deterministic. If the selection method changes,
  // a sort should be explicitly done.
  for (CAGNode *child : *this) {
    if (child->isDirty()) {
      dirtyNodes.push_back(child);
    }
  }

  if (dirtyNodes.empty()) {
    return 0;
  }
  for (auto dirtyNode : dirtyNodes) {
    dirtyNode->clearDirty();
    dirtyNode->propagate(context, config);
  }

  return dirtyNodes.size();
}

void CAGAnchorNode::propagate(SolverContext &solverContext,
                              const TargetConfiguration &config) {
  for (CAGNode *child : *this) {
    child->markDirty();
  }
}

Type CAGAnchorNode::getTransformedType() {
  if (!getUniformMetadata().selectedType) {
    return nullptr;
  }
  return getUniformMetadata().selectedType.castFromExpressedType(
      getOriginalType());
}

void CAGNode::printLabel(raw_ostream &os) const {
  os << "Node<" << static_cast<const void *>(this) << ">";
}

void CAGAnchorNode::printLabel(raw_ostream &os) const {
  getUniformMetadata().printSummary(os);
}

void CAGOperandAnchor::printLabel(raw_ostream &os) const {
  os << "Operand<";
  op->getName().print(os);
  os << "," << operandIdx;
  os << ">";
  CAGAnchorNode::printLabel(os);
}

void CAGResultAnchor::printLabel(raw_ostream &os) const {
  os << "Result<";
  getOp()->getName().print(os);
  os << ">";
  CAGAnchorNode::printLabel(os);
}
