//===- LinalgTransforms.cpp - Linalg transformations as patterns ----------===//
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
// This file implements logic for transforming Linalg operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/LinalgTransforms.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::linalg;

// Marker used as attribute name in generated Linalg rewriting transformations.
constexpr StringRef mlir::linalg::LinalgTransforms::kLinalgTransformMarker;

LogicalResult mlir::linalg::tileLinalgOpAndSetMarker(PatternRewriter &rewriter,
                                                     Operation *op,
                                                     ArrayRef<int64_t> sizes,
                                                     StringRef linalgMarker) {
  auto tileRes = tileLinalgOperation(rewriter, op, sizes);
  if (!tileRes)
    return failure();
  tileRes->op.setAttr(LinalgTransforms::kLinalgTransformMarker,
                      rewriter.getStringAttr(linalgMarker));
  return success();
}

LogicalResult mlir::linalg::tileAndFuseLinalgOpAndSetMarker(
    PatternRewriter &rewriter, Operation *op, ArrayRef<int64_t> sizes,
    ArrayRef<int64_t> operandIndicesToFuse, StringRef linalgMarker) {
  auto tileRes = tileLinalgOperation(rewriter, op, sizes);
  if (!tileRes)
    return failure();
  tileRes->op.setAttr(LinalgTransforms::kLinalgTransformMarker,
                      rewriter.getStringAttr(linalgMarker));
  Aliases aliases;
  auto G = LinalgDependenceGraph::buildDependenceGraph(
      aliases, op->getParentOfType<FuncOp>());
  SmallVector<Operation *, 4> originalProducers;
  for (auto operandIdx : operandIndicesToFuse) {
    auto fusionRes = fuseProducerOf(rewriter, tileRes->op, operandIdx, G);
    if (!fusionRes) {
      // Linalg fusion requires tiled loops to even determine whether it is
      // possible to fuse. As a consequence, the pattern may fail even though a
      // tiled version of op has already been introduced.
      // So we need to remove the tiled version ourselves in case of failure.
      // Another possibility is to ensure the constraints on the pattern
      // guarantee that fusion will occur and just assert here. As we develop
      // more complex patterns we can choose what is best.
      rewriter.eraseOp(tileRes->loops[0]);
      return failure();
    }
    fusionRes->fusedProducer.setAttr(LinalgTransforms::kLinalgTransformMarker,
                                     rewriter.getStringAttr(linalgMarker));
    originalProducers.push_back(fusionRes->originalProducer);
  }

  // The originalProducers can now be safely erased. This is similar to
  // SSA-value use-def but in the world of buffer + structured ops.
  for (auto *originalProducer : originalProducers)
    rewriter.eraseOp(originalProducer);
  return success();
}

bool mlir::linalg::detail::isProducedByOpOfTypeImpl(
    Operation *consumerOp, Value *consumedView,
    llvm::function_ref<bool(Operation *)> isaOpType) {
  LinalgOp consumer = dyn_cast<LinalgOp>(consumerOp);
  if (!consumer)
    return false;

  auto maybeConsumerIndex = consumer.getIndexOfInput(consumedView);
  if (!maybeConsumerIndex)
    return false;

  Aliases aliases;
  auto G = LinalgDependenceGraph::buildDependenceGraph(
      aliases, consumer.getParentOfType<FuncOp>());
  for (auto dependence : G.getDependencesInto(
           consumer, LinalgDependenceGraph::DependenceType::RAW)) {
    auto producer = cast<LinalgOp>(dependence.dependentOpView.op);
    if (!isProducerLastWriteOfView(G, consumer, consumedView, producer))
      continue;
    if (isaOpType(dependence.dependentOpView.op))
      return true;
  }
  return false;
}
