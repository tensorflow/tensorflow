//===- LinalgTransforms.cpp - Linalg transformations as patterns ----------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for transforming Linalg operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/LinalgTransforms.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Utils/Intrinsics.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/VectorOps/VectorOps.h"
#include "mlir/EDSC/Helpers.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <type_traits>

#define DEBUG_TYPE "linalg-transforms"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;
using namespace mlir::linalg::intrinsics;

using llvm::dbgs;
using llvm::SetVector;

// Marker used as attribute name in generated Linalg rewriting transformations.
const StringLiteral mlir::linalg::LinalgTransforms::kLinalgTransformMarker =
    "__internal_linalg_transform__";

LogicalResult mlir::linalg::tileLinalgOpAndSetMarker(
    PatternRewriter &rewriter, Operation *op, ArrayRef<int64_t> sizes,
    StringRef linalgMarker, ArrayRef<unsigned> permutation) {
  assert(permutation.empty() || permutation.size() == sizes.size());
  auto tileRes = tileLinalgOperation(rewriter, op, sizes, permutation);
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
    Operation *consumerOp, Value consumedView,
    function_ref<bool(Operation *)> isaOpType) {
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

static bool hasMultiplyAddBody(linalg::GenericOp op) {
  auto &r = op.region();
  if (r.empty())
    return false;
  if (r.getBlocks().size() != 1)
    return false;
  auto &ops = r.front().getOperations();
  if (ops.size() != 3)
    return false;

  using mlir::matchers::m_Val;
  auto a = m_Val(r.front().getArgument(0));
  auto b = m_Val(r.front().getArgument(1));
  auto c = m_Val(r.front().getArgument(2));
  // TODO(ntv) Update this detection once we have  matcher support for
  // specifying that any permutation of operands matches.
  auto pattern1 = m_Op<YieldOp>(m_Op<AddFOp>(m_Op<MulFOp>(a, b), c));
  auto pattern2 = m_Op<YieldOp>(m_Op<AddFOp>(c, m_Op<MulFOp>(a, b)));
  auto pattern3 = m_Op<YieldOp>(m_Op<AddFOp>(m_Op<MulFOp>(b, a), c));
  auto pattern4 = m_Op<YieldOp>(m_Op<AddFOp>(c, m_Op<MulFOp>(b, a)));
  return pattern1.match(&ops.back()) || pattern2.match(&ops.back()) ||
         pattern3.match(&ops.back()) || pattern4.match(&ops.back());
}

// TODO(ntv) should be Tablegen'd from a single source that generates the op
// itself.
static bool isMatmul(linalg::GenericOp genericOp) {
  auto *ctx = genericOp.getContext();
  auto m = getAffineDimExpr(0, ctx);
  auto n = getAffineDimExpr(1, ctx);
  auto k = getAffineDimExpr(2, ctx);
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {m, k}));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {k, n}));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {m, n}));
  auto maps = ArrayAttr::get({mapA, mapB, mapC}, ctx);
  return genericOp.getNumInputs() == 2 && genericOp.getNumOutputs() == 1 &&
         genericOp.indexing_maps() == maps && hasMultiplyAddBody(genericOp);
}

LogicalResult mlir::linalg::vectorizeGenericOp(PatternRewriter &rewriter,
                                               Operation *op) {
  LLVM_DEBUG(dbgs() << "\n[" DEBUG_TYPE
                       "]: Rewrite linalg op as vector.contract: "
                    << *op << ":\n");

  // TODO(ntv): This is in fact much more general than just vectorization for
  // matmul ops.
  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp || !isMatmul(genericOp))
    return failure();

  // TODO(ntv): non-identity layout.
  auto isStaticMemRefWithIdentityLayout = [](Value v) {
    auto m = v->getType().dyn_cast<MemRefType>();
    if (!m || !m.hasStaticShape() || !m.getAffineMaps().empty())
      return false;
    return true;
  };
  if (!llvm::all_of(genericOp.getInputsAndOutputs(),
                    isStaticMemRefWithIdentityLayout))
    return failure();

  edsc::ScopedContext scope(rewriter, op->getLoc());
  using edsc::intrinsics::std_load;
  using edsc::intrinsics::std_store;
  using vector_contract = edsc::intrinsics::ValueBuilder<vector::ContractionOp>;
  using vector_type_cast = edsc::intrinsics::ValueBuilder<vector::TypeCastOp>;
  auto vA = std_load(vector_type_cast(genericOp.getInput(0)));
  auto vB = std_load(vector_type_cast(genericOp.getInput(1)));
  auto vectorMemRefC = vector_type_cast(genericOp.getOutput(0));
  auto vC = std_load(vectorMemRefC);
  auto vRes = vector_contract(vA, vB, vC, genericOp.indexing_maps(),
                              genericOp.iterator_types());
  std_store(vRes, vectorMemRefC);
  return success();
}

LogicalResult
mlir::linalg::permuteGenericLinalgOp(PatternRewriter &rewriter, Operation *op,
                                     ArrayRef<unsigned> permutation,
                                     StringRef linalgMarker) {
  // If permutation is empty, there is nothing to be done.
  if (permutation.empty())
    return failure();

  auto linOp = cast<LinalgOp>(op);
  auto permutationMap = inversePermutation(
      AffineMap::getPermutationMap(permutation, rewriter.getContext()));
  SmallVector<AffineMap, 4> newIndexingMap;
  auto indexingMaps = linOp.indexing_maps().getValue();
  for (unsigned i = 0, e = linOp.getNumInputsAndOutputs(); i != e; ++i) {
    AffineMap m = indexingMaps[i].cast<AffineMapAttr>().getValue().compose(
        permutationMap);
    newIndexingMap.push_back(m);
  }
  auto itTypes = linOp.iterator_types().getValue();
  SmallVector<Attribute, 4> itTypesVector;
  for (unsigned i = 0, e = itTypes.size(); i != e; ++i)
    itTypesVector.push_back(itTypes[i]);
  applyPermutationToVector(itTypesVector, permutation);
  op->setAttr(getIndexingMapsAttrName(),
              rewriter.getAffineMapArrayAttr(newIndexingMap));
  op->setAttr(getIteratorTypesAttrName(), rewriter.getArrayAttr(itTypesVector));
  op->setAttr(LinalgTransforms::kLinalgTransformMarker,
              rewriter.getStringAttr(linalgMarker));
  linOp.clone(rewriter, linOp.getLoc(), op->getOperands());
  return success();
}

LogicalResult mlir::linalg::linalgOpPromoteSubviews(PatternRewriter &rewriter,
                                                    Operation *op) {
  LinalgOp linOp = dyn_cast<LinalgOp>(op);
  SetVector<Value> subViews;
  for (auto it : linOp.getInputsAndOutputs())
    if (auto sv = dyn_cast_or_null<SubViewOp>(it->getDefiningOp()))
      subViews.insert(sv);
  if (!subViews.empty()) {
    auto resOp = promoteSubViewOperands(rewriter, linOp, subViews);
    return success(resOp);
  }
  return failure();
}
