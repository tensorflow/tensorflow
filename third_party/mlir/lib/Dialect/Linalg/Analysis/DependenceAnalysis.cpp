//===- DependenceAnalysis.cpp - Dependence analysis on SSA views ----------===//
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
// This file implements view-based alias and dependence analyses.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-dependence-analysis"

using namespace mlir;
using namespace mlir::linalg;

using llvm::dbgs;

static StringRef toStringRef(LinalgDependenceGraph::DependenceType dt) {
  switch (dt) {
  case LinalgDependenceGraph::DependenceType::RAW:
    return "RAW";
  case LinalgDependenceGraph::DependenceType::RAR:
    return "RAR";
  case LinalgDependenceGraph::DependenceType::WAR:
    return "WAR";
  case LinalgDependenceGraph::DependenceType::WAW:
    return "WAW";
  default:
    break;
  }
  llvm_unreachable("Unexpected DependenceType");
}

Value Aliases::find(Value v) {
  if (v.isa<BlockArgument>())
    return v;

  auto it = aliases.find(v);
  if (it != aliases.end()) {
    assert(it->getSecond()->getType().isa<MemRefType>() && "Memref expected");
    return it->getSecond();
  }

  while (true) {
    if (v.isa<BlockArgument>())
      return v;
    if (auto alloc = dyn_cast_or_null<AllocOp>(v->getDefiningOp())) {
      if (isStrided(alloc.getType()))
        return alloc.getResult();
    }
    if (auto slice = dyn_cast_or_null<SliceOp>(v->getDefiningOp())) {
      auto it = aliases.insert(std::make_pair(v, find(slice.view())));
      return it.first->second;
    }
    if (auto view = dyn_cast_or_null<ViewOp>(v->getDefiningOp())) {
      auto it = aliases.insert(std::make_pair(v, view.source()));
      return it.first->second;
    }
    if (auto view = dyn_cast_or_null<SubViewOp>(v->getDefiningOp())) {
      v = view.source();
      continue;
    }
    llvm::errs() << "View alias analysis reduces to: " << *v << "\n";
    llvm_unreachable("unsupported view alias case");
  }
}

LinalgDependenceGraph
LinalgDependenceGraph::buildDependenceGraph(Aliases &aliases, FuncOp f) {
  SmallVector<Operation *, 8> linalgOps;
  f.walk([&](LinalgOp op) { linalgOps.push_back(op); });
  return LinalgDependenceGraph(aliases, linalgOps);
}

LinalgDependenceGraph::LinalgDependenceGraph(Aliases &aliases,
                                             ArrayRef<Operation *> ops)
    : aliases(aliases), linalgOps(ops.begin(), ops.end()) {
  for (auto en : llvm::enumerate(linalgOps)) {
    assert(isa<LinalgOp>(en.value()) && "Expected value for LinalgOp");
    linalgOpPositions.insert(std::make_pair(en.value(), en.index()));
  }
  for (unsigned i = 0, e = ops.size(); i < e; ++i) {
    for (unsigned j = i + 1; j < e; ++j) {
      addDependencesBetween(cast<LinalgOp>(ops[i]), cast<LinalgOp>(ops[j]));
    }
  }
}

void LinalgDependenceGraph::addDependenceElem(DependenceType dt,
                                              LinalgOpView indexingOpView,
                                              LinalgOpView dependentOpView) {
  LLVM_DEBUG(dbgs() << "\nAdd dep type " << toStringRef(dt) << ":\t"
                    << *indexingOpView.op << " -> " << *dependentOpView.op);
  (void)toStringRef;
  dependencesFromGraphs[dt][indexingOpView.op].push_back(
      LinalgDependenceGraphElem{dependentOpView, indexingOpView.view});
  dependencesIntoGraphs[dt][dependentOpView.op].push_back(
      LinalgDependenceGraphElem{indexingOpView, dependentOpView.view});
}

LinalgDependenceGraph::dependence_range
LinalgDependenceGraph::getDependencesFrom(
    LinalgOp src, LinalgDependenceGraph::DependenceType dt) const {
  return getDependencesFrom(src.getOperation(), dt);
}

LinalgDependenceGraph::dependence_range
LinalgDependenceGraph::getDependencesFrom(
    Operation *src, LinalgDependenceGraph::DependenceType dt) const {
  auto iter = dependencesFromGraphs[dt].find(src);
  if (iter == dependencesFromGraphs[dt].end())
    return llvm::make_range(nullptr, nullptr);
  return llvm::make_range(iter->second.begin(), iter->second.end());
}

LinalgDependenceGraph::dependence_range
LinalgDependenceGraph::getDependencesInto(
    LinalgOp dst, LinalgDependenceGraph::DependenceType dt) const {
  return getDependencesInto(dst.getOperation(), dt);
}

LinalgDependenceGraph::dependence_range
LinalgDependenceGraph::getDependencesInto(
    Operation *dst, LinalgDependenceGraph::DependenceType dt) const {
  auto iter = dependencesIntoGraphs[dt].find(dst);
  if (iter == dependencesIntoGraphs[dt].end())
    return llvm::make_range(nullptr, nullptr);
  return llvm::make_range(iter->second.begin(), iter->second.end());
}

void LinalgDependenceGraph::addDependencesBetween(LinalgOp src, LinalgOp dst) {
  for (auto srcView : src.getOutputs()) { // W
    // RAW graph
    for (auto dstView : dst.getInputs()) {   // R
      if (aliases.alias(srcView, dstView)) { // if alias, fill RAW
        addDependenceElem(DependenceType::RAW,
                          LinalgOpView{src.getOperation(), srcView},
                          LinalgOpView{dst.getOperation(), dstView});
      }
    }
    // WAW graph
    for (auto dstView : dst.getOutputs()) {  // W
      if (aliases.alias(srcView, dstView)) { // if alias, fill WAW
        addDependenceElem(DependenceType::WAW,
                          LinalgOpView{src.getOperation(), srcView},
                          LinalgOpView{dst.getOperation(), dstView});
      }
    }
  }
  for (auto srcView : src.getInputs()) { // R
    // RAR graph
    for (auto dstView : dst.getInputs()) {   // R
      if (aliases.alias(srcView, dstView)) { // if alias, fill RAR
        addDependenceElem(DependenceType::RAR,
                          LinalgOpView{src.getOperation(), srcView},
                          LinalgOpView{dst.getOperation(), dstView});
      }
    }
    // WAR graph
    for (auto dstView : dst.getOutputs()) {  // W
      if (aliases.alias(srcView, dstView)) { // if alias, fill WAR
        addDependenceElem(DependenceType::WAR,
                          LinalgOpView{src.getOperation(), srcView},
                          LinalgOpView{dst.getOperation(), dstView});
      }
    }
  }
}

SmallVector<Operation *, 8>
LinalgDependenceGraph::findCoveringDependences(LinalgOp srcLinalgOp,
                                               LinalgOp dstLinalgOp) const {
  return findOperationsWithCoveringDependences(
      srcLinalgOp, dstLinalgOp, nullptr,
      {DependenceType::WAW, DependenceType::WAR, DependenceType::RAW});
}

SmallVector<Operation *, 8> LinalgDependenceGraph::findCoveringWrites(
    LinalgOp srcLinalgOp, LinalgOp dstLinalgOp, Value view) const {
  return findOperationsWithCoveringDependences(
      srcLinalgOp, dstLinalgOp, view,
      {DependenceType::WAW, DependenceType::WAR});
}

SmallVector<Operation *, 8> LinalgDependenceGraph::findCoveringReads(
    LinalgOp srcLinalgOp, LinalgOp dstLinalgOp, Value view) const {
  return findOperationsWithCoveringDependences(
      srcLinalgOp, dstLinalgOp, view,
      {DependenceType::RAR, DependenceType::RAW});
}

SmallVector<Operation *, 8>
LinalgDependenceGraph::findOperationsWithCoveringDependences(
    LinalgOp srcLinalgOp, LinalgOp dstLinalgOp, Value view,
    ArrayRef<DependenceType> types) const {
  auto *src = srcLinalgOp.getOperation();
  auto *dst = dstLinalgOp.getOperation();
  auto srcPos = linalgOpPositions.lookup(src);
  auto dstPos = linalgOpPositions.lookup(dst);
  assert(srcPos < dstPos && "expected dst after src in IR traversal order");

  SmallVector<Operation *, 8> res;
  // Consider an intermediate interleaved `interim` op, look for any dependence
  // to an aliasing view on a src -> op -> dst path.
  // TODO(ntv) we are not considering paths yet, just interleaved positions.
  for (auto dt : types) {
    for (auto dependence : getDependencesFrom(src, dt)) {
      auto interimPos = linalgOpPositions.lookup(dependence.dependentOpView.op);
      // Skip if not interleaved.
      if (interimPos >= dstPos || interimPos <= srcPos)
        continue;
      if (view && !aliases.alias(view, dependence.indexingView))
        continue;
      auto *op = dependence.dependentOpView.op;
      LLVM_DEBUG(dbgs() << "\n***Found covering dependence of type "
                        << toStringRef(dt) << ": " << *src << " -> " << *op
                        << " on " << *dependence.indexingView);
      res.push_back(op);
    }
  }
  return res;
}
