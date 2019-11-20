//===- LowerToLoops.cpp - conversion from Linalg library ops to loops------===//
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

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Utils/Intrinsics.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/EDSC/Helpers.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/STLExtras.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/FoldUtils.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;
using namespace mlir::linalg::intrinsics;

using IndexedStdValue = TemplatedIndexedValue<std_load, std_store>;
using edsc::op::operator+;
using edsc::op::operator==;

static SmallVector<ValueHandle, 8>
foldedAffineApplies(OpBuilder &b, Location loc, AffineMap map,
                    ArrayRef<Value *> vals, OperationFolder *folder) {
  assert(map.getNumSymbols() == 0);
  assert(map.getNumInputs() == vals.size());
  SmallVector<ValueHandle, 8> res;
  res.reserve(map.getNumResults());
  auto dims = map.getNumDims();
  for (auto e : map.getResults()) {
    auto exprMap = AffineMap::get(dims, 0, e);
    SmallVector<Value *, 4> operands(vals.begin(), vals.end());
    canonicalizeMapAndOperands(&exprMap, &operands);
    res.push_back(affine_apply(folder, exprMap, operands));
  }
  return res;
}

static SmallVector<Value *, 4> permuteIvs(ArrayRef<Value *> ivs,
                                          Optional<AffineMap> permutation,
                                          OperationFolder *folder) {
  return permutation ? applyMapToValues(ScopedContext::getBuilder(),
                                        ScopedContext::getLocation(),
                                        permutation.getValue(), ivs, folder)
                     : SmallVector<Value *, 4>(ivs.begin(), ivs.end());
}

// Creates a number of ranges equal to the number of results in `map`.
// The returned ranges correspond to the loop ranges, in the proper order, for
// which new loops will be created.
static SmallVector<Value *, 4> emitLoopRanges(OpBuilder &b, Location loc,
                                              AffineMap map,
                                              ArrayRef<Value *> allViewSizes,
                                              OperationFolder *folder);
SmallVector<Value *, 4> emitLoopRanges(OpBuilder &b, Location loc,
                                       AffineMap map,
                                       ArrayRef<Value *> allViewSizes,
                                       OperationFolder *folder) {
  // Apply `map` to get view sizes in loop order.
  auto sizes = applyMapToValues(b, loc, map, allViewSizes, folder);
  // Create a new range with the applied tile sizes.
  ScopedContext scope(b, loc);
  SmallVector<Value *, 4> res;
  for (unsigned idx = 0, e = map.getNumResults(); idx < e; ++idx) {
    res.push_back(range(constant_index(folder, 0), sizes[idx],
                        constant_index(folder, 1)));
  }
  return res;
}

template <typename IndexedValueType, typename LinalgOpType>
class LinalgScopedEmitter {};

template <typename IndexedValueType>
class LinalgScopedEmitter<IndexedValueType, CopyOp> {
public:
  static void emitScalarImplementation(ArrayRef<Value *> allIvs, CopyOp copyOp,
                                       OperationFolder *folder) {
    auto nPar = copyOp.getNumParallelLoops();
    assert(nPar == allIvs.size());
    auto inputIvs =
        permuteIvs(allIvs.take_front(nPar), copyOp.inputPermutation(), folder);
    auto outputIvs =
        permuteIvs(allIvs.take_front(nPar), copyOp.outputPermutation(), folder);
    SmallVector<IndexHandle, 8> iivs(inputIvs.begin(), inputIvs.end());
    SmallVector<IndexHandle, 8> oivs(outputIvs.begin(), outputIvs.end());
    IndexedValueType O(copyOp.getOutput(0)), I(copyOp.getInput(0));
    // Emit the proper scalar assignment, whether we are dealing with a 0-D or
    // an n-D loop nest; with or without permutations.
    // clang-format off
    nPar > 0 ? O(oivs) = I(iivs) :
               O() = I();
    // clang-format on
  }
};

template <typename IndexedValueType>
class LinalgScopedEmitter<IndexedValueType, FillOp> {
public:
  static void emitScalarImplementation(ArrayRef<Value *> allIvs, FillOp fillOp,
                                       OperationFolder *folder) {
    auto nPar = fillOp.getNumParallelLoops();
    assert(nPar == allIvs.size());
    auto ivs =
        SmallVector<IndexHandle, 4>(allIvs.begin(), allIvs.begin() + nPar);
    IndexedValueType O(fillOp.getOutput(0));
    // Emit the proper scalar assignment, whether we are dealing with a 0-D or
    // an n-D loop nest; with or without permutations.
    nPar > 0 ? O(ivs) = ValueHandle(fillOp.getValue())
             : O() = ValueHandle(fillOp.getValue());
  }
};

template <typename IndexedValueType>
class LinalgScopedEmitter<IndexedValueType, DotOp> {
public:
  static void emitScalarImplementation(ArrayRef<Value *> allIvs, DotOp dotOp,
                                       OperationFolder *folder) {
    assert(allIvs.size() == 1);
    IndexHandle r_i(allIvs[0]);
    IndexedValueType A(dotOp.getInput(0)), B(dotOp.getInput(1)),
        C(dotOp.getOutput(0));
    // Emit scalar form.
    C() = C() + A(r_i) * B(r_i);
  }
};

template <typename IndexedValueType>
class LinalgScopedEmitter<IndexedValueType, MatvecOp> {
public:
  static void emitScalarImplementation(ArrayRef<Value *> allIvs,
                                       MatvecOp matvecOp,
                                       OperationFolder *folder) {
    assert(allIvs.size() == 2);
    IndexHandle i(allIvs[0]), r_j(allIvs[1]);
    IndexedValueType A(matvecOp.getInput(0)), B(matvecOp.getInput(1)),
        C(matvecOp.getOutput(0));
    // Emit scalar form.
    C(i) = C(i) + A(i, r_j) * B(r_j);
  }
};

template <typename IndexedValueType>
class LinalgScopedEmitter<IndexedValueType, MatmulOp> {
public:
  static void emitScalarImplementation(ArrayRef<Value *> allIvs,
                                       MatmulOp matmulOp,
                                       OperationFolder *folder) {
    assert(allIvs.size() == 3);
    IndexHandle i(allIvs[0]), j(allIvs[1]), r_k(allIvs[2]);
    IndexedValueType A(matmulOp.getInput(0)), B(matmulOp.getInput(1)),
        C(matmulOp.getOutput(0));
    // Emit scalar form.
    C(i, j) = C(i, j) + A(i, r_k) * B(r_k, j);
  }
};

template <typename IndexedValueType>
class LinalgScopedEmitter<IndexedValueType, ConvOp> {
public:
  static void emitScalarImplementation(ArrayRef<Value *> allIvs, ConvOp convOp,
                                       OperationFolder *folder) {
    auto b = ScopedContext::getBuilder();
    auto loc = ScopedContext::getLocation();
    auto maps = loopToOperandRangesMaps(convOp);
    SmallVector<ValueHandle, 8> fIdx(
        foldedAffineApplies(b, loc, maps[0], allIvs, folder));
    SmallVector<ValueHandle, 8> imIdx(
        foldedAffineApplies(b, loc, maps[1], allIvs, folder));
    SmallVector<ValueHandle, 8> oIdx(
        foldedAffineApplies(b, loc, maps[2], allIvs, folder));
    IndexedValueType F(convOp.filter()), I(convOp.input()), O(convOp.output());
    // Emit scalar form.
    O(oIdx) += F(fIdx) * I(imIdx);
  }
};

// Emits the MLIR for the scalar part of the generic op by:
//   1. Emitting std_load and std_store ops for each input and output
//      view in order. This is achieved by applying the appropriate input or
//      output map to the enclosing induction variables.
//   2. Emitting a call to `op.fun()` that takes as arguments the scalars
//      from point 1. above.
//   3. Emitting std_store to store the results of 2. to the output
//      views.
//
// An example output may resemble:
//
// ```
//    loop.for %i = %c0 to %0 step %c1 {
//      loop.for %j = %c0 to %1 step %c1 {
//        loop.for %k = %c0 to %4 step %c1 {
//          %11 = load %arg0[%i, %j] :
//            memref<?x?xf32, stride_specification>
//          %12 = load %arg1[%i, %j, %k] :
//            memref<?x?x?xf32, stride_specification>
//          %13 = load %arg2[%i, %k, %j] :
//            memref<?x?x?xf32, stride_specification>
//          %14:2 = call @foo(%11, %12, %13) : (f32, f32, f32) -> (f32, f32)
//          store %14#0, %arg1[%i, %j, %k] :
//            memref<?x?x?Xf32, stride_specification>
//          store %14#1, %arg2[%i, %k, %j] :
//            memref<?x?x?Xf32, stride_specification>
//       }
//      }
//    }
// ```
template <typename IndexedValueType>
class LinalgScopedEmitter<IndexedValueType, GenericOp> {
public:
  static void emitScalarImplementation(ArrayRef<Value *> allIvs,
                                       GenericOp genericOp,
                                       OperationFolder *folder) {
    auto b = ScopedContext::getBuilder();
    auto loc = ScopedContext::getLocation();
    using edsc::intrinsics::detail::ValueHandleArray;
    unsigned nInputs = genericOp.getNumInputs();
    unsigned nOutputs = genericOp.getNumOutputs();
    SmallVector<Value *, 4> indexedValues(nInputs + nOutputs);

    // 1.a. Emit std_load from input views.
    for (unsigned i = 0; i < nInputs; ++i) {
      ValueHandleArray indexing(foldedAffineApplies(
          b, loc, genericOp.getInputIndexingMap(i), allIvs, folder));
      indexedValues[i] = std_load(genericOp.getInput(i), indexing);
    }

    // 1.b. Emit std_load from output views.
    for (unsigned i = 0; i < nOutputs; ++i) {
      ValueHandleArray indexing(foldedAffineApplies(
          b, loc, genericOp.getOutputIndexingMap(i), allIvs, folder));
      indexedValues[nInputs + i] = std_load(genericOp.getOutput(i), indexing);
    }

    auto funcOp = genericOp.getFunction();
    if (funcOp) {
      // 2. Emit call.
      Operation *callOp = call(funcOp, indexedValues);
      assert(callOp->getNumResults() == genericOp.getNumOutputs());

      // 3. Emit std_store.
      for (unsigned i = 0; i < nOutputs; ++i) {
        ValueHandleArray indexing(foldedAffineApplies(
            b, loc, genericOp.getOutputIndexingMap(i), allIvs, folder));
        std_store(callOp->getResult(i), genericOp.getOutput(i), indexing);
      }
      return;
    }
    // TODO(ntv): When a region inliner exists, use it.
    // 2. Inline region, currently only works for a single basic block.
    BlockAndValueMapping map;
    auto &block = genericOp.region().front();
    for (auto it : llvm::zip(block.getArguments(), indexedValues))
      map.map(std::get<0>(it), std::get<1>(it));
    for (auto &op : block.without_terminator()) {
      assert(op.getNumRegions() == 0);
      auto *newOp = b.clone(op, map);
      for (auto it : llvm::zip(op.getResults(), newOp->getResults()))
        map.map(std::get<0>(it), std::get<1>(it));
    }

    // 3. Emit std_store.
    auto *yieldOp = cast<YieldOp>(block.back()).getOperation();
    assert(yieldOp->getNumOperands() == nOutputs);
    for (unsigned i = 0; i < nOutputs; ++i) {
      ValueHandleArray indexing(foldedAffineApplies(
          b, loc, genericOp.getOutputIndexingMap(i), allIvs, folder));
      std_store(map.lookup(yieldOp->getOperand(i)), genericOp.getOutput(i),
                indexing);
    }
  }
};

// Emits the MLIR for the scalar part of the indexed generic op by:
//   1. Emitting std_load and std_store ops for each input and output view in
//      order. This is achieved by applying the appropriate input or output map
//      to the enclosing induction variables.
//   2. Emitting a call to `op.fun()` that takes as arguments the induction
//      variables and the scalars from point 1. above.
//   3. Emitting std_store to store the results of 2. to the output views.
//
// An example output may resemble:
//
// ```
//    loop.for %i = %c0 to %0 step %c1 {
//      loop.for %j = %c0 to %1 step %c1 {
//        loop.for %k = %c0 to %4 step %c1 {
//          %11 = load %arg0[%i, %j] :
//            memref<?x?xf32, stride_specification>
//          %12 = load %arg1[%i, %j, %k] :
//            memref<?x?x?xf32, stride_specification>
//          %13 = load %arg2[%i, %k, %j] :
//            memref<?x?x?xf32, stride_specification>
//          %14:2 = call @foo(%i, %j, %k, %11, %12, %13) :
//            (index, index, index, f32, f32, f32) -> (f32, f32)
//          store %14#0, %arg1[%i, %j, %k] :
//            memref<?x?x?Xf32, stride_specification>
//          store %14#1, %arg2[%i, %k, %j] :
//            memref<?x?x?Xf32, stride_specification>
//       }
//      }
//    }
// ```
template <typename IndexedValueType>
class LinalgScopedEmitter<IndexedValueType, IndexedGenericOp> {
public:
  static void emitScalarImplementation(ArrayRef<Value *> allIvs,
                                       IndexedGenericOp indexedGenericOp,
                                       OperationFolder *folder) {
    auto b = ScopedContext::getBuilder();
    auto loc = ScopedContext::getLocation();
    using edsc::intrinsics::detail::ValueHandleArray;
    unsigned nInputs = indexedGenericOp.getNumInputs();
    unsigned nOutputs = indexedGenericOp.getNumOutputs();
    unsigned nLoops = allIvs.size();
    SmallVector<Value *, 4> indexedValues(nLoops + nInputs + nOutputs);

    for (unsigned i = 0; i < nLoops; ++i) {
      indexedValues[i] = allIvs[i];
    }

    // 1.a. Emit std_load from input views.
    for (unsigned i = 0; i < nInputs; ++i) {
      ValueHandleArray indexing(foldedAffineApplies(
          b, loc, indexedGenericOp.getInputIndexingMap(i), allIvs, folder));
      indexedValues[nLoops + i] =
          std_load(indexedGenericOp.getInput(i), indexing);
    }

    // 1.b. Emit std_load from output views.
    for (unsigned i = 0; i < nOutputs; ++i) {
      ValueHandleArray indexing(foldedAffineApplies(
          b, loc, indexedGenericOp.getOutputIndexingMap(i), allIvs, folder));
      indexedValues[nLoops + nInputs + i] =
          std_load(indexedGenericOp.getOutput(i), indexing);
    }

    if (auto funcOp = indexedGenericOp.getFunction()) {
      // 2. Emit call.
      Operation *callOp = call(funcOp, indexedValues);
      assert(callOp->getNumResults() == indexedGenericOp.getNumOutputs());

      // 3. Emit std_store.
      for (unsigned i = 0; i < nOutputs; ++i) {
        ValueHandleArray indexing(foldedAffineApplies(
            b, loc, indexedGenericOp.getOutputIndexingMap(i), allIvs, folder));
        std_store(callOp->getResult(i), indexedGenericOp.getOutput(i),
                  indexing);
      }
      return;
    }
    // TODO(ntv): When a region inliner exists, use it.
    // 2. Inline region, currently only works for a single basic block.
    BlockAndValueMapping map;
    auto &block = indexedGenericOp.region().front();
    for (auto it : llvm::zip(block.getArguments(), indexedValues))
      map.map(std::get<0>(it), std::get<1>(it));
    for (auto &op : block.without_terminator()) {
      assert(op.getNumRegions() == 0);
      auto *newOp = b.clone(op, map);
      for (auto it : llvm::zip(op.getResults(), newOp->getResults()))
        map.map(std::get<0>(it), std::get<1>(it));
    }

    // 3. Emit std_store.
    auto *yieldOp = cast<YieldOp>(block.back()).getOperation();
    assert(yieldOp->getNumOperands() == nOutputs);
    for (unsigned i = 0; i < nOutputs; ++i) {
      ValueHandleArray indexing(foldedAffineApplies(
          b, loc, indexedGenericOp.getOutputIndexingMap(i), allIvs, folder));
      std_store(map.lookup(yieldOp->getOperand(i)),
                indexedGenericOp.getOutput(i), indexing);
    }
  }
};

template <typename LoopType, typename IndexedValueType, typename ConcreteOp>
class LinalgRewritePattern : public RewritePattern {
public:
  explicit LinalgRewritePattern(MLIRContext *context)
      : RewritePattern(ConcreteOp::getOperationName(), /*benefit=*/1, context),
        folder(context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    OpBuilder b(op);
    ScopedContext scope(b, op->getLoc());

    // The flattened loopToOperandRangesMaps is expected to be an invertible
    // permutation map (which is asserted in the inverse calculation).
    auto linalgOp = cast<ConcreteOp>(op);
    auto invertedMap =
        inversePermutation(concatAffineMaps(loopToOperandRangesMaps(linalgOp)));
    if (!invertedMap) {
      LinalgScopedEmitter<IndexedValueType,
                          ConcreteOp>::emitScalarImplementation({}, linalgOp,
                                                                &folder);
      rewriter.eraseOp(op);
      return matchSuccess();
    }

    auto nPar = linalgOp.getNumParallelLoops();
    auto nRed = linalgOp.getNumReductionLoops();
    auto nWin = linalgOp.getNumWindowLoops();
    SmallVector<IndexHandle, 4> allIvs(nPar + nRed + nWin);
    SmallVector<ValueHandle *, 4> allPIvs = makeIndexHandlePointers(allIvs);
    auto loopRanges =
        emitLoopRanges(scope.getBuilder(), scope.getLocation(), invertedMap,
                       getViewSizes(linalgOp), &folder);
    assert(loopRanges.size() == allIvs.size());

    // clang-format off;
    LoopNestRangeBuilder(allPIvs, loopRanges)([&] {
      auto allIvValues = extractValues(allIvs);
      LinalgScopedEmitter<IndexedValueType,
                          ConcreteOp>::emitScalarImplementation(allIvValues,
                                                                linalgOp,
                                                                &folder);
    });
    // clang-format on
    rewriter.eraseOp(op);
    return matchSuccess();
  }

  mutable OperationFolder folder;
};

// Helper classes for type list expansion.
template <typename LoopType, typename IndexedValueType, typename... LinalgOps>
class ConversionList;

template <typename LoopType, typename IndexedValueType>
class ConversionList<LoopType, IndexedValueType> {
public:
  static void build(OwningRewritePatternList &patterns, MLIRContext *ctx) {}
};

template <typename LoopType, typename IndexedValueType, typename ConcreteOp,
          typename... LinalgOps>
class ConversionList<LoopType, IndexedValueType, ConcreteOp, LinalgOps...> {
public:
  static void build(OwningRewritePatternList &patterns, MLIRContext *ctx) {
    patterns
        .insert<LinalgRewritePattern<LoopType, IndexedValueType, ConcreteOp>>(
            ctx);
    ConversionList<LoopType, IndexedValueType, LinalgOps...>::build(patterns,
                                                                    ctx);
  }
};

/// Populate the given list with patterns that convert from Linalg to LLVM.
template <typename LoopType, typename IndexedValueType>
void ForOpRewritePatterns(OwningRewritePatternList &patterns,
                          MLIRContext *ctx) {
  ConversionList<LoopType, IndexedValueType,
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgLibraryOps.cpp.inc"
                 >::build(patterns, ctx);
}

namespace {
template <typename LoopType, typename IndexedValueType>
struct LowerLinalgToLoopsPass
    : public FunctionPass<LowerLinalgToLoopsPass<LoopType, IndexedValueType>> {
  void runOnFunction() override;
};
} // namespace

template <typename LoopType, typename IndexedValueType>
void LowerLinalgToLoopsPass<LoopType, IndexedValueType>::runOnFunction() {
  OwningRewritePatternList patterns;
  ForOpRewritePatterns<LoopType, IndexedValueType>(patterns,
                                                   &this->getContext());

  ConversionTarget target(this->getContext());
  target.addLegalDialect<AffineOpsDialect>();
  target.addLegalDialect<loop::LoopOpsDialect>();
  target.addLegalDialect<StandardOpsDialect>();
  if (failed(applyPartialConversion(this->getFunction(), target, patterns))) {
    this->signalPassFailure();
  }
}

std::unique_ptr<OpPassBase<FuncOp>>
mlir::linalg::createLowerLinalgToLoopsPass() {
  return std::make_unique<
      LowerLinalgToLoopsPass<loop::ForOp, IndexedStdValue>>();
}

static PassRegistration<LowerLinalgToLoopsPass<loop::ForOp, IndexedStdValue>>
    structuredLoopsPass(
        "linalg-lower-to-loops",
        "Lower the operations from the linalg dialect into loops");
