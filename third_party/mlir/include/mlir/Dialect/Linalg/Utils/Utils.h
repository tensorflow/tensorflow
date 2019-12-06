//===- Utils.h - Utilities to support the Linalg dialect --------*- C++ -*-===//
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

#ifndef MLIR_DIALECT_LINALG_UTILS_H_
#define MLIR_DIALECT_LINALG_UTILS_H_

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/EDSC/Helpers.h"

namespace mlir {
class AffineExpr;
class AffineMap;
class OperationFolder;

namespace edsc {

/// A LoopRangeBuilder is a generic NestedBuilder for loop.for operations.
/// More specifically it is meant to be used as a temporary object for
/// representing any nested MLIR construct that is "related to" an mlir::Value*
/// (for now an induction variable).
class LoopRangeBuilder : public NestedBuilder {
public:
  /// Constructs a new loop.for and captures the associated induction
  /// variable. A ValueHandle pointer is passed as the first argument and is the
  /// *only* way to capture the loop induction variable.
  LoopRangeBuilder(ValueHandle *iv, ValueHandle range);
  LoopRangeBuilder(ValueHandle *iv, Value *range);
  LoopRangeBuilder(ValueHandle *iv, SubViewOp::Range range);

  LoopRangeBuilder(const LoopRangeBuilder &) = delete;
  LoopRangeBuilder(LoopRangeBuilder &&) = default;

  LoopRangeBuilder &operator=(const LoopRangeBuilder &) = delete;
  LoopRangeBuilder &operator=(LoopRangeBuilder &&) = default;

  /// The only purpose of this operator is to serve as a sequence point so that
  /// the evaluation of `fun` (which build IR snippets in a scoped fashion) is
  /// scoped within a LoopRangeBuilder.
  ValueHandle operator()(std::function<void(void)> fun = nullptr);
};

/// Helper class to sugar building loop.for loop nests from ranges.
/// This is similar to edsc::AffineLoopNestBuilder except it works on ranges
/// directly. In the current implementation it produces loop.for operations.
class LoopNestRangeBuilder {
public:
  LoopNestRangeBuilder(llvm::ArrayRef<edsc::ValueHandle *> ivs,
                       llvm::ArrayRef<edsc::ValueHandle> ranges);
  LoopNestRangeBuilder(llvm::ArrayRef<edsc::ValueHandle *> ivs,
                       llvm::ArrayRef<Value *> ranges);
  LoopNestRangeBuilder(llvm::ArrayRef<edsc::ValueHandle *> ivs,
                       llvm::ArrayRef<SubViewOp::Range> ranges);
  edsc::ValueHandle operator()(std::function<void(void)> fun = nullptr);

private:
  llvm::SmallVector<LoopRangeBuilder, 4> loops;
};

} // namespace edsc

namespace linalg {
class LinalgDependenceGraph;

struct FusionInfo {
  LinalgOp originalProducer;
  LinalgOp fusedProducer;
};

/// Checks whether the specific `producer` is the last write to exactly the
/// whole `consumedView`. This checks structural dominance, that the dependence
/// is a RAW without any interleaved write to any piece of `consumedView`.
bool isProducerLastWriteOfView(const LinalgDependenceGraph &graph,
                               LinalgOp consumer, Value *consumedView,
                               LinalgOp producer);

/// Checks whether fusing the specific `producer` of the `consumedView` is
/// feasible. This checks `producer` is the last write of `consumedView` and
/// that no interleaved dependence would be violated (RAW, WAR or WAW).
bool isFusableInto(const LinalgDependenceGraph &graph, LinalgOp consumer,
                   Value *consumedView, LinalgOp producer);

/// Fuses producer into consumer if the producer is structurally feasible and
/// the fusion would not violate dependencies.
/// When non-null, the optional pointer `folder` is used to call into the
/// `createAndFold` builder method. If `folder` is null, the regular `create`
/// method is called.
Optional<FusionInfo> fuseProducerOf(OpBuilder &b, LinalgOp consumer,
                                    unsigned consumerIdx,
                                    const LinalgDependenceGraph &graph,
                                    OperationFolder *folder = nullptr);

/// Returns the linearized list of all view dimensions in a linalgOp. Applying
/// the inverse, concatenated loopToOperandRangeMaps to this list allows the
/// derivation of loop ranges for any linalgOp.
template <typename ConcreteOp>
SmallVector<Value *, 8> getViewSizes(ConcreteOp linalgOp) {
  SmallVector<Value *, 8> res;
  for (auto v : linalgOp.getInputsAndOutputs()) {
    MemRefType t = v->getType().template cast<MemRefType>();
    for (unsigned i = 0; i < t.getRank(); ++i)
      res.push_back(edsc::intrinsics::dim(v, i));
  }
  return res;
}

/// Returns the values obtained by applying `map` to the list of values.
/// When non-null, the optional pointer `folder` is used to call into the
/// `createAndFold` builder method. If `folder` is null, the regular `create`
/// method is called.
SmallVector<Value *, 4> applyMapToValues(OpBuilder &b, Location loc,
                                         AffineMap map,
                                         ArrayRef<Value *> values,
                                         OperationFolder *folder = nullptr);

struct TiledLinalgOp {
  LinalgOp op;
  SmallVector<loop::ForOp, 8> loops;
};

/// Performs standalone tiling of a single LinalgOp by `tileSizes`.
/// and permute the loop nest according to `permutation`
/// The permutation is expressed as a list of integers that specify
/// the new ordering of the loop nest. The length of `permutation`
/// must be equal to the length of `tileSizes`.
/// E.g. the permutation `(i,j,k) -> (j,k,i)` will be expressed with
/// `permutation = [1,2,0]`. All values in `permutation` must be
/// integers, in the range 0..`tileSizes.size()` without duplications
/// (i.e. `[1,1,2]` is an invalid permutation). An empty list
/// states for the identity permutation.
/// Returns a struct containing the tiled loops in the specified order
/// and the cloned op if successful, llvm::None otherwise.
/// When non-null, the optional pointer `folder` is used to call into the
/// `createAndFold` builder method. If `folder` is null, the regular `create`
/// method is called.
llvm::Optional<TiledLinalgOp> tileLinalgOp(OpBuilder &b, LinalgOp op,
                                           ArrayRef<Value *> tileSizes,
                                           ArrayRef<unsigned> permutation = {},
                                           OperationFolder *folder = nullptr);

/// Performs standalone tiling of a single LinalgOp by constant `tileSizes`.
/// and permute the loop nest according to `permutation`
/// The permutation is expressed as a list of integers that specify
/// the new ordering of the loop nest. The length of `permutation`
/// must be equal to the length of `tileSizes`.
/// E.g. the permutation `(i,j,k) -> (j,k,i)` will be expressed with
/// `permutation = [1,2,0]`. All values in `permutation` must be
/// integers, in the range 0..`tileSizes.size()` without duplications
/// (i.e. `[1,1,2]` is an invalid permutation). An empty list
/// states for the identity permutation.
/// Returns a struct containing the tiled loops in the specified order
/// and the cloned op if successful, llvm::None otherwise.
/// When non-null, the optional pointer `folder` is used to call into the
/// `createAndFold` builder method. If `folder` is null, the regular `create`
/// method is called.
llvm::Optional<TiledLinalgOp> tileLinalgOp(OpBuilder &b, LinalgOp op,
                                           ArrayRef<int64_t> tileSizes,
                                           ArrayRef<unsigned> permutation = {},
                                           OperationFolder *folder = nullptr);

template <typename... Args>
llvm::Optional<TiledLinalgOp> tileLinalgOperation(OpBuilder &b, Operation *op,
                                                  Args... args) {
  return tileLinalgOp(b, cast<LinalgOp>(op), args...);
}

struct PromotionInfo {
  Value *buffer;
  Value *fullLocalView;
  Value *partialLocalView;
};

/// Promotes the `subViews` into a new buffer allocated at the insertion point
/// `b`. For now, promotion occurs in 3 steps:
///   1. Create a new buffer for a full tile (i.e. not clipped at the boundary).
///   2. Take a full view on the buffer and `linalg.fill` it with zeros (use
///      float zero for now).
///   3. Take a partial slice of the full view in step 2. and copy into it.
/// Infers statically sized buffers from subViews unless `dynamicBuffers` is
/// true.
///
/// Returns a list of PromotionInfo which hold the promoted buffer and the
/// full and partial views indexing into the buffer.
llvm::SmallVector<PromotionInfo, 8>
promoteSubViews(OpBuilder &b, Location loc, ArrayRef<Value *> subViews,
                bool dynamicBuffers = false, OperationFolder *folder = nullptr);

/// Returns all the operands of `linalgOp` that are not views.
/// Asserts that these operands are value types to allow transformations like
/// tiling to just use the values when cloning `linalgOp`.
llvm::SmallVector<Value *, 4> getAssumedNonViewOperands(LinalgOp linalgOp);

} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_UTILS_H_
