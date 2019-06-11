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

#ifndef MLIR_LINALG_UTILS_H_
#define MLIR_LINALG_UTILS_H_

#include "mlir/EDSC/Helpers.h"
#include "mlir/Linalg/IR/LinalgOps.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class OperationFolder;
namespace edsc {

/// A LoopRangeBuilder is a generic NestedBuilder for linalg.for operations.
/// More specifically it is meant to be used as a temporary object for
/// representing any nested MLIR construct that is "related to" an mlir::Value*
/// (for now an induction variable).
class LoopRangeBuilder : public NestedBuilder {
public:
  /// Constructs a new linalg::ForOp and captures the associated induction
  /// variable. A ValueHandle pointer is passed as the first argument and is the
  /// *only* way to capture the loop induction variable.
  LoopRangeBuilder(ValueHandle *iv, ValueHandle range);
  LoopRangeBuilder(ValueHandle *iv, Value *range);

  LoopRangeBuilder(const LoopRangeBuilder &) = delete;
  LoopRangeBuilder(LoopRangeBuilder &&) = default;

  LoopRangeBuilder &operator=(const LoopRangeBuilder &) = delete;
  LoopRangeBuilder &operator=(LoopRangeBuilder &&) = default;

  /// The only purpose of this operator is to serve as a sequence point so that
  /// the evaluation of `fun` (which build IR snippets in a scoped fashion) is
  /// scoped within a LoopRangeBuilder.
  ValueHandle operator()(std::function<void(void)> fun = nullptr);
};

/// Helper class to sugar building linalg.for loop nests from ranges.
/// This is similar to edsc::LoopNestBuilder except it works on ranges directly.
/// In the current implementation it produces linalg.for operations.
class LoopNestRangeBuilder {
public:
  LoopNestRangeBuilder(llvm::ArrayRef<edsc::ValueHandle *> ivs,
                       llvm::ArrayRef<edsc::ValueHandle> ranges);
  LoopNestRangeBuilder(llvm::ArrayRef<edsc::ValueHandle *> ivs,
                       llvm::ArrayRef<Value *> ranges);
  edsc::ValueHandle operator()(std::function<void(void)> fun = nullptr);

private:
  llvm::SmallVector<LoopRangeBuilder, 4> loops;
};

} // namespace edsc

namespace linalg {

// Returns the linearized list of all view dimensions in a linalgOp. Applying
// the inverse, concatenated loopToOperandRangeMaps to this list allows the
// derivation of loop ranges for any linalgOp.
SmallVector<Value *, 8> getViewSizes(LinalgOp &linalgOp);

/// Returns the values obtained by applying `map` to the list of values.
/// Performs simplifications and foldings where possible.
SmallVector<Value *, 4> applyMapToValues(OpBuilder *b, Location loc,
                                         AffineMap map,
                                         ArrayRef<Value *> values,
                                         OperationFolder &state);

struct TiledLinalgOp {
  LinalgOp op;
  SmallVector<ForOp, 8> loops;
};

llvm::Optional<TiledLinalgOp>
tileLinalgOp(LinalgOp op, ArrayRef<Value *> tileSizes, OperationFolder &state);

llvm::Optional<TiledLinalgOp>
tileLinalgOp(LinalgOp op, ArrayRef<int64_t> tileSizes, OperationFolder &state);

} // namespace linalg
} // namespace mlir

#endif // MLIR_LINALG_UTILS_H_
