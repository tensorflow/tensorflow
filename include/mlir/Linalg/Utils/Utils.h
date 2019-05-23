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
#include "mlir/Support/LLVM.h"

namespace mlir {

namespace edsc {
/// Helper class to sugar building loop nests from ranges.
/// This is similar to edsc::LoopNestBuilder except it works on ranges directly.
/// In the current implementation it produces affine.for operations and thus
/// only admits ranges with constant steps.
class LoopNestRangeBuilder {
public:
  LoopNestRangeBuilder(llvm::ArrayRef<edsc::ValueHandle *> ivs,
                       llvm::ArrayRef<edsc::ValueHandle> ranges);
  LoopNestRangeBuilder(llvm::ArrayRef<edsc::ValueHandle *> ivs,
                       llvm::ArrayRef<Value *> ranges);
  edsc::ValueHandle operator()(std::function<void(void)> fun = nullptr);

private:
  llvm::SmallVector<edsc::LoopBuilder, 4> loops;
};

} // namespace edsc

namespace linalg {
class LinalgOp;

/// Helper class to memoize the creation of redundant constants within a given
/// function.
class FunctionConstants {
public:
  FunctionConstants(Function &f) : f(f) {}
  Value *getOrCreateIndex(int64_t v);

private:
  Function &f;
  llvm::SmallDenseMap<int64_t, Value *> map;
};

// Returns the linearized list of all view dimensions in a linalgOp. Applying
// the inverse, concatenated loopToOperandRangeMaps to this list allows the
// derivation of loop ranges for any linalgOp.
SmallVector<Value *, 8> getViewSizes(LinalgOp &linalgOp);

/// Returns the values obtained by applying `map` to the list of values.
/// Performs simplifications and foldings where possible.
SmallVector<Value *, 4> applyMapToValues(FuncBuilder *b, Location loc,
                                         AffineMap map,
                                         ArrayRef<Value *> values,
                                         FunctionConstants &state);

} // namespace linalg
} // namespace mlir

#endif // MLIR_LINALG_UTILS_H_
