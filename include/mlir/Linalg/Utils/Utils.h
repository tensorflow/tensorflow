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
  edsc::ValueHandle operator()(llvm::ArrayRef<edsc::CapturableHandle> stmts);

private:
  llvm::SmallVector<edsc::LoopBuilder, 4> loops;
};

} // namespace edsc

/// Abstracts away the extraction of values of RangeType from the actual op
/// implementation. For each operand of `op`:
///   1. If it is of RangeType, appends it to the result.
///   2. If it is of ViewType, further differentiates between:
///      a. Views that have a defining op, in which cases it appends the ranges
///         of the defining op.
///      b. Views that do not have a defining op, in which case it materializes
///         new range extraction ops to retrieve the range. This is not yet
///         implemented and depends on future operations (e.g. extract_range).
/// Precedence is given to a. over b. because it allows propagating existing
/// values instead of creating new, duplicate, values.
// TODO(ntv): Implement range extraction ops.
SmallVector<Value *, 8> getRanges(Operation *op);

/// Returns a value of ViewType at `b`, `loc` by applying the `ranges` to
/// `viewDefiningOp`. This creates a new op unless `viewDefiningOp` already has
/// the same exact `ranges`, in which case its (unique) result is returned.
Value *createOrReturnView(FuncBuilder *b, Location loc,
                          Operation *viewDefiningOp,
                          llvm::ArrayRef<Value *> ranges);

/// Returns the min/max/step from a RangeType value, depending on `part`:
///   1. If `range` comes from a range defining op, this just returns the proper
///      operand.
///   2. Otherwise (e.g. if range is a function parameter), it materializes new
///      part extraction ops to retrieve the min/max/step. This is not yet
///      implemented and depends on future operations (e.g. extract_min, ...).
/// Precedence is given to 1. over 2. because it allows propagating existing
/// values instead of creating new, duplicate, values.
/// This is used to abstract away the extraction of the min/max/step from a
/// RangeType value.
// TODO(ntv): Implement range extraction ops.
enum class RangePart { Min = 0, Max, Step };
Value *extractRangePart(Value *range, RangePart part);

} // namespace mlir

#endif // MLIR_LINALG_UTILS_H_
