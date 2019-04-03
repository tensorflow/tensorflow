//===- TensorOps.h - Linalg dialect TensorOps operation definition --------===//
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

#ifndef LINALG3_TENSOROPS_H_
#define LINALG3_TENSOROPS_H_

#include "linalg2/TensorOps.h"

namespace linalg {

///
/// Ideally all these functions would go in an Analysis but until
/// TensorContractionBase is templated, they need to remain close enough.
///

/// Takes a `tensorContraction` and a returns an AffineMap that can be used to
/// map ranges to enclosing loops for all the operands' ranges.
template <class ConcreteOp>
mlir::AffineMap operandRangesToLoopsMap(
    linalg::TensorContractionBase<ConcreteOp> &tensorContraction);

/// Takes a `tensorContraction` and returns the ranges of all its operands.
/// When an operand comes from a ViewOp, things are simple:
///   just traverse the indexings and get all the ranges
///     (i.e. drop the rank-reducing indices).
/// In the case of a SliceOp, things are more involved because we need to handle
/// potential rank-reductions.
/// This function abstracts this complexity away and returns all the ranges.
template <class ConcreteOp>
llvm::SmallVector<mlir::Value *, 8>
getRanges(linalg::TensorContractionBase<ConcreteOp> &tensorContraction);

} // namespace linalg

/// The TensorOp-inl.h inclusion pattern is chosen to allow gradual extension of
/// TensorOps by adding implementations as they are needed in the appropriate
/// step in the tutorial.
#include "linalg3/TensorOps-inl.h"

#endif // LINALG3_TENSOROPS_H_
