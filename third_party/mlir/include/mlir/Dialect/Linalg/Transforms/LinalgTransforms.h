//===- LinalgTransforms.h - Linalg transformations as patterns --*- C++ -*-===//
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

#ifndef DIALECT_LINALG_TRANSFORMS_LINALGTRANSFORMS_H_
#define DIALECT_LINALG_TRANSFORMS_LINALGTRANSFORMS_H_

#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace linalg {

// Marker used as attribute name in generated Linalg rewriting transformations.
struct LinalgTransforms {
  static const StringLiteral kLinalgTransformMarker;
};

// Declarative transformation used in tablegen patterns.
// Tiles `op` by `sizes` and sets the attribute `kLinalgTransformMarker` to
// `linalgMarker`.
LogicalResult tileLinalgOpAndSetMarker(PatternRewriter &rewriter, Operation *op,
                                       ArrayRef<int64_t> sizes,
                                       StringRef linalgMarker);

// Declarative transformation used in tablegen patterns.
// Tiles `op` by `sizes`, fuses the producers of `operandIndicesToFuse` and sets
// the attribute `kLinalgTransformMarker` to `linalgMarker`.
LogicalResult tileAndFuseLinalgOpAndSetMarker(
    PatternRewriter &rewriter, Operation *op, ArrayRef<int64_t> sizes,
    ArrayRef<int64_t> operandIndicesToFuse, StringRef linalgMarker);

namespace detail {
// Implementation detail of isProducedByOpOfType avoids the need for explicit
// template instantiations.
bool isProducedByOpOfTypeImpl(Operation *consumerOp, Value *consumedView,
                              llvm::function_ref<bool(Operation *)> isaOpType);
} // namespace detail

// Returns true if the `consumedView` value use in `consumerOp` is produced by
// an op of type `OpTy`. This is used to implement use-def type information on
// buffers.
template <typename OpTy>
bool isProducedByOpOfType(Operation *consumerOp, Value *consumedView) {
  return detail::isProducedByOpOfTypeImpl(
      consumerOp, consumedView, [](Operation *op) { return isa<OpTy>(op); });
}

} // namespace linalg
} // namespace mlir

#endif // DIALECT_LINALG_TRANSFORMS_LINALGTRANSFORMS_H_
