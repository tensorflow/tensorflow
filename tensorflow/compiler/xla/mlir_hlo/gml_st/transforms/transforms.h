/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_TRANSFORMS_H
#define MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_TRANSFORMS_H

#include "gml_st/IR/gml_st_ops.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace linalg {
class LinalgOp;
struct TiledLinalgOp;
struct LinalgTilingOptions;
}  // namespace linalg
}  // namespace mlir

namespace mlir {
namespace gml_st {


bool isZero(Value v);
bool isOne(Value v);

/// Returns true if `candidate`'s offsets are all 0s and strides are all 1s.
bool isIdentityTileOp(TileOp candidate);

/// Returns true if `lhs` and `rhs` are of same static shape.
bool haveSameStaticShape(Value lhs, Value rhs);

/// Perform standalone tiling of a single LinalgOp by `tileSizes`.
/// An empty vector is interpreted as the identity permutation and the
/// transformation returns early.
///
/// Return a struct containing the tiled loops in the specified order
/// and the cloned op if successful, llvm::None otherwise.
FailureOr<linalg::TiledLinalgOp> tileLinalgOp(
    RewriterBase &b, linalg::LinalgOp op,
    const linalg::LinalgTilingOptions &options);

// Sets the attribute to the `op` that indicates that the op was transformed.
void setLabel(Operation *op, StringRef name);

// Removes the attribute that indicates that it was transformed.
void removeLabel(Operation *op, StringRef name);

// Checks if `op` has the attribute that indicates that it was transformed.
bool hasLabel(Operation *op, StringRef name);

// Checks if `op` has the matching label attribute.
bool hasMatchingLabel(Operation *op, StringRef label);

}  // namespace gml_st
}  // namespace mlir

#endif  // MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_TRANSFORMS_H
