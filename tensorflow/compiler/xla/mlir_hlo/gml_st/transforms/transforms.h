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

#ifndef MLIR_HLO_GML_ST_TRANSFORMS_TRANSFORMS_H
#define MLIR_HLO_GML_ST_TRANSFORMS_TRANSFORMS_H

#include "gml_st/IR/gml_st_ops.h"
#include "llvm/ADT/Hashing.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir {

class OpPassManager;

namespace linalg {

class LinalgOp;
struct TiledLinalgOp;
struct LinalgTilingOptions;

}  // namespace linalg
}  // namespace mlir

namespace mlir {
namespace gml_st {

constexpr llvm::StringRef kPerfectlyTiledLoopLabel =
    "__perfectly_tiled_loop_label__";

bool isZero(Value v);
bool isOne(Value v);

template <typename ShapedTy>
bool hasSingleElement(ShapedTy type) {
  return type.hasStaticShape() && type.getNumElements() == 1;
}
bool hasSingleElementOperandsAndResults(Operation *op);

/// Hoist vector.transfer_read/vector.transfer_write pairs out of immediately
/// enclosing gml_st::ForOp iteratively, if the following conditions are true:
///   1. The two ops access the same tensor with the same indices.
///   2. All operands are invariant under the enclosing gml_st::ForOp.
///   3. No uses of the tensor either dominate the transfer_read or are
///   dominated by the transfer_write (i.e. no aliasing between the write and
///   the read across the loop)
/// The transformation follows this logic:
///   1. Look for transfer_write with a single use from ForOp terminator
///   2. Check the uses of the matching block argument and look for a
///   transfer_read with the same indices.
///   3. Check that all the other uses of the tensor argument are either
///   disjoint tensor_read or transfer_write. For transfer_write uses recurse to
///   make sure the new tensor has the same restrictions on its uses.
///   4. Hoist the tensor_read/tensor_write and update the tensor SSA links.
///
/// Example:
///   %for = gml_st.for ... outs (%arg6 = %out: tensor<8x4xf32>) {
///     %tile = gml_st.tile [0, 0] [8, 4] [1, 1] : !gml_st.tile<8x4>
///     ...
///     %read = vector.transfer_read %arg6[%c0, %c0]
///     %compute = foo(%read) : vector<8x4xf32>
///     %write = vector.transfer_write %compute, %arg6[%c0, %c0]
///     gml_st.set_yield %write into %arg6[%tile]
///   } : tensor<8x4xf32>
///
///   will be transformed into:
///
///   %read = vector.transfer_read %out[%c0, %c0]
///   %for = gml_st.for ... outs (%arg6 = %read: vector<8x4xf32>) {
///     %tile = gml_st.tile [0, 0] [8, 4] [1, 1] : !gml_st.tile<8x4>
///     ...
///     %compute = foo(%read) : vector<8x4xf32>
///     gml_st.set_yield %compute into %arg6[%tile]
///   } : vector<8x4xf32>
///   %write = vector.transfer_write %for, %out[%c0, %c0]
///
/// After this transformation the gml_st.ForOp may have unused arguments that
/// can be remove by the canonicalization pass.
void hoistRedundantVectorTransfersOnTensor(func::FuncOp func);

/// Returns true if `candidate`'s offsets are all 0s and strides are all 1s.
bool isIdentitySlice(ValueRange offsets, ValueRange strides);

/// Returns true if `lhs` and `rhs` are of same static shape.
bool haveSameStaticShape(Value lhs, Value rhs);

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

#endif  // MLIR_HLO_GML_ST_TRANSFORMS_TRANSFORMS_H
